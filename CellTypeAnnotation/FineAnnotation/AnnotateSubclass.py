from __future__ import annotations
from pathlib import Path
# import os
# os.environ['R_HOME'] = '/home/arvin/miniconda3/envs/renv/lib/R/'

import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri

from Wrapper import get_adata_from_pred

INPUT_DIR = Path("../Adatas/")
OUTPUT_SUBDIR = "Subclass_Annotated"
MARKERS_CSV = Path(
    "../Adatas/Zeng_markers_with_gene_symbols_subclass.csv")

BROAD_COL = "meta_marker_annotation"
SUBCLASS_COL = "meta_marker_annotation_subclass"
NOT_OF_INTEREST_LABEL = "N/A"
UNASSIGNED_LABEL = "Unassigned"
ENRICHMENT_THRESHOLD = 0.0
N_TOP_GENES = 50  # top marker genes per fine cell type, ranked by AUROC

# Broad cell type -> fine cell types to resolve it into.
CELL_TYPES_OF_INTEREST = {
    "Vascular": ["Endo NN", "Peri NN", "SMC NN", "VLMC NN", "ABC NN"],
    "Immune": ["Microglia NN", "BAM NN", "Monocytes NN", "DC NN", "Lymphoid NN"],
}


def build_fine_marker_csvs(markers_csv, cell_types_of_interest, out_dir):
    """Split the precomputed marker table into one CSV per broad cell type,
    restricted to that broad type's fine cell types.

    This marker table already has both `gene` (Ensembl ID) and `gene_symbol`
    columns plus a `cell_type` column, matching the format FromComputedMarkers.py
    expects, so we just filter rows and keep every column as-is; the gene/
    gene_symbol swap happens in run_metamarkers_annotation, same as there.

    MetaMarkers itself expects a `group` column (it builds an internal
    `group|cell_type` key), which this file doesn't have. Add a constant
    placeholder if it's missing, matching the "all" value the older
    Ensembl-only marker file used.
    """
    markers = pd.read_csv(markers_csv)
    if "group" not in markers.columns:
        markers["group"] = "all"
    marker_paths = {}
    for broad, fine_types in cell_types_of_interest.items():
        subset = markers[markers["cell_type"].isin(fine_types)].copy()
        if subset.empty:
            raise ValueError(f"No marker rows found for {broad!r} among {fine_types}")
        subset_path = out_dir / f"markers_{broad}.csv"
        subset.to_csv(subset_path, index=False)
        marker_paths[broad] = subset_path
    return marker_paths


def run_metamarkers_annotation(query_h5ad, marker_csv, n_top_genes, threshold=ENRICHMENT_THRESHOLD):
    r_code = f'''
        library(MetaMarkers)
        library(SingleCellExperiment)
        library(zellkonverter)
        library(dplyr)

        query_sce <- readH5AD("{query_h5ad}", reader = "R")
        cpm(query_sce) = convert_to_cpm(assay(query_sce, "X"))

        markers <- read.csv("{marker_csv}")
        colnames(markers)[colnames(markers) == "gene"] <- "temp_column"
        colnames(markers)[colnames(markers) == "gene_symbol"] <- "gene"
        colnames(markers)[colnames(markers) == "temp_column"] <- "gene_symbol"

        top_markers <- markers %>%
          group_by(cell_type) %>%
          arrange(desc(auroc), .by_group = TRUE) %>%
          slice_max(auroc, n = {n_top_genes})

        ct_scores <- score_cells(log1p(cpm(query_sce)), top_markers)
        ct_enrichment <- compute_marker_enrichment(ct_scores)
        ct_pred <- assign_cells(ct_scores)

        if ("cell_type" %in% colnames(ct_pred)) {{
          ct_pred$cell_type <- as.character(ct_pred$cell_type)
        }}

        threshold <- {threshold}
        high_conf_pred <- ct_pred[ct_pred$enrichment > threshold, ]
    '''
    try:
        ro.r(r_code)
        annotated_cells = ro.r("high_conf_pred")
    except RRuntimeError as exc:
        print(f"R error while annotating {query_h5ad}: {exc}")
        raise

    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        annotated_df = ro.conversion.rpy2py(annotated_cells)
    return annotated_df


def strip_nn_suffix(series):
    return series.astype(str).str.replace(r"\s*NN$", "", regex=True)


def annotate_adata(adata, sample_name, marker_paths, tmp_dir, output_dir=None):
    """Annotate one already-processed AnnData object and return it."""
    print(f"\n=== Annotating {sample_name} ===")
    adata = adata.copy()
    if BROAD_COL not in adata.obs:
        raise KeyError(f"{sample_name} is missing obs column {BROAD_COL!r}")

    adata.obs[SUBCLASS_COL] = NOT_OF_INTEREST_LABEL
    sample_stem = Path(sample_name).stem
    for broad, fine_types in CELL_TYPES_OF_INTEREST.items():
        mask = adata.obs[BROAD_COL].astype(str) == broad
        n_cells = int(mask.sum())
        if n_cells == 0:
            print(f"  {broad}: no cells found, skipping")
            continue
        print(f"  {broad}: annotating {n_cells:,} cells into {fine_types}")

        subset = adata[mask].copy()
        subset_path = tmp_dir / f"{sample_stem}_{broad}_subset.h5ad"
        subset.write_h5ad(subset_path)
        annotated_df = run_metamarkers_annotation(subset_path, marker_paths[broad], N_TOP_GENES)
        subset_annotated = get_adata_from_pred(subset_path, annotated_df, obs_key=SUBCLASS_COL)
        predicted = subset_annotated.obs[SUBCLASS_COL].where(
            subset_annotated.obs[SUBCLASS_COL].notna(), UNASSIGNED_LABEL)
        adata.obs.loc[predicted.index, SUBCLASS_COL] = predicted.values
        subset_path.unlink(missing_ok=True)

    adata.obs[SUBCLASS_COL] = strip_nn_suffix(adata.obs[SUBCLASS_COL])
    print(f"  First 20 fine cell types: {adata.obs[SUBCLASS_COL].head(20).tolist()}")
    if output_dir is not None:
        output_path = output_dir / sample_name
        adata.write_h5ad(output_path, compression="gzip")
        print(f"  Saved: {output_path}")
    return adata


def annotate_processed_data(processed_data, save_outputs=True):
    """Annotate the in-memory AnnData objects returned by AdataProcessingNoPack."""
    names = processed_data["names"]
    adatas = processed_data["adatas"]
    if len(names) != len(adatas):
        raise ValueError("processed_data['names'] and processed_data['adatas'] must have equal lengths")

    output_dir = INPUT_DIR / OUTPUT_SUBDIR
    tmp_dir = output_dir / "_tmp_subsets"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    marker_paths = build_fine_marker_csvs(MARKERS_CSV, CELL_TYPES_OF_INTEREST, tmp_dir)
    destination = output_dir if save_outputs else None
    annotated_adatas = [
        annotate_adata(adata, name, marker_paths, tmp_dir, destination)
        for name, adata in zip(names, adatas)
    ]
    result = processed_data.copy()
    result["adatas"] = annotated_adatas
    return result


def main(processed_data=None, save_outputs=True):
    if processed_data is None:
        from AdataProcessingNoPack import processed_data as loaded_processed_data
        processed_data = loaded_processed_data

    annotated_data = annotate_processed_data(processed_data, save_outputs=save_outputs)
    if save_outputs:
        print(f"\nDone. Annotated files saved to: {(INPUT_DIR / OUTPUT_SUBDIR).resolve()}")
    return annotated_data


if __name__ == "__main__":
    main()