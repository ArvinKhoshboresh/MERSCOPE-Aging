from __future__ import annotations

import os

THREADS = min(16, max(1, (os.cpu_count() or 2) - 1))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(variable, str(THREADS))

from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy import sparse

from CombinedAdataUMAPQC import (
    CELL_TYPE_COLORS,
    DPI,
    fallback_colors,
    ordered_categories,
    plot_umap,
    prepare_embedding,
)

INPUT_DIR = Path("/inkwell05/arvin/Anirban/FilteredCellAnnotations/")
ANNOTATED_SUBDIR = "Subclass_Annotated"
QC_DIR = INPUT_DIR / ANNOTATED_SUBDIR / "UMAP_QC"
REUSE_CHECKPOINTS = True

BROAD_COL = "meta_marker_annotation"
SUBCLASS_COL = "meta_marker_annotation_subclass"
NOT_OF_INTEREST_LABEL = "N/A"
BROAD_TYPES_OF_INTEREST = ["Vascular", "Immune"]

BROAD_COLORS = {
    "Vascular": CELL_TYPE_COLORS["Vascular"],
    "Immune": CELL_TYPE_COLORS["Immune"],
    NOT_OF_INTEREST_LABEL: "#D9D9D9",
}

AGE_ANNOTATION = {
    "Anterior-SM391_393_Region0-regions_SM393_p74_F.h5ad": "Yng",
    "Anterior-SM391_393_Region1-regions_SM391_p684_F.h5ad": "Old",
    "Anterior-SM395_396_Region1-regions_SM395_P78_M.h5ad": "Yng",
    "Anterior-SM395_396_Region0-SM396_p623_M.h5ad": "Old",
    "Anterior-SM408_414_Region0-regions_SM414_P72_M.h5ad": "Yng",
    "Anterior-SM408_414_Region1-regions_SM408_P662_M.h5ad": "Old",
    "Posterior-SM391_393_Region1-SM391-393_Region1_cleared.h5ad": "Yng",
    "Posterior-SM391_393_Region0-SM391_393_Region0.h5ad": "Old",
    "Posterior-SM395_396_Region1-SM395_396_Region1.h5ad": "Old",
    "Posterior-SM395_396_Region0-SM395-396_Region0_cleared.h5ad": "Yng",
    "Posterior-SM408_414_Region0-SM408_414_Region0_cleared.h5ad": "Old",
    "Posterior-SM408_414_Region1-regions_SM408_414_Region1_cleared.h5ad": "Yng",
}
AGE_ORDER = ["Yng", "Old"]


def load_combined_adata():
    directory = INPUT_DIR / ANNOTATED_SUBDIR
    files = sorted(p for p in directory.glob("*.h5ad") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No annotated .h5ad files found in {directory}")

    prepared = []
    keys = []
    for path in files:
        adata = sc.read_h5ad(path)
        for column in (BROAD_COL, SUBCLASS_COL):
            if column not in adata.obs:
                raise KeyError(f"{path.name} is missing obs column {column!r}")
        if path.name not in AGE_ANNOTATION:
            raise KeyError(f"{path.name} is missing from AGE_ANNOTATION")
        adata.obs["sample"] = path.stem
        adata.obs["age"] = AGE_ANNOTATION[path.name]
        prepared.append(adata)
        keys.append(path.stem)

    combined = ad.concat(prepared, join="inner", merge="same", label="sample_key", keys=keys, index_unique="-")
    combined.obs[BROAD_COL] = combined.obs[BROAD_COL].astype(str)
    combined.obs[SUBCLASS_COL] = combined.obs[SUBCLASS_COL].astype(str)
    print(f"Loaded {combined.n_obs:,} cells across {len(files)} sample(s) and {combined.n_vars:,} shared genes")
    return combined


def compute_and_checkpoint(adata, checkpoint_path, reuse=REUSE_CHECKPOINTS):
    if reuse and checkpoint_path.exists():
        checkpoint = sc.read_h5ad(checkpoint_path)
        if "age" in checkpoint.obs:
            print(f"Loading checkpoint: {checkpoint_path.resolve()}")
            return checkpoint
        print(f"Rebuilding checkpoint without age metadata: {checkpoint_path.resolve()}")

    working = adata.copy()
    prepare_embedding(working)

    checkpoint = ad.AnnData(X=sparse.csr_matrix((working.n_obs, 0), dtype=np.float32), obs=working.obs.copy())
    checkpoint.obsm["X_umap"] = np.asarray(working.obsm["X_umap"], dtype=np.float32)
    checkpoint.uns["source_n_vars"] = int(working.n_vars)
    checkpoint.write_h5ad(checkpoint_path, compression="gzip")
    return checkpoint


def plot_fine_type_composition(counts_by_age_and_broad, colors_by_broad, output_stem):
    broads = list(colors_by_broad)
    ages = [age for age in AGE_ORDER if age in counts_by_age_and_broad]
    fig, axes = plt.subplots(len(ages), len(broads), figsize=(4.4 * len(broads), 8.8), squeeze=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.65, top=0.90)

    for row, age in enumerate(ages):
        for column, broad in enumerate(broads):
            ax = axes[row, column]
            counts = counts_by_age_and_broad[age].get(broad)
            if counts is None or counts.empty:
                ax.set_axis_off()
                continue

            counts = counts.sort_values(ascending=False)
            total = int(counts.sum())
            bottom = 0
            for label, value in counts.items():
                color = colors_by_broad[broad].get(label, "#999999")
                ax.bar([0], [value], bottom=bottom, color=color, width=0.18, label=label)
                fraction = value / total
                if fraction >= 0.04:
                    ax.text(0, bottom + value / 2, f"{fraction:.1%}", ha="center", va="center", fontsize=8)
                bottom += value

            ax.set_title(f"{broad} fine cell type composition", loc="left", pad=28, fontsize=11, fontweight="bold")
            ax.text(0, 1.035, f"{age}: {total:,} cells", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="0.35")
            ax.set_ylabel("Cells")
            ax.set_xlim(-0.11, 0.89)
            ax.set_xticks([])
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.grid(False)
            ax.legend(title="Fine cell type", bbox_to_anchor=(0.25, 1), loc="upper left", frameon=False, fontsize=7, title_fontsize=8)

    fig.savefig(output_stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 13, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.transparent": False,
    })
    QC_DIR.mkdir(parents=True, exist_ok=True)

    combined = load_combined_adata()

    # Plot all cells, colored by broad cell type of interest (Vascular / Immune / N/A).
    full_embedded = compute_and_checkpoint(combined, QC_DIR / "broad_umap_checkpoint.h5ad")
    plot_umap(
        full_embedded, BROAD_COL, "Combined landscape by broad cell type of interest",
        QC_DIR / "01_combined_umap_broad_type",
        preferred_order=BROAD_TYPES_OF_INTEREST + [NOT_OF_INTEREST_LABEL],
        fixed_colors=BROAD_COLORS,
    )

    # Plots one dedicated UMAP per broad type, colored by fine cell type.
    composition_counts = {}
    composition_colors = {}
    next_index = 2
    for broad in BROAD_TYPES_OF_INTEREST:
        subset = combined[combined.obs[BROAD_COL] == broad].copy()
        if subset.n_obs == 0:
            print(f"No {broad} cells found, skipping its UMAP")
            continue

        subset_embedded = compute_and_checkpoint(subset, QC_DIR / f"{broad.lower()}_umap_checkpoint.h5ad")

        categories = ordered_categories(subset_embedded.obs[SUBCLASS_COL])
        colors = fallback_colors(categories)
        plot_umap(
            subset_embedded, SUBCLASS_COL, f"{broad} cells by fine cell type",
            QC_DIR / f"{next_index:02d}_{broad.lower()}_umap_fine_type",
            preferred_order=categories, fixed_colors=colors,
        )
        for age in AGE_ORDER:
            age_counts = subset_embedded.obs.loc[subset_embedded.obs["age"] == age, SUBCLASS_COL].value_counts()
            composition_counts.setdefault(age, {})[broad] = age_counts
        composition_colors[broad] = colors
        next_index += 1

    if composition_counts:
        plot_fine_type_composition(
            composition_counts, composition_colors,
            QC_DIR / f"{next_index:02d}_fine_type_composition",
        )

    print(f"\nSaved checkpoints and figures to: {QC_DIR.resolve()}")


if __name__ == "__main__":
    main()
