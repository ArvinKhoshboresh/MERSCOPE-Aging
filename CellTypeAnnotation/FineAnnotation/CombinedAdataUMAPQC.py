from __future__ import annotations

import os

THREADS = min(16, max(1, (os.cpu_count() or 2) - 1))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(variable, str(THREADS))

from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

CELL_TYPE_COL = "meta_marker_annotation"
UNASSIGNED_LABEL = "Unassigned"
UNASSIGNED_COL = "assignment_status"
QC_DIR = Path("QC")
CHECKPOINT_FILE = QC_DIR / "combined_umap_styling_checkpoint.h5ad"
REUSE_CHECKPOINT = True
RANDOM_SEED = 42
DPI = 600
N_NEIGHBORS = 30
N_PCS = 40
MAX_HVG = 3000
UMAP_MIN_DIST = 0.3
UMAP_SPREAD = 1.0
AXIS_MARGIN_FRACTION = 0.02


CELL_TYPE_COLORS = {
    "IT-ET Glut": "#FA0087", "NP-CT-L6b Glut": "#61E2A4", "OB-CR Glut": "#D00000",
    "DG-IMN Glut": "#16F2F2", "OB-IMN GABA": "#1B4332", "CTX-CGE GABA": "#CCFF33",
    "CTX-MGE GABA": "#F954EE", "CNU GABA": "#70E000", "LSX GABA": "#3283FE",
    "CNU-HYa GABA": "#450099", "HY GABA": "#FF6600", "CNU-HYa Glut": "#90E0EF",
    "HY Glut": "#AA0DFE", "HY Gnrh1 Glut": "#F28266", "HY MM Glut": "#01D669",
    "MH-LH Glut": "#FAA307", "TH Glut": "#0D47A1", "MB Glut": "#007200",
    "MB GABA": "#9EF01A", "MB Dopa": "#38B000", "MB-HB Sero": "#EC4067",
    "P Glut": "#6B5CA5", "MY Glut": "#F0A0FF", "Pineal Glut": "#086375",
    "P GABA": "#72195A", "MY GABA": "#0096C7", "CB GABA": "#FFFB46",
    "CB Glut": "#9D0208", "Astro-Epen": "#594A26", "OPC-Oligo": "#03045E",
    "OEC": "#996B2E", "Vascular": "#858881", "Immune": "#825F45",
}
AGE_COLORS = {"Yng": "#0072B2", "Old": "#D55E00"}
AREA_COLORS = {"Anterior": "#009E73", "Posterior": "#CC79A7"}


def combine_adatas(data):
    adatas, names, ages, areas = data["adatas"], data["names"], data["ages"], data["areas"]
    prepared = []
    keys = []
    for adata, name, age, area in zip(adatas, names, ages, areas, strict=True):
        if CELL_TYPE_COL not in adata.obs:
            raise KeyError(f"{name!r} is missing obs column {CELL_TYPE_COL!r}")
        current = adata.copy()
        current.obs["sample"] = str(name)
        current.obs["age"] = age
        current.obs["area"] = area
        prepared.append(current)
        keys.append(str(name))
    combined = ad.concat(prepared, join="inner", merge="same", label="sample_key", keys=keys, index_unique="-")
    combined.obs[CELL_TYPE_COL] = combined.obs[CELL_TYPE_COL].astype(str)
    return combined


def prepare_embedding(adata):
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(MAX_HVG, adata.n_vars), flavor="seurat", subset=False)
    n_hvg = int(adata.var["highly_variable"].sum())
    n_components = min(N_PCS, adata.n_obs - 1, n_hvg - 1)
    if n_components < 2:
        raise ValueError("At least three cells and three highly variable genes are required")
    sc.pp.pca(adata, n_comps=n_components, use_highly_variable=True, svd_solver="randomized", random_state=RANDOM_SEED)
    sc.pp.neighbors(adata, n_neighbors=min(N_NEIGHBORS, adata.n_obs - 1), n_pcs=n_components,
                    method="umap", transformer="pynndescent", random_state=RANDOM_SEED)
    sc.tl.umap(adata, min_dist=UMAP_MIN_DIST, spread=UMAP_SPREAD, random_state=RANDOM_SEED)


def save_styling_checkpoint(adata):
    checkpoint = ad.AnnData(X=sparse.csr_matrix((adata.n_obs, 0), dtype=np.float32), obs=adata.obs.copy())
    checkpoint.obsm["X_umap"] = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    checkpoint.uns.update({
        "source_n_vars": int(adata.n_vars),
        "embedding_method": "Scanpy CPU on native Windows",
        "log1p_applied": True,
        "normalized_in_this_script": False,
        "n_neighbors": N_NEIGHBORS,
        "n_pcs": N_PCS,
        "max_hvg": MAX_HVG,
        "random_seed": RANDOM_SEED,
    })
    checkpoint.write_h5ad(CHECKPOINT_FILE, compression="gzip")
    return checkpoint


def load_or_compute_embedding():
    if REUSE_CHECKPOINT and CHECKPOINT_FILE.exists():
        print(f"Loading styling checkpoint: {CHECKPOINT_FILE.resolve()}")
        return sc.read_h5ad(CHECKPOINT_FILE)

    from AdataProcessingNoPack import processed_data

    combined = combine_adatas(processed_data)
    print(f"Combined {combined.n_obs:,} cells across {combined.n_vars:,} shared genes")
    prepare_embedding(combined)
    return save_styling_checkpoint(combined)


def point_size(n_cells):
    return float(np.clip(50000 / max(n_cells, 1), 0.5, 15.0))


def ordered_categories(values, preferred_order=None):
    present = set(pd.Series(values).dropna().astype(str))
    if preferred_order is None:
        return sorted(present)
    return [value for value in preferred_order if value in present] + sorted(present - set(preferred_order))


def fallback_colors(categories):
    cmap = plt.get_cmap("turbo")
    positions = np.linspace(0.08, 0.92, max(len(categories), 2))
    return {category: mpl.colors.to_hex(cmap(position)) for category, position in zip(categories, positions)}


def coordinate_limits(coordinates, margin_fraction=AXIS_MARGIN_FRACTION):
    x_min, y_min = np.nanmin(coordinates, axis=0)
    x_max, y_max = np.nanmax(coordinates, axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_margin = max(x_range * margin_fraction, np.finfo(float).eps)
    y_margin = max(y_range * margin_fraction, np.finfo(float).eps)

    return (
        (x_min - x_margin, x_max + x_margin),
        (y_min - y_margin, y_max + y_margin),
    )


def plot_umap(adata, column, title, output_stem, preferred_order=None, fixed_colors=None, legend_columns=1):
    coordinates = np.asarray(adata.obsm["X_umap"])
    categories = ordered_categories(adata.obs[column], preferred_order)
    colors = fallback_colors(categories)
    if fixed_colors:
        colors.update({category: fixed_colors[category] for category in categories if category in fixed_colors})

    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    draw_order = np.random.default_rng(RANDOM_SEED).permutation(adata.n_obs)
    labels = adata.obs[column].astype(str).to_numpy()
    ax.scatter(coordinates[draw_order, 0], coordinates[draw_order, 1], c=[colors[label] for label in labels[draw_order]],
               s=point_size(adata.n_obs), linewidths=0, alpha=0.6, rasterized=False)
    source_n_vars = int(adata.uns.get("source_n_vars", adata.n_vars))
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
    ax.text(0, 1.005, f"{adata.n_obs:,} cells | {source_n_vars:,} shared genes", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="0.35")

    x_limits, y_limits = coordinate_limits(coordinates)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.grid(False)

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=5.5, markerfacecolor=colors[category],
                      markeredgecolor="none", label=category) for category in categories]
    legend = ax.legend(handles=handles, title=column.replace("_", " ").title(), frameon=False,
                       bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, ncol=legend_columns,
                       handletextpad=0.5, columnspacing=1.0, labelspacing=0.45)
    legend.get_title().set_fontweight("bold")

    fig.savefig(output_stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_metadata_summary(adata):
    rows = []
    for column in (CELL_TYPE_COL, "age", "area", "sample"):
        counts = adata.obs[column].astype(str).value_counts()
        rows.extend({"variable": column, "label": label, "n_cells": int(count)} for label, count in counts.items())
    pd.DataFrame(rows).to_csv(QC_DIR / "combined_umap_category_counts.csv", index=False)


def main():
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.titlesize": 13, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.transparent": False,
    })
    QC_DIR.mkdir(parents=True, exist_ok=True)
    combined = load_or_compute_embedding()
    cell_types = combined.obs[CELL_TYPE_COL].astype(str)
    unassigned_mask = cell_types.str.casefold().eq(UNASSIGNED_LABEL.casefold())

    assigned = combined[~unassigned_mask].copy()
    plot_umap(assigned, CELL_TYPE_COL, "Combined transcriptomic landscape by cell type",
              QC_DIR / "01_combined_umap_cell_type_assigned", list(CELL_TYPE_COLORS), CELL_TYPE_COLORS)

    combined.obs[UNASSIGNED_COL] = np.where(unassigned_mask, UNASSIGNED_LABEL, "Other cells")
    assignment_colors = {UNASSIGNED_LABEL: "#404040", "Other cells": "#D9D9D9"}
    plot_umap(combined, UNASSIGNED_COL, "Unassigned cells in the combined transcriptomic landscape",
              QC_DIR / "02_combined_umap_unassigned", ["Other cells", UNASSIGNED_LABEL], assignment_colors)

    plot_umap(combined, "age", "Combined transcriptomic landscape by age",
              QC_DIR / "03_combined_umap_age", ["Yng", "Old"], AGE_COLORS)
    plot_umap(combined, "area", "Combined transcriptomic landscape by anatomical area",
              QC_DIR / "04_combined_umap_area", ["Anterior", "Posterior"], AREA_COLORS)
    save_metadata_summary(combined)
    print(f"Saved checkpoint, figures, and category counts to: {QC_DIR.resolve()}")


if __name__ == "__main__":
    main()
