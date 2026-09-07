import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import scanpy as sc
import pandas as pd

def get_metamarkers_pred(query_h5ad, ref_h5ad):
    # # Set the R environment path
    # os.environ['R_HOME'] = '/home/arvin/miniconda3/envs/renv/lib/R/'

    pandas2ri.activate()

    # query_adata = sc.read_h5ad("/inkwell05/arvin/Anirban/concatenated.h5ad")
    # ref_adata = sc.read_h5ad("/inkwell05/arvin/Zhuang/Zhuang4/Zhuang-ABCA-4-raw-metadata_and_genes.h5ad")
    #
    # # Optional data reduction
    # cut_data = True
    # if cut_data:
    #     np.random.seed(42)
    #     factor = 3000
    #     query_adata = query_adata[np.random.permutation(query_adata.shape[0])[:query_adata.shape[0] // factor]].copy()
    #     ref_adata = ref_adata[np.random.permutation(ref_adata.shape[0])[:ref_adata.shape[0] // factor]].copy()
    #
    # print(query_adata)
    # print(ref_adata)

    ro.r(f'''
        library(MetaMarkers)
        library(SingleCellExperiment)
        library(zellkonverter)
        library(dplyr)
        
        query_h5ad <- {query_h5ad}
        query_sce <- readH5AD(query_h5ad)
        ref_h5ad <- {ref_h5ad}
        ref_sce <- readH5AD(ref_h5ad)
        
        # Function to sample % of cells randomly
        sample_percent <- function(sce) {{
          # Total number of cells
          total_cells <- ncol(sce)
          
          # Number of cells to sample
          sample_size <- round(0.005 * total_cells)
          
          # Randomly sample column indices
          sampled_indices <- sample(seq_len(total_cells), size = sample_size)
          
          # Subset the SingleCellExperiment object
          return(sce[, sampled_indices])
        }}
        
        cut_data <- FALSE
        if (cut_data) {{
          query_sce <- sample_percent(query_sce)
          ref_sce <- sample_percent(ref_sce)
        }}
        
        # Step 1: Normalize data
        cpm(reference_sce) = convert_to_cpm(assay(reference_sce, "X"))
        cpm(query_sce) = convert_to_cpm(assay(query_sce, "X"))
        
        # Step 2: Compute markers for reference dataset
        markers = compute_markers(cpm(reference_sce), reference_sce$class)
        meta_markers = make_meta_markers(markers, common_genes_only = FALSE)
        
        # Step 3: Select top markers (e.g., top 100)
        top_markers = filter(meta_markers, rank <= 100)
        
        print(top_markers)
        
        # Step 4: Score and annotate query cells
        ct_scores = score_cells(log1p(cpm(query_sce)), top_markers)
        ct_enrichment = compute_marker_enrichment(ct_scores)
        ct_pred = assign_cells(ct_scores)
        
        # Optional: Set a threshold for high-confidence annotations
        threshold = 2.0  # Adjust based on your data
        high_conf_pred = ct_pred[ct_pred$enrichment > threshold, ]
        
        print(high_conf_pred)
    ''')

    annotated_cells = ro.r('high_conf_pred')

    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        annotated_df = ro.conversion.rpy2py(annotated_cells)

    return annotated_df

def get_adata_from_pred(query_h5ad, annotated_df):

    adata = sc.read_h5ad(query_h5ad)

    obs_df = pd.DataFrame(adata.obs_names.values, columns=["obs_name"])

    annotated_df.reset_index(inplace=True)
    annotated_df.rename(columns={'index': 'obs_name'}, inplace=True)
    annotated_df['obs_name'] = annotated_df['obs_name'].astype(str)

    merged_df = obs_df.merge(annotated_df[["obs_name", "predicted"]], on="obs_name", how="left")

    adata.obs["meta_marker_annotation"] = merged_df["predicted"].values

    # print(f"For file: {query_h5ad}")
    # print(adata)
    # print(adata.obs["meta_marker_annotation"])
    # print("Sum of unassigned:")
    # print((adata.obs["meta_marker_annotation"] == "unassigned").sum())
    # print()

    return adata

def main():
    annotated_df = get_metamarkers_pred("/inkwell05/arvin/Anirban/SM391_P684_F.h5ad", "/inkwell05/arvin/Zhuang/Zhuang4/Zhuang-ABCA-4-raw-metadata_and_genes.h5ad")
    adata = get_metamarkers_pred("/inkwell05/arvin/Anirban/SM391_P684_F.h5ad", annotated_df)
    print(adata)

if __name__ == '__main__':
    main()
