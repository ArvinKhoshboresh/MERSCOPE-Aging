import os
from Wrapper import get_adata_from_pred
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.rinterface_lib.embedded import RRuntimeError
import sys

def get_metamarkers_pred_from_precomputed_markers(query_h5ad, ref_markers, n_top_genes):

    # Set the R environment path
    # os.environ['R_HOME'] = '/home/arvin/miniconda3/envs/renv/lib/R/'

    pandas2ri.activate()

    r_code = f'''
        library(MetaMarkers)
        library(SingleCellExperiment)
        library(zellkonverter)
        library(dplyr)
    
        query_h5ad <- "{query_h5ad}"
        query_sce <- readH5AD(query_h5ad)
        print(query_sce)
    
        # Function to sample % of cells randomly
        sample_percent <- function(sce) {{
          total_cells <- ncol(sce)
          sample_size <- round(0.01 * total_cells)
          sampled_indices <- sample(seq_len(total_cells), size = sample_size)
          return(sce[, sampled_indices])
        }}
    
        cut_data <- FALSE
        if (cut_data) {{
          query_sce <- sample_percent(query_sce)
        }}
    
        cpm(query_sce) = convert_to_cpm(assay(query_sce, "X"))
    
        markers <- read.csv("{ref_markers}")
        colnames(markers)[colnames(markers) == "gene"] <- "temp_column"
        colnames(markers)[colnames(markers) == "gene_symbol"] <- "gene"
        colnames(markers)[colnames(markers) == "temp_column"] <- "gene_symbol"
    
        top_markers = markers %>% group_by(cell_type) %>% arrange(desc(auroc), .by_group = T) %>% slice_max(auroc, n = {n_top_genes})

        #From the MetaMarker annotation vignette, gives you an annotation prediction for each cell
        ct_scores = score_cells(log1p(cpm(query_sce)), top_markers)
        ct_enrichment = compute_marker_enrichment(ct_scores)
        ct_pred = assign_cells(ct_scores)
    
        # Ensure that the column used in assign_cells or similar functions is a character vector
        if ("cell_type" %in% colnames(ct_pred)) {{
          ct_pred$cell_type <- as.character(ct_pred$cell_type)
        }}
    
        # Optional: Set a threshold for high-confidence annotations
        threshold = 0.0  # Adjust based on your data
        high_conf_pred = ct_pred[ct_pred$enrichment > threshold, ]
    
        #write.csv(high_conf_pred, "/inkwell05/arvin/Anirban/high_conf_pred.csv")
    '''

    try:
        ro.r(r_code)
        annotated_cells = ro.r('high_conf_pred')
    except RRuntimeError as e:
        print("R code parsing error:", e)
        raise

    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        annotated_df = ro.conversion.rpy2py(annotated_cells)

    return annotated_df

def main(adata_path):

    ref_markers = "../Adatas/Zeng_markers_with_symbols.csv"


    annotated_df = get_metamarkers_pred_from_precomputed_markers(adata_path, ref_markers)
    adata_annotated = get_adata_from_pred(adata_path, annotated_df)

    # print(adata_annotated.obs["meta_marker_annotation"])
    # print((adata_annotated.obs["meta_marker_annotation"] == "unassigned").sum())
    return adata_annotated

if __name__ == "__main__":
    adata_path = sys.argv[1]

    adata = main(adata_path)
    print(adata)
    adata.write_h5ad(adata_path)