from Wrapper import get_adata_from_pred
from FromComputedMarkers import get_metamarkers_pred_from_precomputed_markers
import os

def main():

    ref_markers_path = "../Adatas/Zeng_markers_with_symbols.csv"
    adatas_directory = "Cellpose2Adatas/"
    output_directory = "../Adatas/"

    for adata_name in os.listdir(adatas_directory):
        adata_path = os.path.join(adatas_directory, adata_name)

        annotated_df = get_metamarkers_pred_from_precomputed_markers(adata_path, ref_markers_path, 50)
        annotated_adata = get_adata_from_pred(adata_path, annotated_df)

        annotated_adata.write_h5ad(output_directory + adata_name)


if __name__ == '__main__':
    main()
