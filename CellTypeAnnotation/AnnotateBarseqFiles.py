from Wrapper import get_adata_from_pred
from FromComputedMarkers import get_metamarkers_pred_from_precomputed_markers

def main():

    barseq_prefix = "/inkwell05/arvin/barseq/"

    barseq_files = [
        "controls/Control_1.mat.h5ad",
        "controls/Control_2.mat.h5ad",
        "controls/Control_3.mat.h5ad",
        "controls/Control_4.mat.h5ad",
        "enucleated/Enucleated_1.mat.h5ad",
        "enucleated/Enucleated_2.mat.h5ad",
        "enucleated/Enucleated_3.mat.h5ad",
        "enucleated/Enucleated_4.mat.h5ad"]

    ref_markers_path = "/inkwell05/arvin/MetaMarkersRef/Zeng_markers_with_symbols.csv"

    for barseq_file in barseq_files:
        annotated_df = get_metamarkers_pred_from_precomputed_markers(barseq_prefix + barseq_file, ref_markers_path)
        annotated_adata = get_adata_from_pred(barseq_prefix + barseq_file, annotated_df)

        annotated_adata.obs =  annotated_adata.obs.drop(columns=['H1'])

        annotated_adata.write_h5ad("/inkwell05/arvin/barseq_annotated/" + barseq_file)


if __name__ == '__main__':
    main()
