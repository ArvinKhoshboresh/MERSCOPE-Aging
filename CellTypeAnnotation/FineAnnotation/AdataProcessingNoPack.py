import os
import anndata as ad
import scanpy as sc


def load_and_process_adatas(data_source: str, age_annotation: dict, skip_normalization: bool = False) -> dict:
    """
    Load and process AnnData files from a specified source directory.

    Parameters:
    -----------
    data_source : str
        Path to the directory containing AnnData files
    age_annotation : dict
        Dictionary mapping file names to age groups
    skip_normalization : bool
        If True, skip library size normalization

    Returns:
    --------
    dict
        A dictionary containing processed AnnData information
     """
    names = []
    adatas = []
    ages = []
    areas = []

    for filename in sorted(os.listdir(data_source)):
        if filename.endswith('.h5ad'):
            filepath = os.path.join(data_source, filename)

            try:
                adata = ad.read_h5ad(filepath)

                if 'Brain Region Name' in adata.obs:
                    adata = adata[~adata.obs['Brain Region Name'].isna()]

                if adata.n_obs > 0:
                    adata = adata[adata.X.sum(axis=1) > 0].copy()
                    if not skip_normalization:
                        sc.pp.normalize_total(adata, target_sum=1)

                    names.append(filename)
                    adatas.append(adata)
                    ages.append(age_annotation.get(filename, 'Unknown'))
                    areas.append('Anterior' if 'Anterior' in filename else 'Posterior')
                else:
                    print(f"Skipping {filename}: No cells remaining after filtering.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    processed_data = {
        'names': names,
        'adatas': adatas,
        'ages': ages,
        'areas': areas
    }

    return processed_data

adatas_source = "../Adatas/"

age_annotation = {

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
    "Posterior-SM408_414_Region1-regions_SM408_414_Region1_cleared.h5ad": "Yng"
}


processed_data = load_and_process_adatas(adatas_source, age_annotation)