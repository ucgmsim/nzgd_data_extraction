import numpy as np
from pathlib import Path
import toml
import loading_funcs_for_nzgd_data
import xlrd
from tqdm import tqdm
import natsort

parquet_output_dir = Path("/home/arr65/data/nzgd/standard_format/cpt/data")
metadata_output_dir = Path("/home/arr65/data/nzgd/standard_format/cpt/metadata")

parquet_output_dir.mkdir(exist_ok=True, parents=True)
metadata_output_dir.mkdir(exist_ok=True, parents=True)

# Define possible names for each column
possible_xls_cols = [
    ["Depth [m]", "Depth", "Depth (m)"],
    ["Cone resistance (qc) in MPa", "qc [MPa]",  "Tip resistance", "qc Clean (MPa)", "Qc [MPa]"],
    ["Sleeve friction (fs) in MPa", "fs [MPa]", "Local friction","fs Clean (MPa)", "Fs [KPa]"],
    ["Tip resistance", "Local friction", "Pore shoulder","u2 Clean (MPa)", "Dynamic pore pressure (u2) in MPa", "u2 [MPa]", "U2 [KPa]"]
]


downloaded_files = Path("/home/arr65/data/nzgd/downloaded_files/cpt")

meta_successfully_loaded = []
meta_failed_to_load = []

record_counter = 0

for record_dir in tqdm(natsort.natsorted(list(downloaded_files.glob("*")))):
#for record_dir in tqdm(natsort.natsorted(list(downloaded_files.glob("*")))[0:100]):
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_1")]:
#for record_dir in failed_ags_loads:

    record_counter += 1
    if record_counter % 100 == 0:
        np.savetxt(metadata_output_dir / "successfully_loaded.txt", np.array(meta_successfully_loaded), fmt="%s")
        np.savetxt(metadata_output_dir / "failed_to_load.txt", np.array(meta_failed_to_load), fmt="%s")

    has_loaded_a_file_for_this_record = False

    ### ags files
    files_to_try = list(record_dir.glob("*.ags")) + list(record_dir.glob("*.AGS"))
    if len(files_to_try) > 0:
        for file_to_try in files_to_try:
            try:
                record_df = loading_funcs_for_nzgd_data.load_ags(file_to_try)
                record_df.attrs["original_file_name"] = file_to_try.name
                record_df.to_parquet(parquet_output_dir / f"{record_dir.name}.parquet")
                meta_successfully_loaded.append(file_to_try.name)

                has_loaded_a_file_for_this_record = True
                continue

            ## If the ags file is missing data, KeyError or UnboundLocalError will be raised
            except(KeyError, UnboundLocalError) as e:
                meta_failed_to_load.append(f"{record_dir.name}, {file_to_try.name}, {str(e)}")
                pass

    if has_loaded_a_file_for_this_record:
        continue

    ### xls files
    files_to_try = list(record_dir.glob("*.xls")) + list(record_dir.glob("*.XLS")) + \
                   list(record_dir.glob("*.xlsx")) + list(record_dir.glob("*.XLSX"))

    if len(files_to_try) > 0:
        for file_to_try in files_to_try:
            try:
                record_df = loading_funcs_for_nzgd_data.load_xls_file_brute_force(file_to_try, possible_xls_cols)
                record_df.attrs["original_file_name"] = file_to_try.name
                record_df.to_parquet(parquet_output_dir / f"{record_dir.name}.parquet")
                meta_successfully_loaded.append(file_to_try.name)
                has_loaded_a_file_for_this_record = True
                continue
            except(ValueError, xlrd.compdoc.CompDocError, Exception) as e:
                meta_failed_to_load.append(f"{record_dir.name}, {file_to_try.name}, {e}")
                pass

    if not has_loaded_a_file_for_this_record:
        meta_failed_to_load.append(f"{record_dir.name}, Did_not_attempt_to_load_any_files, ")

np.savetxt(metadata_output_dir / "successfully_loaded.txt", np.array(meta_successfully_loaded), fmt="%s")
np.savetxt(metadata_output_dir / "failed_to_load.txt", np.array(meta_failed_to_load), fmt="%s")
