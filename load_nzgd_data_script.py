from multiprocessing.managers import Value

from python_ags4 import AGS4
from typing import Union
from pathlib import Path
import pandas as pd
import natsort
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from itertools import product
import toml
import loading_funcs_for_nzgd_data

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

meta_info = {"loaded_records_and_files": {}, "failed_ags_loads": {}, "failed_xls_loads": {},
             "all_loading_attempts_failed": []}

for record_dir in tqdm(natsort.natsorted(list(downloaded_files.glob("*")))):
#for record_dir in tqdm(natsort.natsorted(list(downloaded_files.glob("*")))[0:100]):
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_225072")]:
#for record_dir in failed_ags_loads:

    #record_dir = Path(str(record_dir)).parent


    has_loaded_a_file_for_this_record = False

    file_types = [file.suffix for file in record_dir.glob("*")]

    ### ags files
    files_to_try = list(record_dir.glob("*.ags")) + list(record_dir.glob("*.AGS"))
    if len(files_to_try) > 0:
        for file_to_try in files_to_try:
            try:
                record_df = loading_funcs_for_nzgd_data.load_ags(file_to_try)
                record_df.original_file_name = file_to_try.name
                record_df.to_parquet(parquet_output_dir / f"{record_dir.name}.parquet")
                meta_info["loaded_records_and_files"][record_dir.name] = file_to_try.name
                has_loaded_a_file_for_this_record = True
                continue

            ## If the ags file is missing data, KeyError or UnboundLocalError will be raised
            except(KeyError, UnboundLocalError):
                meta_info["failed_ags_loads"][record_dir.name] = file_to_try.name
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
                record_df.original_file_name = file_to_try.name
                record_df.to_parquet(parquet_output_dir / f"{record_dir.name}.parquet")
                meta_info["loaded_records_and_files"][record_dir.name] = file_to_try.name
                has_loaded_a_file_for_this_record = True
                continue
            except(ValueError):
                meta_info["failed_xls_loads"][record_dir.name] = file_to_try.name
                pass

    if not has_loaded_a_file_for_this_record:
        meta_info["all_loading_attempt_failed"].append(record_dir.name)

### save metadata to toml file on every iteration in case of failure
with open(metadata_output_dir /  "record_standardization_metadata.toml", "w") as f:
    toml.dump(meta_info, f)


# ## save the failed ags loads
# np.savetxt("/home/arr65/data/nzgd/debugging_output/ags_loading_failed.txt", records_with_failed_ags_load, fmt="%s")
#
# ## save the failed xls loads
# np.savetxt("/home/arr65/data/nzgd/debugging_output/xls_loading_failed.txt", records_with_failed_xls_load, fmt="%s")
#
# np.savetxt("/home/arr65/data/nzgd/debugging_output/records_not_loaded.txt", records_not_loaded, fmt="%s")

#print()