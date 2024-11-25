import pandas as pd
from pathlib import Path
import natsort
import random
import subprocess
import os
import time
import numpy as np

# spreadsheet_format_df = pd.read_csv("/home/arr65/data/nzgd/standard_format_batch1/cpt/metadata/spreadsheet_format_description.csv")
# spreadsheet_format_df_only_headers = spreadsheet_format_df[["depth_col_name_in_original_file","adopted_cone_resistance_column_name_in_original_file",
# "adopted_sleeve_friction_column_name_in_original_file","adopted_porewater_pressure_column_name_in_original_file"]]
#
#
# spreadsheet_format_df_only_headers_no_dup = spreadsheet_format_df_only_headers.drop_duplicates()



# failed_df = pd.read_csv("/home/arr65/data/nzgd/standard_format_batch50/cpt/metadata/all_failed_loads.csv")
#
# print()
#
# failed_df = failed_df.iloc[225:]
#
# idx = np.where(failed_df["record_name"] == "CPT_58993")

# xls_scpts = list(Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/scpt").rglob("*.xls"))
# random.shuffle(xls_scpts)

#xls_scpts = list(Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt").rglob("*"))

inferred_cm_to_m_names = np.loadtxt("/home/arr65/data/nzgd/resources/inferred_cm_to_m_names.txt",dtype=str)

files_to_open = []

for record_name in inferred_cm_to_m_names:
    file_list_with_pdfs = list(Path(f"/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt/{record_name}").rglob("*"))

    files_to_open_for_this_cpt = [file for file in file_list_with_pdfs if file.suffix != ".pdf"]
    files_to_open.extend(files_to_open_for_this_cpt)


#for index, row in failed_df.iterrows():
for index, xls_scpt_ffp in enumerate(files_to_open):
    # print(row["record_name"])
    #
    # load_path = Path(f"/home/arr65/data/nzgd/downloaded_files/cpt/{row['record_name']}/{row['file_name']}")

    load_path = xls_scpt_ffp

    command_str = f"libreoffice --calc {load_path}"

    # result = subprocess.run(command_str, capture_output=False, text=False)

    os.system(command_str)

    input("Press Enter to continue...")

