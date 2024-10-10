import pandas as pd
from pathlib import Path
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



failed_df = pd.read_csv("/home/arr65/data/nzgd/standard_format_batch1/cpt/metadata/all_failed_loads_no_ags.csv")

failed_df = failed_df.iloc[225:]

idx = np.where(failed_df["record_name"] == "CPT_58993")

for index, row in failed_df.iterrows():
    print(row["record_name"])

    load_path = Path(f"/home/arr65/data/nzgd/downloaded_files/cpt/{row['record_name']}/{row['file_name']}")

    command_str = f"libreoffice --calc {load_path}"

    # result = subprocess.run(command_str, capture_output=False, text=False)

    os.system(command_str)

    input("Press Enter to continue...")

    print()
