import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from natsort import natsort_keygen, natsorted
import shutil


### Create a dataframe containing only the information related to how the data was extracted from the spreadsheets

# extracted_data_dir = Path("/home/arr65/data/nzgd/extracted_cpt_and_scpt_data/cpt/extracted_data_per_record")
#
# extracted_files = list(extracted_data_dir.glob("*.parquet"))
#
# info_df = pd.DataFrame()
#
# for extracted_file in tqdm(extracted_files):
#     info_df_for_record = pd.read_parquet(extracted_file, columns=['record_name', 'original_file_name',
#        'sheet_in_original_file', 'header_row_index',
#        'adopted_Depth_column_name_in_original_file',
#        'adopted_qc_column_name_in_original_file',
#        'adopted_fs_column_name_in_original_file',
#        'adopted_u_column_name_in_original_file', 'investigation_number'])
#
#     ### Drop rows of info_df_for_record if ".ags" is in the "original_file_name" column
#     info_df_for_record = info_df_for_record[~info_df_for_record["original_file_name"].str.contains(".ags")]
#
#     info_df_for_record = info_df_for_record.drop_duplicates(keep='first')
#
#     info_df = pd.concat([info_df,
#                              info_df_for_record],
#                         ignore_index=True)
#
# info_df.to_csv(Path("/home/arr65/data/nzgd/extracted_cpt_and_scpt_data/cpt") / "spreadsheet_extraction_info_df.csv", index=False)

info_df = pd.read_csv(Path("/home/arr65/data/nzgd/extracted_cpt_and_scpt_data/cpt") / "spreadsheet_extraction_info_df.csv")

info_df = info_df[info_df["investigation_number"] == 0]

info_df.sort_values(by="record_name",
                    key=natsort_keygen(),
                    inplace=True)
###################################################

# Identify the 10 most common duplicates
common_duplicates = info_df.duplicated(subset=["header_row_index",
                                               "adopted_Depth_column_name_in_original_file",
                                               "adopted_qc_column_name_in_original_file",
                                               "adopted_fs_column_name_in_original_file",
                                               "adopted_u_column_name_in_original_file",
                                               "investigation_number"], keep=False)

# Filter the DataFrame to keep only these duplicates
common_duplicates_df = info_df[common_duplicates]

# Get the 10 most common duplicates
top_60_common_duplicates = common_duplicates_df.groupby(["header_row_index",
                                                         "adopted_Depth_column_name_in_original_file",
                                                         "adopted_qc_column_name_in_original_file",
                                                         "adopted_fs_column_name_in_original_file",
                                                         "adopted_u_column_name_in_original_file",
                                                         "investigation_number"]).size().nlargest(60).index

# Filter the DataFrame to keep only the top 10 most common duplicates
info_df = info_df[info_df.set_index(["header_row_index",
                                     "adopted_Depth_column_name_in_original_file",
                                     "adopted_qc_column_name_in_original_file",
                                     "adopted_fs_column_name_in_original_file",
                                     "adopted_u_column_name_in_original_file",
                                     "investigation_number"]).index.isin(top_60_common_duplicates)]

# Keep only the first occurrence of these duplicates
info_df = info_df.drop_duplicates(keep='first', subset=["header_row_index",
                                                        "adopted_Depth_column_name_in_original_file",
                                                        "adopted_qc_column_name_in_original_file",
                                                        "adopted_fs_column_name_in_original_file",
                                                        "adopted_u_column_name_in_original_file",
                                                        "investigation_number"])

info_df.to_csv("/home/arr65/data/nzgd/extracted_cpt_and_scpt_data/cpt/top_60_most_common_spreadsheet_format_duplicates.csv", index=False)

raw_data_path = Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt")

destination_path = Path("/home/arr65/data/nzgd/resources/example_spreadsheets/most_common_spreadsheet_formats")

for idx, row in info_df.iterrows():

    source_path = raw_data_path / row["record_name"] / row["original_file_name"]

    #shutil.copy(source_path, destination_path)

### Copy the first N spreadsheets to a new directory
# N = 50
#
# raw_data_path = Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt")
# cpt_dir_list = natsorted(list(raw_data_path.glob("*")))
# cpt_dir_list = cpt_dir_list[:N]
#
# destination_path = Path("/home/arr65/data/nzgd/resources/example_spreadsheets/first_spreadsheets")
#
# for cpt_dir in cpt_dir_list:
#
#     xlsx_files = list(cpt_dir.glob("*.xlsx"))
#     xls_files = list(cpt_dir.glob("*.xls"))
#     csv_files = list(cpt_dir.glob("*.csv"))
#     txt_files = list(cpt_dir.glob("*.txt"))
#
#     if len(xlsx_files) > 0:
#         shutil.copy(xlsx_files[0], destination_path)
#         continue
#     elif len(xls_files) > 0:
#         shutil.copy(xls_files[0], destination_path)
#         continue
#     elif len(csv_files) > 0:
#         shutil.copy(csv_files[0], destination_path)
#         continue
#     elif len(txt_files) > 0:
#         shutil.copy(txt_files[0], destination_path)
#         continue

### Copy N random spreadsheets to a new directory
N = 60

raw_data_path = Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt")
cpt_dir_list = list(raw_data_path.glob("*"))

import random
random.shuffle(cpt_dir_list)

cpt_dir_list = cpt_dir_list[:N]

destination_path = Path("/home/arr65/data/nzgd/resources/example_spreadsheets/random_spreadsheets")

for cpt_dir in cpt_dir_list:

    xlsx_files = list(cpt_dir.glob("*.xlsx"))
    xls_files = list(cpt_dir.glob("*.xls"))
    csv_files = list(cpt_dir.glob("*.csv"))
    txt_files = list(cpt_dir.glob("*.txt"))

    if len(xlsx_files) > 0:
        shutil.copy(xlsx_files[0], destination_path)
        continue
    elif len(xls_files) > 0:
        shutil.copy(xls_files[0], destination_path)
        continue
    elif len(csv_files) > 0:
        shutil.copy(csv_files[0], destination_path)
        continue
    elif len(txt_files) > 0:
        shutil.copy(txt_files[0], destination_path)
        continue








