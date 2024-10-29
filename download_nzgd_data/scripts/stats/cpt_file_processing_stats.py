import numpy as np
import pandas as pd
from pathlib import Path

index_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_23102024_1042.csv")

# get all unique values and the count of each value in the column "Type" of index_df
unique_values = index_df["Type"].value_counts()

print()

print()
output_path = Path("/home/arr65/data/nzgd/processed_data/cpt")

summary_df = pd.read_csv(output_path / "metadata/loading_summary.csv")

summary_of_spreadsheet_loads = summary_df[((summary_df["loaded_file_type"]==".xls") |
                                            (summary_df["loaded_file_type"]==".xlsx") |
                                            (summary_df["loaded_file_type"]==".csv") |
                                            (summary_df["loaded_file_type"]==".txt"))]

summary_of_spreadsheet_loads.to_csv(output_path / "metadata/cpt_loaded_from_spreadsheet_in_NZGD_Investigation_Report_23102024_1042.csv", index=False)

print()
only_has_pdf = summary_df[summary_df["only_has_pdf"] == True]
print()


all_files_that_failed_to_load = pd.read_csv(output_path / "metadata/all_failed_loads.csv")
print()
# select rows from not_loaded_records where the column "only_has_pdf" is True
summary_of_failed_loads = summary_df[summary_df["file_was_loaded"] == False]
summary_of_successful_loads = summary_df[summary_df["file_was_loaded"] == True]

num_cpts_with_no_records = summary_of_failed_loads.shape[0]
num_only_pdf = summary_of_failed_loads["only_has_pdf"].sum()

spreadsheet_format_description_df = pd.read_csv(output_path / "metadata/spreadsheet_format_description.csv")
spreadsheet_format_description_df.drop(columns=["header_row_index","record_name","file_name"], inplace=True)

# get number of unique rows in spreadsheet_format_description_df
spreadsheet_format_description_df_unique = spreadsheet_format_description_df.drop_duplicates()
spreadsheet_format_description_df_unique.to_csv(output_path / "metadata/spreadsheet_format_description_unique.csv", index=False)

print()

one_failed_load_per_cpt_record = all_files_that_failed_to_load.drop_duplicates(subset="record_name", keep="last")

one_failed_load_per_failed_cpt_record = one_failed_load_per_cpt_record[one_failed_load_per_cpt_record["record_name"].isin(summary_of_failed_loads["record_name"])]

one_failed_load_per_failed_cpt_record_no_ags = one_failed_load_per_failed_cpt_record[one_failed_load_per_failed_cpt_record["file_type"] != ".ags"]
one_failed_load_per_failed_cpt_record_no_ags = one_failed_load_per_failed_cpt_record_no_ags[one_failed_load_per_failed_cpt_record_no_ags["category"] == "missing_columns"]


# one_failed_load_per_failed_cpt_record.to_csv(output_path / "metadata/one_failed_load_per_failed_cpt_record.csv", index=False)

# count the number of unique categories in the column "category" of one_failed_load_per_failed_cpt_record
#num_unique_categories = one_failed_load_per_failed_cpt_record["category"].nunique()

## select rows where "u" is in the column "description" of one_failed_load_per_failed_cpt_record
u_idx = one_failed_load_per_failed_cpt_record_no_ags["details"].str.contains("u")

pore_pressure_idx = one_failed_load_per_failed_cpt_record_no_ags["details"].str.contains("pore pressure")

non_numeric_index = one_failed_load_per_failed_cpt_record_no_ags["details"].str.contains("numeric")
opus_note = one_failed_load_per_failed_cpt_record_no_ags["details"].str.contains("has only one line with first cell of Historical CPT data has been uploaded by Opus under engagement through the MBIE")
multiple_columns = one_failed_load_per_failed_cpt_record_no_ags["details"].str.contains("columns")
column_idx = one_failed_load_per_failed_cpt_record_no_ags["details"].str.contains("column")
corruption_idx = one_failed_load_per_failed_cpt_record_no_ags["details"].str.contains("corruption")
no_raw_data_supplied_idx = one_failed_load_per_failed_cpt_record_no_ags["details"].str.contains("No raw data supplied")



u_idx2 = u_idx.copy()


u_idx2[non_numeric_index] = False
u_idx2[opus_note] = False
u_idx2[multiple_columns] = False
u_idx2[column_idx] = False
u_idx2[corruption_idx] = False
u_idx2[no_raw_data_supplied_idx] = False

missing_u_or_porepressure = one_failed_load_per_failed_cpt_record_no_ags[u_idx2 | pore_pressure_idx]




missing_u_or_porepressure.to_csv(output_path / "metadata/missing_u_or_porepressure.csv", index=False)




print()


print()