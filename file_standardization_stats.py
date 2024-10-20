import numpy as np
import pandas as pd
from pathlib import Path

output_path = Path("/home/arr65/Downloads")


summary_df = pd.read_csv(output_path / "metadata/loading_summary.csv")
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
spreadsheet_format_description_df.drop(columns=["record_name","file_name"], inplace=True)

# get number of unique rows in spreadsheet_format_description_df
spreadsheet_format_description_df_unique = spreadsheet_format_description_df.drop_duplicates()


one_failed_load_per_cpt_record = all_files_that_failed_to_load.drop_duplicates(subset="record_name", keep="last")

one_failed_load_per_failed_cpt_record = one_failed_load_per_cpt_record[one_failed_load_per_cpt_record["record_name"].isin(summary_of_failed_loads["record_name"])]

# count the number of unique categories in the column "category" of one_failed_load_per_failed_cpt_record
num_unique_categories = one_failed_load_per_failed_cpt_record["category"].nunique()

print()