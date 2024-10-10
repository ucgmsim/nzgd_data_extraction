import numpy as np
import pandas as pd

summary_df = pd.read_csv("/home/arr65/data/nzgd/standard_format_batch1/cpt/metadata/loading_summary.csv")
failed_df = pd.read_csv("/home/arr65/data/nzgd/standard_format_batch1/cpt/metadata/all_failed_loads.csv")

failed_df_no_ags = failed_df[failed_df["category"] != "bad_ags"]
failed_df_no_ags.to_csv("/home/arr65/data/nzgd/standard_format_batch1/cpt/metadata/failed_df_no_ags.csv")

failed_record_names = failed_df["record_name"]


failed_record_names_duplicates = failed_record_names[failed_record_names.duplicated()]

multiple_load_failures = failed_df[failed_df["record_name"].isin(failed_record_names_duplicates)]

multiple_load_failures_no_ags = multiple_load_failures[multiple_load_failures["category"] != "bad_ags"]

multiple_load_failures_no_ags_names = multiple_load_failures_no_ags["record_name"]

multiple_load_failures_no_ags_duplicate_names = multiple_load_failures_no_ags_names[multiple_load_failures_no_ags_names.duplicated()]

multiple_load_failures.to_csv("/home/arr65/data/nzgd/standard_format_batch1/cpt/metadata/multiple_load_failures.csv")



len(failed_record_names)




#left_joined = failed_df.join(summary_df, on="record_name", how="left", rsuffix="_summary")

records_with_only_pdf_idx = ((summary_df["num_pdf_files"] > 0) & (summary_df["num_cpt_files"] == 0) &
                (summary_df["num_ags_files"] == 0) & (summary_df["num_xls_files"] == 0) &
                (summary_df["num_xlsx_files"] == 0) & (summary_df["num_csv_files"] == 0) &
                (summary_df["num_txt_files"] == 0) & (summary_df["num_other_files"] == 0))

only_pdf_and_cpt = ((summary_df["num_pdf_files"] > 0) & (summary_df["num_cpt_files"] > 0) &
                (summary_df["num_ags_files"] == 0) & (summary_df["num_xls_files"] == 0) &
                (summary_df["num_xlsx_files"] == 0) & (summary_df["num_csv_files"] == 0) &
                (summary_df["num_txt_files"] == 0) & (summary_df["num_other_files"] == 0))


## insert  records_with_only_pdf_idx into summary_df
summary_df.insert(2, "has_only_pdf", records_with_only_pdf_idx)

summary_df_only_failed = summary_df[summary_df["file_was_loaded"] == False]

summary_df_with_spreadsheet_file_loads = summary_df_only_failed[summary_df_only_failed["has_only_pdf"] == False]
summary_df_with_spreadsheet_file_loads.to_csv("/home/arr65/data/nzgd/standard_format_batch1/cpt/metadata/summary_df_with_spreadsheet_file_loads.csv")


# joined_df = pd.merge(failed_df, summary_df, on="record_name", how="outer")
# joined_df.to_csv("/home/arr65/data/nzgd/standard_format_batch1/cpt/metadata/outer_joined.csv")

failed_files_outer_merge_df = pd.merge(summary_df_with_spreadsheet_file_loads, failed_df, on="record_name", how="outer")
failed_files_outer_merge_df.to_csv("/home/arr65/data/nzgd/standard_format_batch1/cpt/metadata/failed_files_outer_merge_df.csv")


num_only_pdf = records_with_only_pdf_idx.sum()

print()