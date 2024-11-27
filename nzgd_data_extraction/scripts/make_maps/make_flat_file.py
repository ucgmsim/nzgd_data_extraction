from pathlib import  Path
import pandas as pd

nzgd_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_08112024_1017.csv")
# drop columns X and Y
nzgd_df.drop(columns=["X","Y"], inplace=True)

print()


spt_vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/spt/spt_vs30.csv")


# merge spt_vs30_df using column "record_name" with nzgd_df using column "ID", keeping only the rows that match in both dataframes
merged_df = pd.merge(spt_vs30_df, nzgd_df, how="inner", left_on="record_name", right_on="ID")

## Make a new column that is the concatenation of strings in columns 'spt_vs_correlation' and 'vs30_correlation'
merged_df["spt_vs_correlation_and_vs30_correlation"] = merged_df["spt_vs_correlation"] + "_" + merged_df["vs30_correlation"]

#merged_df.set_index(['record_name', 'spt_vs_correlation_and_vs30_correlation'], inplace=True)

merged_df.to_parquet(Path("/home/arr65/src/nzgd_map_from_webplate/instance/spt_vs30.parquet"))

print()

# merged_df.to_csv(Path("/home/arr65/src/nzgd_map_from_webplate/instance/spt_vs30.csv"), index=False)
# print()





#ll = pd.read_parquet("/home/arr65/src/nzgd_map_from_webplate/instance/intensity_measures.parquet")


print()

