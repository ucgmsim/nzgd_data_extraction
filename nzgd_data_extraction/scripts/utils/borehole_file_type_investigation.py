from pathlib import Path
import pandas as pd
from tqdm import tqdm

borehole_path = Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/borehole")

borehole_dirs = list(borehole_path.glob("*"))

count_df = pd.DataFrame(columns=["borehole_name","num_pdf_files","num_ags_files","num_other_files"])

# for borehole_dir in tqdm(borehole_dirs):
#
#     count_df = pd.concat([count_df,
#                pd.DataFrame({"borehole_name":borehole_dir.name,
#                              "num_pdf_files":len(list(borehole_dir.glob("*.pdf"))),
#                              "num_ags_files":len(list(borehole_dir.glob("*.ags"))),
#                              "num_other_files":len(list(borehole_dir.glob("*"))) - len(list(borehole_dir.glob("*.pdf"))) - len(list(borehole_dir.glob("*.ags")))},
#                             index=[0])],
#                          ignore_index=True)
#
# count_df.to_csv("/home/arr65/data/nzgd/resources/borehole_file_type_count.csv",index=False)

count_df = pd.read_csv("/home/arr65/data/nzgd/resources/borehole_file_type_count.csv")

info_df = pd.read_csv("/home/arr65/data/nzgd/resources/nzgd_index_files/csv_files/NZGD_Investigation_Report_08112024_1017.csv")

merged_df = pd.merge(count_df, info_df,left_on="borehole_name",right_on="ID",how="left")

investigation_date = merged_df["InvestigationDate"].to_datetime()

print()
