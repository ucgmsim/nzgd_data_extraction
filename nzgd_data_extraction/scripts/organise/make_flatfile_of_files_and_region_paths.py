"""
Make a flat file that contains all file names linked to their record names and region paths.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from nzgd_data_extraction.lib import organise

record_name_to_raw_files = defaultdict(list)
record_name_to_processed_files = defaultdict(list)


record_type_to_path = {
    "CPT": "cpt",
    "SCPT": "scpt",
    "BH": "borehole",
    "VsVp": "vsvp",
}

processed_spt_df = pd.read_parquet("/home/arr65/data/nzgd/processed_data/spt/extracted_spt_data.parquet")

processed_spt_df.reset_index(inplace=True)
processed_spt_df['record_name'] = processed_spt_df['NZGD_ID'].apply(lambda x: f'BH_{x}')

region_df_path = Path("/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/regions_NZGD_Investigation_Report_08112024_1017.csv")

region_df = pd.read_csv(region_df_path)
region_df.rename(
    columns={
        "record_id": "record_name",
        "district": "region",
        "territor_1": "district",
        "major_na_2": "city",
        "name_ascii": "suburb",
    },
    inplace=True,
)
region_df = region_df[["record_name", "region", "district", "city", "suburb"]]
region_df = region_df.map(organise.replace_chars)

print("Finding all raw files")
all_raw_files = list(Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd").rglob("*"))
all_raw_files = [file for file in all_raw_files if (file.is_file() and ("last_updated" not in file.name))]
for file in all_raw_files:
    record_name_to_raw_files[file.parent.name].append(file.name)

print("Finding all processed files")
all_processed_files = list(Path("/home/arr65/data/nzgd/processed_data").rglob("*.parquet"))
all_processed_files = [file.name for file in all_processed_files if file.is_file()]

for file in all_processed_files:
    if file == "extracted_spt_data.parquet":
        continue
    else:
        record_name_to_processed_files[file.split(".")[0]].append(file)

for record_name in processed_spt_df['record_name'].unique():
    record_name_to_processed_files[record_name].append("extracted_spt_data.parquet")

file_names_df = pd.DataFrame()

print("Matching all files to record names")
for row_index, row in tqdm(region_df.iterrows(), total=len(region_df)):

    type_for_path = record_type_to_path[row["record_name"].split("_")[0]]

    region_path = Path(row["region"]) / row["district"] / row["city"] / row["suburb"]



    for raw_file in record_name_to_raw_files[row["record_name"]]:

        full_path = Path("raw_from_nzgd") / type_for_path / region_path / raw_file

        file_names_df = pd.concat([file_names_df,
                                   pd.DataFrame({"record_name": row["record_name"],
                                                 "file_type": "raw",
                                                 "file": raw_file,
                                                 "region_path":str(region_path),
                                                 "full_path":str(full_path)}, index=[0])],
                                  ignore_index=True)

    if row["record_name"] in record_name_to_processed_files:

        for processed_file in record_name_to_processed_files[row["record_name"]]:

            if "BH" in row["record_name"]:
                full_path = Path("processed_data") / "spt" / "extracted_spt_data.parquet"
            else:
                full_path = Path("processed_data") / type_for_path / region_path / processed_file


            file_names_df = pd.concat([file_names_df,
                                       pd.DataFrame({"record_name": row["record_name"],
                                                     "file_type": "processed",
                                                     "file": processed_file,
                                                     "region_path":str(region_path),
                                                     "full_path":str(full_path)}, index=[0])],
                                      ignore_index=True)

file_names_df.to_csv(f"/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/"
                     f"file_names_linked_to_{region_df_path.name}", index=False)
