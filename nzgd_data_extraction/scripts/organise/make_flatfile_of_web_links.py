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

link_prefix = "https://quakecoresoft.canterbury.ac.nz"

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

## Drop names with "Other" in them as they do not correspond to any desired types of records
region_df = region_df[~region_df["record_name"].str.contains("Other")]
region_df.set_index("record_name", inplace=True)

file_names_df = pd.DataFrame({"raw_file_links" : None,
                                "processed_file_links" : None},
                                index = region_df.index)

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

print("Matching all files to record names")
for record_name in tqdm(file_names_df.index):

    type_for_path = record_type_to_path[record_name.split("_")[0]]

    region_path = Path(region_df.loc[record_name,"region"] ) / region_df.loc[record_name,"district"] / region_df.loc[record_name,"city"] / region_df.loc[record_name,"suburb"]
    full_path_to_file = Path(link_prefix) / Path("raw_from_nzgd") / type_for_path / region_path / record_name

    file_names_df.at[record_name, "raw_file_links"] = []
    for raw_file in record_name_to_raw_files[record_name]:
        file_names_df.at[record_name, "raw_file_links"].append(str(full_path_to_file / raw_file))

    file_names_df.at[record_name, "processed_file_links"] = []
    for processed_file in record_name_to_processed_files[record_name]:
        if "BH" in record_name:
            full_file_path = Path(link_prefix) / "processed" / "spt" / "extracted_spt_data.parquet"
        else:
            full_file_path = Path(link_prefix) / "processed" / type_for_path / region_path / processed_file
        file_names_df.at[record_name, "processed_file_links"].append(str(full_file_path))

file_names_df.reset_index(inplace=True)
file_names_df.to_parquet(f"/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/"
                     f"file_paths_names_linked_to_{region_df_path.stem}.parquet", index=False)