"""
Make a flat file for SPT data
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from nzgd_data_extraction.lib import organise


nzgd_df = pd.read_csv(
    "/home/arr65/data/nzgd/resources/nzgd_index_files/csv_files/NZGD_Investigation_Report_08112024_1017.csv"
)
### drop columns X and Y
nzgd_df.drop(columns=["X", "Y", "Coordinate System","URL"], inplace=True)
nzgd_df.rename(
    columns={
        "ID": "record_name",
        "Type": "type",
        "OriginalReference": "original_reference",
        "InvestigationDate": "investigation_date",
        "TotalDepth": "total_depth",
        "PublishedDate": "published_date",
        "Latitude": "latitude",
        "Longitude": "longitude",
    },
    inplace=True,
)

region_df = pd.read_csv(
    "/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/regions_NZGD_Investigation_Report_08112024_1017.csv"
)

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

#####################################################################
file_path_df = pd.read_csv(
    "/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/file_names_linked_to_regions_NZGD_Investigation_Report_08112024_1017.csv"
)
links_df_for_merge = region_df[["record_name"]].copy()
links_df_for_merge["raw_files"] = None
links_df_for_merge["processed_files"] = None

links_df_for_merge.set_index("record_name", inplace=True)

for idx, row in tqdm(links_df_for_merge.iterrows()):


    raw_file_links = file_path_df[(file_path_df["record_name"] == idx) &
                                  (file_path_df["file_type"] == "raw")]["link"].to_list()

    processed_file_links = file_path_df[(file_path_df["record_name"] == idx) &
                                  (file_path_df["file_type"] == "processed")]["link"].to_list()

    links_df_for_merge.at[idx, "raw_files"] = raw_file_links
    links_df_for_merge.at[idx, "processed_files"] = processed_file_links

#####################################################################

foster_vs30_df = pd.read_csv(
    "/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/foster_vs30_at_nzgd_locations_NZGD_Investigation_Report_08112024_1017.csv"
)
foster_vs30_df.rename(
    columns={
        "ID": "record_name",
        "vs30": "foster_2019_vs30",
        "vs30_std": "foster_2019_vs30_std",
    },
    inplace=True,
)

spt_vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/spt/spt_vs30.csv")
scpt_vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/scpt/metadata/vs30_estimates_from_cpt.csv")
cpt_vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/vs30_estimates_from_cpt.csv")

vs30_df = pd.concat([spt_vs30_df, scpt_vs30_df, cpt_vs30_df], ignore_index=True)

## Make a new column that is the concatenation of strings in columns 'spt_vs_correlation' and 'vs30_correlation'
# spt_vs30_df["spt_vs_correlation_and_vs30_correlation"] = (
#     spt_vs30_df["spt_vs_correlation"] + "_" + spt_vs30_df["vs30_correlation"]
# )

### Merge DataFrames using the record_name column
merged_df = pd.merge(nzgd_df, region_df, on="record_name")
merged_df = pd.merge(merged_df, foster_vs30_df, on="record_name")
merged_df = pd.merge(merged_df, spt_vs30_df, on="record_name")

merged_df["vs30_log_residual"] = np.log(merged_df["vs30"]) - np.log(
    merged_df["foster_2019_vs30"]
)

merged_df["raw_files"] = np.nan
merged_df["processed_files"] = np.nan

merged_df.set_index("record_name", inplace=True)

merged_df.to_parquet(
    Path("/home/arr65/src/nzgd_map_from_webplate/instance/spt_vs30.parquet")
)
