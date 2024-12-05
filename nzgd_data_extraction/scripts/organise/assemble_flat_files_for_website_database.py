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

nzgd_df = nzgd_df[(nzgd_df["Type"] == "Borehole") | (nzgd_df["Type"] == "CPT") | (nzgd_df["Type"] == "SCPT")]
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

borehole_num_code = 0
cpt_num_code = 1
scpt_num_code = 2

# Add the new column "shape" based on the conditions
nzgd_df["type_number_code"] = nzgd_df["type"].map({
    "Borehole": borehole_num_code,
    "CPT": cpt_num_code,
    "SCPT": scpt_num_code
})

region_df = pd.read_csv(
    "/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/regions_NZGD_Investigation_Report_08112024_1017.csv"
)
## Drop velocity profiles (VsVp)
region_df = region_df[~region_df["record_id"].str.contains("VsVp")]
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
foster_vs30_df = foster_vs30_df[foster_vs30_df["record_name"].isin(nzgd_df["record_name"])]

file_links_df = pd.read_parquet("/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/"
                            "file_paths_names_linked_to_regions_NZGD_Investigation_Report_08112024_1017.parquet")
file_links_df = file_links_df[file_links_df["record_name"].isin(nzgd_df["record_name"])]


spt_vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/spt/spt_vs30.csv")
scpt_vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/scpt/metadata/vs30_estimates_from_cpt.csv")
cpt_vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/vs30_estimates_from_cpt.csv")

vs30_df = pd.concat([spt_vs30_df, scpt_vs30_df, cpt_vs30_df], ignore_index=True)

### If vs30_df["vs30"] > 1000, replace vs30_df["vs30"] and vs30_df["vs30_std"] with np.nan
vs30_df.loc[vs30_df["vs30"] > 1000, ["vs30", "vs30_std"]] = np.nan

## Make a new column that is the concatenation of strings in columns 'spt_vs_correlation' and 'vs30_correlation'
# spt_vs30_df["spt_vs_correlation_and_vs30_correlation"] = (
#     spt_vs30_df["spt_vs_correlation"] + "_" + spt_vs30_df["vs30_correlation"]
# )

### Merge DataFrames using the record_name column
df_from_record_locations = pd.merge(nzgd_df, region_df, on="record_name")
df_from_record_locations = pd.merge(df_from_record_locations, foster_vs30_df, on="record_name")
df_from_record_locations = pd.merge(df_from_record_locations, file_links_df, on="record_name")

merged_df = pd.merge(df_from_record_locations, vs30_df, on="record_name", how="outer")

merged_df["vs30_log_residual"] = np.log(merged_df["vs30"]) - np.log(
    merged_df["foster_2019_vs30"]
)

#merged_df.to_csv("/home/arr65/data/nzgd/resources/website_database.csv")


merged_df.to_parquet(
    Path("/home/arr65/src/nzgd_map_from_webplate/instance/website_database.parquet")
)
