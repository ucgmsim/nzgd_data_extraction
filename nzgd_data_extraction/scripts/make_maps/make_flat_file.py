from pathlib import  Path
import pandas as pd
import re
import numpy as np

def replace_chars(old_string):
    # Regular expression to replace " " (space), ' (apostrophe), "," (comma), and "/" (forward slash) characters
    chars_to_replace = r"[ ',/]"
    return re.sub(chars_to_replace, "_", old_string)

nzgd_df = pd.read_csv("/home/arr65/data/nzgd/resources/nzgd_index_files/csv_files/NZGD_Investigation_Report_08112024_1017.csv")
# drop columns X and Y
nzgd_df.drop(columns=["X","Y", "Coordinate System"], inplace=True)
nzgd_df.rename(columns={"ID": "record_name",
                        "Type": "type",
                        "OriginalReference": "original_reference",
                        "InvestigationDate": "investigation_date",
                        "TotalDepth": "total_depth",
                        "PublishedDate": "published_date",
                        "Latitude": "latitude",
                        "Longitude": "longitude",
                        "URL":"nzgd_url"}, inplace=True)

region_df = pd.read_csv("/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/regions_NZGD_Investigation_Report_08112024_1017.csv")
region_df.rename(columns={"record_id": "record_name",
                          "district": "region",
                          "territor_1":"district",
                          "major_na_2":"city",
                          "name_ascii":"suburb"}, inplace=True)
region_df = region_df[["record_name","region","district","city","suburb"]]
region_df = region_df.map(replace_chars)


foster_vs30_df = pd.read_csv("/home/arr65/data/nzgd/resources/metadata_from_nzgd_location/foster_vs30_at_nzgd_locations_NZGD_Investigation_Report_08112024_1017.csv")
foster_vs30_df.rename(columns={"ID": "record_name",
                               "vs30": "foster_2019_vs30",
                               "vs30_std":"foster_2019_vs30_std"}, inplace=True)

spt_vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/spt/spt_vs30.csv")
spt_vs30_df.rename(columns={'ID': 'record_name',
                            "error":"error_from_data",
                            "Vs30":"vs30_from_data",
                            "Vs30_sd":"vs30_std_from_data"}, inplace=True)

# Merge DataFrames
merged_df = pd.merge(nzgd_df, region_df, on="record_name")
merged_df = pd.merge(merged_df, foster_vs30_df, on="record_name")
merged_df = pd.merge(merged_df, spt_vs30_df, on="record_name")


## Make a new column that is the concatenation of strings in columns 'spt_vs_correlation' and 'vs30_correlation'
merged_df["spt_vs_correlation_and_vs30_correlation"] = merged_df["spt_vs_correlation"] + "_" + merged_df["vs30_correlation"]

#merged_df["log_vs30_from_data_minus_log_vs30_from_foster_2019"] = np.log(merged_df["vs30_from_data"]) - np.log(merged_df["foster_2019_vs30"])
merged_df["vs30_log_residual"] = np.log(merged_df["vs30_from_data"]) - np.log(merged_df["foster_2019_vs30"])


merged_df.to_parquet(Path("/home/arr65/src/nzgd_map_from_webplate/instance/spt_vs30.parquet"))

