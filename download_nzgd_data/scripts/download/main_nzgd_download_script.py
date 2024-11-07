"""
This script downloads the data from the NZGD website.
"""

import os
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import enum
import download_nzgd_data.download.download_config as cfg

config = cfg.Config()

import download_nzgd_data.download.nzgd_download_helper_functions as nzgd_download_helper_functions

class DownloadMode(enum.StrEnum):

    files = "files"
    meta_data = "meta_data"

start_time = time.time()

high_level_download_dir = Path(config.get_value("high_level_download_dir"))
download_subdir = high_level_download_dir / config.get_value("download_subdir")
download_subdir.mkdir(parents=True, exist_ok=True)

download_metadata_dir = high_level_download_dir / "downloads_metadata" / config.get_value("download_subdir")

downloaded_record_note_per_record_dir = download_metadata_dir / "downloaded_record_note_per_record"
downloaded_record_note_per_record_dir.mkdir(parents=True, exist_ok=True)

download_mode = DownloadMode.files

previous_download_dir = Path(config.get_value("previous_download_dir"))
type_subdirs = list(previous_download_dir.glob("*"))

downloaded_records_path = []
for subdir in type_subdirs:
    if subdir.is_dir():
        downloaded_records_path.extend(list(subdir.glob("*")))

downloaded_records = [record.name for record in downloaded_records_path]

url_df = pd.read_csv(config.get_value("data_lookup_index"))

url_df = url_df[
    (url_df["Type"] == "CPT")
    | (url_df["Type"] == "SCPT")
    | (url_df["Type"] == "Borehole")
    | (url_df["Type"] == "VsVp")
][["ID", "URL"]]

# Remove records that have already been downloaded
url_df = url_df[~url_df["ID"].isin(downloaded_records)]

## Drop the small number of rows with "Other" in the "ID" column as these do not follow the standard naming convention.
url_df = url_df[~url_df["ID"].str.contains("Other")]

# Define the login URL and the list of URLs to scrape
login_url = config.get_value("login_url")

### Load the last processed index if it exists
if os.path.exists(downloaded_record_note_per_record_dir):
    notes_per_record_list = os.listdir(downloaded_record_note_per_record_dir)
    if len(notes_per_record_list) > 0:
        last_processed_index = sorted(notes_per_record_list)[-1]
else:
    last_processed_index = 0

print("starting downloads")
if download_mode == DownloadMode.files:

    with Pool(processes=config.get_value("number_of_processes")) as pool:
        pool.starmap(
            nzgd_download_helper_functions.process_df_row, url_df.iterrows()
        )

if download_mode == DownloadMode.meta_data:

    with Pool(processes=config.get_value("number_of_processes")) as pool:
        pool.starmap(
            nzgd_download_helper_functions.get_metadata_from_nzgd_record_page, url_df.iterrows()
        )

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:.2f} hours")
