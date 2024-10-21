"""
This script downloads the data from the NZGD website.
"""

import os
import time
from multiprocessing import Pool
from pathlib import Path


import pandas as pd
from dotenv import load_dotenv

import enum
import config as cfg

config = cfg.Config()

import nzgd_download_helper_functions

class DownloadMode(enum.StrEnum):

    files = "files"
    meta_data = "meta_data"

start_time = time.time()

Path(config.get_value("high_level_download_dir")).mkdir(parents=True, exist_ok=True)
Path(config.get_value("downloaded_record_note_per_record")).mkdir(parents=True, exist_ok=True)
Path(config.get_value("name_to_files_dir_per_record")).mkdir(parents=True, exist_ok=True)
Path(config.get_value("name_to_link_str_dir_per_record")).mkdir(parents=True, exist_ok=True)
Path(config.get_value("name_to_metadata_dir_per_record")).mkdir(parents=True, exist_ok=True)

download_mode = DownloadMode.files

previous_download_dir = Path("/home/arr65/data/nzgd/downloads_and_metadata/25092024/downloaded_files")
type_subdirs = list(previous_download_dir.glob("*"))

downloaded_records_path = []
for subdir in type_subdirs:
    downloaded_records_path.extend(list(subdir.glob("*")))

downloaded_records = [record.name for record in downloaded_records_path]

# downloaded_records = os.listdir("/home/arr65/data/nzgd/downloaded_files/download_run_3") + \
#                      os.listdir("/home/arr65/data/nzgd/downloaded_files/download_run_4") + \
#                      os.listdir("/home/arr65/data/nzgd/downloaded_files/download_run_5") + \
#                      os.listdir("/home/arr65/data/nzgd/downloaded_files/download_run_6") + \
#                      os.listdir("/home/arr65/data/nzgd/downloaded_files/download_run_7")

url_df = pd.read_csv(config.get_value("data_lookup_index"))


url_df = url_df[
    (url_df["Type"] == "CPT")
    | (url_df["Type"] == "SCPT")
    | (url_df["Type"] == "Borehole")
    | (url_df["Type"] == "VsVp")
][["ID", "URL"]]

### Only the velocity profiles
# url_df = url_df[
#     (url_df["Type"] == "VsVp")
# ][["ID", "URL"]]

# Remove records that have already been downloaded
#url_df = url_df[~url_df["ID"].isin(downloaded_records)]

# Load environment variables from .env_nzgd file
load_dotenv(".env_nzgd")

# Define the login URL and the list of URLs to scrape
login_url = config.get_value("login_url")

# Get login credentials from environment variables
username_str = os.getenv("NZGD_USERNAME")
password_str = os.getenv("NZGD_PASSWORD")

# Set up the download directory
# high_level_download_dir = Path(config.get_value("high_level_download_dir"))
# os.makedirs(high_level_download_dir, exist_ok=True)

# create directories
# os.makedirs(config.get_value("downloaded_record_note_per_record"), exist_ok=True)
# os.makedirs(config.get_value("name_to_files_dir_per_record"), exist_ok=True)
# os.makedirs(config.get_value("name_to_link_str_dir_per_record"), exist_ok=True)


### Load the last processed index if it exists
if os.path.exists(config.get_value("downloaded_record_note_per_record")):
    notes_per_record_list = os.listdir(config.get_value("downloaded_record_note_per_record"))
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
