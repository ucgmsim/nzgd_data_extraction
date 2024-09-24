import os
import time
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm


import pandas as pd
from dotenv import load_dotenv

import config as cfg

config = cfg.Config()

import nzgd_download_helper_functions

start_time = time.time()

downloaded_records = os.listdir("/home/arr65/data/nzgd/downloaded_files/download_run_3")
url_df = pd.read_csv(config.get_value("data_lookup_index"))

url_df = url_df[
    (url_df["Type"] == "CPT")
    | (url_df["Type"] == "SCPT")
    | (url_df["Type"] == "Borehole")
][["ID", "URL"]]

# Remove records that have already been downloaded
url_df = url_df[~url_df["ID"].isin(downloaded_records)]

# Load environment variables from .env_nzgd file
load_dotenv(".env_nzgd")

# Define the login URL and the list of URLs to scrape
login_url = config.get_value("login_url")

# Get login credentials from environment variables
username_str = os.getenv("NZGD_USERNAME")
password_str = os.getenv("NZGD_PASSWORD")

# Set up the download directory
high_level_download_dir = Path(config.get_value("high_level_download_dir"))
os.makedirs(high_level_download_dir, exist_ok=True)

# create directories
os.makedirs(config.get_value("downloaded_record_note_per_record"), exist_ok=True)
os.makedirs(config.get_value("name_to_files_dir_per_record"), exist_ok=True)
os.makedirs(config.get_value("name_to_link_str_dir_per_record"), exist_ok=True)

### Load the last processed index if it exists
if os.path.exists(config.get_value("downloaded_record_note_per_record")):
    notes_per_record_list = os.listdir(config.get_value("downloaded_record_note_per_record"))
    if len(notes_per_record_list) > 0:
        last_processed_index = sorted(notes_per_record_list)[-1]
else:
    last_processed_index = 0

with Pool(processes=config.get_value("number_of_processes")) as pool:
    pool.starmap(
        nzgd_download_helper_functions.process_chunk, url_df.iterrows()
    )

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:.2f} hours")
