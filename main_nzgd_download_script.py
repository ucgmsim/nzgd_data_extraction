import os
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import config as cfg

config = cfg.Config()

import nzgd_download_helper_functions

start_time = time.time()


url_df = pd.read_csv(config.get_value("data_lookup_index"))

url_df = url_df[
    (url_df["Type"] == "CPT")
    | (url_df["Type"] == "SCPT")
    | (url_df["Type"] == "Borehole")
][["ID", "URL"]]

url_df = url_df[
    config.get_value("dataframe_start_index") : config.get_value("dataframe_end_index")
]

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

# Set up the state directory
state_dir = high_level_download_dir.parent.parent / config.get_value(
    "last_attempted_download_dir"
)
os.makedirs(state_dir, exist_ok=True)

name_to_files_highest_dir = (
    high_level_download_dir.parent.parent / "name_to_files_dicts"
)
os.makedirs(name_to_files_highest_dir, exist_ok=True)

name_to_link_str_highest_dir = (
    high_level_download_dir.parent.parent / "name_to_link_str_dicts"
)
os.makedirs(name_to_link_str_highest_dir, exist_ok=True)

# Divide data_urls into chunks
data_url_chunks = nzgd_download_helper_functions.chunkify_dataframe(
    url_df, config.get_value("number_of_processes")
)

with Pool(processes=config.get_value("number_of_processes")) as pool:
    pool.starmap(
        nzgd_download_helper_functions.process_chunk, enumerate(data_url_chunks)
    )

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:.2f} hours")
