

import matplotlib.pyplot as plt

import pandas as pd
import scipy
import random

from pathlib import Path

import time

import download_nzgd_data.validation.helpers as helpers


start_time = time.time()


old_data_dir = Path("/home/arr65/vs30_data_input_data/parquet")
new_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")

check_output_dir = Path("/home/arr65/data/nzgd/check_output")
check_output_dir.mkdir(parents=True, exist_ok=True)


import pandas as pd

df = pd.read_csv("/home/arr65/data/nzgd/check_output/results.csv")

inconsistent_record_names = df.loc[0,"inconsistent_record_names"]

inconsistent_record_names = inconsistent_record_names.split(" ")

for record_name in inconsistent_record_names:
    residual, old_df, interpolated_df, new_df = helpers.get_residual(record_name = record_name, old_data_ffp = old_data_dir, new_data_ffp=new_data_dir,make_plot=True)




