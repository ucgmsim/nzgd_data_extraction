
import matplotlib.pyplot as plt

import pandas as pd
# import scipy
# import random

from pathlib import Path
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
import multiprocessing
import time
import natsort




#import download_nzgd_data.validation.load_sql_db as load_sql_db
import download_nzgd_data.validation.helpers as helpers

import numpy as np
from tqdm import tqdm

import functools

start_time = time.time()


old_data_dir = Path("/home/arr65/vs30_data_input_data/parquet/data")
new_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")

check_output_dir = Path("/home/arr65/data/nzgd/validation_checks/processed_data")
check_output_dir.mkdir(parents=True, exist_ok=True)

print("getting record names")

### Find the record_ids that are in both the old and new data directories
old_ids = natsort.natsorted([file.stem for file in old_data_dir.glob("*.parquet")])
new_ids = natsort.natsorted([file.stem for file in new_data_dir.glob("*.parquet")])
ids_to_check = [id for id in new_ids if id in old_ids]

# ids_to_check = ["CPT_143345"]

print("starting check")

#allowed_percentages_not_close_to_zero = np.array([5])
#allowed_percentages_not_close_to_zero = np.arange(5,50,5)
#allowed_percentages_not_close_to_zero = np.array([5])

allowed_percentages_not_close_to_zero = 10.0
max_allowed_resid_as_pc_of_mean_vect = np.array([1,3,5,10,15,20,25,30,50,75])
#max_allowed_resid_as_pc_of_mean_vect = np.array([10,50,100])
#max_allowed_resid_as_pc_of_mean_vect = np.array([50,60])#

concat_results_df = pd.DataFrame()

#for index, allowed_percentage_not_close_to_zero in tqdm(enumerate(allowed_percentages_not_close_to_zero)):
for index, max_allowed_resid_as_pc_of_old_range in tqdm(enumerate(max_allowed_resid_as_pc_of_mean_vect),
                                                   total=len(max_allowed_resid_as_pc_of_mean_vect)):

    check_residual_partial = functools.partial(helpers.check_residual,
                                               old_data_ffp=old_data_dir,
                                               new_data_ffp=new_data_dir,
                                               max_allowed_resid_as_pc_of_old_range = max_allowed_resid_as_pc_of_old_range,
                                               allowed_percent_not_close_to_zero=allowed_percentages_not_close_to_zero)

    with multiprocessing.Pool(processes=8) as pool:
        residuals_ok_for_record = pool.map(check_residual_partial, ids_to_check)

    inconsistent_record_names = list(np.array(ids_to_check)[~np.array(residuals_ok_for_record)])
    num_inconsistent_records = len(inconsistent_record_names)

    inconsistent_record_names_str = " ".join(inconsistent_record_names)

    results_df = pd.DataFrame({"allowed_percentages_not_close_to_zero": [allowed_percentages_not_close_to_zero],
                               "max_allowed_resid_as_pc_of_old_range": [max_allowed_resid_as_pc_of_old_range],
                               "num_inconsistent_records": [num_inconsistent_records],
                               "num_records_in_both_old_and_new": [len(ids_to_check)],
                               "percent_inconsistent_records": [100*num_inconsistent_records/len(ids_to_check)],
                               "inconsistent_record_names":[inconsistent_record_names_str]})

    concat_results_df = pd.concat([concat_results_df,results_df],ignore_index=True)

concat_results_df.to_csv(check_output_dir / "results.csv")

print(f"time taken: {(time.time()-start_time)/60} minutes")

