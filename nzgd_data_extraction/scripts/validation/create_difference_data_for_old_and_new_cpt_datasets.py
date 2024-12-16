
import matplotlib.pyplot as plt

import pandas as pd
# import scipy
# import random
from datetime import datetime

from pathlib import Path
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
import multiprocessing
import time
import natsort
import matplotlib.pyplot as plt




#import download_nzgd_data.validation.load_sql_db as load_sql_db
import nzgd_data_extraction.validation.helpers as helpers

import numpy as np
from tqdm import tqdm

import functools

start_time = time.time()

current_date = datetime.now().date()

# Convert the date to a string
current_date_str = current_date.strftime("%Y-%m-%d")

old_data_dir = Path("/home/arr65/vs30_data_input_data/parquet/data")
new_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/extracted_data_per_record")

check_output_dir = Path(f"/home/arr65/data/nzgd/validation_checks/{current_date_str}/processed_data")
check_output_dir.mkdir(parents=True, exist_ok=True)

print("getting record names")

### Find the record_ids that are in both the old and new data directories
old_ids = natsort.natsorted([file.stem for file in old_data_dir.glob("*.parquet")])
new_ids = natsort.natsorted([file.stem for file in new_data_dir.glob("*.parquet")])
ids_in_both_old_and_new_datasets = [id for id in new_ids if id in old_ids]

### To output the record_names that are in both the old and new datasets
ids_to_check_df = pd.DataFrame({"record_names_in_old_and_new_datasets":ids_in_both_old_and_new_datasets})
ids_to_check_df.to_csv(Path("/home/arr65/data/nzgd/resources") / "record_names_in_old_and_new_datasets.csv",index=False)
#ids_to_check_df = pd.read_csv(Path("/home/arr65/data/nzgd/resources") / "record_names_in_old_and_new_datasets.csv")
# ids_in_both_old_and_new_datasets = ids_to_check_df["record_names_in_old_and_new_datasets"].to_list()

num_points_in_new = []
num_points_in_old = []

print("starting comparison")

for id in tqdm(ids_in_both_old_and_new_datasets):
    new_df = pd.read_parquet(new_data_dir / f"{id}.parquet")
    old_df = pd.read_parquet(old_data_dir / f"{id}.parquet")

    num_points_in_new.append(len(new_df))
    num_points_in_old.append(len(old_df))

num_points_df = pd.DataFrame({"record_name":ids_in_both_old_and_new_datasets,
                                "num_points_in_new":num_points_in_new,
                                "num_points_in_old":num_points_in_old})
num_points_df.to_csv(check_output_dir / "num_points_in_new_and_old.csv")

plt.figure()
plt.scatter(num_points_in_old, num_points_in_new, s = 1)
plt.plot([0,5000],[0,5000],color="red",linestyle="--",label="one-to-one line")
plt.xlabel("num points in old version")
plt.ylabel("num points in new version")
plt.title("number of datapoints for each CPT in the new and old datasets")
plt.xlim((0,1580))
plt.legend()
plt.savefig(check_output_dir / "num_points_in_new_and_old.png",dpi=500)
plt.close()
print()


# ids_in_both_old_and_new_datasets = ["CPT_143345"]

print("starting check")

#allowed_percentages_not_close_to_zero = np.array([5])
#allowed_percentages_not_close_to_zero = np.arange(5,50,5)
#allowed_percentages_not_close_to_zero = np.array([5])

allowed_percentages_not_close_to_zero = 10.0
#max_allowed_resid_as_pc_of_mean_vect = np.array([5])
max_allowed_resid_as_pc_of_mean_vect = np.array([1,3,5,10,15,20])
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
        residuals_ok_for_record = pool.map(check_residual_partial, ids_in_both_old_and_new_datasets)

    inconsistent_record_names = list(np.array(ids_in_both_old_and_new_datasets)[~np.array(residuals_ok_for_record)])
    num_inconsistent_records = len(inconsistent_record_names)

    inconsistent_record_names_str = " ".join(inconsistent_record_names)

    results_df = pd.DataFrame({"allowed_percentages_not_close_to_zero": [allowed_percentages_not_close_to_zero],
                               "max_allowed_resid_as_pc_of_old_range": [max_allowed_resid_as_pc_of_old_range],
                               "num_inconsistent_records": [num_inconsistent_records],
                               "num_records_in_both_old_and_new": [len(ids_in_both_old_and_new_datasets)],
                               "percent_inconsistent_records": [100*num_inconsistent_records/len(ids_in_both_old_and_new_datasets)],
                               "inconsistent_record_names":[inconsistent_record_names_str]})

    concat_results_df = pd.concat([concat_results_df,results_df],ignore_index=True)

concat_results_df.to_csv(check_output_dir / "inconsistent_records_with_variable_thresholds.csv")

print(f"time taken: {(time.time()-start_time)/60} minutes")

