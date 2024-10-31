
import matplotlib.pyplot as plt

import pandas as pd
import scipy
import random

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import multiprocessing
import time
import natsort




import download_nzgd_data.validation.load_sql_db as load_sql_db
import download_nzgd_data.validation.helpers as helpers

import numpy as np
from tqdm import tqdm

import functools

start_time = time.time()


old_data_dir = Path("/home/arr65/vs30_data_input_data/parquet")
new_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")

check_output_dir = Path("/home/arr65/data/nzgd/check_output")
check_output_dir.mkdir(parents=True, exist_ok=True)

print("getting record names")

# old_ids = natsort.natsorted([file.with_suffix("").name for file in old_data_dir.glob("*.parquet")])
# new_ids = natsort.natsorted([file.with_suffix("").name for file in new_data_dir.glob("*.parquet")])
#
# ids_to_check_with_empty = [id for id in new_ids if id in old_ids]
# ids_to_check_with_empty_df = pd.DataFrame({"ids_to_check":ids_to_check_with_empty})
#
#
# # ids_to_check = pd.read_parquet(check_output_dir / "ids_to_check.parquet")["ids_to_check"]
# # df = pd.DataFrame({"ids_to_check":ids_to_check})
# # df.to_parquet(check_output_dir / "ids_to_check.parquet")
# # print()
# #
# ids_to_check_with_empty_df.to_parquet(check_output_dir / "ids_to_check_with_empty.parquet")
#
#
# #ids_to_check.remove("CPT_104077")
#
# ids_to_check = []
# new_empty_parquet_files = []
#
# for id in ids_to_check_with_empty:
#     new_df = pd.read_parquet(new_data_dir / f"{id}.parquet")
#     if new_df.size != 0:
#         ids_to_check.append(id)
#     else:
#         new_empty_parquet_files.append(id)
#
# df_ids_to_check = pd.DataFrame({"ids_to_check":ids_to_check})
# df_new_empty_parquet_files = pd.DataFrame({"new_empty_parquet_files":new_empty_parquet_files})
#
# df_ids_to_check.to_parquet(check_output_dir / "ids_to_check.parquet")
# df_ids_to_check.to_parquet(check_output_dir / "ids_with_new_empty_parquet_files.parquet")

ids_to_check = pd.read_parquet(check_output_dir / "ids_to_check.parquet")["ids_to_check"].to_list()

#ids_to_check = ids_to_check[0:100]


#ids_to_check = ids_to_check[0:1000]
#ids_to_check = ["CPT_13"]
#ids_to_check = ["CPT_104077"]
#ids_to_check = ["CPT_146038"]
#print()






#new_ids = new_ids[37:39]
# for record_name in tqdm(new_ids):
#
#     #residual, old_df, interpolated_df, new_df = helpers.get_residual(record_name = record_name, old_data_ffp = old_data_dir, new_data_ffp=new_data_dir,make_plot=True)
#
#     test = helpers.check_residual(record_name = record_name, old_data_ffp = old_data_dir, new_data_ffp=new_data_dir, allowed_percent_not_close_to_zero=5)
#


print("starting check")

#allowed_percentages_not_close_to_zero = np.array([5])
#allowed_percentages_not_close_to_zero = np.arange(5,50,5)
#allowed_percentages_not_close_to_zero = np.array([5])

allowed_percentages_not_close_to_zero = 10.0
max_allowed_resid_as_pc_of_mean_vect = np.arange(10,110,10)
#max_allowed_resid_as_pc_of_mean_vect = np.array([50,60])#
num_inconsistent_records = np.zeros_like(max_allowed_resid_as_pc_of_mean_vect)

concat_results_df = pd.DataFrame()

#for index, allowed_percentage_not_close_to_zero in tqdm(enumerate(allowed_percentages_not_close_to_zero)):
for index, max_allowed_resid_as_pc_of_mean in tqdm(enumerate(max_allowed_resid_as_pc_of_mean_vect),total=len(max_allowed_resid_as_pc_of_mean_vect)):

    check_residual_partial = functools.partial(helpers.check_residual,
                                               old_data_ffp=old_data_dir,
                                               new_data_ffp=new_data_dir,
                                               max_allowed_resid_as_pc_of_mean = max_allowed_resid_as_pc_of_mean,
                                               allowed_percent_not_close_to_zero=allowed_percentages_not_close_to_zero)

    with multiprocessing.Pool(processes=8) as pool:
        record_checks = pool.map(check_residual_partial, ids_to_check)
    num_inconsistent_records[index] = sum(~np.array(record_checks))

    inconsistent_record_names = list(np.array(ids_to_check)[~np.array(record_checks)])

    inconsistent_record_names_str = " ".join(inconsistent_record_names)

    results_df = pd.DataFrame({"allowed_percentages_not_close_to_zero": [allowed_percentages_not_close_to_zero],
                               "max_allowed_resid_as_pc_of_mean": [max_allowed_resid_as_pc_of_mean],
                            "num_inconsistent_records": [num_inconsistent_records[index]],
                            "num_records_in_old_and_new": [len(ids_to_check)],
                            "percent_inconsistent_records": [100*num_inconsistent_records[index]/len(ids_to_check)],
                             "inconsistent_record_names":[inconsistent_record_names_str]})
                             #  "inconsistent_record_names": ["placeholder"]})

    concat_results_df = pd.concat([concat_results_df,results_df],ignore_index=True)

concat_results_df.to_csv(check_output_dir / "results.csv")

print(f"time taken: {(time.time()-start_time)/60} minutes")

