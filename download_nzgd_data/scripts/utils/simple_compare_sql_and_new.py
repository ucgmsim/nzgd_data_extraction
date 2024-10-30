
import matplotlib.pyplot as plt

import pandas as pd
import scipy
import random

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import multiprocessing
import time




import download_nzgd_data.validation.load_sql_db as load_sql_db
import download_nzgd_data.validation.helpers as helpers

import numpy as np
from tqdm import tqdm

import functools

start_time = time.time()


old_data_dir = Path("/home/arr65/vs30_data_input_data/sql")
old_data_as_parquet_dir = Path("/home/arr65/vs30_data_input_data/parquet")



## new_converted_data_dir = Path("/home/arr65/data/nzgd/processed_data_copy/cpt/data")

## helpers.convert_old_sql_to_parquet(old_data_dir, old_data_as_parquet_dir)

print("converted old data to parquet")


# engine = create_engine(f"sqlite:///{old_data_dir}/nz_cpt.db")
# DBSession = sessionmaker(bind=engine)
# session = DBSession()
#
# make_df_list_partial = functools.partial(helpers.make_df_list, new_converted_data_dir=new_converted_data_dir, session=session)
#
# cpt_locs = load_sql_db.cpt_locations(session)
#
# cpt_locs = cpt_locs[:100]
#
# initial_num_cpts = len(cpt_locs)
#
# inconsistent_cpts = []
# old_cpt_not_in_new = []
#
# print("loading CPTs")
#
# new_ids = [file.with_suffix("").name for file in new_converted_data_dir.glob("*.parquet")]
#
# record_ids_in_old = []
#
# #cpt_locs = cpt_locs[152:154]
#
# ids_in_both = []
#
# for cpt_loc in tqdm(cpt_locs):
#
#     record_ids_in_old.append(cpt_loc.name)
#     if cpt_loc.name in new_ids:
#         ids_in_both.append(cpt_loc.name)
#
#
# list_of_sql_dfs = helpers.get_list_of_sql_dfs(cpt_locs, session)
#
# print()

# with multiprocessing.Pool(processes=6) as pool:
#     dfs = pool.map(make_df_list_partial, cpt_locs)
#
# with multiprocessing.Pool(processes=6) as pool:
#     results = pool.map(helpers.check_for_consistency, dfs)
#
# print(f"time taken: {(time.time()-start_time)/60} minutes")





    ############################################################

        # fig, axes = plt.subplots(3, 1)
        #
        # linewidth1 = 10
        # linewidth2 = 4
        # linewidth3 = 2
        #
        # axes[0].plot(new_df_upper["Depth"],new_df_upper["qc"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        # axes[0].plot(new_df_lower["Depth"],new_df_lower["qc"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        # axes[0].plot(old_df["Depth"], old_df["qc"], linestyle="-", color="red",label="old", linewidth=linewidth3,marker="o")
        # axes[0].legend()
        # axes[0].set_xlabel("Depth (m)")
        # axes[0].set_ylabel("qc (MPa)")
        # axes[0].set_title(f"{cpt_loc.name}")
        #
        # #####################################################
        #
        # axes[1].plot(new_df_upper["Depth"],new_df_upper["fs"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        # axes[1].plot(new_df_lower["Depth"],new_df_lower["fs"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        # axes[1].plot(old_df["Depth"], old_df["fs"], linestyle="-", color="red",label="old", linewidth=linewidth2)
        # axes[1].legend()
        # axes[1].set_xlabel("Depth (m)")
        # axes[1].set_ylabel("qc (MPa)")
        #
        # #####################################################
        #
        # axes[2].plot(new_df_upper["Depth"],new_df_upper["u"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        # axes[2].plot(new_df_lower["Depth"],new_df_lower["u"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        # axes[2].plot(old_df["Depth"], old_df["u"], linestyle="-", color="red",label="old", linewidth=linewidth2)
        # axes[2].legend()
        # axes[2].set_xlabel("Depth (m)")
        # axes[2].set_ylabel("qc (MPa)")
        #
        # plt.subplots_adjust(hspace=0.0)
        #
        # plt.show()
        #
        # print()