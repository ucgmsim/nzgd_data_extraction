
import matplotlib.pyplot as plt

import pandas as pd
import scipy
import random

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker



import download_nzgd_data.validation.load_sql_db as load_sql_db
import download_nzgd_data.validation.helpers as helpers

import numpy as np
from tqdm import tqdm

old_data_dir = Path("/home/arr65/vs30_data_input_data/sql")
new_converted_data_dir = Path("/home/arr65/data/nzgd/processed_data_copy/cpt/data")

engine = create_engine(f"sqlite:///{old_data_dir}/nz_cpt.db")
DBSession = sessionmaker(bind=engine)
session = DBSession()

cpt_locs = load_sql_db.cpt_locations(session)
initial_num_cpts = len(cpt_locs)

inconsistent_cpts = []
old_cpt_not_in_new = []

print("loading CPTs")

record_ids_in_old = []

#cpt_locs = cpt_locs[152:154]


for cpt_loc in tqdm(cpt_locs):
    record_ids_in_old.append(cpt_loc.name)


#random.shuffle(cpt_locs)

for row_n, cpt_loc in tqdm(enumerate(cpt_locs), total=len(cpt_locs)):

    cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)

    if cpt_records.size == 0:
        inconsistent_cpts.append(cpt_loc.name)
        continue

    elif cpt_records.shape[0] < 4:
        inconsistent_cpts.append(cpt_loc.name)
        continue

    old_df = pd.DataFrame(cpt_records, columns=["Depth", "qc", "fs", "u"])

    parquet_to_load = new_converted_data_dir / f"{cpt_loc.name}.parquet"

    if not parquet_to_load.exists():
        old_cpt_not_in_new.append(cpt_loc.name)
        continue

    new_df_with_metadata = pd.read_parquet(parquet_to_load)
    new_df_with_metadata = new_df_with_metadata[new_df_with_metadata["multiple_measurements"] == 0]
    new_df = new_df_with_metadata.drop(columns=["multiple_measurements", "record_name", "latitude", "longitude"])

    band_width_percent = 30

    new_df_absolute_value = new_df.copy()
    new_df_absolute_value[["qc","fs","u"]] *= (band_width_percent/100)



    new_df_upper = new_df.copy()
    new_df_lower = new_df.copy()

    new_df_upper[["qc","fs","u"]] += new_df_absolute_value[["qc","fs","u"]]
    new_df_lower[["qc", "fs", "u"]] -= new_df_absolute_value[["qc", "fs", "u"]]

    interpolated_new_df_upper = helpers.get_interpolated_df(helpers.organise_with_depth_range(old_df, new_df_upper))
    interpolated_new_df_lower = helpers.get_interpolated_df(helpers.organise_with_depth_range(old_df, new_df_lower))

    old_lower_than_new_upper = old_df[["qc","fs","u"]] < interpolated_new_df_upper[["qc","fs","u"]]
    old_higher_than_new_lower = old_df[["qc","fs","u"]] > interpolated_new_df_lower[["qc","fs","u"]]

    old_within_band = old_lower_than_new_upper & old_higher_than_new_lower
    #100 * (len(old_within_band) - old_within_band.sum()) / len(old_within_band)
    percent_not_within_band = 100*(len(old_within_band) - old_within_band.sum()) / len(old_within_band)

    allowed_percent_of_points_not_within_band = 10

    if (percent_not_within_band < allowed_percent_of_points_not_within_band).all():
        print(f"cpt: {cpt_loc.name}")
        print(f"old_df: {old_df[~old_within_band]}")
        print(f"new_df_upper: {interpolated_new_df_upper[~old_within_band]}")
        print(f"new_df_lower: {interpolated_new_df_lower[~old_within_band]}")
        print()

    ############################################################

        fig, axes = plt.subplots(3, 1)

        linewidth1 = 10
        linewidth2 = 4
        linewidth3 = 2

        axes[0].plot(new_df_upper["Depth"],new_df_upper["qc"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        axes[0].plot(new_df_lower["Depth"],new_df_lower["qc"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        axes[0].plot(old_df["Depth"], old_df["qc"], linestyle="-", color="red",label="old", linewidth=linewidth3,marker="o")
        axes[0].legend()
        axes[0].set_xlabel("Depth (m)")
        axes[0].set_ylabel("qc (MPa)")
        axes[0].set_title(f"{cpt_loc.name}")

        #####################################################

        axes[1].plot(new_df_upper["Depth"],new_df_upper["fs"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        axes[1].plot(new_df_lower["Depth"],new_df_lower["fs"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        axes[1].plot(old_df["Depth"], old_df["fs"], linestyle="-", color="red",label="old", linewidth=linewidth2)
        axes[1].legend()
        axes[1].set_xlabel("Depth (m)")
        axes[1].set_ylabel("qc (MPa)")

        #####################################################

        axes[2].plot(new_df_upper["Depth"],new_df_upper["u"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        axes[2].plot(new_df_lower["Depth"],new_df_lower["u"], linestyle="-", color="blue",label="new", linewidth=linewidth3,marker="*")
        axes[2].plot(old_df["Depth"], old_df["u"], linestyle="-", color="red",label="old", linewidth=linewidth2)
        axes[2].legend()
        axes[2].set_xlabel("Depth (m)")
        axes[2].set_ylabel("qc (MPa)")

        plt.subplots_adjust(hspace=0.0)

        plt.show()

        print()