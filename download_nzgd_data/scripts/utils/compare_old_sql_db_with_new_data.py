
import matplotlib.pyplot as plt

import pandas as pd


from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker



import download_nzgd_data.validation.load_sql_db as load_sql_db
import download_nzgd_data.validation.helpers as helpers

import numpy as np
from tqdm import tqdm

## old_to_new_match_df = pd.read_csv("/home/arr65/data/nzgd/stats_plots/sung_id_with_nzgd_match.csv")


old_data_dir = Path("/home/arr65/vs30_data_input_data/sql")
new_converted_data_dir = Path("/home/arr65/data/nzgd/processed_data_copy/cpt/data")

engine = create_engine(f"sqlite:///{old_data_dir}/nz_cpt.db")
DBSession = sessionmaker(bind=engine)
session = DBSession()

cpt_locs = load_sql_db.cpt_locations(session)
initial_num_cpts = len(cpt_locs)

# cpt_locs = cpt_locs[0:100]

inconsistent_cpts = []
old_cpt_not_in_new = []

print("loading CPTs")

record_ids_in_old = []

#cpt_locs = cpt_locs[0:100]

for cpt_loc in tqdm(cpt_locs):
    record_ids_in_old.append(cpt_loc.name)

for row_n, cpt_loc in tqdm(enumerate(cpt_locs), total=len(cpt_locs)):

    if row_n % 1000 == 0:  # print every 1000
        print(f"{row_n + 1}/{len(cpt_locs)}: {cpt_loc.name}")

    cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)

    old_df = pd.DataFrame(cpt_records, columns=["Depth", "qc", "fs", "u"])


    parquet_to_load = new_converted_data_dir / f"{cpt_loc.name}.parquet"

    if not parquet_to_load.exists():
        old_cpt_not_in_new.append(cpt_loc.name)
        continue

    new_df_with_metadata = pd.read_parquet(parquet_to_load)
    new_df_with_metadata = new_df_with_metadata[new_df_with_metadata["multiple_measurements"] == 0]
    new_df = new_df_with_metadata.drop(columns=["multiple_measurements", "record_name", "latitude", "longitude"])
    # new_df.loc[:, "Depth"] += 1

    organised_with_depth_range = helpers.organise_with_depth_range(old_df, new_df)

    interpolated_df = helpers.get_interpolated_df(organised_with_depth_range)

    residual = interpolated_df - organised_with_depth_range.shortest_depth_range

    data_range = organised_with_depth_range.shortest_depth_range.max() - organised_with_depth_range.shortest_depth_range.min()

    relative_residual = residual / data_range

    # residual = scipy.stats.sigmaclip(residual.to_numpy, low=5, high=5)

    fractional_residual = residual / organised_with_depth_range.shortest_depth_range

    max_allowed_fractional_residual = 1e-2
    num_mismatch_points = (fractional_residual > max_allowed_fractional_residual).sum()

    fraction_mismatch_points = num_mismatch_points / fractional_residual.shape[0]

    ### 3 sigma should include 99.7% of the data

    fraction_of_mismatched_points_threshold = 1e-2

    if any(fraction_mismatch_points > fraction_of_mismatched_points_threshold):
        # import matplotlib as mpl
        # mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

        fig, axes = plt.subplots(2, 3)

        axes[0, 0].plot(interpolated_df["Depth"], interpolated_df["qc"], label="interpolated with largest depth range")
        axes[0, 0].plot(organised_with_depth_range.shortest_depth_range["Depth"],
                        organised_with_depth_range.shortest_depth_range["qc"], linestyle="--",
                        label="original with shortest depth range")

        axes[0, 1].plot(interpolated_df["Depth"], interpolated_df["fs"], label="interpolated with largest depth range")
        axes[0, 1].plot(organised_with_depth_range.shortest_depth_range["Depth"],
                        organised_with_depth_range.shortest_depth_range["fs"], linestyle="--",
                        label="original with shortest depth range")

        axes[0, 2].plot(interpolated_df["Depth"], interpolated_df["u"], label="interpolated with largest depth range")
        axes[0, 2].plot(organised_with_depth_range.shortest_depth_range["Depth"],
                        organised_with_depth_range.shortest_depth_range["u"], linestyle="--",
                        label="original with shortest depth range")

        ###################################################################

        denominator_of_range_adjustment = 4

        axes[1, 0].set_title(f"{fraction_mismatch_points["qc"]:.2e}")
        axes[1, 0].plot(interpolated_df["Depth"], fractional_residual["qc"], linestyle="-")
        axes[1, 0].set_ylim(axes[0, 0].get_ylim() - axes[0, 0].get_ylim()[1] / denominator_of_range_adjustment)

        axes[1, 1].set_title(f"{fraction_mismatch_points["fs"]:.2e}")
        axes[1, 1].plot(interpolated_df["Depth"], fractional_residual["fs"])
        axes[1, 1].set_ylim(axes[1, 1].get_ylim() - axes[1, 1].get_ylim()[1] / denominator_of_range_adjustment)

        axes[1, 2].set_title(f"{fraction_mismatch_points["u"]:.2e}")
        axes[1, 2].plot(interpolated_df["Depth"], fractional_residual["u"])
        axes[1, 2].set_ylim(axes[1, 2].get_ylim() - axes[1, 2].get_ylim()[1] / denominator_of_range_adjustment)

        plt.subplots_adjust(hspace=0.5, wspace=0.5)

        plt.show()
        print()

print()
