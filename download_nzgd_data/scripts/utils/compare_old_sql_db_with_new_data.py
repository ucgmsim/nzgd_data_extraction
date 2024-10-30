
import matplotlib.pyplot as plt

import pandas as pd
import scipy

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

inconsistent_cpts = []
old_cpt_not_in_new = []

print("loading CPTs")

record_ids_in_old = []

#cpt_locs = cpt_locs[152:154]

for cpt_loc in tqdm(cpt_locs):
    record_ids_in_old.append(cpt_loc.name)

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
    # new_df.loc[:, "Depth"] += 1



    residual = interpolated_df - organised_with_depth_range.shortest_depth_range

    qc_sigma_clip_mask = helpers.sigma_clip_indices(residual["qc"].values, 3)
    fs_sigma_clip_mask = helpers.sigma_clip_indices(residual["fs"].values, 3)
    u_sigma_clip_mask = helpers.sigma_clip_indices(residual["u"].values, 3)

    rel_qc_resid = np.abs(residual["qc"].values[qc_sigma_clip_mask]) / np.median(organised_with_depth_range.shortest_depth_range["qc"].values[qc_sigma_clip_mask])
    rel_fs_resid = np.abs(residual["fs"].values[fs_sigma_clip_mask]) / np.median(organised_with_depth_range.shortest_depth_range["fs"].values[fs_sigma_clip_mask])
    rel_u_resid = np.abs(residual["u"].values[u_sigma_clip_mask]) / np.median(organised_with_depth_range.shortest_depth_range["u"].values[u_sigma_clip_mask])

    rel_qc_resid_depth = organised_with_depth_range.shortest_depth_range["Depth"].values[qc_sigma_clip_mask]
    rel_fs_resid_depth = organised_with_depth_range.shortest_depth_range["Depth"].values[fs_sigma_clip_mask]
    rel_u_resid_depth = organised_with_depth_range.shortest_depth_range["Depth"].values[u_sigma_clip_mask]

    # print()
    #
    # data_median = organised_with_depth_range.shortest_depth_range.median()
    #
    # relative_residual = np.abs(residual) / data_median
    #
    # # residual = scipy.stats.sigmaclip(residual.to_numpy, low=5, high=5)
    #
    # chi2 = np.sum(residual**2/organised_with_depth_range.shortest_depth_range)
    #
    # max_allowed_fractional_residual = 3e-2
    # num_mismatch_points = (np.abs(relative_residual) > max_allowed_fractional_residual).sum()
    #
    # fraction_mismatch_points = num_mismatch_points / relative_residual.shape[0]
    #
    # ### 3 sigma should include 99.7% of the data
    #
    # fraction_of_mismatched_points_threshold = 3e-2

    #if any(fraction_mismatch_points > fraction_of_mismatched_points_threshold):
    if any(rel_qc_resid > 3e-2) or any(rel_fs_resid > 3e-2) or any(rel_u_resid > 3e-2):
        # import matplotlib as mpl
        # mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

        fig, axes = plt.subplots(2, 3, sharex=True)


        axes[0, 0].plot(organised_with_depth_range.largest_depth_range["Depth"],
                        organised_with_depth_range.largest_depth_range["qc"], linestyle="--", marker="+", color="grey",label="new")
        axes[0, 0].plot(organised_with_depth_range.shortest_depth_range["Depth"],
                        organised_with_depth_range.shortest_depth_range["qc"], linestyle="--", color="green", marker="o", markersize=10, markerfacecolor='none', markeredgecolor='green', label="old")

        axes[0, 0].plot(interpolated_df["Depth"], interpolated_df["qc"], linestyle="--", marker="x", color="red",label="interp")
        axes[0, 0].legend()
        axes[0, 0].set_xlabel("Depth (m)")
        axes[0, 0].set_ylabel("qc (MPa)")

        axes[0, 1].plot(organised_with_depth_range.largest_depth_range["Depth"],
                        organised_with_depth_range.largest_depth_range["fs"], linestyle="--", marker="+", color="grey",label="new")
        axes[0, 1].plot(organised_with_depth_range.shortest_depth_range["Depth"],
                        organised_with_depth_range.shortest_depth_range["fs"], linestyle="--", color="green", marker="o", markersize=10, markerfacecolor='none', markeredgecolor='green', label="old")
        axes[0, 1].plot(interpolated_df["Depth"], interpolated_df["fs"], linestyle="--", marker="x", color="red",label="interp")
        axes[0, 1].legend()
        axes[0, 1].set_xlabel("Depth (m)")
        axes[0, 1].set_ylabel("fs (MPa)")

        axes[0, 1].plot(organised_with_depth_range.largest_depth_range["Depth"],
                        organised_with_depth_range.largest_depth_range["u"], linestyle="--", marker="+", color="grey",label="new")
        axes[0, 1].plot(organised_with_depth_range.shortest_depth_range["Depth"],
                        organised_with_depth_range.shortest_depth_range["u"], linestyle="--", color="green", marker="o", markersize=10, markerfacecolor='none', markeredgecolor='green', label="old")
        axes[0, 1].plot(interpolated_df["Depth"], interpolated_df["u"], linestyle="--", marker="x", color="red",label="interp")
        axes[0, 1].legend()
        axes[0, 1].set_xlabel("Depth (m)")
        axes[0, 1].set_ylabel("u (MPa)")

        ###################################################################

        axes[1, 0].plot(rel_qc_resid_depth, rel_qc_resid, linestyle="--",marker="o",markersize=5)
        axes[1, 0].set_xlabel("Depth (m)")
        axes[1, 0].set_ylabel("qc abs(resid)/med (MPa)")

        print()

        axes[1, 1].plot(rel_fs_resid_depth, rel_fs_resid, linestyle="--",marker="o",markersize=5)
        axes[1, 1].set_xlabel("Depth (m)")
        axes[1, 1].set_ylabel("fs abs(resid)/med (MPa)")

        axes[1, 2].plot(rel_u_resid_depth, rel_u_resid, linestyle="--",marker="o",markersize=5)
        axes[1, 2].set_xlabel("Depth (m)")
        axes[1, 2].set_ylabel("qc abs(resid)/med (MPa)")

        plt.show()
        print()

        # denominator_of_range_adjustment = 4
        #
        # axes[1, 0].set_title(f"{fraction_mismatch_points["qc"]:.2e}")
        # axes[1, 0].plot(interpolated_df["Depth"], relative_residual["qc"], linestyle="--",marker="o",markersize=5)
        # #axes[1, 0].set_ylim(axes[0, 0].get_ylim() - axes[0, 0].get_ylim()[1] / denominator_of_range_adjustment)
        # axes[1, 0].set_xlabel("Depth (m)")
        # axes[1, 0].set_ylabel("qc abs(resid)/med (MPa)")
        #
        # axes[1, 1].set_title(f"{fraction_mismatch_points["fs"]:.2e}")
        # axes[1, 1].plot(interpolated_df["Depth"], relative_residual["fs"], linestyle="--",marker="o",markersize=5)
        # #axes[1, 1].set_ylim(axes[1, 1].get_ylim() - axes[1, 1].get_ylim()[1] / denominator_of_range_adjustment)
        # axes[1, 1].set_xlabel("Depth (m)")
        # axes[1, 1].set_ylabel("fs abs(resid)/med (MPa)")
        #
        # axes[1, 2].set_title(f"{fraction_mismatch_points["u"]:.2e}")
        # axes[1, 2].plot(interpolated_df["Depth"], relative_residual["u"], linestyle="--",marker="o",markersize=5)
        # #axes[1, 2].set_ylim(axes[1, 2].get_ylim() - axes[1, 2].get_ylim()[1] / denominator_of_range_adjustment)
        # axes[1, 2].set_xlabel("Depth (m)")
        # axes[1, 2].set_ylabel("qc abs(resid)/med (MPa)")

        ###################################################################
        #
        # axes[2, 0].set_title(f"{fraction_mismatch_points["qc"]:.2e}")
        # axes[2, 0].plot(interpolated_df["Depth"], residual["qc"], linestyle="--",marker="o",markersize=5)
        # axes[2, 0].set_xlabel("Depth (m)")
        # axes[2, 0].set_ylabel("qc resid (MPa)")
        #
        # axes[2, 1].set_title(f"{fraction_mismatch_points["fs"]:.2e}")
        # axes[2, 1].plot(interpolated_df["Depth"], residual["fs"], linestyle="--",marker="o",markersize=5)
        # axes[2, 1].set_xlabel("Depth (m)")
        # axes[2, 1].set_ylabel("fs resid (MPa)")
        #
        # axes[2, 2].set_title(f"{fraction_mismatch_points["u"]:.2e}")
        # axes[2, 2].plot(interpolated_df["Depth"], residual["u"], linestyle="--",marker="o",markersize=5)
        # axes[2, 2].set_xlabel("Depth (m)")
        # axes[2, 2].set_ylabel("qc resid (MPa)")
        #
        # plt.subplots_adjust(hspace=0.2, wspace=0.2)



print()
