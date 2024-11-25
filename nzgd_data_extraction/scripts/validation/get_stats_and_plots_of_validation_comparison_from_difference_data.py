

import matplotlib.pyplot as plt

import pandas as pd
import scipy
import random

import numpy as np

from pathlib import Path

from sqlalchemy import column
from tqdm import tqdm
from datetime import datetime

import time

import nzgd_data_extraction.validation.helpers as helpers


start_time = time.time()

current_date = datetime.now().date()

# Convert the date to a string
current_date_str = current_date.strftime("%Y-%m-%d")

old_data_dir = Path("/home/arr65/vs30_data_input_data/parquet/data")
new_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")

data_validation_check_output_dir = Path(f"/home/arr65/data/nzgd/validation_checks/{current_date_str}/processed_data")

plot_output_dir = data_validation_check_output_dir / "plots"
plot_output_dir.mkdir(parents=True, exist_ok=True)

identified_issues = data_validation_check_output_dir / "identified_issues"
identified_issues.mkdir(parents=True, exist_ok=True)

data_validation_check_output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(f"{data_validation_check_output_dir}/inconsistent_records_with_variable_thresholds.csv")
record_names_in_both_old_and_new_datasets = pd.read_csv("/home/arr65/data/nzgd/resources/record_names_in_old_and_new_datasets.csv")["record_names_in_old_and_new_datasets"].to_list()


inconsistent_record_names = df.loc[0,"inconsistent_record_names"]
inconsistent_record_names = inconsistent_record_names.split(" ")

uncategorized_issues = []

old_id_qc_0 = []
old_id_fs_0 = []
old_id_u_0 = []

old_id_qc_less_than_zero = []
old_id_fs_less_than_zero = []

new_id_qc_0 = []
new_id_fs_0 = []
new_id_u_0 = []

new_id_qc_less_than_0 = []
new_id_fs_less_than_0 = []

new_id_qc_10x = []
new_id_fs_10x = []
new_id_u_10x = []

old_id_qc_10x = []
old_id_fs_10x = []
old_id_u_10x = []

inconsistent_record_counter = 0

for record_name in tqdm(inconsistent_record_names):

    residual, old_df, interpolated_df, new_df = helpers.get_residual(record_name = record_name,
                                                                     old_data_ffp = old_data_dir,
                                                                     new_data_ffp=new_data_dir)

    if np.isclose(old_df["u"],0).all():
        old_id_u_0.append(record_name)
        plot_subdir = plot_output_dir / "old_u_all_zeros"
    elif np.isclose(new_df["u"],0).all():
        new_id_u_0.append(record_name)
        plot_subdir = plot_output_dir / "new_u_all_zeros"

    elif np.isclose(old_df["qc"],0).all():
        old_id_qc_0.append(record_name)
        plot_subdir = plot_output_dir / "old_qc_all_zeros"
    elif old_df["qc"].min() < 0:
        old_id_qc_less_than_zero.append(record_name)
        plot_subdir = plot_output_dir / "old_qc_less_than_0"

    elif np.isclose(new_df["qc"],0).all():
        new_id_qc_0.append(record_name)
        plot_subdir = plot_output_dir / "new_qc_all_zeros"
    elif new_df["qc"].min() < 0:
        new_id_qc_less_than_0.append(record_name)
        plot_subdir = plot_output_dir / "new_qc_less_than_0"

    elif np.isclose(old_df["fs"],0).all():
        old_id_fs_0.append(record_name)
        plot_subdir = plot_output_dir / "old_fs_all_zeros"
    elif old_df["fs"].min() < 0:
        old_id_fs_less_than_zero.append(record_name)
        plot_subdir = plot_output_dir / "old_fs_less_than_0"

    elif np.isclose(new_df["fs"],0).all():
        new_id_fs_0.append(record_name)
        plot_subdir = plot_output_dir / "new_fs_all_zeros"
    elif new_df["fs"].min() < 0:
        new_id_fs_less_than_0.append(record_name)
        plot_subdir = plot_output_dir / "new_fs_less_than_0"

     ### if the max of one data set is 10 times the max of the other data set
    elif new_df["qc"].max() > 10*old_df["qc"].max():
        new_id_qc_10x.append(record_name)
        plot_subdir = plot_output_dir / "new_qc_10x"
    elif new_df["fs"].max() > 10*old_df["fs"].max():
        new_id_fs_10x.append(record_name)
        plot_subdir = plot_output_dir / "new_fs_10x"
    elif new_df["u"].max() > 10*old_df["u"].max():
        new_id_u_10x.append(record_name)
        plot_subdir = plot_output_dir / "new_u_10x"
    elif old_df["qc"].max() > 10*new_df["qc"].max():
        old_id_qc_10x.append(record_name)
        plot_subdir = plot_output_dir / "old_qc_10x"
    elif old_df["fs"].max() > 10*new_df["fs"].max():
        old_id_fs_10x.append(record_name)
        plot_subdir = plot_output_dir / "old_fs_10x"
    elif old_df["u"].max() > 10*new_df["u"].max():
        old_id_u_10x.append(record_name)
        plot_subdir = plot_output_dir / "old_u_10x"

    else:
        plot_subdir = plot_output_dir / "uncategorized_issues"
        uncategorized_issues.append(record_name)
    plot_subdir.mkdir(parents=True, exist_ok=True)
    helpers.plot_residual(residual, old_df, interpolated_df, new_df, record_name=record_name, plot_output_dir=plot_subdir)
    inconsistent_record_counter += 1


def make_summary_df_line(description, num_points_inconsistent_array, num_inconsistent_records, num_old_with_u_0, num_all_records=34663.0):
    summary_df_line = pd.DataFrame({"description": [description],
                  "number_of_records_with_this_issue": [num_points_inconsistent_array],
                  "number_of_inconsistent_records": [num_inconsistent_records],
                  "number_of_records_both_in_old_and_new_for_comparison": [num_all_records],
                  "number_as_percent_of_inconsistent_records": [100.0*num_points_inconsistent_array/num_inconsistent_records],
                  "number_as_percent_of_all_records_in_both_old_and_new": [100.0*num_points_inconsistent_array/num_all_records],
                  "number_as_percent_of_inconsistent_records_excluding_old_with_constant_u = 0": [100.0*num_points_inconsistent_array/(num_inconsistent_records-num_old_with_u_0)],
                  "number_as_percent_of_all_records_in_both_old_and_new_excluding_old_with_constant_u = 0": [100*num_points_inconsistent_array/(num_all_records-num_old_with_u_0)]})

    return summary_df_line

summary_df = pd.concat([
    make_summary_df_line("number with constant u = 0 in old data", len(old_id_u_0), len(inconsistent_record_names), np.nan), ### subtituting np.nan for num_old_with_u_0 is not applicable since we use it to find the percentage excluding the old records with constant u = 0 (which would exclude itself)
    make_summary_df_line("number with constant qc = 0 in old data", len(old_id_qc_0), len(inconsistent_record_names), np.nan),
    make_summary_df_line("number with constant fs = 0 in old data", len(old_id_fs_0), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("number with constant u = 0 in new data", len(new_id_u_0), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("number with constant qc = 0 in new data", len(new_id_qc_0), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("number with constant fs = 0 in new data", len(new_id_fs_0), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("number in old data with u > (10*new u)", len(old_id_u_10x), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("number old data with qc > (10*new qc)", len(old_id_qc_10x), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("number in old data with fs > (10*new fs)", len(old_id_fs_10x), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("number in new data with u > (10*old u)", len(new_id_u_10x), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("number in new data with qc > (10*old qc)", len(new_id_qc_10x), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("number in new with fs > (10*old fs)", len(new_id_fs_10x), len(inconsistent_record_names), len(old_id_u_0)),
    make_summary_df_line("uncategorized issues", len(uncategorized_issues), len(inconsistent_record_names), len(old_id_u_0))],
    ignore_index=True)

summary_df.to_csv(identified_issues / "summary.csv")

## Save new_id_qc_0 as a text file in the directory identified_issues
np.savetxt(identified_issues / "old_id_qc_0.txt", np.array(old_id_qc_0), fmt="%s")
np.savetxt(identified_issues / "new_id_qc_0.txt", np.array(new_id_qc_0), fmt="%s")
np.savetxt(identified_issues / "old_id_fs_0.txt", np.array(old_id_fs_0), fmt="%s")
np.savetxt(identified_issues / "new_id_fs_0.txt", np.array(new_id_fs_0), fmt="%s")
np.savetxt(identified_issues / "old_id_u_0.txt", np.array(old_id_u_0), fmt="%s")
np.savetxt(identified_issues / "new_id_u_0.txt", np.array(new_id_u_0), fmt="%s")

np.savetxt(identified_issues / "new_id_qc_10x.txt", np.array(new_id_qc_10x), fmt="%s")
np.savetxt(identified_issues / "new_id_fs_10x.txt", np.array(new_id_fs_10x), fmt="%s")
np.savetxt(identified_issues / "new_id_u_10x.txt", np.array(new_id_u_10x), fmt="%s")

np.savetxt(identified_issues / "old_id_qc_10x.txt", np.array(old_id_qc_10x), fmt="%s")
np.savetxt(identified_issues / "old_id_fs_10x.txt", np.array(old_id_fs_10x), fmt="%s")
np.savetxt(identified_issues / "old_id_u_10x.txt", np.array(old_id_u_10x), fmt="%s")
