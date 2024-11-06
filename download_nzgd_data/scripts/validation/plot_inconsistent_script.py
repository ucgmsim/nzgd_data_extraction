

import matplotlib.pyplot as plt

import pandas as pd
import scipy
import random

import numpy as np

from pathlib import Path
from tqdm import tqdm

import time

import download_nzgd_data.validation.helpers as helpers


start_time = time.time()


old_data_dir = Path("/home/arr65/vs30_data_input_data/parquet/data")
new_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")



data_validation_check_output_dir = Path("/home/arr65/data/nzgd/validation_checks/processed_data")

plot_output_dir = data_validation_check_output_dir / "plots"
plot_output_dir.mkdir(parents=True, exist_ok=True)

identified_issues = data_validation_check_output_dir / "identified_issues"
identified_issues.mkdir(parents=True, exist_ok=True)

data_validation_check_output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(f"{data_validation_check_output_dir}/results.csv")

inconsistent_record_names = df.loc[0,"inconsistent_record_names"]
inconsistent_record_names = inconsistent_record_names.split(" ")

count_of_inconsistent_plots = 0

new_id_qc_0 = []
old_id_qc_0 = []

new_id_fs_0 = []
old_id_fs_0 = []

new_id_u_0 = []
old_id_u_0 = []

new_id_qc_10x = []
new_id_fs_10x = []
new_id_u_10x = []

old_id_qc_10x = []
old_id_fs_10x = []
old_id_u_10x = []

for record_name in tqdm(inconsistent_record_names):

    residual, old_df, interpolated_df, new_df = helpers.get_residual(record_name = record_name,
                                                                     old_data_ffp = old_data_dir,
                                                                     new_data_ffp=new_data_dir)

    if np.isclose(old_df["qc"],0).all():
        old_id_qc_0.append(record_name)
        plot_subdir = plot_output_dir / "old_qc_all_zeros"
    elif np.isclose(new_df["qc"],0).all():
        new_id_qc_0.append(record_name)
        plot_subdir = plot_output_dir / "new_qc_all_zeros"
    elif np.isclose(old_df["fs"],0).all():
        old_id_fs_0.append(record_name)
        plot_subdir = plot_output_dir / "old_fs_all_zeros"
    elif np.isclose(new_df["fs"],0).all():
        new_id_fs_0.append(record_name)
        plot_subdir = plot_output_dir / "new_fs_all_zeros"
    elif np.isclose(old_df["u"],0).all():
        old_id_u_0.append(record_name)
        plot_subdir = plot_output_dir / "old_u_all_zeros"
    elif np.isclose(new_df["u"],0).all():
        new_id_u_0.append(record_name)
        plot_subdir = plot_output_dir / "new_u_all_zeros"

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
        plot_subdir = plot_output_dir / "no_cols_all_zeros"
    plot_subdir.mkdir(parents=True, exist_ok=True)

    helpers.plot_residual(residual, old_df, interpolated_df, new_df, record_name=record_name, plot_output_dir=plot_subdir)

    count_of_inconsistent_plots += 1

print(f"count_of_inconsistent_plots: {count_of_inconsistent_plots}")

print(f"percent of inconsistent records with constant u = 0 in old data: {100*len(old_id_qc_0)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with constant u = 0 in new data: {100*len(new_id_qc_0)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with constant fs = 0 in old data: {100*len(old_id_fs_0)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with constant fs = 0 in new data: {100*len(new_id_fs_0)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with constant qc = 0 in old data: {100*len(old_id_u_0)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with constant qc = 0 in new data: {100*len(new_id_u_0)/len(inconsistent_record_names)}")

print(f"percent of inconsistent records with qc > 10*old qc in new data: {100*len(new_id_qc_10x)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with fs > 10*old fs in new data: {100*len(new_id_fs_10x)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with u > 10*old u in new data: {100*len(new_id_u_10x)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with qc > 10*new qc in old data: {100*len(old_id_qc_10x)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with fs > 10*new fs in old data: {100*len(old_id_fs_10x)/len(inconsistent_record_names)}")
print(f"percent of inconsistent records with u > 10*new u in old data: {100*len(old_id_u_10x)/len(inconsistent_record_names)}")



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
