import natsort
import pandas as pd
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import sys

vs_calc_path = Path("/home/arr65/src/vs30/VsViewer")
sys.path.append(str(vs_calc_path))

def batch_list(input_list, num_batches):
    """
    Splits the input list into the specified number of batches.

    Parameters:
    input_list (list): The list to be split into batches.
    num_batches (int): The number of batches.

    Returns:
    list: A list of lists, where each inner list is a batch.
    """
    batch_size = len(input_list) // num_batches
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

#np.seterr(divide='ignore', invalid='ignore')

data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")
meta_dir = data_dir.parent / "metadata"



all_files = list(data_dir.glob("*.parquet"))

batched_file_list = batch_list(all_files, 8)

# for i in batched_file_list:
#     print(len(i))

batch_index_to_do = 2  ### For 8 batches, this can be up to 8
file_list = batched_file_list[batch_index_to_do]

output_dir = meta_dir / "batch_vs30_results_bad_files"
output_dir.mkdir(parents=True, exist_ok=True)

file_name = f"batch_{batch_index_to_do}_vs30_results.csv"

#file_list.reverse()



#vs30s = []

file_list = [Path("/home/arr65/data/nzgd/processed_data/cpt/data/CPT_198490.parquet")]

results_df = pd.DataFrame(columns=["cpt_name", "vs30", "time_taken"])

for file_index, file in tqdm(enumerate(file_list), total=len(file_list)):

    print(f"Processing file {file.stem}")

    file_start_time = time.time()

    cpt_df_repeat_measures = pd.read_parquet(file)

    ###################################

    max_depth_row = cpt_df_repeat_measures[cpt_df_repeat_measures["Depth"] == cpt_df_repeat_measures["Depth"].max()]
    multiple_measurements_index = max_depth_row["multiple_measurements"].values[0]
    cpt_df = cpt_df_repeat_measures[cpt_df_repeat_measures["multiple_measurements"] == multiple_measurements_index]

    ### fs == 0 results in infinite values due to log(Fr) in calc_cpt_params
    cpt_df = cpt_df[cpt_df["fs"] > 0]

    if cpt_df.size == 0:
        raise ValueError("No valid measurements in the CPT")

    cpt = vs_calc.CPT(
        cpt_df["record_name"].values[0],
        cpt_df["Depth"].values,
        cpt_df["qc"].values,
        cpt_df["fs"].values,
        cpt_df["u"].values)

    vs30_correlation = "boore_2011"
    cpt_vs_correlation = "andrus_2007"

    try:
        cpt_vs_profile = vs_calc.VsProfile.from_cpt(cpt, cpt_vs_correlation)
        cpt_vs_profile.vs30_correlation = vs30_correlation
        vs30 = cpt_vs_profile.vs30
        vs30_sd = cpt_vs_profile.vs30_sd
    except:
        vs30 = np.nan
        vs30_sd = np.nan

    time_taken_for_file = time.time() - file_start_time

    results_df_row = pd.DataFrame(
        {
            "cpt_name": [cpt.name],
            "cpt_correlation": [cpt_vs_correlation],
            "vs30_correlation": [vs30_correlation],
            "vs30": [vs30],
            "vs30_sd": [vs30_sd],
            "time_taken_ms": [time_taken_for_file / 1000.0]
        }
    )

    results_df = pd.concat([results_df, results_df_row], ignore_index=True)

    if file_index % 10 == 0:
        results_df.to_csv(output_dir / f"vs30_batch_{batch_index_to_do}.csv")




