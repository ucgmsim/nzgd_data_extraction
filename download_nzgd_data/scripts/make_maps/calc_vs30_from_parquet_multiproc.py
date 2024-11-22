import pandas as pd
import numpy as np
from pathlib import Path
import functools
import multiprocessing as mp
from tqdm import tqdm
import time
import sys

import natsort as natsort

vs_calc_path = Path("/home/arr65/src/vs30/VsViewer")
sys.path.append(str(vs_calc_path))

import vs_calc

def filter_out_files_with_insufficient_depth(file_paths, metadata_dir):
    sufficient_depths = []
    insufficient_depths = []

    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        if df["Depth"].max() > 5:
            sufficient_depths.append(file_path)
        else:
            insufficient_depths.append(file_path)

    record_ids_sufficient_depth = [file.stem for file in sufficient_depths]
    record_ids_insufficient_depth = [file.stem for file in insufficient_depths]

    pd.DataFrame(record_ids_sufficient_depth, columns=["record_name_sufficient_depth"]).to_csv(metadata_dir / "record_ids_sufficient_depth.csv")
    pd.DataFrame(record_ids_insufficient_depth, columns=["record_name_insufficient_depth"]).to_csv(metadata_dir / "record_ids_insufficient_depth.csv")

    return sufficient_depths

def calc_vs30_from_filename(file_path, cpt_vs_correlation, vs30_correlation):

    try:
        cpt_df_repeat_measures = pd.read_parquet(file_path)

        max_depth_row = cpt_df_repeat_measures[cpt_df_repeat_measures["Depth"] == cpt_df_repeat_measures["Depth"].max()]
        multiple_measurements_index = max_depth_row["multiple_measurements"].values[0]
        cpt_df = cpt_df_repeat_measures[cpt_df_repeat_measures["multiple_measurements"] == multiple_measurements_index]
        cpt_df = cpt_df[cpt_df["fs"] > 0]

        if cpt_df.size == 0:
            raise ValueError("No valid measurements in the CPT")

        cpt = vs_calc.CPT(
            cpt_df["record_name"].values[0],
            cpt_df["Depth"].values,
            cpt_df["qc"].values,
            cpt_df["fs"].values,
            cpt_df["u"].values)

        cpt_vs_profile = vs_calc.VsProfile.from_cpt(cpt, cpt_vs_correlation)
        cpt_vs_profile.vs30_correlation = vs30_correlation

        return pd.DataFrame({
            "record_name": cpt.name,
            "exception": None,
            "cpt_vs_correlation": cpt_vs_correlation,
            "vs30_correlation": vs30_correlation,
            "vs30": cpt_vs_profile.vs30,
            "vs30_sd": cpt_vs_profile.vs30_sd,
            "calculation_time_ms": time.time() - start_time,
            "latitude": cpt_df["latitude"].values[0],
            "longitude": cpt_df["longitude"].values[0],
            "min_depth_m": cpt_df["Depth"].min(),
            "max_depth_m": cpt_df["Depth"].max(),
            "depth_span_m": cpt_df["Depth"].max() - cpt_df["Depth"].min(),
            "num_depth_levels": cpt_df["Depth"].size
        }, index=[0])

    except Exception as e:
        return pd.DataFrame({
            "record_name": cpt.name,
            "exception": e,
            "cpt_vs_correlation": cpt_vs_correlation,
            "vs30_correlation": vs30_correlation,
            "vs30": None,
            "vs30_sd": None,
            "calculation_time_ms": time.time() - start_time,
            "latitude": cpt_df["latitude"].values[0],
            "longitude": cpt_df["longitude"].values[0],
            "min_depth_m": cpt_df["Depth"].min(),
            "max_depth_m": cpt_df["Depth"].max(),
            "depth_span_m": cpt_df["Depth"].max() - cpt_df["Depth"].min(),
            "num_depth_levels": cpt_df["Depth"].size
        }, index=[0])



if __name__ == "__main__":

    start_time = time.time()

    np.seterr(divide='ignore', invalid='ignore')

    parquet_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")
    metadata_dir = parquet_dir.parent / "metadata"

    # if Path(metadata_dir / "record_ids_sufficient_depth.csv").exists():
    #     record_ids = natsort.natsorted(pd.read_csv(metadata_dir / "record_ids_sufficient_depth.csv")["record_name_sufficient_depth"].to_list())
    #     file_paths = [parquet_dir / f"{record_id}.parquet" for record_id in record_ids]
    # else:
    #     file_paths = list(parquet_dir.glob("*.parquet"))
    #     file_paths = filter_out_files_with_insufficient_depth(file_paths, metadata_dir)

    file_paths = list(parquet_dir.glob("*.parquet"))

    # cpt_vs_correlations = list(vs_calc.cpt_vs_correlations.CPT_CORRELATIONS.keys())
    # vs30_correlations = list(vs_calc.vs30_correlations.VS30_CORRELATIONS.keys())

    # cpt_vs_correlations = cpt_vs_correlations[0:1]
    # vs30_correlations = vs30_correlations[1:]

    cpt_vs_correlations = ["andrus_2007_pleistocene", "andrus_2007_holocene"]
    vs30_correlations = ["boore_2004"]

    results = []
    for vs30_correlation in vs30_correlations:
        for cpt_vs_correlation in cpt_vs_correlations:

            description_text = f"Calculating Vs30 using {vs30_correlation} and {cpt_vs_correlation}"
            print(description_text)

            calc_vs30_from_filename_partial = functools.partial(calc_vs30_from_filename,
                                                      cpt_vs_correlation=cpt_vs_correlation,
                                                      vs30_correlation=vs30_correlation)
            #num_workers = mp.cpu_count() - 1
            num_workers = 8
            with mp.Pool(processes=num_workers) as pool:
                results.extend(list(tqdm(pool.imap(calc_vs30_from_filename_partial, file_paths),
                                    total=len(file_paths))))

    pd.concat(results, ignore_index=True).to_csv(metadata_dir / f"vs30_estimates_from_data_andrus.csv", index=False)

    print()
    print(f"Total time taken: {(time.time() - start_time)/3600} hours")