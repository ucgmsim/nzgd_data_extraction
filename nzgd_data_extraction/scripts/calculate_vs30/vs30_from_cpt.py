import pandas as pd
import numpy as np
from pathlib import Path
import functools
import multiprocessing as mp
from tqdm import tqdm
import time
import sys
from nzgd_data_extraction.lib import processing_helpers

import natsort as natsort

vs_calc_path = Path("/home/arr65/src/vs30/VsViewer")
sys.path.append(str(vs_calc_path))

import vs_calc

def calc_vs30_from_filename(file_path: Path, cpt_vs_correlation: str, vs30_correlation: str) -> pd.DataFrame:
    """
    Calculate Vs30 from a given CPT file.

    Parameters
    ----------
    file_path : Path
        The path to the CPT file in Parquet format.
    cpt_vs_correlation : str
        The correlation method to use for CPT to Vs conversion.
    vs30_correlation : str
        The correlation method to use for Vs30 calculation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Vs30 calculation results and metadata.
    """

    # Read the CPT data from the Parquet file
    cpt_df_repeat_measures = pd.read_parquet(file_path)

    # Find the row with the maximum depth and get the index for multiple measurements
    max_depth_row = cpt_df_repeat_measures[cpt_df_repeat_measures["Depth"] == cpt_df_repeat_measures["Depth"].max()]
    multiple_measurements_index = max_depth_row["multiple_measurements"].values[0]

    # Filter the DataFrame to include only the rows with the same multiple measurements index
    cpt_df = cpt_df_repeat_measures[cpt_df_repeat_measures["multiple_measurements"] == multiple_measurements_index]
    cpt_df = cpt_df[cpt_df["fs"] > 0]

    try:
        # Raise an error if there are no valid measurements
        if cpt_df.size == 0:
            raise ValueError("No valid measurements in the CPT")

        # Create a CPT object from the DataFrame
        cpt = vs_calc.CPT(
            cpt_df["record_name"].values[0],
            cpt_df["Depth"].values,
            cpt_df["qc"].values,
            cpt_df["fs"].values,
            cpt_df["u"].values)

        # Create a VsProfile from the CPT object using the specified correlation
        cpt_vs_profile = vs_calc.VsProfile.from_cpt(cpt, cpt_vs_correlation)
        cpt_vs_profile.vs30_correlation = vs30_correlation

        # Calculate Vs30 and its standard deviation
        vs30 = cpt_vs_profile.vs30
        vs30_std = cpt_vs_profile.vs30_sd
        error = np.nan

    except Exception as e:
        # Handle any exceptions by setting Vs30 and its standard deviation to NaN
        vs30 = np.nan
        vs30_std = np.nan
        error = e

    # Create a DataFrame with the Vs30 calculation results and metadata
    cpt_vs30_df = pd.DataFrame(
                 {"record_name": cpt_df["record_name"].values[0],
                  "record_type": "cpt",
                  "processing_error": error,
                  "max_depth": cpt_df["Depth"].max(),
                  "min_depth": cpt_df["Depth"].min(),
                  "depth_span": cpt_df["Depth"].max() - cpt_df["Depth"].min(),
                  "num_depth_levels": cpt_df["Depth"].size,
                  "vs30": vs30,
                  "vs30_std": vs30_std,
                  "vs30_correlation": vs30_correlation,
                  "cpt_vs_correlation": cpt_vs_correlation,
                  "spt_vs_correlation": np.nan,
                  "spt_used_soil_info": np.nan,
                  "spt_hammer_type": np.nan,
                  "spt_borehole_diameter": np.nan},
                 index=[0])

    return cpt_vs30_df

if __name__ == "__main__":

    start_time = time.time()

    for investigation_type in [processing_helpers.InvestigationType.cpt, processing_helpers.InvestigationType.scpt]:

        ## The code to calculate vs30 from CPT data (vs_calc) produces tens of thousands of divide by zero and invalid
        ## value warnings that are suppressed.
        np.seterr(divide='ignore', invalid='ignore')

        parquet_dir = Path(f"/home/arr65/data/nzgd/processed_data/{investigation_type}/data")
        metadata_dir = parquet_dir.parent / "metadata"

        file_paths = list(parquet_dir.glob("*.parquet"))

        # cpt_vs_correlations = ["andrus_2007_pleistocene","andrus_2007_holocene", "andrus_2007_tertiary_age_cooper_marl",
        #                        "robertson_2009","hegazy_2006","mcgann_2015","mcgann_2018"]
        #
        # vs30_correlations = ["boore_2004", "boore_2011"]

        cpt_vs_correlations = ["andrus_2007_pleistocene", "robertson_2009", "mcgann_2018"]
        vs30_correlations = ["boore_2004", "boore_2011"]

        results = []
        for vs30_correlation in vs30_correlations:
            for cpt_vs_correlation in cpt_vs_correlations:

                description_text = f"Calculating Vs30 using {vs30_correlation} and {cpt_vs_correlation}"
                print(description_text)

                calc_vs30_from_filename_partial = functools.partial(calc_vs30_from_filename,
                                                          cpt_vs_correlation=cpt_vs_correlation,
                                                          vs30_correlation=vs30_correlation)
                num_workers = 8
                with mp.Pool(processes=num_workers) as pool:
                    results.extend(list(tqdm(pool.imap(calc_vs30_from_filename_partial, file_paths),
                                        total=len(file_paths))))

        pd.concat(results, ignore_index=True).to_csv(metadata_dir / f"vs30_estimates_from_cpt.csv", index=False)

    print(f"Total time taken: {(time.time() - start_time)/3600} hours")