import pandas as pd
import numpy as np
import numpy.typing as npt
from pathlib import Path
import functools
import multiprocessing as mp
from scipy.stats import lognorm, norm
from tqdm import tqdm
import time
import sys
from nzgd_data_extraction.lib import processing_helpers

import natsort as natsort

vs_calc_path = Path("/home/arr65/src/vs30/VsViewer")
sys.path.append(str(vs_calc_path))

import vs_calc


def compute_partial_correlation(depth, vs, vs_sd, n_to_sample) -> tuple[npt.NDArray, npt.NDArray]:
    # Pick a value for constant a
    rho_0 = 0.02
    h_0 = 5
    a = np.log(rho_0) / (-1 * h_0)
    # Find the lag distance between each pair of values
    h = abs(np.subtract.outer(depth, depth))
    # Form the correlation rho, which is covariance divided by standard deviation
    # (in the range from 0 to 1)
    rho = np.exp(-1 * a * h)
    # Form the covariance matrix (population standDev * rho * population standDev)
    sd = np.diag(vs_sd.reshape(-1))
    cov = np.dot(sd, np.dot(rho, sd))
    # Cholesky Decomposition to get lower triangular matrix
    cholesky_cov = np.linalg.cholesky(cov)
    
    # Generate random uncorrelated values with depth~(0,1)
    uncorrelated_random_sample = norm.rvs(loc=0, scale=1, size=((len(depth)), n_to_sample))
    
    # Multiply by cholesky(cov) to generate random correlated values
    random_correlated_variables = cholesky_cov @ uncorrelated_random_sample
    # Adding the means to the random correlated variables
    random_ln_vs = (np.log(vs).reshape(len(vs),1) + random_correlated_variables)
    # Return generated samples and log standard deviation
    random_vs = np.exp(random_ln_vs)
    sigma = np.std(random_ln_vs)

    return random_vs, sigma

def sample_vs_profiles_from_distribution(vs_df: pd.DataFrame, n_to_sample: int = 50, correlation_flag="partial"):
    
    """
    Sample vs_df["Vs"] profiles from the provided mean and standard deviation at each depth level
    
    Parameters
    ----------
    vs_df : pd.DataFrame
    n_to_sample

    Returns
    -------

    """
    
    ### Claire Dong's code forces all depths < 1.5
    ### to have constant vs_df["Vs"]. Is that needed?

    # assume the vs_df["Vs"] in the first 1m is constant
    # i = 0
    # while z[i] < 1.5:
    #     i += 1
    # vs_df["Vs"][0:i] = vs_df["Vs"][i]

    # -----------------Compute random selected vs_df["Vs"]---------------------

    if correlation_flag == 0:
        ln_vs = np.log(vs_df["Vs"])
        randomly_sampled_ln_vs = norm.rvs(loc=ln_vs, scale=vs_df["Vs_SD"], size=(len(vs_df["Vs"]), n_to_sample))
        sigma = np.std(randomly_sampled_ln_vs)
        randomly_sampled_vs = np.exp(randomly_sampled_ln_vs)

    elif correlation_flag == 1:
        # Generate the first vs_df["Vs"] and applt its variance to the rest
        ln_vs = np.log(vs_df["Vs"])
        randomly_sampled_ln_vs = norm.rvs(loc=ln_vs[0], scale=vs_df["Vs_SD"][0], size=n_to_sample)
        z_score = (randomly_sampled_ln_vs - ln_vs[0]) / vs_df["Vs_SD"][0]
        randomly_sampled_ln_vs = vs_df["Vs_SD"] * z_score + ln_vs
        randomly_sampled_vs = np.exp(randomly_sampled_ln_vs)
        sigma = np.std(randomly_sampled_ln_vs)

    else:
        randomly_sampled_vs, sigma = compute_partial_correlation(vs_df["Depth"].values, vs_df["Vs"].values, vs_df["Vs_SD"].values, n_to_sample)

    return randomly_sampled_vs, sigma


def calc_vs30_from_filename(file_path: Path, cpt_vs_correlation: str, vs30_correlation: str,
                            randomly_sampled_vs_profiles_output_dir: Path = None) -> pd.DataFrame:
    """
    Calculate Vs30 from a given CPT file.

    Parameters
    ----------
    file_path : Path
        The path to the CPT file in Parquet format.
    cpt_vs_correlation : str
        The correlation method to use for CPT to vs_df["Vs"] conversion.
    vs30_correlation : str
        The correlation method to use for Vs30 calculation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Vs30 calculation results and metadata.
    """

    # Read the CPT data from the Parquet file
    cpt_df_repeat_measures = pd.read_parquet(file_path)
    # rename column measurement_index to investigation_number
    cpt_df_repeat_measures = cpt_df_repeat_measures.rename(columns={"measurement_index": "investigation_number"})

    # Find the row with the maximum depth and get the index for multiple measurements
    max_depth_row = cpt_df_repeat_measures[cpt_df_repeat_measures["Depth"] == cpt_df_repeat_measures["Depth"].max()]
    investigation_number = max_depth_row["investigation_number"].values[0]

    # Filter the DataFrame to include only the rows with the same multiple measurements index
    cpt_df = cpt_df_repeat_measures[cpt_df_repeat_measures["investigation_number"] == investigation_number]
    cpt_df = cpt_df[cpt_df["fs"] > 0]

    n_to_sample = 10

    try:
        ## Raise an error if there are no valid measurements
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

        vs_df = cpt_vs_profile.to_dataframe()


        randomly_sampled_vs_profiles, sigma = sample_vs_profiles_from_distribution(vs_df, n_to_sample=n_to_sample, correlation_flag="partial")
        if randomly_sampled_vs_profiles_output_dir:
            np.save(randomly_sampled_vs_profiles_output_dir / f"{file_path.stem}_random_vs_profiles.npy",
                    randomly_sampled_vs_profiles)


        vs30_from_sampled_profiles = np.zeros(n_to_sample)
        for vs_profile_idx in range(n_to_sample):
            vs_profile_for_idx = vs_calc.VsProfile(name=file_path.stem,
                                                   depth=vs_df["Depth"].values,
                                                   vs=randomly_sampled_vs_profiles[:, vs_profile_idx],
                                                   vs_sd=vs_df["Vs_SD"].values,
                                                   vs_correlation=vs30_correlation,
                                                   vs30_correlation=vs30_correlation)

            vs30_from_sampled_profiles[vs_profile_idx] = vs_profile_for_idx.vs30

        vs30 = np.mean(vs30_from_sampled_profiles)
        vs30_std = np.std(vs30_from_sampled_profiles)
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
                  "num_sampled_vs_profiles": n_to_sample,
                  "vs30_correlation": vs30_correlation,
                  "cpt_vs_correlation": cpt_vs_correlation,
                  "spt_vs_correlation": np.nan,
                  "spt_used_soil_info": np.nan,
                  "spt_hammer_type": np.nan,
                  "spt_borehole_diameter": np.nan},
                 index=[0])

    cpt_vs30_df.to_parquet(randomly_sampled_vs_profiles_output_dir / f"{file_path.stem}_vs30.parquet")

    return cpt_vs30_df

if __name__ == "__main__":

    start_time = time.time()

    #for investigation_type in [processing_helpers.InvestigationType.cpt, processing_helpers.InvestigationType.scpt]:
    for investigation_type in [processing_helpers.InvestigationType.cpt]:

        ## The code to calculate vs30 from CPT data (vs_calc) produces tens of thousands of divide by zero and invalid
        ## value warnings that are suppressed.
        np.seterr(divide='ignore', invalid='ignore')

        #parquet_dir = Path(f"/home/arr65/data/nzgd/processed_data/{investigation_type}/data")
        parquet_dir = Path(f"/home/arr65/data/nzgd/processed_data/{investigation_type}/extracted_data_per_record")
        vs_profiles_dir = Path("/home/arr65/data/nzgd/robust_vs30/randomly_sampled_velocity_profiles")

        vs30_output_dir = Path("/home/arr65/data/nzgd/robust_vs30/vs30_estimates")
        
        vs_profiles_dir.mkdir(exist_ok=True, parents=True)
        vs30_output_dir.mkdir(exist_ok=True, parents=True)

        file_paths = natsort.natsorted(list(parquet_dir.glob("*.parquet")))
        file_paths = file_paths[:10]

        # cpt_vs_correlations = ["andrus_2007_pleistocene","andrus_2007_holocene", "andrus_2007_tertiary_age_cooper_marl",
        #                        "robertson_2009","hegazy_2006","mcgann_2015","mcgann_2018"]

        cpt_vs_correlations = ["mcgann_2015"]

        #vs30_correlations = ["boore_2004", "boore_2011"]
        vs30_correlations = ["boore_2011"]

        # cpt_vs_correlations = ["andrus_2007_pleistocene", "robertson_2009", "mcgann_2018"]
        # vs30_correlations = ["boore_2004", "boore_2011"]

        results = []
        for vs30_correlation in vs30_correlations:
            for cpt_vs_correlation in cpt_vs_correlations:

                description_text = f"Calculating Vs30 using {vs30_correlation} and {cpt_vs_correlation}"
                print(description_text)

                calc_vs30_from_filename_partial = functools.partial(calc_vs30_from_filename,
                                                          cpt_vs_correlation=cpt_vs_correlation,
                                                          vs30_correlation=vs30_correlation,
                                                          randomly_sampled_vs_profiles_output_dir=vs_profiles_dir)
                num_workers = 8
                with mp.Pool(processes=num_workers) as pool:
                    results.extend(list(tqdm(pool.imap(calc_vs30_from_filename_partial, file_paths),
                                        total=len(file_paths))))

        pd.concat(results, ignore_index=True).to_csv(vs30_output_dir / f"vs30_estimates_from_cpt.csv", index=False)

    print(f"Total time taken: {(time.time() - start_time)/3600} hours")