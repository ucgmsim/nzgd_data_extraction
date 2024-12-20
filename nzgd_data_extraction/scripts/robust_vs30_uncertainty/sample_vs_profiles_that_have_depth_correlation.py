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


def claire_dong_compute_partial_correlation(depth, vs, vs_sd, num_to_sample) -> tuple[npt.NDArray, npt.NDArray]:
    # Pick a value for constant a
    # rho_0 = 0.02
    # h_0 = 5
    # a = np.log(rho_0) / (-1 * h_0)
    # Find the lag distance between each pair of values
    h = abs(np.subtract.outer(depth, depth))
    # Form the correlation rho, which is covariance divided by standard deviation
    # (in the range from 0 to 1)
    #rho = np.exp(-1 * a * h)
    ######################################################
    ### Andrew's rho calculation
    sill = 0.8
    range_m = 3
    gamma = sill * (1 - np.exp(-3 * h / range_m))
    rho = sill - gamma
    ################################################

    # Form the covariance matrix (population standDev * rho * population standDev)
    sd = np.diag(vs_sd.reshape(-1))
    cov = np.dot(sd, np.dot(rho, sd))
    # Cholesky Decomposition to get lower triangular matrix
    cholesky_cov = np.linalg.cholesky(cov)
    
    # Generate random uncorrelated values with depth~(0,1)
    uncorrelated_random_sample = norm.rvs(loc=0, scale=1, size=((len(depth)), num_to_sample))

    # Multiply by cholesky(cov) to generate random correlated values
    random_correlated_variables = cholesky_cov @ uncorrelated_random_sample
    # Adding the means to the random correlated variables
    random_ln_vs = (np.log(vs).reshape(len(vs),1) + random_correlated_variables)
    # Return generated samples and log standard deviation
    random_vs = np.exp(random_ln_vs)
    sigma = np.std(random_ln_vs)

    return random_vs, sigma


def calculate_semivariogram_model_gamma_h(h: npt.NDArray, sill: float, range_m: float) -> npt.NDArray:
    """
    Calculate the semivariogram model gamma(h) for a given separation distance h, range r, and sill s.

    Parameters
    ----------
    h : npt.NDArray
        The separation distance h.
    range_m : float
        The range parameter, r, is the distance at which the semivariogram, gamma(h), is equal to 95% of the sill
        (i.e., the distance at which 95% of the correlation is lost).
    sill : float
        The sill parameter, s, is equal to the variance of Z = delta W  (the residual of the within event term for ground motions)

    Returns
    -------
    npt.NDArray
        The semivariogram model gamma(h).
    """

    return sill * (1 - np.exp(-3 * h / range_m))


def andrew_compute_partial_correlation(depth, vs, vs_sd, num_to_sample) -> tuple[npt.NDArray, npt.NDArray]:

    ### Defining rho from semivariogram_model ###
    # Find the lag distance between each pair of values
    h = abs(np.subtract.outer(depth, depth))
    sill = 0.8
    gamma = calculate_semivariogram_model_gamma_h(h=h, sill=sill, range_m=3)
    covariance = sill - gamma
    rho = covariance / sill

    ### The rest of the code is the same as Claire Dong's
    #################################

    # Form the covariance matrix (population standDev * rho * population standDev)
    sd = np.diag(vs_sd.reshape(-1))
    cov = np.dot(sd, np.dot(rho, sd))
    # Cholesky Decomposition to get lower triangular matrix
    cholesky_cov = np.linalg.cholesky(cov)

    # Generate random uncorrelated values with depth~(0,1)
    uncorrelated_random_sample = norm.rvs(loc=0, scale=1, size=((len(depth)), num_to_sample))

    # Multiply by cholesky(cov) to generate random correlated values
    random_correlated_variables = cholesky_cov @ uncorrelated_random_sample
    # Adding the means to the random correlated variables
    random_ln_vs = (np.log(vs).reshape(len(vs), 1) + random_correlated_variables)
    # Return generated samples and log standard deviation
    random_vs = np.exp(random_ln_vs)
    sigma = np.std(random_ln_vs)

    return random_vs, sigma


def sample_vs_profiles_from_distribution(vs_df: pd.DataFrame, num_to_sample: int = 50):
    
    """
    Sample vs_df["Vs"] profiles from the provided mean and standard deviation at each depth level
    
    Parameters
    ----------
    vs_df : pd.DataFrame
    num_to_sample

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


    # ln_vs = np.log(vs_df["Vs"].values)
    # vs_sd = vs_df["Vs_SD"].values

    randomly_sampled_vs, sigma = andrew_compute_partial_correlation(vs_df["Depth"].values, vs_df["Vs"].values, vs_df["Vs_SD"].values, num_to_sample)
    #randomly_sampled_vs, sigma = claire_dong_compute_partial_correlation(vs_df["Depth"].values, vs_df["Vs"].values, vs_df["Vs_SD"].values, num_to_sample)


    return randomly_sampled_vs, sigma


def calc_vs30_from_filename(file_path: Path, cpt_vs_correlation: str,
                            vs30_correlation: str,
                            num_to_sample = 10,
                            vs30_per_record_dir : Path = None,
                            randomly_sampled_vs_profiles_per_record_dir: Path = None) -> pd.DataFrame:
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
        vs_df.to_csv(randomly_sampled_vs_profiles_per_record_dir / f"{file_path.stem}_vs_profiles.csv", index=False)


        randomly_sampled_vs_profiles, sigma = sample_vs_profiles_from_distribution(vs_df, num_to_sample=num_to_sample)

        if randomly_sampled_vs_profiles_per_record_dir:
            vs_profiles_df = pd.DataFrame(columns=["record_name", "depth", "vs", "vs_sigma"], index=[0], dtype="object")
            vs_profiles_df.at[0, "record_name"] = cpt_df["record_name"].values[0]
            vs_profiles_df.at[0, "depth"] = vs_df["Depth"].values
            vs_profiles_df.at[0, "num_sampled_profiles"] = num_to_sample
            vs_profiles_df.at[0, "vs"] = randomly_sampled_vs_profiles.tolist()
            vs_profiles_df.at[0, "vs_sigma"] = sigma

            vs_profiles_df.to_parquet(randomly_sampled_vs_profiles_per_record_dir / f"{file_path.stem}_random_vs_profiles.parquet")

        vs30_from_sampled_profiles = np.zeros(num_to_sample)
        vs30_std_from_sampled_profiles = np.zeros(num_to_sample)

        for vs_profile_idx in range(num_to_sample):
            vs_profile_for_idx = vs_calc.VsProfile(name=file_path.stem,
                                                   depth=vs_df["Depth"].values,
                                                   vs=randomly_sampled_vs_profiles[:, vs_profile_idx],
                                                   vs_sd=vs_df["Vs_SD"].values,
                                                   vs_correlation=vs30_correlation,
                                                   vs30_correlation=vs30_correlation)

            vs30_from_sampled_profiles[vs_profile_idx] = vs_profile_for_idx.vs30
            vs30_std_from_sampled_profiles[vs_profile_idx] = vs_profile_for_idx.vs30_sd

        mean_vs30 = np.mean(vs30_from_sampled_profiles)
        sd_vs30 = np.std(vs30_from_sampled_profiles)

        mean_vs30sd = np.mean(vs30_std_from_sampled_profiles)
        sd_vs30sd = np.std(vs30_std_from_sampled_profiles)

        error = np.nan

    except Exception as e:
        # Handle any exceptions by setting Vs30 and its standard deviation to NaN
        mean_vs30 = np.nan
        sd_vs30 = np.nan
        mean_vs30sd = np.nan
        sd_vs30sd = np.nan
        error = str(e)

    ### Create a DataFrame with the Vs30 calculation results and metadata
    cpt_vs30_df = pd.DataFrame(
                 {"record_name": cpt_df["record_name"].values[0],
                  "record_type": "cpt",
                  "processing_error": error,
                  "max_depth": cpt_df["Depth"].max(),
                  "min_depth": cpt_df["Depth"].min(),
                  "depth_span": cpt_df["Depth"].max() - cpt_df["Depth"].min(),
                  "num_depth_levels": cpt_df["Depth"].size,
                  "mean_vs30": mean_vs30,
                  "sd_vs30": sd_vs30,
                  "mean_vs30sd": mean_vs30sd,
                  "sd_vs30sd":sd_vs30sd,
                  "num_sampled_vs_profiles": num_to_sample,
                  "vs30_correlation": vs30_correlation,
                  "cpt_vs_correlation": cpt_vs_correlation,
                  "spt_vs_correlation": np.nan,
                  "spt_used_soil_info": np.nan,
                  "spt_hammer_type": np.nan,
                  "spt_borehole_diameter": np.nan},
                 index=[0])

    if vs30_per_record_dir:
        cpt_vs30_df.to_parquet(vs30_per_record_dir / f"{file_path.stem}_vs30.parquet")

    return cpt_vs30_df

if __name__ == "__main__":

    start_time = time.time()

    #for investigation_type in [processing_helpers.InvestigationType.cpt, processing_helpers.InvestigationType.scpt]:
    for investigation_type in [processing_helpers.InvestigationType.cpt]:

        ## The code to calculate vs30 from CPT data (vs_calc) produces tens of thousands of divide by zero and invalid
        ## value warnings that are suppressed.
        np.seterr(divide='ignore', invalid='ignore')

        num_to_sample = 1

        #cpt_data_dir = Path(f"/home/arr65/data/nzgd/processed_data/{investigation_type}/data")
        #cpt_data_dir = Path(f"/home/arr65/data/nzgd/processed_data/{investigation_type}/extracted_data_per_record")
        cpt_data_dir = Path(f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/extracted_cpt_data/extracted_data_per_record")

        #vs_profiles_dir = Path(f"/home/arr65/data/nzgd/robust_vs30/{investigation_type}/randomly_sampled_velocity_profiles")
        vs_profiles_dir = Path(f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/McGann2015_velocity_profiles_from_cpt")
        #vs30_output_dir = Path(f"/home/arr65/data/nzgd/robust_vs30/{investigation_type}/vs30_estimate_per_record")
        vs30_output_dir = Path(f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/"
                               f"McGann2015_Bore2004_Vs30")
        
        vs_profiles_dir.mkdir(exist_ok=True, parents=True)
        vs30_output_dir.mkdir(exist_ok=True, parents=True)

        file_paths = natsort.natsorted(list(cpt_data_dir.glob("*.parquet")))
        #file_paths = file_paths[4:6]
        #file_paths = file_paths[0:2]

        # cpt_vs_correlations = ["andrus_2007_pleistocene","andrus_2007_holocene", "andrus_2007_tertiary_age_cooper_marl",
        #                        "robertson_2009","hegazy_2006","mcgann_2015","mcgann_2018"]

        cpt_vs_correlations = ["mcgann_2015"]

        #vs30_correlations = ["boore_2004", "boore_2011"]
        vs30_correlations = ["boore_2004"]

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
                                                          num_to_sample = num_to_sample,
                                                          vs30_per_record_dir = vs30_output_dir,
                                                          randomly_sampled_vs_profiles_per_record_dir=vs_profiles_dir)

                num_workers = 8
                with mp.Pool(processes=num_workers) as pool:
                    results.extend(list(tqdm(pool.imap(calc_vs30_from_filename_partial, file_paths),
                                        total=len(file_paths))))

        pd.concat(results, ignore_index=True).to_csv(vs30_output_dir.parent / f"vs30_estimates_from_cpt.csv", index=False)

    print(f"Total time taken: {(time.time() - start_time)/3600} hours")