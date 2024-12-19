import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skgstat import Variogram
import time
import toml
from tqdm import tqdm
from itertools import combinations
from numba import jit


import numpy as np

def dynamic_variogram(coords, values, n_bins=10):
    """
    Compute a variogram dynamically by binning pairwise distances on-the-fly.

    Parameters:
        coords (np.ndarray): Array of shape (N, 2) with point coordinates.
        values (np.ndarray): Array of shape (N,) with values at the points.
        n_bins (int): Number of lag bins.

    Returns:
        bin_edges (np.ndarray): Edges of the bins.
        variogram (np.ndarray): Semi-variance for each bin.
    """
    n_points = len(coords)
    max_dist = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Iterate through all pairs
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Compute distance
            dist = np.linalg.norm(coords[i] - coords[j])

            # Find the bin
            bin_idx = np.digitize(dist, bin_edges) - 1
            if 0 <= bin_idx < n_bins:
                # Update bin statistics
                diff = values[i] - values[j]
                bin_sums[bin_idx] += diff ** 2
                bin_counts[bin_idx] += 1

    # Compute semi-variance
    variogram = 0.5 * bin_sums / bin_counts
    return bin_edges, bin_counts, variogram

def dynamic_variogram_andrew_mod(coords, values, n_bins=10):
    """
    Compute a variogram dynamically by binning pairwise distances on-the-fly.

    Parameters:
        coords (np.ndarray): Array of shape (N, 2) with point coordinates.
        values (np.ndarray): Array of shape (N,) with values at the points.
        n_bins (int): Number of lag bins.

    Returns:
        bin_edges (np.ndarray): Edges of the bins.
        variogram (np.ndarray): Semi-variance for each bin.
    """
    n_points = len(coords)
    max_dist = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Iterate through all pairs
    for i in tqdm(range(n_points)):

        dist = np.linalg.norm(coords[i] - coords)

        # Find the bin
        bin_idx = np.digitize(dist, bin_edges) - 1
        if 0 <= bin_idx < n_bins:
            # Update bin statistics
            diff = values[i] - values[j]
            bin_sums[bin_idx] += diff ** 2
            bin_counts[bin_idx] += 1

    # Compute semi-variance
    variogram = 0.5 * bin_sums / bin_counts
    return bin_edges, variogram


def efficient_variogram_with_progress(coords, values, n_bins=10):
    """
    Compute a variogram dynamically by binning pairwise distances efficiently,
    with a progress bar.

    Parameters:
        coords (np.ndarray): Array of shape (N, 2) with point coordinates.
        values (np.ndarray): Array of shape (N,) with values at the points.
        n_bins (int): Number of lag bins.

    Returns:
        bin_edges (np.ndarray): Edges of the bins.
        variogram (np.ndarray): Semi-variance for each bin.
    """
    n_points = len(coords)
    max_dist = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Use combinations to iterate over pairs without creating a full matrix
    total_pairs = n_points * (n_points - 1) // 2
    for i, j in tqdm(combinations(range(n_points), 2), total=total_pairs, desc="Computing Variogram"):
        # Compute distance
        dist = np.linalg.norm(coords[i] - coords[j])

        # Find the bin
        bin_idx = np.digitize(dist, bin_edges) - 1
        if 0 <= bin_idx < n_bins:
            # Update bin statistics
            diff = values[i] - values[j]
            bin_sums[bin_idx] += diff ** 2
            bin_counts[bin_idx] += 1

    # Compute semi-variance
    variogram = 0.5 * bin_sums / np.maximum(bin_counts, 1)  # Avoid division by zero
    return bin_edges, variogram

import numpy as np
from tqdm import tqdm

def hybrid_variogram(coords, values, n_bins=10):
    """
    Compute a variogram using a hybrid approach:
    vectorizing over one point at a time to save memory.

    Parameters:
        coords (np.ndarray): Array of shape (N, 2) with point coordinates.
        values (np.ndarray): Array of shape (N,) with values at the points.
        n_bins (int): Number of lag bins.

    Returns:
        bin_edges (np.ndarray): Edges of the bins.
        variogram (np.ndarray): Semi-variance for each bin.
    """
    n_points = len(coords)
    max_dist = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Iterate over each point
    for i in tqdm(range(n_points), desc="Computing Variogram"):
        # Get the current point and its value
        current_point = coords[i]
        current_value = values[i]

        # Compute distances and squared value differences to all other points
        distances = np.linalg.norm(coords[i + 1:] - current_point, axis=1)
        squared_differences = (values[i + 1:] - current_value) ** 2

        # Bin the distances
        bin_indices = np.digitize(distances, bin_edges) - 1

        # Aggregate statistics for each bin
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            bin_sums[bin_idx] += np.sum(squared_differences[mask])
            bin_counts[bin_idx] += np.sum(mask)

    # Compute semi-variance
    variogram = 0.5 * bin_sums / np.maximum(bin_counts, 1)  # Avoid division by zero
    return bin_edges, variogram


def make_semivariogram_and_output(base_output_dir:Path, name:str, input_df:pd.DataFrame, binning_method:str):

    combined_path = base_output_dir / binning_method / name
    combined_path.mkdir(exist_ok=True, parents=True)

    plt.figure()
    plt.plot(input_df["inferred_vs"], input_df["depth_m"], '.', color="tab:orange", label="Inferred Vs from CPT")
    plt.plot(input_df["measured_vs"],input_df["depth_m"], '.', color="tab:blue", label="Measured Vs")
    plt.legend()
    plt.ylabel("depth (m)")
    plt.xlabel("Vs (m/s)")
    plt.gca().invert_yaxis()
    plt.savefig(combined_path / f"{name}_measured_and_inferred_vs.png", dpi=500)
    plt.close()

    plt.figure()
    plt.plot(input_df["depth_m"],input_df["ln_measured_minus_ln_inferred"], '.')
    plt.xlabel("depth (m)")
    plt.ylabel("ln(measured_vs) - ln(cpt_inferred_vs)")
    plt.ylim(-3.1,3.1)
    plt.savefig(combined_path / f"{name}_ln_residual.png", dpi=500)
    plt.close()

    log_resid_df = input_df.sort_values(by="depth_m")

    print("Doing Variogram...")

    selected_n_lags = 100

    variogram_calc_start_time = time.time()

    semivar = Variogram(log_resid_df["depth_m"].values,
                      log_resid_df["ln_measured_minus_ln_inferred"].values,
                      normalize=False,
                      n_lags = selected_n_lags,
                      model = "exponential",
                      sparse=True,
                      bin_func = binning_method)

    fitted_semivariogram_df = Variogram.to_DataFrame(semivar, n=selected_n_lags)

    fitted_semivariogram_df["bins"] = semivar.bins
    fitted_semivariogram_df["bin_count"] = semivar.bin_count

    empirical_semivar = Variogram.get_empirical(semivar)
    fitted_semivariogram_df["empirical_bins"] = empirical_semivar[0]
    fitted_semivariogram_df["empirical_semivariance"] = empirical_semivar[1]

    fitted_semivariogram_df.to_csv(combined_path / f"{name}_semivar.csv",index=False)

    semivar_plot = semivar.plot(show=False)
    semivar_plot.savefig(combined_path / f"{name}_semivariogram.png",dpi=500)

    # dist_diff_plot = semivar.distance_difference_plot(show=False)
    # dist_diff_plot.savefig(base_output_dir / name / f"{name}_distance_difference_plot.png",dpi=500)

    describe_dict = Variogram.describe(semivar)
    with open(combined_path / "description.toml", "w") as toml_file:
        toml.dump(describe_dict, toml_file)

    print(f"Variogram calculation took {(time.time() - variogram_calc_start_time)/60} mins")
    plt.close("all")


#@jit
def dynamic_variogram_andrew_mod(depth, values, n_bins):
    """
    Compute a variogram dynamically by binning pairwise distances on-the-fly.

    Parameters:
        depth (np.ndarray): Array of shape (N, 2) with point coordinates.
        values (np.ndarray): Array of shape (N,) with values at the points.
        n_bins (int): Number of lag bins.

    Returns:
        bin_edges (np.ndarray): Edges of the bins.
        variogram (np.ndarray): Semi-variance for each bin.
    """
    n_points = len(depth)
    #max_dist = np.linalg.norm(depth.max(axis=0) - depth.min(axis=0))
    max_dist = np.max(depth) - np.min(depth)
    bin_edges = np.linspace(0, max_dist, n_bins)
    bin_sum_square_diffs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Iterate through all pairs
    for i in range(n_points):
        for j in range(n_points):
            # Compute distance
            #dist = np.linalg.norm(depth[i] - depth[j])
            dist = np.abs(depth[i] - depth[j])

            # Find the bin
            bin_idx = np.digitize(dist, bin_edges) - 1

            # Update bin statistics
            bin_sum_square_diffs[bin_idx] += (values[i] - values[j])** 2
            bin_counts[bin_idx] += 1

    # Compute semi-variance
    #variogram = 0.5 * bin_sums / bin_counts
    semivariogram = (1 / (2 * bin_counts)) * bin_sum_square_diffs

    return bin_edges, bin_counts, semivariogram


measured_vs_profile_path = Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/reformatted_for_easier_loading/vsProf042015")
vs_profile_from_cpt_path = Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/chris_mcgann_vs_from_cpt/mcgann_2015_vs_profiles")

base_output_dir = Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis")
base_output_dir.mkdir(exist_ok=True, parents=True)

measured_vs_profile_files = list(measured_vs_profile_path.glob("*.dat"))

custom_header = ['depth', 'vs']

cpt_dir = Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/reformatted_for_easier_loading/smsCPTdata")
cpt_dirs = list(cpt_dir.glob("*"))

ids_with_cpt = [cpt_dir.name for cpt_dir in cpt_dirs]
#ids_with_cpt = ["NNBS"]
#ids_with_cpt = ["SHLC"]
#ids_with_cpt = ["CHHC"]


log_resid_df = pd.DataFrame(columns=["record_name","depth_m","measured_vs","inferred_vs","ln_measured_minus_ln_inferred"])

for file in measured_vs_profile_files:

    measured_vs_depth_boundaries_df = pd.read_csv(file, sep=r"\s+", header=None, names=custom_header)

    print(f"Doing file: {file.name}")
    id_code = file.stem[0:4]

    if id_code not in ids_with_cpt:
        print(f"{id_code} is missing CPT data")
        continue

    from_cpt = pd.read_csv(
        f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/chris_mcgann_vs_from_cpt/mcgann_2015_vs_profiles/{id_code}_vs_profiles.csv")

    num_investigations = from_cpt["investigation_number"].nunique()

    for investigation_number in range(num_investigations):

        investigation_df = from_cpt[from_cpt["investigation_number"]==investigation_number]
        investigation_df = investigation_df[investigation_df["Vs"]>0]

        measured_vs = np.copy(investigation_df["Depth"].values)
        measured_vs[:] = np.nan

        for i in range(len(measured_vs_depth_boundaries_df["depth"])-1):

            depth_1 = measured_vs_depth_boundaries_df["depth"].iloc[i]
            depth_2 = measured_vs_depth_boundaries_df["depth"].iloc[i+1]

            measured_depth_indices = np.where((investigation_df["Depth"] >= depth_1) & (investigation_df["Depth"] <= depth_2))

            measured_vs[measured_depth_indices] = measured_vs_depth_boundaries_df["vs"].iloc[i]

        log_resid = np.log(measured_vs) - np.log(investigation_df["Vs"])

        current_cpt_df = pd.DataFrame({"record_name":id_code,
                                                    "depth_m":investigation_df["Depth"].values,
                                                    "measured_vs":measured_vs,
                                                    "inferred_vs":investigation_df["Vs"],
                                                    "ln_measured_minus_ln_inferred":log_resid})

        #make_semivariogram_and_output(base_output_dir, name=f"{file.stem}_investigation_{investigation_number}", input_df=current_cpt_df)

        log_resid_df = pd.concat([log_resid_df,
                                    current_cpt_df],
                                     ignore_index=True)

# n = 2
# # Remove every nth row
# df = log_resid_df.drop(log_resid_df.index[::n])
#
log_resid_df = log_resid_df.sort_values(by="depth_m")

#log_resid_df = log_resid_df.iloc[::2]
#log_resid_df = log_resid_df.iloc[::2]
#log_resid_df = log_resid_df.iloc[::2]
#log_resid_df = log_resid_df.iloc[::2]
#log_resid_df = log_resid_df.iloc[::2]
#log_resid_df = log_resid_df.iloc[::2]
#log_resid_df = log_resid_df.iloc[::2]
# log_resid_df = log_resid_df.iloc[::2]


### Remove half of points 8 times to get it to about 11GB

#binning_methods = ["even", "uniform", "fd", "sturges", "scott", "doane", "sqrt","kmeans", "ward"]
binning_methods = ["even", "uniform"]

for binning_method in binning_methods:

    print(f"Doing binning method: {binning_method}")

    start_time = time.time()

    make_semivariogram_and_output(base_output_dir, name="combined", input_df=log_resid_df, binning_method=binning_method)

    print(f"Semivariogram calc took {(time.time() - start_time)/60} mins")




# bin_edges, bin_counts, semivariogram = dynamic_variogram_andrew_mod(log_resid_df["depth_m"].values, log_resid_df["depth_m"].values, n_bins=100)
# np.savetxt(base_output_dir / "combined" / "numba_bin_edges.txt", bin_edges)
# np.savetxt(base_output_dir / "combined" / "numba_bin_counts.txt", bin_counts)
# np.savetxt(base_output_dir / "combined" / "numba_semivariogram.txt", semivariogram)

# v_df = pd.DataFrame({"bin_edges":bin_edges,"bin_counts":bin_counts,"variogram":variogram})
# v_df.to_csv(base_output_dir / "combined" / "numba_variogram.csv",index=False)



print()
