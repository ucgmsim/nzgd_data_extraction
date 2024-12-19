import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#bin_methods = ["even", "uniform", "fd", "sturges", "scott", "doane", "sqrt", "kmeans"]
bin_methods = ["even", "uniform", "fd", "sturges", "scott", "doane", "sqrt", "kmeans"]

for bin_method in bin_methods:

    df1 = pd.read_parquet(f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/{bin_method}/combined/combined_semivar.parquet")
    #df1 = pd.read_parquet("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/fd/combined/combined_semivar.parquet")

    binning_method_id = df1.attrs["binning_method"]
    n_bins = df1.attrs["n_lags"]

    if binning_method_id == "even":
        bin_description_title = f"{n_bins} evenly spaced bins"

    if binning_method_id == "uniform":
        bin_description_title = f"{n_bins} with an equal number of pairs in each bin"

    if binning_method_id == "fd":
        bin_description_title = f"{n_bins} bins found using the Freedman-Diaconis estimator"

    if binning_method_id == "sturges":
        bin_description_title = f"{n_bins} bins found using Sturge's rule"

    if binning_method_id == "scott":
        bin_description_title = f"{n_bins} bins found using Scott's rule"

    if binning_method_id == "doane":
        bin_description_title = f"{n_bins} bins found using Doane's extension to Scott's rule"

    if binning_method_id == "sqrt":
        bin_description_title = f"{n_bins} where nbins is found as the square-root of distance"

    if binning_method_id == "kmeans":
        bin_description_title = f"{n_bins} bins found using k-means clustering"

    # numba_bins = np.loadtxt("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/combined/numba_bin_edges.txt")
    # numba_semivar = np.loadtxt("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/combined/numba_semivariogram.txt")


    # plt.plot(df1["empirical_bins"], df1["empirical_semivariance"])
    # # plt.plot(numba_bins, numba_semivar, label="manual calc")
    # plt.show()

    # Create a figure and two subplots (top and bottom panels)
    #fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 10]}, figsize=(10, 8))
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 10]}, figsize=(5, 4))

    # Top panel: Bar plot
    ax1.bar(df1["empirical_bins"], df1["bin_count"])
    #ax1.set_xlabel('Empirical Bins')
    ax1.set_xlabel('Pair separation distance (m)')
    ax1.set_ylabel('Number of\npairs in bin')
    ax1.set_title(bin_description_title)

    # Bottom panel: Scatter plot
    ax2.scatter(df1["empirical_bins"], df1["empirical_semivariance"])
    #ax2.set_xlabel('Empirical Bins')
    ax2.set_xlabel('Pair separation distance (m)')
    ax2.set_ylabel('Empirical Semivariance')
    #ax2.set_title('Scatter Plot of Empirical Bins vs Empirical Semivariance')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.savefig(f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/plots/{bin_method}_empirical_semivariogram.png")

    # Show the plot
    plt.show()


    print()