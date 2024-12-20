import pandas as pd
import matplotlib.pyplot as plt



#bin_methods = ["even", "uniform", "fd", "sturges", "scott", "doane", "sqrt", "kmeans"]
bin_methods = ["even", "uniform"]

for bin_method in bin_methods:

    ### For plotting semivariograms generated on the local computer
    # df = pd.read_parquet(f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/{bin_method}/combined/combined_semivar.parquet")

    ### For plotting semivariograms generated on hypocentre and transferred with SFTP
    df = pd.read_parquet(
        f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/hypocentre/{bin_method}/combined/combined_semivar.parquet")

    binning_method_id = df.attrs["binning_method"]
    n_bins = df.attrs["n_lags"]

    if binning_method_id == "even":
        bin_description_title = f"{n_bins} evenly spaced bins"

    if binning_method_id == "uniform":
        bin_description_title = f"{n_bins} bins with an equal number of pairs in each bin"

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

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 10]}, figsize=(5, 4))

    ax1.bar(df["empirical_bins"], df["bin_count"])
    ax1.set_xlabel('Pair separation distance (m)')
    ax1.set_ylabel('Number of\npairs in bin')
    ax1.set_title(bin_description_title)

    ax2.scatter(df["empirical_bins"], df["empirical_semivariance"])
    ax2.set_xlabel('Pair separation distance (m)')
    ax2.set_ylabel('Empirical Semivariance')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.savefig(f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/plots/{bin_method}_empirical_semivariogram.png")

    plt.show()


    print()