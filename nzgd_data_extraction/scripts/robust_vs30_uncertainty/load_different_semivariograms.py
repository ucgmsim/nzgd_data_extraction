import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/even/combined/combined_semivar.csv")

# numba_bins = np.loadtxt("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/combined/numba_bin_edges.txt")
# numba_semivar = np.loadtxt("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis/combined/numba_semivariogram.txt")


# plt.plot(df1["empirical_bins"], df1["empirical_semivariance"])
# # plt.plot(numba_bins, numba_semivar, label="manual calc")
# plt.show()

# Create a figure and two subplots (top and bottom panels)
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 10]}, figsize=(10, 8))

# Top panel: Bar plot
ax1.bar(df1["empirical_bins"], df1["bin_count"])
#ax1.set_xlabel('Empirical Bins')
ax1.set_xlabel('Pair separation distance (m)')
ax1.set_ylabel('Number of\npairs in bin')
#ax1.set_title('Bar Plot of Empirical Bins vs Bin Count')

# Bottom panel: Scatter plot
ax2.scatter(df1["empirical_bins"], df1["empirical_semivariance"])
#ax2.set_xlabel('Empirical Bins')
ax2.set_xlabel('Pair separation distance (m)')
ax2.set_ylabel('Empirical Semivariance')
#ax2.set_title('Scatter Plot of Empirical Bins vs Empirical Semivariance')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()


print()