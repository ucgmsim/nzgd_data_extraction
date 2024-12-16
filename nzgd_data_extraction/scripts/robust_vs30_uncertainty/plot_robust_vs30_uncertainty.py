import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

vs30_df = pd.read_csv("/home/arr65/data/nzgd/robust_vs30/cpt/vs30_estimates_from_cpt.csv")

mean_vs30 = vs30_df["mean_vs30"].values
std_vs30 = vs30_df["sd_vs30"].values

pc_uncertainty = 100*std_vs30/mean_vs30

plt.scatter(mean_vs30, pc_uncertainty)
plt.xlabel("Mean Vs30 (m/s)")
plt.ylabel("Vs30 Uncertainty (%)")
plt.show()

pc_uncertainty[pc_uncertainty > 20] = np.nan
vs30_std_relative_to_boore_2004_sigma = vs30_df["sd_vs30"].values/vs30_df["mean_vs30sd"].values
vs30_std_relative_to_boore_2004_sigma = vs30_std_relative_to_boore_2004_sigma[np.isfinite(vs30_std_relative_to_boore_2004_sigma)]
vs30_std_relative_to_boore_2004_sigma[vs30_std_relative_to_boore_2004_sigma>1000] = np.nan


plt.figure()
plt.hist(pc_uncertainty, bins = 100)
plt.xlabel("Vs30 Uncertainty (%)")
plt.ylabel("Number of Records")
plt.show()

plt.figure()
plt.hist(vs30_std_relative_to_boore_2004_sigma, bins = 100)
plt.xlabel("(standard deviation of Vs30 samples)/(Boore 2004 Vs30 sigma)")
plt.ylabel("Number of Records")
plt.show()

###########


print()
