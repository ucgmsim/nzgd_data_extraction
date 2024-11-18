from pygmt_helper import plotting
import pandas as pd
import pygmt
import toml
import numpy as np
import matplotlib.pyplot as plt
from qcore import coordinates

import pygmt
from pathlib import Path

record_names_in_old_dataset = pd.read_csv("/home/arr65/data/nzgd/resources/record_names_in_old_dataset.csv")["record_names_in_old_dataset"].to_list()

geotiff_file = Path("/home/arr65/data/nzgd/resources/vs30map_data_2023_geotiff/vs30map_data/combined.tif")
info = pygmt.grdinfo(grid=geotiff_file)
## Downsample the GeoTIFF using grdsample to a lower resolution for easier plotting
downsampled_grid = pygmt.grdsample(grid=geotiff_file, spacing=1000)

vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/vs30_from_data_and_model.csv")

vs30_df.drop_duplicates(subset=["record_name"],keep='first',inplace=True)
vs30_df = vs30_df[~vs30_df["record_name"].isin(record_names_in_old_dataset)]

vs30_latlon = vs30_df[["latitude","longitude"]].to_numpy()

vs30_nztm = coordinates.wgs_depth_to_nztm(vs30_latlon)
vs30_df.loc[:,"nztm_y"] = vs30_nztm[:, 0]
vs30_df.loc[:,"nztm_x"] = vs30_nztm[:, 1]
vs30_df.loc[:,"ln_cptvs30_minus_modelvs30"] = np.log(vs30_df["vs30"]) - np.log(vs30_df["model_vs30"])

## Make a histogram of vs30 residual distribution and set limits for the colorbar
resid_no_nan = vs30_df["ln_cptvs30_minus_modelvs30"].dropna()
colorbar_percent_extreme_to_exclude = 3
resid_max = np.percentile(resid_no_nan, 100-colorbar_percent_extreme_to_exclude/2)
resid_min = np.percentile(resid_no_nan, colorbar_percent_extreme_to_exclude/2)

counts, bins, patches = plt.hist(vs30_df["ln_cptvs30_minus_modelvs30"], bins=30)
plt.xlabel("ln(CPT Vs30) - ln(Model Vs30)")
plt.ylabel("count")
plt.vlines([resid_min, resid_max],0,np.max(counts),colors="red")

plt.savefig("/home/arr65/data/nzgd/plots/vs30_residual_histogram.png",dpi=500)

resid_no_nan = vs30_df["ln_cptvs30_minus_modelvs30"].dropna()
colorbar_percent_extreme_to_exclude = 3
resid_max = np.percentile(resid_no_nan, 100-colorbar_percent_extreme_to_exclude/2)
resid_min = np.percentile(resid_no_nan, colorbar_percent_extreme_to_exclude/2)

# Plot the GeoTIFF in the background
fig = pygmt.Figure()


fig.grdimage(
    grid=downsampled_grid,          # Path to the GeoTIFF file
    cmap="viridis",             # Color map for the data
    frame=True,                  # Add frame to the map
)
fig.colorbar(frame='af+lBackground Vs30 model (m/s)')  # Label the color bar as needed
pygmt.makecpt(cmap="haxby", series=[resid_min, resid_max])

fig.plot(
    x=vs30_df["nztm_x"],
    y=vs30_df["nztm_y"],
    fill=vs30_df["ln_cptvs30_minus_modelvs30"],
    cmap=True,
    style="c0.08c",
    pen="black")

fig.colorbar(frame="a0.1+lln(CPT Vs30) - ln(Model Vs30) (discrete points)",position="JML+o2c/0c")
fig.savefig(Path("/home/arr65/data/nzgd/plots") / "vs30_from_geotiff.png", dpi=500)
