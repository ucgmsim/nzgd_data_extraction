from pygmt_helper import plotting
import pandas as pd
import pygmt
import toml
import numpy as np
import matplotlib.pyplot as plt
from qcore import coordinates

import pygmt
from pathlib import Path

# Define the file path to your GeoTIFF file
#geotiff_file = "path/to/your/geotiff_file.tif"

record_names_in_old_and_new_datasets = pd.read_csv("/home/arr65/data/nzgd/resources/record_names_in_old_and_new_datasets.csv")

geotiff_file = Path("/home/arr65/data/nzgd/resources/vs30map_data_2023_geotiff/vs30map_data/combined.tif")
info = pygmt.grdinfo(grid=geotiff_file)
# Downsample the GeoTIFF to inspect it
downsampled_grid = pygmt.grdsample(grid=geotiff_file, spacing=1000)

# # Convert the GeoTIFF to a numpy array of x, y, and z values
# grid_data = pygmt.grd2xyz(grid=geotiff_file, output_type="numpy")
#
# # Create a pandas DataFrame from the numpy array
# grid_data_df = pd.DataFrame(grid_data, columns=["x", "y", "value"])
#
# # drop rows of the grid_data_df that have NaN values in the value column
# grid_data_df_dropna = grid_data_df.dropna(subset=["value"])

# print()

##################################################################

# Convert the downsampled grid to a numpy array
# grid_data = pygmt.grd2xyz(grid=downsampled_grid, output_type="numpy")
#
# # Check for NaNs in the grid data
# nan_count = np.isnan(grid_data[:, 2]).sum()
# total_count = grid_data.shape[0]
#
# print(f"Total points in grid: {total_count}")
# print(f"Number of NaNs in grid: {nan_count}")
# print(f"Percentage of NaNs: {100 * nan_count / total_count:.2f}%")



######################################################################

vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/vs30_calculations.csv")

# drop the column 'Unnamed: 0'
#vs30_df = vs30_df[(vs30_df["cpt_vs_correlation"] == "andrus_2007") & (vs30_df["vs30_correlation"] == "boore_2011")]
# drop duplicate rows except for the first occurrence
vs30_df.drop_duplicates(subset=["cpt_name"],keep='first',inplace=True)
vs30_df = vs30_df[~vs30_df["cpt_name"].isin(record_names_in_old_and_new_datasets["record_names_in_old_and_new_datasets"])]

vs30_latlon = vs30_df[["latitude","longitude"]].to_numpy()

vs30_nztm = coordinates.wgs_depth_to_nztm(vs30_latlon)

vs30_df.loc[:,"nztm_y"] = vs30_nztm[:, 0]
vs30_df.loc[:,"nztm_x"] = vs30_nztm[:, 1]
############################################
#
# x_min = 1060050
# x_max = 2120050
#
# y_min = 4730050
# y_max = 6250050
#
#
# within_bounds = vs30_df[
#     (vs30_df["nztm_x"] >= x_min) & (vs30_df["nztm_x"] <= x_max) &
#     (vs30_df["nztm_y"] >= y_min) & (vs30_df["nztm_y"] <= y_max)
# ]
#
# print()
#
# print(f"Points within bounds: {len(within_bounds)} out of {len(vs30_df)}")
#
# print()


############################################

test = pygmt.grdtrack(points=vs30_df[["nztm_x", "nztm_y"]],
                            grid=geotiff_file, newcolname="model_vs30",
                              radius=r"10+e")

print(100*np.sum(np.isfinite(vs30_df["model_vs30"]))/len(vs30_df["model_vs30"]))

print()



vs30_df.loc[:,"model_vs30"] = model_vs30_df["model_vs30"]
vs30_df.loc[:,"ln_cptvs30_minus_modelvs30"] = np.log(vs30_df["vs30"]) - np.log(vs30_df["model_vs30"])



print()

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

print()


resid_no_nan = vs30_df["ln_cptvs30_minus_modelvs30"].dropna()
colorbar_percent_extreme_to_exclude = 3
resid_max = np.percentile(resid_no_nan, 100-colorbar_percent_extreme_to_exclude/2)
resid_min = np.percentile(resid_no_nan, colorbar_percent_extreme_to_exclude/2)

# Step 1: Downsample the GeoTIFF using grdsample to a lower resolution
downsampled_grid = pygmt.grdsample(
    grid=geotiff_file,
    spacing=1000  # Set desired grid spacing for lower resolution
)

### All of NZ
region = [166, 179, -47.5, -34.1]

# Plot the GeoTIFF in the background
fig = pygmt.Figure()


fig.grdimage(
    grid=downsampled_grid,          # Path to the GeoTIFF file
    cmap="viridis",             # Color map for the data
    frame=True,                  # Add frame to the map
)
fig.colorbar(frame='af+lBackground Vs30 model (m/s)')  # Label the color bar as needed

# resid_colormin = -1.2
# resid_colormax = 1.0
pygmt.makecpt(cmap="haxby", series=[resid_min, resid_max])
#pygmt.makecpt(cmap="haxby", series=[resid_min, resid_max])
#pygmt.makecpt(cmap="haxby")
fig.plot(
    x=vs30_df["nztm_x"],
    y=vs30_df["nztm_y"],
    fill=vs30_df["ln_cptvs30_minus_modelvs30"],
    cmap=True,
    style="c0.08c",
    pen="black")

fig.colorbar(frame="a0.1+lln(CPT Vs30) - ln(Model Vs30) (discrete points)",position="JML+o2c/0c")
fig.savefig(Path("/home/arr65/data/nzgd/plots") / "vs30_from_geotiff.png", dpi=500)

# Show the plot
#fig.show()
