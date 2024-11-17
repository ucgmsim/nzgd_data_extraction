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


geotiff_file = Path("/home/arr65/data/nzgd/resources/vs30map_data_2023_geotiff/vs30map_data/combined.tif")
info = pygmt.grdinfo(grid=geotiff_file)

vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/vs30_calculations.csv")
# drop the column 'Unnamed: 0'
#vs30_df = vs30_df[(vs30_df["cpt_vs_correlation"] == "andrus_2007") & (vs30_df["vs30_correlation"] == "boore_2011")]
# drop duplicate rows except for the first occurrence
vs30_df.drop_duplicates(subset=["cpt_name"],keep='first',inplace=True)


vs30_latlon = vs30_df[["latitude","longitude"]].to_numpy()

vs30_nztm = coordinates.wgs_depth_to_nztm(vs30_latlon)

vs30_df.loc[:,"nztm_y"] = vs30_nztm[:, 0]
vs30_df.loc[:,"nztm_x"] = vs30_nztm[:, 1]

model_vs30_df = pygmt.grdtrack(points=vs30_df[["nztm_x", "nztm_y"]],
                            grid=geotiff_file, newcolname="model_vs30")

vs30_df.loc[:,"model_vs30"] = model_vs30_df["model_vs30"]
vs30_df.loc[:,"ln_cptvs30_minus_modelvs30"] = np.log(vs30_df["vs30"]) - np.log(vs30_df["model_vs30"])

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

resid_colormin = -1.2
resid_colormax = 1.0
pygmt.makecpt(cmap="haxby", series=[resid_colormin, resid_colormax])
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
print()