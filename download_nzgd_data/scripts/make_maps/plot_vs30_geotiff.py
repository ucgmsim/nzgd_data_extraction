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
vs30_latlon = vs30_df[["latitude", "longitude"]].to_numpy()

vs30_nztm = coordinates.wgs_depth_to_nztm(vs30_latlon)

vs30_df["nztm_y"] = vs30_nztm[:, 0]
vs30_df["nztm_x"] = vs30_nztm[:, 1]

# Step 1: Downsample the GeoTIFF using grdsample to a lower resolution
downsampled_grid = pygmt.grdsample(
    grid=geotiff_file,
    spacing=1000  # Set desired grid spacing for lower resolution
)

### All of NZ
region = [166, 179, -47.5, -34.1]



# Create a new figure
fig = pygmt.Figure()

fig.grdimage(
    grid=downsampled_grid,          # Path to the GeoTIFF file
    cmap="viridis",             # Color map for the data
    frame=True,                  # Add frame to the map
)
fig.colorbar(frame='af+lVs30 (m/s)')  # Label the color bar as needed

# Plot the GeoTIFF
### This doesn't work
# fig.grdimage(
#     grid=downsampled_grid,          # Path to the GeoTIFF file
#     cmap="viridis",             # Color map for the data
#     frame=True,                  # Add frame to the map
#     projection="M6i",           # Set the map projection
#     region=region,               # Set the region to plot
# )

pygmt.makecpt(cmap="mag", series=[vs30_df["vs30"].min(), vs30_df["vs30"].max()])
fig.plot(
    x=vs30_df["nztm_x"],
    y=vs30_df["nztm_y"],
    fill=vs30_df["vs30"],
    cmap=True,
    style="c0.05c")

fig.colorbar(position="JML+o0.5c/0c")
fig.savefig(Path("/home/arr65/data/nzgd/plots") / "vs30_from_geotiff.png", dpi=500)


# Show the plot
#fig.show()
print()