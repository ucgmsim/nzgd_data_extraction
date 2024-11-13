# from pathlib import Path
# from pygmt_helper import plotting
# import pandas as pd
# import pygmt
# import toml
# import numpy as np
# import matplotlib.pyplot as plt

import pygmt
from pathlib import Path

# Define the file path to your GeoTIFF file
#geotiff_file = "path/to/your/geotiff_file.tif"


geotiff_file = "/home/arr65/Downloads/vs30map_data_2023/vs30map_data/combined.tif"

info = pygmt.grdinfo(grid=geotiff_file)


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

# Plot the GeoTIFF
### This doesn't work
# fig.grdimage(
#     grid=downsampled_grid,          # Path to the GeoTIFF file
#     cmap="viridis",             # Color map for the data
#     frame=True,                  # Add frame to the map
#     projection="M6i",           # Set the map projection
#     region=region,               # Set the region to plot
# )

# Optionally, add a color bar
fig.colorbar(frame='af+lVs30 (m/s)')  # Label the color bar as needed
fig.savefig(Path("/home/arr65/data/nzgd/plots") / "vs30_from_geotiff.png", dpi=500)


# Show the plot
#fig.show()
print()