from pathlib import Path
from pygmt_helper import plotting
import pandas as pd
import pygmt
import toml
import numpy as np
import matplotlib.pyplot as plt

metadata_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/metadata")
output_ffp = metadata_dir / "new_vs30_residuals.png"

residual_df = pd.read_csv(metadata_dir / "new_vs30_resdiuals.csv")

#residual_df = residual_df.iloc[0:100]

residual = residual_df["residual_new_minus_old"]
ln_residual = residual_df["log_residual_ln_new_minus_ln_old"]

## Make a histogram of ln_residual to inform the limits of the colorbar
plt.hist(ln_residual, bins=100)
plt.xlabel("ln_residual")
plt.ylabel("count")
plt.savefig(metadata_dir / "ln_residual_histogram.png",dpi=500)

min_color = -1.4
max_color = 0.7

print()

### All of NZ
region = [166, 179, -47.5, -34.1]

### CHC
# lat = -43.5320
# lon = 172.6366
# dlat = 0.05
# dlon = 0.15
# region = [lon-dlon, lon+dlon, lat-dlat, lat+dlat]



map_data_ffp = Path("/home/arr65/src/qcore/qcore/data")


# Path to the map data
# Is part of the qcore package, once installed it
# can be found under qcore/qcore/data
# Set to None for lower quality map, but much faster plotting time
map_data_ffp = None
#map_data_ffp = Path("/home/arr65/src/qcore/qcore/data")

# If true, then use the high resolution topography
# This will further increase plot time, and only has an
# effect if map_data_ffp is set
use_high_res_topo = False
# List of events to highlight

### End Config
# Load map data

map_data = (
    None
    if map_data_ffp is None
    else plotting.NZMapData.load(map_data_ffp, high_res_topo=use_high_res_topo)
)
print("generating figure...")
min_lon, max_lon, min_lat, max_lat = region
# Generate the figure
fig = plotting.gen_region_fig(
    region=(min_lon, max_lon, min_lat, max_lat),
    map_data=map_data,
    config_options=dict(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xx",
        MAP_FRAME_PEN="thinner,black",
        FONT_ANNOT_PRIMARY="20p,Helvetica,black",
    ),
)

# print("plotting points...")
# fig.plot(
#     x=residual_df["record_lon"],
#     y=residual_df["record_lat"],
#     style="d0.3c",
#     fill=ln_residual,
#     cmap="viridis")

pygmt.makecpt(cmap="viridis", series=[min_color, max_color])

print("plotting points...")
fig.plot(
    x=residual_df["record_lon"],
    y=residual_df["record_lat"],
    fill=ln_residual,
    cmap=True,
    style="c0.1c",
    pen="black")

print("adding colorbar...")
fig.colorbar(frame="xaf+llog residual (ln(new) - ln(old))")

# fig.colorbar(
#     frame='af+l"Log Residual (ln)"'
# )

print("saving figure...")

fig.savefig(
    output_ffp,
    dpi=1200,
    anti_alias=True,
)


print()

