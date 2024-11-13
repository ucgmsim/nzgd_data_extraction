from pathlib import Path
from pygmt_helper import plotting
import pandas as pd
import pygmt
import toml
import numpy as np
import matplotlib.pyplot as plt


estimated_vs30_dir = Path("/home/arr65/data/nzgd/resources/vs30_map")
output_ffp = Path("/home/arr65/data/nzgd/plots") / "old_vs30_model.png"
# new_data_dir = Path("/home/arr65/data/nzgd/processed_data_copy/cpt/data")
#
# old_data_dir = Path("/home/arr65/vs30_data_input_data/parquet/data")
#
# metadata_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/metadata")

# num_procs = 8

### Setting the station_id to be the record_name for compatibility with functions that use record_name
estimated_vs30_ll_df = pd.read_csv(estimated_vs30_dir/"non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll",
                                sep=" ",
                                header=None,
                                names=["longitude",
                                       "latitude",
                                       "record_name"])
estimated_vs30_value_df = pd.read_csv(estimated_vs30_dir/"non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30",
                                   sep=" ", header=None, names=["record_name", "vs30"])

### Merge the two dataframes
vs30_df = estimated_vs30_value_df.merge(estimated_vs30_ll_df, on=["record_name"], how="left")

### Make a histogram of vs30_df["vs30"]
plt.hist(vs30_df["vs30"], bins=500)
plt.xlabel("vs30")
plt.ylabel("count")
#plt.xlim(200, 750)
plt.savefig(output_ffp.parent / "vs30_histogram.png",dpi=500)
print()
#################################################
##########################################################################


### All of NZ
region = [166, 179, -47.5, -34.1]

### CHC
# lat = -43.5320
# lon = 172.6366
# dlat = 0.05
# dlon = 0.15
# region = [lon-dlon, lon+dlon, lat-dlat, lat+dlat]
# block_size = 0.001
# block_data = pygmt.blockmean(
#     x=vs30_df["longitude"],
#     y=vs30_df["latitude"],
#     z=vs30_df["vs30"],
#     region=region,
#     spacing=block_size)

print()

# Generate the grid
# grid = pygmt.surface(
#     x=vs30_df["longitude"],
#     y=vs30_df["latitude"],
#     z=vs30_df["vs30"],
#     region=region,
#     spacing=0.001,  # Adjust spacing based on your map's scale
# )

# grid = pygmt.surface(data=block_data,
#     region=region,
#     spacing=block_size,  # Adjust spacing based on your map's scale
# )

print()


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

#pygmt.makecpt(cmap="viridis", series=[vs30_df["vs30"].min(), vs30_df["vs30"].max()])
pygmt.makecpt(cmap="viridis", series=[200, 700])

print("plotting points...")
fig.plot(
    x=vs30_df["longitude"],
    y=vs30_df["latitude"],
    fill=vs30_df["vs30"],
    cmap=True,
    style="c0.05c")

# print("adding colorbar...")
# fig.colorbar(frame="xaf+llog residual (ln(new) - ln(old))")

# fig.grdimage(
#     grid=grid,
#     cmap="viridis",  # Choose a color map (you can use any PyGMT colormap)
#     shading=True     # Optional: adds shading to give a 3D effect
# )
# fig.colorbar(frame='af+l"Vs30 (m/s)"')

fig.colorbar(
    frame='af+l"Vs30 (m/s)"'
)

print("saving figure...")

fig.savefig(
    output_ffp,
    dpi=1200,
    anti_alias=True,
)


print()




print()