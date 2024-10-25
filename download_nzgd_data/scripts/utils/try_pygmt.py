from pathlib import Path
from pygmt_helper import plotting
import pandas as pd
import toml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

record_id_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_25092024_1043.csv")
categorized_record_ids = toml.load("/home/arr65/data/nzgd/stats_plots/categorized_record_ids.toml")
sung_id_with_nzgd_match_df = pd.read_csv("/home/arr65/data/nzgd/stats_plots/sung_id_with_nzgd_match.csv")

record_id_df_digitized_borehole = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Digitized data Borehole"])]
record_id_df_digitized_cpt = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Digitized data CPT"])]
record_id_df_digitized_scpt = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Digitized data SCPT"])]
record_id_df_digitized_vsvp = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Digitized data VsVp"])]

record_id_df_not_digitized_borehole = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Non-data Borehole"])]
record_id_df_not_digitized_cpt = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Non-data CPT"])]
record_id_df_not_digitized_scpt = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Non-data SCPT"])]

#################################


# Convert the 'InvestigationDate' column to datetime format
# record_id_df_digitized_cpt['InvestigationDate'] = pd.to_datetime(record_id_df_digitized_cpt['InvestigationDate'], format='%d/%b/%Y')
# record_id_df_digitized_cpt['PublishedDate'] = pd.to_datetime(record_id_df_digitized_cpt['PublishedDate'], format='%d/%b/%Y')
#
# df_sorted_InvestigationDate = record_id_df_digitized_cpt.sort_values(by='InvestigationDate')
# df_sorted_PublishedDate = record_id_df_digitized_cpt.sort_values(by='PublishedDate')
#
# print()
#
#
# fig, ax = plt.subplots()
# # Plot the data
# ax.plot(df_sorted_PublishedDate["PublishedDate"], np.cumsum(np.ones(len(df_sorted_PublishedDate))),marker='.', linestyle="")
# #ax.plot(df_sorted_InvestigationDate["InvestigationDate"], np.cumsum(np.ones(len(df_sorted_InvestigationDate))),marker='.', linestyle="")
#
#
# # Format the x-axis to show dates nicely
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.YearLocator(1)) # Show every year
# #ax.xaxis.set_major_locator(mdates.YearLocator(5))
#
#
# # Rotate date labels for better readability
# plt.gcf().autofmt_xdate()
#
# plt.xlabel('Publication date')
# #plt.xlabel('Investigation date')
# plt.ylabel('number of CPT records')
#
# plt.savefig("/home/arr65/data/nzgd/stats_plots/cpt_publication_date.png", dpi=500)
# #plt.savefig("/home/arr65/data/nzgd/stats_plots/cpt_investigation_date.png", dpi=500)
#
#
# plt.show()
#
#
# fig, ax = plt.subplots()
# # Plot the data
# ax.plot(df_sorted_PublishedDate["PublishedDate"], df_sorted_PublishedDate["InvestigationDate"] ,marker='.', linestyle="")
# plt.xlabel('Publication date')
# plt.ylabel('Investiation date')
#
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.xaxis.set_major_locator(mdates.YearLocator(1)) # Show every year
#
# ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.yaxis.set_major_locator(mdates.YearLocator(1)) # Show every year
#
# plt.gcf().autofmt_xdate()
# #plt.gcf().autofmt_ydate()
#
# plt.savefig("/home/arr65/data/nzgd/stats_plots/investigation_date_vs_publication_date.png", dpi=500)
# print()



### All NZ
# region = [166, 179, -47.5, -34.1]


### CHC
# lat = -43.5320
# lon = 172.6366
#
# dlat = 0.05
# dlon = 0.15

### Up North
lat = -37.96
lon = 177
dlat = 0.02
dlon = 0.05

region = [lon-dlon, lon+dlon, lat-dlat, lat+dlat]
# output_ffp = Path("/home/arr65/data/nzgd/map_plots/CHC_data.png")


# Path to the map data
# Is part of the qcore package, once installed it
# can be found under qcore/qcore/data
# Set to None for lower quality map, but much faster plotting time
#map_data_ffp = None
map_data_ffp = Path("/home/arr65/src/qcore/qcore/data")

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

output_ffp = Path("/home/arr65/data/nzgd/map_plots/up_north_VsVp.png")

# fig.plot(x=record_id_df_not_digitized_borehole["Longitude"], y=record_id_df_not_digitized_borehole["Latitude"], style="c0.35c", fill="#2ca02c", label="only pdf")
# fig.plot(x=record_id_df_digitized_borehole["Longitude"], y=record_id_df_digitized_borehole["Latitude"], style="d0.3c", fill="#ff7f0e", label="data")

# fig.plot(x=record_id_df_digitized_scpt["Longitude"], y=record_id_df_digitized_scpt["Latitude"], style="d0.3c", fill="#ff7f0e", label="data")
# fig.plot(x=record_id_df_not_digitized_scpt["Longitude"], y=record_id_df_not_digitized_scpt["Latitude"], style="c0.25c", fill="#2ca02c", label="only pdf")

# fig.plot(x=record_id_df_digitized_cpt["Longitude"], y=record_id_df_digitized_cpt["Latitude"], style="d0.3c", fill="#ff7f0e", label="data")
# fig.plot(x=record_id_df_not_digitized_cpt["Longitude"], y=record_id_df_not_digitized_cpt["Latitude"], style="c0.25c", fill="#2ca02c", label="only pdf")

# fig.plot(x=record_id_df_digitized_cpt["Longitude"], y=record_id_df_digitized_cpt["Latitude"], style="d0.3c", fill="DARKMAGENTA", label="new CPT dataset")
# fig.plot(x=sung_id_with_nzgd_match_df["lon"], y=sung_id_with_nzgd_match_df["lat"], style="c0.35c", fill="forestgreen", label="old CPT dataset")

fig.plot(x=record_id_df_digitized_vsvp["Longitude"], y=record_id_df_digitized_vsvp["Latitude"], style="d0.3c", fill="#ff7f0e", label="data")



#fig.legend(position="JTL+jTL+o0.2c", box="+gwhite")
fig.legend()
fig.savefig(
    output_ffp,
    dpi=900,
    anti_alias=True,
)

