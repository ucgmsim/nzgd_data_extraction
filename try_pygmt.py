from pathlib import Path
from pygmt_helper import plotting
import pandas as pd
import toml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

record_id_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_25092024_1043.csv")
categorized_record_ids = toml.load("/home/arr65/data/nzgd/stats_plots/categorized_record_ids.toml")

record_id_df_digitized_borehole = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Digitized data Borehole"])]
record_id_df_digitized_cpt = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Digitized data CPT"])]
record_id_df_digitized_scpt = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Digitized data SCPT"])]
record_id_df_digitized_vsvp = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Digitized data VsVp"])]

# record_id_df_non_digitized_borehole = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Non-data Borehole"])]
# record_id_df_non_digitized_cpt = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Non-data CPT"])]
# record_id_df_non_digitized_scpt = record_id_df[record_id_df["ID"].isin(categorized_record_ids["Non-data SCPT"])]

#################################


# Convert the 'InvestigationDate' column to datetime format
record_id_df_digitized_cpt['InvestigationDate'] = pd.to_datetime(record_id_df_digitized_cpt['InvestigationDate'], format='%d/%b/%Y')
record_id_df_digitized_cpt['PublishedDate'] = pd.to_datetime(record_id_df_digitized_cpt['PublishedDate'], format='%d/%b/%Y')

df_sorted_InvestigationDate = record_id_df_digitized_cpt.sort_values(by='InvestigationDate')
df_sorted_PublishedDate = record_id_df_digitized_cpt.sort_values(by='PublishedDate')

print()


fig, ax = plt.subplots()
# Plot the data
ax.plot(df_sorted_PublishedDate["PublishedDate"], np.cumsum(np.ones(len(df_sorted_PublishedDate))),marker='.', linestyle="")
#ax.plot(df_sorted_InvestigationDate["InvestigationDate"], np.cumsum(np.ones(len(df_sorted_InvestigationDate))),marker='.', linestyle="")


# Format the x-axis to show dates nicely
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.YearLocator(1)) # Show every year
#ax.xaxis.set_major_locator(mdates.YearLocator(5))


# Rotate date labels for better readability
plt.gcf().autofmt_xdate()

plt.xlabel('Publication date')
#plt.xlabel('Investigation date')
plt.ylabel('number of CPT records')

plt.savefig("/home/arr65/data/nzgd/stats_plots/cpt_publication_date.png", dpi=500)
#plt.savefig("/home/arr65/data/nzgd/stats_plots/cpt_investigation_date.png", dpi=500)


plt.show()


fig, ax = plt.subplots()
# Plot the data
ax.plot(df_sorted_PublishedDate["PublishedDate"], df_sorted_PublishedDate["InvestigationDate"] ,marker='.', linestyle="")
plt.xlabel('Publication date')
plt.ylabel('Investiation date')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.YearLocator(1)) # Show every year

ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.yaxis.set_major_locator(mdates.YearLocator(1)) # Show every year

plt.gcf().autofmt_xdate()
#plt.gcf().autofmt_ydate()

plt.savefig("/home/arr65/data/nzgd/stats_plots/investigation_date_vs_publication_date.png", dpi=500)
print()

# Sort the DataFrame by the 'InvestigationDate' column



#date_string = "12/Dec/2013"

# datetimes = []

# for date_string in record_id_df_digitized_cpt["InvestigationDate"]:
#     date_format = "%d/%b/%Y"
#     # Convert the string to a datetime object
#     date_object = datetime.datetime.strptime(date_string, date_format)
#     datetimes.append(date_object)

print()

#################################

sung_cpt_names = np.loadtxt("/home/arr65/data/nzgd/stats_plots/sung_cpt_names.txt", dtype=str)

cpt_df = record_id_df[record_id_df["Type"] == "CPT"]
cpt_not_in_sung_df = cpt_df[~cpt_df["ID"].isin(sung_cpt_names)]
cpt_in_sung_df = cpt_df[cpt_df["ID"].isin(sung_cpt_names)]

num_sung_scpt = 0
for scpt_name in sung_cpt_names:
    if "SCPT" in scpt_name:
        num_sung_scpt += 1

print()

### All NZ
region = [164, 180, -49, -33.]
output_ffp = Path("/home/arr65/data/nzgd/map_plots/All_NZ_new_cpt_data.png")

### CHC
# lat = -43.5320
# lon = 172.6366
#
# dlat = 0.1
# dlon = 0.1
#
# region = [lon-dlon, lon+dlon, lat-dlat, lat+dlat]
# output_ffp = Path("/home/arr65/data/nzgd/map_plots/CHC_data.png")


# Path to the map data
# Is part of the qcore package, once installed it
# can be found under qcore/qcore/data
# Set to None for lower quality map, but much faster plotting time
map_data_ffp = None
#map_data_ffp = Path("/home/arr65/src/qcore/qcore/data")

# If true, then use the high resolution topography
# This will further increase plot time, and only has an
# effect if map_data_ffp is set
#use_high_res_topo = True
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
        FONT_ANNOT_PRIMARY="6p,Helvetica,black",
    ),
)

# fig.plot(x=record_id_df_digitized_cpt["Longitude"], y=record_id_df_digitized_cpt["Latitude"], style="c0.1c", fill="red", label="CPT")
# fig.plot(x=record_id_df_digitized_scpt["Longitude"], y=record_id_df_digitized_scpt["Latitude"], style="d0.1c", fill="orange", label="SCPT")
# fig.plot(x=record_id_df_digitized_borehole["Longitude"], y=record_id_df_digitized_borehole["Latitude"], style="s0.1c", fill="blue", label="Borehole")
#fig.plot(x=record_id_df_digitized_vsvp["Longitude"], y=record_id_df_digitized_vsvp["Latitude"], style="t0.1c", fill="green", label="VsVp")

fig.plot(x=cpt_not_in_sung_df["Longitude"], y=cpt_not_in_sung_df["Latitude"], style="t0.1c", fill="blue", label="new CPT dataset")
fig.plot(x=cpt_in_sung_df["Longitude"], y=cpt_in_sung_df["Latitude"], style="c0.05c", fill="orange", label="old CPT dataset")


fig.legend()

fig.savefig(
    output_ffp,
    dpi=900,
    anti_alias=True,
)