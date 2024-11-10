"""
This script generates the html.index file for the NZGD HTTP server on Hypocentre.
"""

import folium
from folium.plugins import MarkerCluster, Search
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from download_nzgd_data.lib import map
import branca

import time

start_time = time.time()

vs30_df = pd.read_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/parquet_vs30_results.csv")
vs30_correlation = "Boore2011"
cpt_vs_correlation = "Andrus2007"

print()
### drop rows of the vs30_df that have NaN values in the vs30 or vs30_std columns
vs30_df = vs30_df.dropna(subset=["vs30", "vs30_sd"])
print()

vs30_dict = {}
for row_index, row in vs30_df.iterrows():
    vs30_dict[row["cpt_name"]] = row["vs30"]





print()

max_num_records = None

record_id_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_23102024_1042.csv")
hypo_base_dir = Path("/home/arr65/data/nzgd/small_hypocentre_mirror/nzgd")

record_id_df = record_id_df[record_id_df["Type"].isin(["CPT", "SCPT", "Borehole", "VsVp"])]

if max_num_records:
    record_id_df = record_id_df.iloc[:max_num_records]

date_of_last_nzgd_retrieval = Path("/home/arr65/data/nzgd/hypocentre_mirror/nzgd/date_of_last_nzgd_retrieval.txt").read_text().strip("\n")

raw_nzgd_files = map.get_files_with_relative_paths(processed_files=False,
                           file_root_directory=hypo_base_dir / "raw_from_nzgd",
                           relative_to = hypo_base_dir,
                           max_num_records=max_num_records)

processed_files = map.get_files_with_relative_paths(processed_files=True,
                            file_root_directory=hypo_base_dir / "processed",
                            relative_to = hypo_base_dir,
                            max_num_records=max_num_records)

processed_metadata = map.get_processed_metadata(
    file_root_directory=hypo_base_dir / "processed",
    max_num_records=max_num_records)

# Make an empty map centered on New Zealand
m = folium.Map(location=[-41.2728,173.2994], tiles="OpenStreetMap", zoom_start=6)

# Create a MarkerCluster object
marker_cluster = MarkerCluster().add_to(m)

date_of_last_ngzd_retrieval = f"""
     <div style="position: fixed; 
                 bottom: 10px; left: 10px; width: auto; height: auto; 
                 z-index: 9999;
                 font-size: 24px; color: black; background-color: rgba(255, 255, 255, 0.7);
                 padding: 10px; border-radius: 5px; font-weight: bold;">
        NZGD last retrieved on {date_of_last_nzgd_retrieval}
     </div>
     """

browse_link_text = f"""
        <div style="position: fixed;
                    bottom: 10px; right: 10px; width: auto; height: auto;
                    z-index: 9999;
                    font-size: 24px; color: black; background-color: rgba(255, 255, 255, 0.7);
                    padding: 10px; border-radius: 5px; font-weight: bold;">
            <a href="processed/">Browse all processed data by region</a> <br> 
            <a href="raw_from_nzgd/">Browse all raw data by region</a> <br>
            <a href="https://www.dropbox.com/scl/fo/9y9x4bpwsbd7fwnzmsm00/AM0ZTcQ0x_D2T4GZRjU81hE?rlkey=bbm5067njdlppwtazcgu1xk3b&st=c0a5n99f&dl=0">Browse files in Dropbox (recommended for large downloads)</a>
        </div>
        """

# Create the HTML for the legend
legend_html = '''
     <div style="
     position: fixed; 
     bottom: 150px; left: 50px; width: 150px; height: 150px; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey;
     border-radius: 5px;
     padding: 10px;
     ">
     <h4>Legend</h4>
     <div style="display: flex; align-items: center;">
         <div style="height: 12px; width: 12px; background-color: blue; border-radius: 50%; margin-right: 5px;"></div>
         <span>CPT</span>
     </div>
     <div style="display: flex; align-items: center;">
         <div style="height: 12px; width: 12px; background-color: red; margin-right: 5px; border: 1px solid red;"></div>
         <span>SCPT</span>
     </div>
     <div style="display: flex; align-items: center;">
         <div style="height: 0; width: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 12px solid green; margin-right: 5px;"></div>
         <span>Borehole</span>
     </div>
     <div style="display: flex; align-items: center;">
         <div style="height: 12px; width: 12px; color: yellow;">★</div>
         <span>VsVp</span>
     </div>
     </div>
     '''

# legend_html = """
# <div style="
#     position: fixed;
#     bottom: 50px;
#     left: 50px;
#     width: 150px;
#     height: 60px;
#     background-color: white;
#     border:2px solid grey;
#     z-index:9999;
#     font-size:14px;
#     padding: 10px;
#     ">
#     <b>Legend</b><br>
#     <i class="fa fa-circle" style="color:blue"></i>&nbsp;Blue Circle Marker<br>
# </div>
# """

m.get_root().html.add_child(folium.Element(date_of_last_ngzd_retrieval))
m.get_root().html.add_child(folium.Element(browse_link_text))
m.get_root().html.add_child(branca.element.Element(legend_html))

print("Adding markers to map")
for row_index, row in tqdm(record_id_df.iterrows(), total=record_id_df.shape[0]):
    if row["ID"] not in raw_nzgd_files:
        continue

    icon = folium.Icon(icon="star", color="purple", prefix="fa") #row["Type"] == "VsVp"
    if row["Type"] == "CPT":
        icon = folium.Icon(icon="circle", color="blue", prefix="fa")
    elif row["Type"] == "SCPT":
        icon = folium.Icon(icon="square", color="red", prefix="fa")
    elif row["Type"] == "Borehole":
        icon = folium.Icon(icon="play", color="green", prefix="fa")

    popup_html = f"<h2>{row['ID']}</h2><br>"

    if row["ID"] in processed_metadata:
        popup_html += f"<h4>Metadata:</h4><br>"
        if row["ID"] in vs30_dict:
            popup_html += f"Vs30 = {vs30_dict[row['ID']]} m/s (using {vs30_correlation} and {cpt_vs_correlation})<br>"
        else:
            popup_html += f"Vs30 not available.<br>"
        popup_html += f"max depth = {processed_metadata[row['ID']].max_depth} m<br>"
        popup_html += f"min depth = {processed_metadata[row['ID']].min_depth} m<br><br>"
    else:
        popup_html += f"No metadata available.<br><br>"

    if row["ID"] in processed_files:
        popup_html += f"<h4>Processed files:</h4><br>"
        for processed_file in processed_files[row["ID"]]:
            popup_html += f"<a href='{processed_file}'>{processed_file.name}</a><br>"
    else:
        popup_html += f"No processed NZGD files available.<br>"

    popup_html += "<br><h4>Raw NZGD files:</h4><br>"

    for raw_file in raw_nzgd_files[row["ID"]]:
        popup_html += f"<a href='{raw_file}'>{raw_file.name}</a><br>"

    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_html,
        icon=icon,
        name=row["ID"]
    ).add_to(marker_cluster)

# Add Search plugin to search within the MarkerCluster based on popup text
search = Search(
    layer=marker_cluster,
    geom_type="Point",
    placeholder="Search for a record name (e.g., CPT_1)",
    collapsed=False,
    search_label="name"
).add_to(m)

print()
print("Saving map")

m.save(hypo_base_dir / "index.html")


end_time = time.time()

time_taken = end_time - start_time

print(f"Time taken: {time_taken/60} minutes")
