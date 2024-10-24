"""
This script generates the html.index file for the NZGD HTTP server on Hypocentre.
"""
import folium
from folium.plugins import MarkerCluster
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from download_nzgd_data.lib import map

import time

start_time = time.time()

record_id_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_23102024_1042.csv")

date_of_last_nzgd_retrieval = Path("/home/arr65/data/nzgd/hypocentre_mirror/nzgd/date_of_last_nzgd_retrieval.txt").read_text().strip("\n")

raw_nzgd_files = map.get_files_with_relative_paths(processed_files=False,
                           file_root_directory=Path("/home/arr65/data/nzgd/hypocentre_mirror/nzgd/raw_from_nzgd"),
                           relative_to = Path("/home/arr65/data/nzgd/hypocentre_mirror/nzgd"))

processed_files = map.get_files_with_relative_paths(processed_files=True,
                            file_root_directory=Path("/home/arr65/data/nzgd/hypocentre_mirror/nzgd/processed"),
                            relative_to = Path("/home/arr65/data/nzgd/hypocentre_mirror/nzgd"))

processed_metadata = map.get_processed_metadata(
    file_root_directory=Path("/home/arr65/data/nzgd/hypocentre_mirror/nzgd/processed"))


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
            <a href="raw_from_nzgd/">Browse all raw data by region</a>            
        </div>
        """

m.get_root().html.add_child(folium.Element(date_of_last_ngzd_retrieval))
m.get_root().html.add_child(folium.Element(browse_link_text))

print()
print("Adding markers to map")
for row_index, row in tqdm(record_id_df.iterrows(), total=record_id_df.shape[0]):

    if row["ID"] not in raw_nzgd_files:
        continue

    popup_html = f"<h1>{row['ID']}</h1><br>"

    if row["ID"] in processed_metadata:
        popup_html += f"<h4>Metadata:</h4><br>"
        popup_html += f"max depth = {processed_metadata[row['ID']].max_depth}<br>"
        popup_html += f"min depth = {processed_metadata[row['ID']].min_depth}<br><br>"
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
    ).add_to(marker_cluster)

print()
print("Saving map")

m.save('/home/arr65/data/nzgd/hypocentre_mirror/nzgd/index.html')


end_time = time.time()

time_taken = end_time - start_time

print(f"Time taken: {time_taken/60} minutes")
