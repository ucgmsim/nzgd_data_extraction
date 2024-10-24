# import the library
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import toml
from pathlib import Path
from tqdm import tqdm

import time

start_time = time.time()

record_id_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_23102024_1042.csv")

directory = Path("/home/arr65/data/nzgd/downloads_and_metadata/raw_from_nzgd")

relative_to_dir = Path("/home/arr65/data/nzgd/downloads_and_metadata/")

# Recursively get all files
all_files = [file for file in directory.rglob('*') if file.is_file()]

record_id_to_files = {}
for file in all_files:
    if file.parent.name in record_id_to_files:
        record_id_to_files[file.parent.name].append(file.relative_to(relative_to_dir))
    else:
        record_id_to_files[file.parent.name] = [file.relative_to(relative_to_dir)]

#record_id_df = record_id_df[record_id_df["Type"] == "VsVp"]
#record_id_df = record_id_df[record_id_df["Type"] == "CPT"]

#categorized_record_ids = toml.load("/home/arr65/data/nzgd/stats_plots/categorized_record_ids.toml")


#record_id_df = record_id_df[record_id_df["ID"].isin(categorized_record_ids["VsVp"])]


print()

# Make an empty map
m = folium.Map(location=[-41.2728,173.2994], tiles="OpenStreetMap", zoom_start=6)

# Create a MarkerCluster object
marker_cluster = MarkerCluster().add_to(m)


# Make a data frame with dots to show on the map
# data = pd.DataFrame({
#    'lon':[-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],
#    'lat':[-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],
#    'name':['Buenos Aires', 'Paris', 'melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador'],
#    'value':[10, 12, 40, 70, 23, 43, 100, 43]
# }, dtype=str)



# add marker one by one on the map
# for i in range(0,len(record_id_df)):
#    folium.Marker(
#       location=[record_id_df.iloc[i]['Latitude'], record_id_df.iloc[i]['Longitude']],
#       popup=record_id_df.iloc[i]['ID'],
#    ).add_to(m)

# add marker one by one on the map
#for i in range(0,len(record_id_df)):
#for i in range(0,2):

# record_id_df = record_id_df.iloc[:100]

# late_update_text = '''
#      <div style="position: fixed;
#                  top: 10px; left: 10px; width: 100%; height: auto;
#                  z-index: 9999;
#                  font-size: 24px; color: black; background-color: rgba(255, 255, 255, 0.7);
#                  padding: 10px; border-radius: 5px; font-weight: bold;">
#         Last updated 23 October 2024
#      </div>
#      '''
# m.get_root().html.add_child(folium.Element(late_update_text))

for row_index, row in tqdm(record_id_df.iterrows(), total=record_id_df.shape[0]):

    if row["ID"] not in record_id_to_files:
        continue

    files = record_id_to_files[row["ID"]]

    popup_html = f"<h1>{row['ID']}</h1><br>Raw NZGD files:<br>"
    for file in files:
        popup_html += f"<a href='{file}'>{file.name}</a><br>"

    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_html,
    ).add_to(marker_cluster)


m.save('/home/arr65/data/nzgd/downloads_and_metadata/map_with_all_nzgd_links.html')


end_time = time.time()

time_taken = end_time - start_time

print(f"Time taken: {time_taken/3600} hours")

# Make an empty map


#### Custom HTML markers

# n = folium.Map(location=[20,0], tiles="OpenStreetMap", zoom_start=2)
# add marker one by one on the map
# for i in range(0,len(data)):
#     html=f"""
#         <h1> {data.iloc[i]['name']}</h1>
#         <p>You can use any html here! Let's do a list:</p>
#         <ul>
#             <li>Item 1</li>
#             <li>Item 2</li>
#         </ul>
#         </p>
#         <p>And that's a <a href="https://python-graph-gallery.com">link</a></p>
#         """
#     iframe = folium.IFrame(html=html, width=200, height=200)
#     popup = folium.Popup(iframe, max_width=2650)
#     folium.Marker(
#         location=[data.iloc[i]['lat'], data.iloc[i]['lon']],
#         popup=popup,
#         icon=folium.DivIcon(html=f"""
#             <div><svg>
#                 <circle cx="50" cy="50" r="40" fill="#69b3a2" opacity=".4"/>
#                 <rect x="35", y="35" width="30" height="30", fill="red", opacity=".3"
#             </svg></div>""")
#     ).add_to(n)