# import the library
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import toml


record_id_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_25092024_1043.csv")

#record_id_df = record_id_df[record_id_df["Type"] == "VsVp"]
record_id_df = record_id_df[record_id_df["Type"] == "CPT"]

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
for i in range(0,2):

   popup_html = "test test test test test <a href='raw_from_nzgd/borehole/Canterbury/Christchurch_City/Christchurch/Addington/BH_1761/Borehole_1761_RAW01.pdf'> string here</a>"
   #record_id_df.iloc[i]['ID']

   folium.Marker(
      location=[record_id_df.iloc[i]['Latitude'], record_id_df.iloc[i]['Longitude']],
      popup=popup_html,
   ).add_to(marker_cluster)


# for index, row in df.iterrows():
#     folium.Marker(location=[row['Latitude'], row['Longitude']]).add_to(marker_cluster)


m.save('/home/arr65/data/nzgd/downloads_and_metadata/test_map.html')

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