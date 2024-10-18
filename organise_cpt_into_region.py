import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

# Load the shapefile
district_gdf = gpd.read_file("/home/arr65/data/nzgd/shapefiles/lds-nz-land-districts-SHP_WGS84_EPSG_4326/nz-land-districts.shp")
suburbs_gdf = gpd.read_file("/home/arr65/data/nzgd/shapefiles/lds-nz-suburbs-and-localities-SHP_WGS84_EPSG_4326/nz-suburbs-and-localities.shp")

nzgd_index_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_25092024_1043.csv")
nzgd_index_df = nzgd_index_df[nzgd_index_df["Type"]=="CPT"]


#nzgd_index_df = nzgd_index_df.iloc[:10]

latitude_array = nzgd_index_df["Latitude"].values
longitude_array = nzgd_index_df["Longitude"].values

found_region = []
found_suburb = []

found_suburbs = gpd.GeoDataFrame()

for index, row in tqdm(nzgd_index_df.iterrows(), total=nzgd_index_df.shape[0]):

    point = gpd.GeoDataFrame([{'geometry': Point(row["Longitude"], row["Latitude"])}], crs="EPSG:4326")

    # Perform a spatial join to find the region
    district_result = gpd.sjoin(point, district_gdf, how="left", predicate="within")
    suburb_result = gpd.sjoin(point, suburbs_gdf, how="left", predicate="within")

    suburb_result.insert(0, "record_id", row["ID"])
    suburb_result.insert(1, "district", district_result.name)

    found_suburbs = pd.concat([found_suburbs,suburb_result], ignore_index=True)

    # Print the result


#final_result = pd.concat([nzgd_index_df, found_suburbs], axis=1)


found_suburbs.to_csv("/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_region_classification.csv", index=False)
