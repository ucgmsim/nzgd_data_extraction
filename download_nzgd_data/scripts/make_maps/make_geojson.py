from shapely.geometry import Point
import geopandas as gpd

import pandas as pd
from pathlib import Path
from tqdm import tqdm

import time


metadata_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/metadata")
resources_dir = Path("/home/arr65/data/nzgd/resources")
vs30_from_data_df = pd.read_csv(metadata_dir / "vs30_estimates_from_data.csv")


#vs30_from_data_df = vs30_from_data_df.loc[vs30_from_data_df["vs30_correlation"] == "boore_2011"]
vs30_from_data_df = vs30_from_data_df.loc[vs30_from_data_df["vs30_correlation"] == "boore_2004"]

### Only keep rows that do not have an exception
vs30_from_data_df = vs30_from_data_df[vs30_from_data_df["exception"].isna()]

vs30_from_data_df.drop_duplicates(subset=["record_name"],keep='first',inplace=True)

record_names = set(vs30_from_data_df["record_name"].to_list())

# Create a list of Point objects
geometry = [Point(xy) for xy in zip(vs30_from_data_df['longitude'], vs30_from_data_df['latitude'])]

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(vs30_from_data_df, geometry=geometry, crs="EPSG:4326")

hypo_base_dir = Path("/home/arr65/data/nzgd/hypocentre_mirror/nzgd")

raw_data_dir = hypo_base_dir / "raw_from_nzgd"

raw_cpt_data_dir = raw_data_dir / "cpt"

cpt_files = list(raw_cpt_data_dir.rglob("*.pdf"))

#cpt_files = cpt_files[0:1000]


corresponding_pdf_url = []

preceding_url = "https://quakecoresoft.canterbury.ac.nz"

# for cpt_file in tqdm(cpt_files):

gdf = gdf.iloc[0:10]

for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):

    for cpt_file in cpt_files:

        if cpt_file.parent.name == row["record_name"]:

            url_str = f"{preceding_url}/{cpt_file.relative_to(hypo_base_dir)}"

            corresponding_pdf_url.append(url_str)

            break

gdf.loc[:, "pdf_url"] = corresponding_pdf_url

gdf.to_file(resources_dir / "cpt_test_url.geojson", driver="GeoJSON")

print()



