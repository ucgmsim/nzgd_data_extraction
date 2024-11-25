import rasterio
from rasterio.sample import sample_gen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from qcore import coordinates


nzgd_index_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_08112024_1017.csv")

data_latlon = nzgd_index_df[["Latitude", "Longitude"]].to_numpy()


##################################
### Calculate coordinates in NZTM

nzgd_nztm = coordinates.wgs_depth_to_nztm(data_latlon)
nzgd_index_df.loc[:, "nztm_y"] = nzgd_nztm[:, 0]
nzgd_index_df.loc[:, "nztm_x"] = nzgd_nztm[:, 1]

############################

#vs30_from_data = vs30_from_data.iloc[0:10]

# Path to the GeoTIFF file
geotiff_path = Path("/home/arr65/data/nzgd/resources/NSHM2022_NoG6G13")
file_name = "combined.tif"

xy_iterable = []
for i in range(nzgd_index_df.shape[0]):
    xy_iterable.append((nzgd_index_df["nztm_x"][i], nzgd_index_df["nztm_y"][i]))

# Open the GeoTIFF file
with rasterio.open(geotiff_path / file_name) as dataset:
    # Read the dataset's metadata
    metadata = dataset.meta
    print("Metadata:", metadata)

    # Get the number of bands
    num_bands = dataset.count
    print(f"Number of bands: {num_bands}")

    ## Print descriptions of each band
    for i in range(1, num_bands + 1):
        band_description = dataset.descriptions[i - 1]
        print(f"Band {i} description: {band_description}")

    ## Interpolate the value at the given coordinates
    progress_bar = tqdm(total=len(xy_iterable))
    interpolated_values = []
    for val in sample_gen(dataset, xy_iterable):
        interpolated_values.append(val)
        progress_bar.update(1)

vs30_vs30std = np.array(interpolated_values)

nzgd_index_df.loc[:, "vs30"] = vs30_vs30std[:, 0]
nzgd_index_df.loc[:, "vs30_std"] = vs30_vs30std[:, 1]

nzgd_index_df.to_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/foster_vs30_at_nzgd_locations.csv", index=False)
