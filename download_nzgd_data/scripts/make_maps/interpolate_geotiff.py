import rasterio
from rasterio.sample import sample_gen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from qcore import coordinates


vs30_from_data = pd.read_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/vs30_estimates_from_data.csv")

data_latlon = vs30_from_data[["latitude", "longitude"]].to_numpy()


##################################
### Calculate coordinates in NZTM

vs30_latlon = vs30_from_data[["latitude", "longitude"]].to_numpy()

vs30_nztm = coordinates.wgs_depth_to_nztm(vs30_latlon)
vs30_from_data.loc[:, "nztm_y"] = vs30_nztm[:, 0]
vs30_from_data.loc[:, "nztm_x"] = vs30_nztm[:, 1]

############################

#vs30_from_data = vs30_from_data.iloc[0:10]

# Path to the GeoTIFF file
geotiff_path = Path("/home/arr65/data/nzgd/resources/NSHM2022_NoG6G13")
file_name = "combined.tif"

xy_iterable = []
for i in range(vs30_from_data.shape[0]):
    xy_iterable.append((vs30_from_data["nztm_x"][i], vs30_from_data["nztm_y"][i]))

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

vs30_from_data.loc[:, "vs30"] = vs30_vs30std[:, 0]
vs30_from_data.loc[:, "vs30_std"] = vs30_vs30std[:, 1]

vs30_from_model = vs30_from_data[["record_name", "nztm_x", "nztm_y", "vs30", "vs30_std"]]
vs30_from_model.to_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/vs30_from_model.csv", index=False)

print()

        # print()
        # print(f"Interpolated value at ({x}, {y}): {val}")

    # interp_vs30 = np.array(list(sample_gen(dataset, vs30_from_data[["nztm_x", "nztm_y"]].to_numpy(), indexes=1))).flatten()

    # band1 = dataset.read(1)
    # # Plot the map
    # fig, ax = plt.subplots(figsize=(10, 10))
    # cax = ax.imshow(band1, cmap='viridis', extent=(dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top))
    # fig.colorbar(cax, ax=ax, label='Band 1 Values')
    #
    # # Plot the points
    # ax.scatter(vs30_from_data["nztm_x"], vs30_from_data["nztm_y"], color='red', marker='o', label='Interpolated Points')
    # ax.legend()
    #
    # plt.xlabel('NZTM X')
    # plt.ylabel('NZTM Y')
    # plt.title('Map with Interpolated Points')
    #
    # subset_xrange = np.max(vs30_from_data["nztm_x"]) - np.min(vs30_from_data["nztm_x"])
    # subset_yrange = np.max(vs30_from_data["nztm_y"]) - np.min(vs30_from_data["nztm_y"])
    #
    # range_scaling_factor = 0.5
    #
    # dx = range_scaling_factor * subset_xrange
    # dy = range_scaling_factor * subset_yrange
    #
    #
    # plt.xlim([np.min(vs30_from_data["nztm_x"])-dx, np.max(vs30_from_data["nztm_x"])+dx])
    # plt.ylim([np.min(vs30_from_data["nztm_y"])-dy, np.max(vs30_from_data["nztm_y"])+dy])
    #
    # plt.show()
    #
    # print()

## Create a plt


    # # Print descriptions of each band
    # for i in range(1, num_bands + 1):
    #     band_description = dataset.descriptions[i - 1]
    #     print(f"Band {i} description: {band_description}")


    # # Read the first band of the dataset
    # band1 = dataset.read(1)
    # print("Band 1 data:", band1)

    # # Interpolate the value at the given coordinates
    # for val in sample_gen(dataset, [(x, y)]):
    #     print(f"Interpolated value at ({x}, {y}): {val}")



# print()
# band_number = 1
# # Interpolate the value at the given coordinates for the specified band
# for val in sample_gen(dataset, [(x, y)], indexes=band_number):
#     print(f"Interpolated value at ({x}, {y}) for band {band_number}: {val}")