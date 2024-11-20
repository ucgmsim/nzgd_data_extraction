import rasterio
from rasterio.sample import sample_gen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

y = 5.181262e+06
x = 1.576467e+06


# Path to the GeoTIFF file
geotiff_path = '/home/arr65/data/nzgd/resources/vs30map_data_2023_geotiff/vs30map_data/combined.tif'

# Open the GeoTIFF file
with rasterio.open(geotiff_path) as dataset:
    # Read the dataset's metadata
    metadata = dataset.meta
    print("Metadata:", metadata)

    # Get the number of bands
    num_bands = dataset.count
    print(f"Number of bands: {num_bands}")

    interp_vs30 = np.array(list(sample_gen(dataset, vs30_from_data[["nztm_x", "nztm_y"]].to_numpy(), indexes=1))).flatten()

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