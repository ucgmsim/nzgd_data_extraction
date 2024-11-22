import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import enum
from datetime import date



class CPTCorrelation(enum.StrEnum):
    andrus_2007_holocene = "andrus_2007_holocene"
    andrus_2007_pleistocene = "andrus_2007_pleistocene"
    andrus_2007_tertiary_age_cooper_marl = "andrus_2007_tertiary_age_cooper_marl"
    robertson_2009 = "robertson_2009"
    hegazy_2006 = "hegazy_2006"
    mcgann_2015 = "mcgann_2015"
    mcgann_2018 = "mcgann_2018"

class Vs30Correlation(enum.StrEnum):
    boore_2011 = "boore_2011"
    boore_2004 = "boore_2004"

class DataSubset(enum.StrEnum):
    new_and_old = "new_and_old"
    only_old = "only_old"
    only_new = "only_new"

cpt_correlation = CPTCorrelation.andrus_2007_pleistocene
vs30_correlation = Vs30Correlation.boore_2004
data_subset = DataSubset.new_and_old
min_acceptable_max_depth_m = 20

metadata_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/metadata")
output_dir = metadata_dir / "residual_plots" / str(date.today()) / f"{cpt_correlation}_{vs30_correlation}_{data_subset}_data_min_max_depth_{min_acceptable_max_depth_m}m"
output_dir.mkdir(parents=True, exist_ok=True)

vs30_from_data = pd.read_csv(metadata_dir / "vs30_estimates_from_data_andrus.csv")
vs30_from_data.rename(columns={"vs30": "vs30_from_data"}, inplace=True)
vs30_from_data.rename(columns={"vs30_sd": "vs30_std_from_data"}, inplace=True)

vs30_from_model = pd.read_csv(metadata_dir / "foster_vs30_at_nzgd_locations.csv")
vs30_from_model.rename(columns={"vs30": "vs30_from_model"}, inplace=True)
vs30_from_model.rename(columns={"vs30_std": "vs30_std_from_model"}, inplace=True)


vs30_from_data2 = vs30_from_data[vs30_from_data["cpt_vs_correlation"] == cpt_correlation]
print()
vs30_from_data = vs30_from_data[vs30_from_data["max_depth_m"] > min_acceptable_max_depth_m]
vs30_from_data = vs30_from_data.dropna(subset=["vs30_from_data", "vs30_std_from_data"])
print()
## Check if all rows in column record_name of vs30_from_data are unique
if vs30_from_data["record_name"].nunique() != vs30_from_data.shape[0]:
    raise ValueError("Some record_names in vs30_from_data are not unique. Please filter out duplicate record_names"
                     "and try again.")

# Merge the dataframes on the matching columns
vs30_df = pd.merge(vs30_from_data, vs30_from_model, how="inner", left_on="record_name", right_on="ID")
vs30_df.loc[:,"ln_cpt_vs30_minus_ln_foster_vs30"] = np.log(vs30_df["vs30_from_data"]) - np.log(vs30_df["vs30_from_model"])



#####################################################################################

vs30_df = vs30_df.dropna(subset=["ln_cpt_vs30_minus_ln_foster_vs30"])

exclude_highest_and_lowest_percentile = 1

## Make a histogram of Vs30 values
counts, bins, patches = plt.hist(vs30_df[["vs30_from_data", "vs30_from_model"]], bins=250,
                                 label=["CPT-inferred Vs30", "Foster model Vs30"], histtype='step', stacked=False)
plt.xlim(100,750)
plt.legend()

plt.xlabel("Vs30 (m/s)")
plt.ylabel("count")
plt.title("Vs30 values from CPT-inferred and Foster model at corresponding locations", fontsize=10)

plt.savefig(output_dir / f"Vs30_values_{data_subset}_dataset.png", dpi=500)
plt.close()

#######################################################################################
########################################################################################



resid_colorbar_min = np.percentile(vs30_df["ln_cpt_vs30_minus_ln_foster_vs30"],
                                   exclude_highest_and_lowest_percentile)
resid_colorbar_max = np.percentile(vs30_df["ln_cpt_vs30_minus_ln_foster_vs30"],
                                   100 - exclude_highest_and_lowest_percentile)

## Make a histogram of ln_residual to inform the limits of the colorbar
counts, bins, patches = plt.hist(vs30_df["ln_cpt_vs30_minus_ln_foster_vs30"], bins=100)

plt.xlabel("log residual")
plt.ylabel("count")
plt.title("log residuals in Vs30 between Foster model\n"
          f"and CPT-inference using {cpt_correlation} and {vs30_correlation}\n"
          f"for {len(vs30_df)} records. "
          f"Median = {np.median(vs30_df['ln_cpt_vs30_minus_ln_foster_vs30']):.3f}.",
          fontsize=10)

plt.vlines([resid_colorbar_min, resid_colorbar_max], 0, np.max(counts), colors="red", linestyles="dashed",
           label=f"colorbar limits on map\n(excluding highest and lowest {exclude_highest_and_lowest_percentile}% of values)")
plt.legend(fontsize=8)

plt.savefig(output_dir / f"residual_hist_{data_subset}_dataset.png", dpi=500)
plt.close()

######################################################################################

geotiff_path = Path("/home/arr65/data/nzgd/resources/NSHM2022_NoG6G13")
file_name = "combined.tif"
with rasterio.open(geotiff_path / file_name) as dataset:
    band1 = dataset.read(1)
    extent = dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top

# Create a custom colormap
custom_cmap = plt.get_cmap('viridis_r').copy()
custom_cmap.set_bad(color='gray')  # Set the color for NaN values

# 10, 15
fig, ax = plt.subplots(figsize=(12,10))
# Hide all tick labels
ax.set_xticklabels([])
ax.set_yticklabels([])

cax = ax.imshow(band1, cmap=custom_cmap, extent=extent,
                vmin=100, vmax=800)
fig.colorbar(cax, ax=ax, label=r'Foster et al. (2019) Vs$_{30}$ (ms$^{-1}$)')

# seismic
cax2 = ax.scatter(
    vs30_df["nztm_x"],
    vs30_df["nztm_y"],
    c=vs30_df["ln_cpt_vs30_minus_ln_foster_vs30"],
    s=10,
    marker='o',
    cmap="seismic_r",
    vmin=resid_colorbar_min,
    vmax=resid_colorbar_max,
    edgecolor='black',  # Add black edge around markers
    linewidth = 0.5  # Reduce the thickness of the black edge

)
cbar2 = fig.colorbar(cax2, ax=ax, location='left', label=r'log(residual) [$\ln(CPT_{vs30} - \ln(\mathrm{Foster}_{vs30})$] (discrete points)')

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

fig.savefig(output_dir / f"residual_map_{data_subset}_dataset.png", dpi=500)
plt.close()

print()
#





#vs30_df.to_csv(metadata_dir / "matched_vs30_from_model.csv", index=False)

## Match vs30_from_data dataframe column record_name with vs30_from_model dataframe column ID and keep only rows of vs30_from_model that have a match in vs30_from_data






print()
#
#
#     # Plot the map
#     fig, ax = plt.subplots(figsize=(10, 10))
#     cax = ax.imshow(band1, cmap='viridis', extent=(dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top))
#     fig.colorbar(cax, ax=ax, label='Band 1 Values')
#
#     # Plot the points
#     ax.scatter(vs30_from_data["nztm_x"], vs30_from_data["nztm_y"], color='red', marker='o', label='Interpolated Points')
#     ax.legend()
#
#     plt.xlabel('NZTM X')
#     plt.ylabel('NZTM Y')
#     plt.title('Map with Interpolated Points')
#
#     subset_xrange = np.max(vs30_from_data["nztm_x"]) - np.min(vs30_from_data["nztm_x"])
#     subset_yrange = np.max(vs30_from_data["nztm_y"]) - np.min(vs30_from_data["nztm_y"])
#
#     range_scaling_factor = 0.5
#
#     dx = range_scaling_factor * subset_xrange
#     dy = range_scaling_factor * subset_yrange
#
#
#     plt.xlim([np.min(vs30_from_data["nztm_x"])-dx, np.max(vs30_from_data["nztm_x"])+dx])
#     plt.ylim([np.min(vs30_from_data["nztm_y"])-dy, np.max(vs30_from_data["nztm_y"])+dy])
#
#     plt.show()
#
#     print()
#
# # Create a plt
#
#
#     # Print descriptions of each band
#     for i in range(1, num_bands + 1):
#         band_description = dataset.descriptions[i - 1]
#         print(f"Band {i} description: {band_description}")
#
#
#     # Read the first band of the dataset
#     band1 = dataset.read(1)
#     print("Band 1 data:", band1)
#
#     # Interpolate the value at the given coordinates
#     for val in sample_gen(dataset, [(x, y)]):
#         print(f"Interpolated value at ({x}, {y}): {val}")
#
#
#
# print()
# band_number = 1
# # Interpolate the value at the given coordinates for the specified band
# for val in sample_gen(dataset, [(x, y)], indexes=band_number):
#     print(f"Interpolated value at ({x}, {y}) for band {band_number}: {val}")