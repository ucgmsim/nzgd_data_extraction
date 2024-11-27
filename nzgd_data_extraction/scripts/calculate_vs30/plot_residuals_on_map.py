import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import enum
from datetime import date


output_dir = Path("/home/arr65/data/nzgd/vs30_plots")
output_file_name = "vs30_residuals.png"






output_dir = metadata_dir / "residual_plots" / str(date.today()) / f"{cpt_correlation}_{vs30_correlation}_{data_subset}_data_min_max_depth_{min_acceptable_max_depth_m}m"
output_dir.mkdir(parents=True, exist_ok=True)

# vs30_from_model = pd.read_csv(metadata_dir / "foster_vs30_at_nzgd_locations.csv")
# vs30_from_model.rename(columns={"vs30": "vs30_from_model"}, inplace=True)
# vs30_from_model.rename(columns={"vs30_std": "vs30_std_from_model"}, inplace=True)

vs30_from_model = pd.read_csv(metadata_dir / "vs30_from_Foster_geotiff_and_sung_resampled_txt.csv")
vs30_from_model.rename(columns={"model_vs30_from_closest_point": "vs30_from_model"}, inplace=True)
vs30_from_model.rename(columns={"vs30_std": "vs30_std_from_model"}, inplace=True)

vs30_from_data = pd.read_csv(metadata_dir / "vs30_estimates_from_cpt.csv")
vs30_from_data.rename(columns={"vs30": "vs30_from_data"}, inplace=True)
vs30_from_data.rename(columns={"vs30_sd": "vs30_std_from_data"}, inplace=True)

vs30_from_data = vs30_from_data[vs30_from_data["cpt_vs_correlation"] == cpt_correlation]
vs30_from_data = vs30_from_data[vs30_from_data["vs30_correlation"] == vs30_correlation]
vs30_from_data = vs30_from_data.dropna(subset=["vs30_from_data"])

# print()
# if data_subset == DataSubset.only_old:
#     vs30_from_data = vs30_from_data[vs30_from_data["record_name"].isin(record_names_in_old_dataset)]
# elif data_subset == DataSubset.only_new:
#     vs30_from_data = vs30_from_data[~vs30_from_data["record_name"].isin(record_names_in_old_dataset)]

vs30_from_data = vs30_from_data[vs30_from_data["max_depth_m"] < 100]
vs30_from_data_all_max_depths = vs30_from_data.copy()

## Filter out records with a maximum depth less than min_acceptable_max_depth_m
vs30_from_data = vs30_from_data[vs30_from_data["max_depth_m"] > min_acceptable_max_depth_m]

# Merge the dataframes on the matching columns
vs30_df = pd.merge(vs30_from_data, vs30_from_model, how="inner", left_on="record_name", right_on="ID")
vs30_df.loc[:,"ln_cpt_vs30_minus_ln_foster_vs30"] = np.log(vs30_df["vs30_from_data"]) - np.log(vs30_df["vs30_from_model"])
vs30_df = vs30_df.dropna(subset=["ln_cpt_vs30_minus_ln_foster_vs30"])

vs30_df_only_new = vs30_df[~vs30_df["record_name"].isin(record_names_in_old_dataset)].copy()
vs30_df_only_old = vs30_df[vs30_df["record_name"].isin(record_names_in_old_dataset)].copy()
vs30_df_new_and_old = vs30_df.copy()

max_depths_arr = np.linspace(10,50,100)
num_surviving_only_new = get_num_surviving(max_depths_arr, vs30_df_only_new)
num_surviving_only_old = get_num_surviving(max_depths_arr, vs30_df_only_old)
num_surviving_new_and_old = get_num_surviving(max_depths_arr, vs30_df_new_and_old)

## Check if all rows in column record_name of vs30_from_data are unique
if vs30_df_new_and_old["record_name"].nunique() != vs30_df_new_and_old.shape[0]:
    raise ValueError("Some record_names in vs30_from_data are not unique. Please filter out duplicate record_names"
                     "and try again.")

hist_data = [vs30_df_only_new["ln_cpt_vs30_minus_ln_foster_vs30"].values,
                vs30_df_only_old["ln_cpt_vs30_minus_ln_foster_vs30"].values,
                vs30_df_new_and_old["ln_cpt_vs30_minus_ln_foster_vs30"].values]

hist_labels = [f"Only new records\nnum points={len(hist_data[0])},median={np.median(hist_data[0]):.2f}",
               f"Only old records\nnum points={len(hist_data[1])}, median={np.median(hist_data[1]):.2f}",
               f"New and old records\nnum points={len(hist_data[2])}, median={np.median(hist_data[2]):.2f}"]

##############################################################################################3

plt.hist(hist_data, bins=30, label=hist_labels, histtype='step',
         stacked=False, fill=False)
plt.legend(fontsize=8)
plt.xlabel("log residual")
plt.ylabel("count")
plt.xlim(-2,2)

plt.savefig(output_dir / f"hist_all_data_subsets.png", dpi=500)
plt.close()

# make a histogram of the maximum depth values
plt.hist(vs30_from_data["max_depth_m"], bins=100)
plt.xlabel("Maximum depth (m)")
plt.ylabel("count")
plt.title("Histogram of maximum depth values in the CPT dataset")
plt.savefig(output_dir / f"max_depth_histogram_{data_subset}_dataset.png", dpi=500)
plt.close()

### Plot number of records surviving vs minimum required maximum depth
plt.semilogy(max_depths_arr, num_surviving_only_new,linestyle='-', marker='.',label="Only new records")
plt.semilogy(max_depths_arr, num_surviving_only_old,linestyle='-', marker='.',label="Only old records")
plt.semilogy(max_depths_arr, num_surviving_new_and_old,linestyle='-', marker='.',label="New and old records")
plt.xlabel("Minimum required maximum depth (m)")
plt.ylabel("Number of records surviving")
# put grid lines on the plot
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig(output_dir / f"log_plot_num_surviving_vs_max_depth_{data_subset}_dataset.png", dpi=500)
plt.close()

plt.plot(max_depths_arr, num_surviving_only_new,linestyle='-', marker='.',label="Only new records")
plt.plot(max_depths_arr, num_surviving_only_old,linestyle='-', marker='.',label="Only old records")
plt.plot(max_depths_arr, num_surviving_new_and_old,linestyle='-', marker='.',label="New and old records")
plt.xlabel("Minimum required maximum depth (m)")
plt.ylabel("Number of records surviving")
# put grid lines on the plot
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig(output_dir / f"num_surviving_vs_max_depth_{data_subset}_dataset.png", dpi=500)
plt.close()

######################################################################################################

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

exclude_highest_and_lowest_percentile = 1
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

map_text = f"""Log residuals for {data_subset} data with 
a minimum acceptable maximum depth of {min_acceptable_max_depth_m} m
(total of {len(vs30_df)} records).
CPT-inferred Vs30 values based on 
{cpt_correlation} and {vs30_correlation}.
Median log residual = {np.median(vs30_df['ln_cpt_vs30_minus_ln_foster_vs30']):.3f}."""

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
fig.colorbar(cax, ax=ax, label=r'Foster et al. (2019) Vs$_{30}$ (m/s) (background map)')

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

ax.text(1.1e6, 6e6, map_text, fontsize=10, color='white')


plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

fig.savefig(output_dir / f"residual_map_{data_subset}_dataset.png", dpi=500)
plt.close()

