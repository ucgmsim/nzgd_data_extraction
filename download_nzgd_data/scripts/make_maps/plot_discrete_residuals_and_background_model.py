from pathlib import Path
from pygmt_helper import plotting
import pandas as pd
import pygmt
import enum
import toml
import numpy as np
import matplotlib.pyplot as plt

from qcore import coordinates

class CPTCorrelation(enum.StrEnum):
    andrus_2007 = "andrus_2007"
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

#data_subset = DataSubset.only_old
#data_subset = DataSubset.only_new
#data_subset = DataSubset.new_and_old



metadata_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/metadata")
data_vs30_df = pd.read_csv(metadata_dir / "vs30_estimates_from_data.csv")

cpt_correlation = CPTCorrelation.andrus_2007
#vs30_correlation = Vs30Correlation.boore_2011
vs30_correlation = Vs30Correlation.boore_2004

record_names_in_old_dataset = pd.read_csv("/home/arr65/data/nzgd/resources/record_names_in_old_dataset.csv")["record_names_in_old_dataset"].to_list()


geotiff_file = Path("/home/arr65/data/nzgd/resources/vs30map_data_2023_geotiff/vs30map_data/combined.tif")
info = pygmt.grdinfo(grid=geotiff_file)
## Downsample the GeoTIFF using grdsample to a lower resolution for easier plotting
downsampled_grid = pygmt.grdsample(grid=geotiff_file, spacing=1000)

output_dir = metadata_dir / "residual_plots" / f"{cpt_correlation}_{vs30_correlation}"
output_dir.mkdir(exist_ok=True, parents=True)

data_vs30_df = data_vs30_df[data_vs30_df["exception"].isna()]

data_vs30_df = data_vs30_df[(data_vs30_df["cpt_vs_correlation"] == cpt_correlation) &
                            (data_vs30_df["vs30_correlation"] == vs30_correlation)]

residuals_from_variations = []
for data_subset in [DataSubset.only_old, DataSubset.new_and_old,  DataSubset.only_new]:
#for data_subset in [DataSubset.only_old]:

    if data_subset == DataSubset.only_old:
        data_vs30_for_dataset_df = data_vs30_df[data_vs30_df["record_name"].isin(record_names_in_old_dataset)]
    elif data_subset == DataSubset.only_new:
        data_vs30_for_dataset_df = data_vs30_df[~data_vs30_df["record_name"].isin(record_names_in_old_dataset)]
    else:
        data_vs30_for_dataset_df = data_vs30_df

    model_vs30_df = pd.read_csv(metadata_dir / "matched_vs30_from_model.csv")

    vs30_from_data_and_model_df = data_vs30_for_dataset_df.merge(model_vs30_df,left_on = "record_name", right_on="record_name_from_data", how="left")

    vs30_from_data_and_model_df.loc[:,"ln_data_vs30_minus_ln_model_vs30"] = np.log(vs30_from_data_and_model_df["vs30"]) - np.log(vs30_from_data_and_model_df["model_vs30"])

    ##################################
    ### Calculate coordinates in NZTM

    vs30_latlon = vs30_from_data_and_model_df[["latitude","longitude"]].to_numpy()

    vs30_nztm = coordinates.wgs_depth_to_nztm(vs30_latlon)
    vs30_from_data_and_model_df.loc[:,"nztm_y"] = vs30_nztm[:, 0]
    vs30_from_data_and_model_df.loc[:,"nztm_x"] = vs30_nztm[:, 1]

    ############################



    ## drop rows with NaN values in the residuals
    vs30_from_data_and_model_df = vs30_from_data_and_model_df.dropna(subset=["ln_data_vs30_minus_ln_model_vs30"])

    exclude_highest_and_lowest_percentile = 1

    resid_colorbar_min = np.percentile(vs30_from_data_and_model_df["ln_data_vs30_minus_ln_model_vs30"], exclude_highest_and_lowest_percentile)
    resid_colorbar_max = np.percentile(vs30_from_data_and_model_df["ln_data_vs30_minus_ln_model_vs30"], 100-exclude_highest_and_lowest_percentile)

    ## Make a histogram of ln_residual to inform the limits of the colorbar
    counts, bins, patches = plt.hist(vs30_from_data_and_model_df["ln_data_vs30_minus_ln_model_vs30"], bins=100)
    residuals_from_variations.append(vs30_from_data_and_model_df["ln_data_vs30_minus_ln_model_vs30"].values)

    plt.xlabel("log residual")
    plt.ylabel("count")
    plt.title("log residuals in Vs30 between Foster model\n"
                f"and data prediction using {cpt_correlation} and {vs30_correlation}\n"
                f"for {len(vs30_from_data_and_model_df)} records. "
                f"Median = {np.median(vs30_from_data_and_model_df['ln_data_vs30_minus_ln_model_vs30']):.3f}.",
              fontsize=10)

    plt.vlines([resid_colorbar_min, resid_colorbar_max],0,np.max(counts),colors="red",linestyles="dashed",
    label=f"colorbar limits on map\n(excluding highest and lowest {exclude_highest_and_lowest_percentile}% of values)")
    plt.legend(fontsize=8)

    plt.savefig(output_dir / f"residual_hist_{data_subset}_dataset.png",dpi=500)
    plt.close()

    # Plot the GeoTIFF in the background
    fig = pygmt.Figure()

    # viridis, cubhelix
    fig.grdimage(
        grid=downsampled_grid,          # Path to the GeoTIFF file
        cmap="viridis",             # Color map for the data
        frame=True,                  # Add frame to the map
    )
    fig.colorbar(frame='af+lBackground Vs30 model (m/s)')  # Label the color bar as needed
    ## haxby, buda,
    pygmt.makecpt(cmap="plasma", series=[resid_colorbar_min, resid_colorbar_max])

    fig.plot(
        x=vs30_from_data_and_model_df["nztm_x"],
        y=vs30_from_data_and_model_df["nztm_y"],
        fill=vs30_from_data_and_model_df["ln_data_vs30_minus_ln_model_vs30"],
        cmap=True,
        style="c0.08c",
        pen="black")

    map_text1 = f"Showing residuals for {len(vs30_from_data_and_model_df)} records."
    map_text2 = f"Using {cpt_correlation} and {vs30_correlation}"
    map_text3 = "to estimate Vs30 from data."
    map_text4 = f"Median log residual = {np.median(vs30_from_data_and_model_df['ln_data_vs30_minus_ln_model_vs30']):.3f}."

    def plot_text_line_on_pygmt(y_position, text):

        fig.text(
            text=text,
            x=1.1e6,  # X position
            y=y_position,  # Y position
            font="8p,Helvetica-Bold,white",  # Font size, type, and color
            justify="TL",  # Text alignment (top left)
        )

    plot_text_line_on_pygmt(6.1e6, map_text1)
    plot_text_line_on_pygmt(6.05e6, map_text2)
    plot_text_line_on_pygmt(6.0e6, map_text3)
    plot_text_line_on_pygmt(5.95e6, map_text4)
    fig.colorbar(frame="a0.1+lln(Data Vs30) - ln(Model Vs30) (discrete points)",position="JML+o2c/0c")
    fig.savefig(output_dir / f"residual_map_{data_subset}_dataset.png", dpi=500)
    plt.close()

plt.figure()
plt.hist(residuals_from_variations, bins=100, label=["old data", "new and old data", "new data"],histtype='step', stacked=False, fill=False)
plt.legend()
plt.xlabel("log residual")
plt.ylabel("count")
plt.savefig(output_dir / "combined_hist.png", dpi=500)
plt.close()

