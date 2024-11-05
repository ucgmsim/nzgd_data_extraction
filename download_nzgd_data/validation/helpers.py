"""
Functions to help with the validation of CPT data.
"""

from dataclasses import dataclass
import pandas as pd
from scipy import interpolate
import numpy as np
from pathlib import Path
import download_nzgd_data.validation.load_sql_db as load_sql_db
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt

def get_residual(record_name: str, old_data_ffp: Path, new_data_ffp: Path, make_plot=None) -> pd.DataFrame:

    old_df = pd.read_parquet(old_data_ffp / f"{record_name}.parquet")
    old_df = old_df.drop(columns=["record_name", "latitude", "longitude"])
    new_df = pd.read_parquet(new_data_ffp / f"{record_name}.parquet")

    if new_df.size == 0:
        return False

    new_df = new_df[new_df["multiple_measurements"] == 0]
    new_df = new_df.drop(columns=["multiple_measurements", "record_name", "latitude", "longitude"])

    interpolated_df = get_interpolated_df(old_df, new_df)

    #residual = np.log(interpolated_df) - np.log(old_df)
    residual = interpolated_df - old_df

    if make_plot:
        make_plot.mkdir(parents=True, exist_ok=True)
        plot_residual(residual, old_df, interpolated_df, new_df, record_name=record_name, plot_output_dir=make_plot)

    return residual, old_df, interpolated_df, new_df


def check_residual(record_name: str, old_data_ffp: Path, new_data_ffp: Path, max_allowed_resid_as_pc_of_mean:float, allowed_percent_not_close_to_zero: float) -> bool:

    residual, old_df, interpolated_df, new_df = get_residual(record_name = record_name, old_data_ffp = old_data_ffp, new_data_ffp=new_data_ffp)

    old_df_range = old_df.max()[["qc","fs","u"]] - old_df.min()[["qc","fs","u"]]

    resid_close_to_zero = residual.abs()[["qc","fs","u"]] <= (max_allowed_resid_as_pc_of_mean/100)*old_df_range

    percent_resid_close_to_zero = 100*resid_close_to_zero.sum()/resid_close_to_zero.shape[0]

    resid_close_to_zero_check = percent_resid_close_to_zero >= allowed_percent_not_close_to_zero

    if (resid_close_to_zero_check).all():
        return True

    else:
        return False


def plot_residual(residual, old_df, interpolated_df, new_df,record_name = None, plot_output_dir=None) -> None:
    """
    Plot the residuals of the old and new data.

    Parameters
    ----------
    old_df : pd.DataFrame
        The old DataFrame.
    new_df : pd.DataFrame
        The new DataFrame.
    """

    fig, axes = plt.subplots(2, 3, sharex=True)

    axes[0, 0].plot(new_df["Depth"],new_df["qc"], linestyle="--", marker="+", color="grey",label="new")
    axes[0, 0].plot(old_df["Depth"],old_df["qc"], linestyle="--", color="green", marker="o",
                    markersize=10, markerfacecolor='none', markeredgecolor='green', label="old")
    axes[0, 0].plot(interpolated_df["Depth"], interpolated_df["qc"], linestyle="--", marker="x", color="red",
                    label="interp")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Depth (m)")
    axes[0, 0].set_ylabel("qc (MPa)")

    axes[0, 1].plot(new_df["Depth"],new_df["fs"], linestyle="--", marker="+", color="grey",
                    label="new")
    axes[0, 1].plot(old_df["Depth"],old_df["fs"], linestyle="--", color="green", marker="o",
                    markersize=10, markerfacecolor='none', markeredgecolor='green', label="old")
    axes[0, 1].plot(interpolated_df["Depth"], interpolated_df["fs"], linestyle="--", marker="x", color="red",
                    label="interp")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Depth (m)")
    axes[0, 1].set_ylabel("fs (MPa)")

    axes[0, 2].plot(new_df["Depth"],new_df["u"], linestyle="--", marker="+", color="grey",
                    label="new")
    axes[0, 2].plot(old_df["Depth"],
                    old_df["u"], linestyle="--", color="green", marker="o",
                    markersize=10, markerfacecolor='none', markeredgecolor='green', label="old")
    axes[0, 2].plot(interpolated_df["Depth"], interpolated_df["u"], linestyle="--", marker="x", color="red",
                    label="interp")
    axes[0, 2].legend()
    axes[0, 2].set_xlabel("Depth (m)")
    axes[0, 2].set_ylabel("u (MPa)")

    ###################################################################

    axes[1, 0].plot(old_df["Depth"], residual["qc"], linestyle="--",marker="o",markersize=5)
    axes[1, 0].set_xlabel("Depth (m)")
    axes[1, 0].set_ylabel("qc resid")

    axes[1, 1].plot(old_df["Depth"], residual["fs"], linestyle="--",marker="o",markersize=5)
    axes[1, 1].set_xlabel("Depth (m)")
    axes[1, 1].set_ylabel("fs resid")

    axes[1, 2].plot(old_df["Depth"], residual["u"], linestyle="--",marker="o",markersize=5)
    axes[1, 2].set_xlabel("Depth (m)")
    axes[1, 2].set_ylabel("u resid")

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    if record_name == "CPT_23719":
        print()

    if record_name:
        plt.savefig(Path(plot_output_dir)/f"{record_name}.png", dpi=500)
    else:
        plt.savefig("/home/arr65/data/nzgd/plots/inconsistent_cpt_records/redisuals.png", dpi=500)






def convert_old_sql_to_parquet(original_old_data_dir: Path, old_data_as_parquet_dir: Path) -> None:

    """
    Convert the old SQL database to parquet files.

    Parameters
    ----------
    old_data_dir : Path
        The directory containing the old SQL database.
    new_data_dir : Path
        The directory where the new parquet files will be saved.
    session : sqlalchemy.orm.session.Session
        The session to the old SQL database.
    """

    engine = create_engine(f"sqlite:///{original_old_data_dir}/nz_cpt.db")
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    old_data_as_parquet_dir.mkdir(parents=True, exist_ok=True)

    cpt_locs = load_sql_db.cpt_locations(session)

    for cpt_loc in tqdm(cpt_locs):

        cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)

        if cpt_records.size == 0:
            continue

        elif cpt_records.shape[0] < 4:
            continue

        df = pd.DataFrame(cpt_records, columns=["Depth", "qc", "fs", "u"])

        df.to_parquet(old_data_as_parquet_dir / f"{cpt_loc.name}.parquet")

def get_list_of_sql_dfs(cpt_locs: list, session) -> list[pd.DataFrame]:

    dfs = []

    for cpt_loc in tqdm(cpt_locs):

        cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)

        if cpt_records.size == 0:
            continue

        elif cpt_records.shape[0] < 4:
            continue

        old_df = pd.DataFrame(cpt_records, columns=["Depth", "qc", "fs", "u"])

        dfs.append(old_df)

    return dfs


def make_df_list(cpt_locs:list, new_converted_data_dir:Path):

    dfs = []

    for row_n, cpt_loc in enumerate(cpt_locs):

        parquet_to_load = new_converted_data_dir / f"{cpt_loc.name}.parquet"

        if not parquet_to_load.exists():
            continue

        cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)

        if cpt_records.size == 0:
            continue

        elif cpt_records.shape[0] < 4:
            continue

        old_df = pd.DataFrame(cpt_records, columns=["Depth", "qc", "fs", "u"])

        dfs.append((old_df, parquet_to_load))

    return dfs










# def check_for_consistency(record_name: str, band_width_percent:float, allowed_percent_of_points_not_within_band: float) -> bool:
#
#     old_df = dfs[0]
#     new_df = dfs[1]
#
#     new_df_upper = new_df.copy()
#     new_df_lower = new_df.copy()
#
#     new_df_upper[["qc","fs","u"]] *= 1+(band_width_percent/100)
#     new_df_lower[["qc", "fs", "u"]] -= 1-(band_width_percent/100)
#
#     interpolated_new_df_upper = get_interpolated_df(organise_with_depth_range(old_df, new_df_upper))
#     interpolated_new_df_lower = get_interpolated_df(organise_with_depth_range(old_df, new_df_lower))
#
#     old_lower_than_new_upper = old_df[["qc","fs","u"]] < interpolated_new_df_upper[["qc","fs","u"]]
#     old_higher_than_new_lower = old_df[["qc","fs","u"]] > interpolated_new_df_lower[["qc","fs","u"]]
#
#     old_within_band = old_lower_than_new_upper & old_higher_than_new_lower
#     percent_not_within_band = 100*(len(old_within_band) - old_within_band.sum()) / len(old_within_band)
#
#     if (percent_not_within_band < allowed_percent_of_points_not_within_band).all():
#         return False
#     else:
#         return True






@dataclass
class OrganizedWithDepthRange:
    """
    Keeps track of which DataFrames have the largest and smallest depth ranges.
    """

    largest_depth_range: pd.DataFrame
    shortest_depth_range: pd.DataFrame


def organise_with_depth_range(df1: pd.DataFrame, df2: pd.DataFrame) -> OrganizedWithDepthRange:
    """
    Selects the DataFrame with the largest depth range.

    This function compares the depth ranges of two DataFrames and returns the one with the larger range.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame containing a 'Depth' column.
    df2 : pd.DataFrame
        DataFrame containing a 'Depth' column.

    Returns
    -------
    OrganizedWithDepthRange
        An instance of OrganizedWithDepthRange to indicate which Dataframes have the largest and shortest
        depth ranges.
    """

    d1_range = df1["Depth"].max() - df1["Depth"].min()
    d2_range = df2["Depth"].max() - df2["Depth"].min()

    if d1_range > d2_range:
        return OrganizedWithDepthRange(largest_depth_range=df1, shortest_depth_range=df2)
    else:
        return OrganizedWithDepthRange(largest_depth_range=df2, shortest_depth_range=df1)


def get_interpolated_df(old_df, new_df) -> pd.DataFrame:

    """
    Interpolates the DataFrame with the largest depth range onto the DataFrame with the smallest depth range so that
    every point in the smallest depth range has a corresponding point in the largest depth range.

    Parameters
    ----------
    old_df : pd.DataFrame
        The old CPT data from the SQL database.

    new_df : pd.DataFrame
        The new CPT data from the parquet files.

    Returns
    -------
    pd.DataFrame: The DataFrame with the largest depth range with interpolated onto the Depth values of the DataFrame
    with the smallest depth range.
    """

    qc_interp = interpolate.interp1d(new_df["Depth"], new_df["qc"], kind="linear", bounds_error=False, fill_value=np.nan)
    fs_interp = interpolate.interp1d(new_df["Depth"], new_df["fs"], kind="linear", bounds_error=False, fill_value=np.nan)
    u_interp = interpolate.interp1d(new_df["Depth"], new_df["u"], kind="linear", bounds_error=False, fill_value=np.nan)

    interpolated_df = old_df.copy()

    interpolated_df.loc[:,"qc"] = qc_interp(interpolated_df["Depth"])
    interpolated_df.loc[:,"fs"] = fs_interp(interpolated_df["Depth"])
    interpolated_df.loc[:,"u"] = u_interp(interpolated_df["Depth"])

    return interpolated_df




def OLD_get_interpolated_df(organised_dfs: OrganizedWithDepthRange) -> pd.DataFrame:

    """
    Interpolates the DataFrame with the largest depth range onto the DataFrame with the smallest depth range so that
    every point in the smallest depth range has a corresponding point in the largest depth range.

    Parameters
    ----------
    organised_dfs : OrganizedWithDepthRange
        An instance of OrganizedWithDepthRange containing the DataFrames.

    Returns
    -------
    pd.DataFrame: The DataFrame with the largest depth range with interpolated onto the Depth values of the DataFrame
    with the smallest depth range.
    """

    qc_interp = interpolate.interp1d(organised_dfs.largest_depth_range["Depth"], organised_dfs.largest_depth_range["qc"], kind="linear", bounds_error=False)
    fs_interp = interpolate.interp1d(organised_dfs.largest_depth_range["Depth"], organised_dfs.largest_depth_range["fs"], kind="linear", bounds_error=False)
    u_interp = interpolate.interp1d(organised_dfs.largest_depth_range["Depth"], organised_dfs.largest_depth_range["u"], kind="linear", bounds_error=False)

    interpolated_df = organised_dfs.shortest_depth_range.copy()

    interpolated_df.loc[:,"qc"] = qc_interp(interpolated_df["Depth"])
    interpolated_df.loc[:,"fs"] = fs_interp(interpolated_df["Depth"])
    interpolated_df.loc[:,"u"] = u_interp(interpolated_df["Depth"])

    return interpolated_df



def sigma_clip_indices(data: np.array, n_sigma: int) -> np.array:
    """
    Apply a 3-sigma clip to the residual data.

    Parameters
    ----------
    data : np.array
        The data to be clipped.

    Returns
    -------
    tuple

    """

    data = np.abs(data)

    keep_indices_bool_mask = data < np.median(data) + (n_sigma * np.std(data))


    return keep_indices_bool_mask
