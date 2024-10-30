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


def check_for_consistency(dfs: tuple[pd.DataFrame, pd.DataFrame], band_width_percent:float , allowed_percent_of_points_not_within_band: float) -> bool:

    old_df = dfs[0]
    new_df = dfs[1]

    new_df_upper = new_df.copy()
    new_df_lower = new_df.copy()

    new_df_upper[["qc","fs","u"]] *= 1+(band_width_percent/100)
    new_df_lower[["qc", "fs", "u"]] -= 1-(band_width_percent/100)

    interpolated_new_df_upper = get_interpolated_df(organise_with_depth_range(old_df, new_df_upper))
    interpolated_new_df_lower = get_interpolated_df(organise_with_depth_range(old_df, new_df_lower))

    old_lower_than_new_upper = old_df[["qc","fs","u"]] < interpolated_new_df_upper[["qc","fs","u"]]
    old_higher_than_new_lower = old_df[["qc","fs","u"]] > interpolated_new_df_lower[["qc","fs","u"]]

    old_within_band = old_lower_than_new_upper & old_higher_than_new_lower
    percent_not_within_band = 100*(len(old_within_band) - old_within_band.sum()) / len(old_within_band)

    if (percent_not_within_band < allowed_percent_of_points_not_within_band).all():
        return False
    else:
        return True






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

def get_interpolated_df(organised_dfs: OrganizedWithDepthRange) -> pd.DataFrame:

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
