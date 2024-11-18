"""
Functions for getting information required by the map on the Hypocentre NZGD HTTP server.
"""

from pathlib import Path
from collections import defaultdict, namedtuple
from tqdm import tqdm
import pandas as pd
import functools
from typing import Optional
import multiprocessing

from qcore import coordinates, geo

MetaData = namedtuple("MetaData", ["max_depth", "min_depth"])

def get_files_with_relative_paths(
    processed_files: bool, file_root_directory: Path, relative_to: Path, max_num_records = None
) -> dict[str, list[Path]]:
    """
    Get all files in a directory and its subdirectories and match them to record IDs.

    Parameters
    ----------
    processed_files : bool
        True if getting processed files, False if getting raw files.
    file_root_directory : Path
        The root directory containing the files.
    relative_to : Path
        The directory to which the file paths should be relative.

    Returns
    -------
    dict
        A dictionary with record IDs as keys and lists of file paths as values.
    """
    # Recursively get all files
    print("Recursively getting all files")
    all_files = [
        file for file in tqdm(list(file_root_directory.rglob("*"))) if file.is_file()
    ]

    if max_num_records:
        all_files = all_files[:max_num_records]

    print("Matching all files to record IDs")
    record_id_to_files = defaultdict(list)
    for file in tqdm(all_files):
        record_id = file.stem if processed_files else file.parent.name
        if relative_to:
            record_id_to_files[record_id].append(file.relative_to(relative_to))
        else:
            record_id_to_files[record_id].append(file)

    return record_id_to_files


def get_processed_metadata(file_root_directory: Path, max_num_records = None) -> dict[str, MetaData]:
    """
    Get metadata for processed files in a directory and its subdirectories.

    Parameters
    ----------
    file_root_directory : Path
        The root directory containing the processed files.

    Returns
    -------
    dict
        A dictionary with record IDs as keys and MetaData named tuples as values.
    """

    record_id_to_metadata = {}

    # Recursively get all files
    print("Recursively getting all files")
    all_files = [
        file for file in tqdm(list(file_root_directory.rglob("*"))) if file.is_file()
    ]

    if max_num_records:
        all_files = all_files[:max_num_records]

    for file in tqdm(all_files):
        record_df = pd.read_parquet(file)
        record_id_to_metadata[file.stem] = MetaData(
            max_depth=record_df.attrs["max_depth"],
            min_depth=record_df.attrs["min_depth"],
        )

    return record_id_to_metadata


def calc_dist_to_closest_cpt(long_lat_to_consider_df: pd.DataFrame, all_long_lat_df: pd.DataFrame) -> pd.DataFrame:

    """
    Calculates the distance between a single CPT and its closest neighbour.

    Parameters
    ----------
    long_lat_to_consider_df : pd.Series
        A Pandas Series containing the record name, longitude, and latitude of a single CPT.

    all_long_lat_df : pd.DataFrame
        A DataFrame containing the record name, longitude, and latitude of all CPTs from which to find the
        closest neighbour.

    Returns
    -------
    pd.DataFrame
           A DataFrame with the following columns:
            - record_name: the name of the record
            - distance_to_closest_cpt_km: the distance to the closest CPT in km
            - closest_cpt_name: the name of the closest CPT
            - lon: the longitude of the CPT
            - lat: the latitude of the CPT
            - closest_cpt_lon: the longitude of the closest CPT
            - closest_cpt_lat: the latitude of the closest CPT
    """

    all_cpt_names = all_long_lat_df["model_grid_point_name"].values
    # If the CPT to consider is among the CPTs from which to find the closest neighbour, filter it out
    all_bool_mask = all_cpt_names != long_lat_to_consider_df["record_name"]

    needed_rows_long_lat_df = all_long_lat_df[all_bool_mask]

    nd_array_lon_lat = needed_rows_long_lat_df[["model_longitude","model_latitude"]].to_numpy()
    idx, d = geo.closest_location(
        locations=nd_array_lon_lat, lon=long_lat_to_consider_df["longitude"], lat=long_lat_to_consider_df["latitude"]
    )

    closest_dist_df = pd.DataFrame(
        {
            "record_name_from_data": long_lat_to_consider_df["record_name"],
            "distance_to_closest_model_grid_point_km": d,
            "closest_model_grid_point_name": str(all_cpt_names[all_bool_mask][idx]),
            "longitude_from_data": long_lat_to_consider_df["longitude"],
            "latitude_from_data": long_lat_to_consider_df["latitude"],
            "closest_model_grid_point_longitude": nd_array_lon_lat[idx, 0],
            "closest_model_grid_point_latitude": nd_array_lon_lat[idx, 1],
        }, index = [0]
    )

    return closest_dist_df


def calc_all_closest_cpt_dist(
    lon_lat_to_consider_df: pd.DataFrame, all_lon_lat_df: pd.DataFrame, n_procs: Optional[int] = 1
) -> pd.DataFrame:
    """
    For each record in `lon_lat_to_consider_df`, finds the closest record in `all_lon_lat_df`.

    Parameters
    ----------
    lon_lat_to_consider_df : pd.DataFrame
        DataFrame containing records for which the closest record in `all_lon_lat_df` should be found.
        Has columns record_name, longitude, and latitude.

    all_lon_lat_df : pd.DataFrame
        DataFrame containing the records from which to find the closest record to each record in
        `lon_lat_to_consider_df`.
    n_procs : int
        The number of processes to use for the calculation

    Returns
    -------
    pd.DataFrame
           A DataFrame with the following columns:
            - record_name: the name of the record
            - distance_to_closest_cpt_km: the distance to the closest CPT in km
            - closest_cpt_name: the name of the closest CPT
            - lon: the longitude of the CPT
            - lat: the latitude of the CPT
            - closest_cpt_lon: the longitude of the closest CPT
            - closest_cpt_lat: the latitude of the closest CPT
    """

    list_of_rows_as_series = []
    for index, row in lon_lat_to_consider_df.iterrows():
        list_of_rows_as_series.append(row)

    with multiprocessing.Pool(processes=n_procs) as pool:
        closest_cpt_df_list = list(tqdm(
            pool.imap(functools.partial(calc_dist_to_closest_cpt, all_long_lat_df=all_lon_lat_df), list_of_rows_as_series),
            total=len(list_of_rows_as_series)
        ))

    return pd.concat(closest_cpt_df_list, ignore_index=True)