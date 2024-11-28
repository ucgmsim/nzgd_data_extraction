"""
Functions for organising the downloaded and processed NZGD data into folders based on geographical regions.
"""

import functools
import multiprocessing
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


def find_region(
    df_row: pd.Series, district_gdf: pd.DataFrame, suburbs_gdf: pd.DataFrame
) -> pd.DataFrame:
    """
    Finds the region for a given point based on its geographical coordinates.

    Parameters
    ----------
    df_row : pd.Series
        A row from the DataFrame containing the point's coordinates (Longitude, Latitude) and ID.
    district_gdf : pd.DataFrame
        GeoDataFrame containing the district boundaries.
    suburbs_gdf : pd.DataFrame
        GeoDataFrame containing the suburb boundaries.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the region information for the given point, including the record ID and district name.
    """

    # Create a GeoDataFrame for the point using its longitude and latitude
    point = gpd.GeoDataFrame(
        [{"geometry": Point(df_row["Longitude"], df_row["Latitude"])}], crs="EPSG:4326"
    )

    # Perform a spatial join to find the district containing the point
    district_result_df = gpd.sjoin(point, district_gdf, how="left", predicate="within")

    # Perform a spatial join to find the suburb containing the point
    suburb_result_df = gpd.sjoin(point, suburbs_gdf, how="left", predicate="within")

    # Insert the record ID and district name into the result DataFrame
    suburb_result_df.insert(0, "record_id", df_row["ID"])
    suburb_result_df.insert(1, "district", district_result_df.name)

    return suburb_result_df


def find_regions(
    latest_nzgd_index_file_path: Path,
    district_shapefile_path: Path,
    suburbs_shapefile_path: Path,
    region_classification_output_dir: Path,
    num_procs: int,
    previous_region_file_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Finds the region for each point in the NZGD index and outputs the results to a CSV file.

    Parameters
    ----------
    latest_nzgd_index_file_path : Path
        Path to the NZGD index CSV file.
    district_shapefile_path : Path
        Path to the district shapefile.
    suburbs_shapefile_path : Path
        Path to the suburbs shapefile.
    region_classification_output_dir : Path
        Directory to save the output CSV file.
    num_procs : int
        Number of processes to use for parallel processing.
    previous_region_file_path : Path, optional
        If a path to a previous region classification file is provided, the regions will be found for the new points only.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the regions found for each point in the NZGD index.
    """

    start_time = time.time()

    region_classification_output_dir.mkdir(exist_ok=True, parents=True)

    region_file = (
        region_classification_output_dir / f"regions_{latest_nzgd_index_file_path.stem}.csv"
    )
    if region_file.exists():
        print("Region file already exists. Loading existing file.")
        return pd.read_csv(region_file)

    latest_nzgd_index_df = pd.read_csv(latest_nzgd_index_file_path)
    latest_nzgd_index_df = latest_nzgd_index_df[
        (latest_nzgd_index_df["Type"] == "CPT")
        | (latest_nzgd_index_df["Type"] == "SCPT")
        | (latest_nzgd_index_df["Type"] == "Borehole")
        | (latest_nzgd_index_df["Type"] == "VsVp")
    ]

    if previous_region_file_path:
        previous_region_file = pd.read_csv(previous_region_file_path)
        latest_nzgd_index_df = latest_nzgd_index_df[~latest_nzgd_index_df["ID"].isin(previous_region_file["record_id"])]

    district_gdf = gpd.read_file(district_shapefile_path)
    suburbs_gdf = gpd.read_file(suburbs_shapefile_path)

    find_regions_partial = functools.partial(
        find_region, district_gdf=district_gdf, suburbs_gdf=suburbs_gdf
    )
    df_rows_as_list = [row for index, row in latest_nzgd_index_df.iterrows()]

    with multiprocessing.Pool(processes=num_procs) as pool:
        found_suburbs_df = pd.concat(
            list(tqdm(pool.imap(find_regions_partial, df_rows_as_list), total=len(df_rows_as_list))), ignore_index=True
        )

    ## Blank fields get values of np.nan so they are replaced with "unclassified"
    found_suburbs_df.fillna("unclassified", inplace=True)

    ## Combine with the previous region classification file if it exists
    if previous_region_file_path:
        found_suburbs_df = pd.concat([previous_region_file, found_suburbs_df])

    if region_classification_output_dir:
        found_suburbs_df.to_csv(
            region_classification_output_dir / f"regions_{latest_nzgd_index_file_path.stem}.csv",
            index=False
        )

    end_time = time.time()

    print(f"Time taken to find regions: {(end_time - start_time)/60} minutes")

    return found_suburbs_df


def get_recursive_list_of_nzgd_files(root_dir):
    """
    Get a list of the downloaded records from the download root directory.

    Parameters
    ----------
    root_dir : Path
        The root directory of the records.

    Returns
    -------
    list
        A list of the files under the root directory.
    """

    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
    type_dirs = [record_dir for record_dir in root_dir.iterdir() if record_dir.is_dir()]

    downloaded_records = []
    for type_dir in type_dirs:
        downloaded_records.extend(
            [record_dir for record_dir in type_dir.iterdir() if record_dir.is_dir()]
        )

    return downloaded_records


def organise_records_into_regions(
    processed_data: bool,
    dry_run: bool,
    unorganised_root_dir_to_copy_from: Path,
    organised_root_dir_to_copy_to: Path,
    region_df: pd.DataFrame,
) -> None:
    """
    Organise records into regions based on the provided region DataFrame from Land Information New Zealand (LINZ).

    Parameters
    ----------
    processed_data : bool
        Flag indicating if the data has been processed.
    dry_run : bool
        If True, only print the actions without performing them.
    unorganised_root_dir_to_copy_from : Path
        The root directory containing unorganised records.
    organised_root_dir_to_copy_to : Path
        The root directory where organised records will be copied.
    region_df : pd.DataFrame
        DataFrame containing region information for each record.
    """

    record_type_to_path = {
        "CPT": "cpt",
        "SCPT": "scpt",
        "BH": "borehole",
        "VsVp": "vsvp",
    }

    paths_to_already_organized_files = list(organised_root_dir_to_copy_to.rglob("*"))
    already_organized_records = [record_dir.parent.name for record_dir in paths_to_already_organized_files if record_dir.is_file()]

    # Regular expression to replace " " (space), ' (apostrophe), "," (comma), and "/" (forward slash) characters
    chars_to_replace = r"[ ',/]"

    # Get the list of records to copy based on the processed_data flag
    if processed_data:
        paths_to_records_to_copy = list(
            unorganised_root_dir_to_copy_from.glob("*.parquet")
        )
        record_ids_to_copy = [
            record_dir.stem for record_dir in paths_to_records_to_copy
        ]

    else:
        paths_to_records_to_copy = get_recursive_list_of_nzgd_files(
            unorganised_root_dir_to_copy_from
        )
        record_ids_to_copy = [
            record_dir.name for record_dir in paths_to_records_to_copy]

    ## Only copy records that have not already been organised
    record_ids_to_copy = list(set(record_ids_to_copy) - set(already_organized_records))

    # Create a dictionary mapping record IDs to their paths
    downloaded_record_dict = dict(zip(record_ids_to_copy, paths_to_records_to_copy))

    # Iterate over each row in the region DataFrame
    for index, row in tqdm(region_df.iterrows(), total=region_df.shape[0]):
        if row["record_id"] not in record_ids_to_copy:
            continue

        # Extract the record type from the record ID
        record_type = row["record_id"].split("_")[0]

        # Replace certain characters in the region names
        district_txt = re.sub(chars_to_replace, "_", row["district"])
        territor_1_txt = re.sub(chars_to_replace, "_", row["territor_1"])
        major_na_2_txt = re.sub(chars_to_replace, "_", row["major_na_2"])
        name_ascii_txt = re.sub(chars_to_replace, "_", row["name_ascii"])

        # Construct the destination path for the record
        destination_path = (
            Path(organised_root_dir_to_copy_to)
            / record_type_to_path[record_type]
            / district_txt
            / territor_1_txt
            / major_na_2_txt
            / name_ascii_txt
        )

        # Create the destination directory if it does not exist
        destination_path.mkdir(parents=True, exist_ok=True)

        if dry_run:
            # Print the action that would be performed in dry run mode
            print(
                f"will copy {downloaded_record_dict[row["record_id"]]} to {destination_path / downloaded_record_dict[row["record_id"]].name}"
            )

        else:
            # Copy the record to the destination path
            if processed_data:
                shutil.copy(
                    downloaded_record_dict[row["record_id"]],
                    destination_path / downloaded_record_dict[row["record_id"]].name,
                )
            else:
                shutil.copytree(
                    downloaded_record_dict[row["record_id"]],
                    destination_path / downloaded_record_dict[row["record_id"]].name,
                )

    return None


def replace_folder_with_tar_xz(folder_path: Path) -> None:
    """
    Compress a folder into a .tar.xz archive and delete the original folder.

    Parameters
    ----------
    folder_path : Path
        The path to the folder to be compressed.
    """

    print(f"Compressing folder: {folder_path}")
    ## Example of folder_path:
    ## /home/arr65/data/nzgd/dropbox_mirror/nzgd/processed/cpt/Canterbury/Christchurch_City
    working_dir = folder_path.parent
    relative_path = folder_path.relative_to(working_dir)
    output_tar_xz_file_name = relative_path.with_suffix(".tar.xz")

    ## The - is used with tar send its output to stdout. The | then passes it to xz.
    ## The xz argument -T0 uses all available processors to compress the file (can be changed to -T1, -T2, etc. for
    ## one, two processors, etc.)
    terminal_command = (
        f"tar -cf - ./{relative_path} | xz -T0 > ./{output_tar_xz_file_name}"
    )
    subprocess.run(terminal_command, cwd=folder_path.parent, shell=True)

    # Delete the original folder after compression
    shutil.rmtree(folder_path)

    return None
