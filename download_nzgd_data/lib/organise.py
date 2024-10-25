"""
Functions for organising the downloaded and processed NZGD data into folders based on geographical regions.
"""

import re
import shutil
import tarfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


def find_regions(
    nzgd_index_path: Path,
    district_shapefile_path: Path,
    suburbs_shapefile_path: Path,
    region_classification_output_dir: Path,
) -> pd.DataFrame:
    """
    Finds the region for each point in the NZGD index and outputs the results to a CSV file.

    Parameters
    ----------
    nzgd_index_path : Path
        Path to the NZGD index CSV file.
    district_shapefile_path : Path
        Path to the district shapefile.
    suburbs_shapefile_path : Path
        Path to the suburbs shapefile.
    region_classification_output_dir : Optional[Union[str, Path]]
        Directory to save the output CSV file. If None, the file is not saved.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the regions found for each point in the NZGD index.
    """

    region_classification_output_dir.mkdir(exist_ok=True, parents=True)

    region_file = (
        region_classification_output_dir / f"regions_{nzgd_index_path.stem}.csv"
    )
    if region_file.exists():
        print("Region file already exists. Loading existing file.")
        return pd.read_csv(region_file)

    nzgd_index_df = pd.read_csv(nzgd_index_path)
    district_gdf = gpd.read_file(district_shapefile_path)
    suburbs_gdf = gpd.read_file(suburbs_shapefile_path)

    found_suburbs_df = gpd.GeoDataFrame()

    for index, row in tqdm(nzgd_index_df.iterrows(), total=nzgd_index_df.shape[0]):

        point = gpd.GeoDataFrame(
            [{"geometry": Point(row["Longitude"], row["Latitude"])}], crs="EPSG:4326"
        )

        # Perform a spatial join to find the region
        district_result_df = gpd.sjoin(
            point, district_gdf, how="left", predicate="within"
        )
        suburb_result_df = gpd.sjoin(point, suburbs_gdf, how="left", predicate="within")

        suburb_result_df.insert(0, "record_id", row["ID"])
        suburb_result_df.insert(1, "district", district_result_df.name)

        found_suburbs_df = pd.concat(
            [found_suburbs_df, suburb_result_df], ignore_index=True
        )

    if region_classification_output_dir:
        found_suburbs_df.to_csv(
            region_classification_output_dir / f"regions_{nzgd_index_path.stem}.csv"
        )

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

    # Regular expression to replace " " (space), "," (comma), and "/" (forward slash) characters
    chars_to_replace = r"[ ,/]"

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
            record_dir.name for record_dir in paths_to_records_to_copy
        ]

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
    # Define the output .tar.xz path
    output_tar_xz = folder_path.with_suffix(".tar.xz")

    # TODO: rewrite this to just write the following command in the terminal as the -T0 does multiprocessing on all cores
    # TODO: tar -cf - [dir] | xz -T0 > [output_file_name]
    ## Needed becaues it gets stuck on Christchurch due it being 42GB (more than half of the data) and was
    ## just chugging through it with one processor. By using this command, all processors will work on the same file

    # Compress the folder
    with tarfile.open(output_tar_xz, "w:xz") as tar:
        tar.add(folder_path, arcname=folder_path.name)

    # Delete the original folder after compression
    shutil.rmtree(folder_path)

    return None
