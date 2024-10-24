from pathlib import Path
from typing import Optional, Union
import re
import shutil

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


def find_regions(
    nzgd_index_path: Union[str, Path],
    district_shapefile_path: Union[str, Path],
    suburbs_shapefile_path: Union[str, Path],
    region_classifcation_output_dir: Optional[Union[str, Path]],
) -> pd.DataFrame:
    """
    Finds the region for each point in the NZGD index and outputs the results to a CSV file.

    Parameters
    ----------
    nzgd_index_path : Union[str, Path]
        Path to the NZGD index CSV file.
    district_shapefile_path : Union[str, Path]
        Path to the district shapefile.
    suburbs_shapefile_path : Union[str, Path]
        Path to the suburbs shapefile.
    region_classifcation_output_dir : Optional[Union[str, Path]]
        Directory to save the output CSV file. If None, the file is not saved.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the regions found for each point in the NZGD index.
    """

    if isinstance(nzgd_index_path, str):
        nzgd_index_path = Path(nzgd_index_path)
    if isinstance(district_shapefile_path, str):
        district_shapefile_path = Path(district_shapefile_path)
    if isinstance(suburbs_shapefile_path, str):
        suburbs_shapefile_path = Path(suburbs_shapefile_path)
    if isinstance(region_classifcation_output_dir, str):
        region_classifcation_output_dir = Path(region_classifcation_output_dir)
    region_classifcation_output_dir.mkdir(exist_ok=True, parents=True)

    region_file = region_classifcation_output_dir / f"regions_{nzgd_index_path.stem}.csv"
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

    if region_classifcation_output_dir:
        found_suburbs_df.to_csv(region_classifcation_output_dir / f"regions_{nzgd_index_path.stem}.csv")

    return found_suburbs_df


def get_recursive_list_of_nzgd_files(root_dir):
    """
    Get a list of the downloaded records from the download root directory.

    Parameters
    ----------
    root_dir : Union[str, Path]
        The root directory where the downloaded records are stored.

    Returns
    -------
    list
        A list of the downloaded records.
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
    analysis_ready_data: bool,
    dry_run: bool,
    unorganised_root_dir_to_copy_from: Union[str, Path],
    organised_root_dir_to_copy_to: Union[str, Path],
    region_df: pd.DataFrame,
) -> None:
    """
    Creates the directory structure to organise the NZGD data into regions.

    Parameters
    ----------
    copy_to_organised_root_dir : Union[str, Path]
        Root of the directory structure.
    region_df : pd.DataFrame
        Contains the region of each record in the NZGD index
        (can be obtained from the find_regions function).
    """

    record_type_to_path = {
        "CPT": "cpt",
        "SCPT": "scpt",
        "BH": "borehole",
        "VsVp": "vsvp",
    }

    if isinstance(organised_root_dir_to_copy_to, str):
        organised_root_dir_to_copy_to = Path(organised_root_dir_to_copy_to)
    if isinstance(unorganised_root_dir_to_copy_from, str):
        unorganised_root_dir_to_copy_from = Path(unorganised_root_dir_to_copy_from)

    chars_to_replace = r"[ ,/]"

    if analysis_ready_data:
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

    downloaded_record_dict = dict(zip(record_ids_to_copy, paths_to_records_to_copy))
    for index, row in tqdm(region_df.iterrows(), total=region_df.shape[0]):
        if row["record_id"] not in record_ids_to_copy:
            continue

        record_type = row["record_id"].split("_")[0]

        district_txt = re.sub(chars_to_replace, "_", row["district"])
        territor_1_txt = re.sub(chars_to_replace, "_", row["territor_1"])
        major_na_2_txt = re.sub(chars_to_replace, "_", row["major_na_2"])
        name_ascii_txt = re.sub(chars_to_replace, "_", row["name_ascii"])

        destination_path = (
            Path(organised_root_dir_to_copy_to)
            / record_type_to_path[record_type]
            / district_txt
            / territor_1_txt
            / major_na_2_txt
            / name_ascii_txt
        )

        destination_path.mkdir(parents=True, exist_ok=True)

        if dry_run:
            print(
                f"will copy {downloaded_record_dict[row["record_id"]]} to {destination_path / downloaded_record_dict[row["record_id"]].name}"
            )

        else:
            if analysis_ready_data:
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
