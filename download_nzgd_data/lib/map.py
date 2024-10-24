"""
Functions for getting information required by the map on the Hypocentre NZGD HTTP server.
"""

from pathlib import Path
from collections import defaultdict, namedtuple
from tqdm import tqdm
import pandas as pd

MetaData = namedtuple("MetaData", ["max_depth", "min_depth"])

def get_files_with_relative_paths(
    processed_files: bool, file_root_directory: Path, relative_to: Path
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
    print()
    print("Matching all files to record IDs")
    record_id_to_files = defaultdict(list)
    for file in tqdm(all_files):
        record_id = file.stem if processed_files else file.parent.name
        if relative_to:
            record_id_to_files[record_id].append(file.relative_to(relative_to))
        else:
            record_id_to_files[record_id].append(file)

    return record_id_to_files


def get_processed_metadata(file_root_directory: Path) -> dict[str, MetaData]:
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

    for file in tqdm(all_files):
        record_df = pd.read_parquet(file)
        record_id_to_metadata[file.stem] = MetaData(
            max_depth=record_df.attrs["max_depth"],
            min_depth=record_df.attrs["min_depth"],
        )

    return record_id_to_metadata
