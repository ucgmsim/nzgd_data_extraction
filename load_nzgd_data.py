"""
Functions for loading data from the New Zealand Geotechnical Database (NZGD).
"""

from python_ags4 import AGS4
from typing import Union
from pathlib import Path
import pandas as pd

def load_ags(file_path: Union[Path, str]):
    """
    Load an AGS file.

    Parameters
    ----------
    file_path : Path or str
        The path to the AGS file.

    Returns
    -------
    pandas.DataFrame
        The CPT data from the AGS file.
    """

    tables, headings = AGS4.AGS4_to_dataframe(file_path)

    loaded_data_df = pd.DataFrame({
        "depth_m": tables["SCPT"]["SCPT_DPTH"],
        "qc_mpa": tables["SCPT"]["SCPT_RES"],
        "fs_mpa": tables["SCPT"]["SCPT_FRES"],
        "u_mpa": tables["SCPT"]["SCPT_PWP2"]
    })

    ### The first two rows are dropped as they contain header information from the ags file
    return loaded_data_df.apply(pd.to_numeric, errors='coerce').dropna()


if __name__ == "__main__":
    load_ags("/home/arr65/data/nzgd/downloaded_files/manually_organized/ags/CPT_1_AGS01.ags")

    ags_path = Path("/home/arr65/data/nzgd/downloaded_files/manually_organized/ags")

    for ags_index, ags_file in enumerate(sorted(ags_path.glob('*.ags'))):

        if ags_index % 100 == 0:
            print(f"Trying to load AGS file {ags_index + 1} of {len(list(ags_path.glob('*.ags')))}")
            print(f"AGS file  {ags_file} processed")

        try:
            load_ags(ags_file)
        except:
            print(f"failed to load AGS file index {ags_index + 1}, {ags_file}")






