"""
Functions for loading data from the New Zealand Geotechnical Database (NZGD).
"""

from python_ags4 import AGS4
from typing import Union
from pathlib import Path
import pandas as pd
import natsort
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from itertools import product

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
        "u_mpa": tables["SCPT"]["SCPT_PWP2"]  ## Assuming dynamic pore pressure (u2) in MPa ???
    })

    ### The first two rows are dropped as they contain header information from the ags file
    return loaded_data_df.apply(pd.to_numeric, errors='coerce').dropna()


def load_xls_basic(file_path: Union[Path, str]):
    """
    Load an XLS file.

    Parameters
    ----------
    file_path : Path or str
        The path to the XLS file.

    Returns
    -------
    pandas.DataFrame
        The CPT data from the XLS file.
    """

    sheet_names = pd.ExcelFile(file_path).sheet_names

    print(sheet_names)

    print()

    if ("REPORT" in sheet_names) and ("Field Sheet" in sheet_names) and ("WORKSHEET" in sheet_names):

        raise IndexError("xls too difficult to load")

    if "Data" in sheet_names:
        loaded_data_df = pd.read_excel(file_path, sheet_name="Data",header=0)



        needed_col_names_and_cols = pd.DataFrame({"depth_m": loaded_data_df["Depth [m]"],
                                                  "qc_mpa": loaded_data_df["Cone resistance (qc) in MPa"],
                                                  "fs_mpa": loaded_data_df["Sleeve friction (fs) in MPa"],
                                                  "u_mpa": loaded_data_df["Dynamic pore pressure (u2) in MPa"]})

    elif "CPTU" in sheet_names:
        loaded_data_df = pd.read_excel(file_path, sheet_name="CPTU",skiprows=32, header=0)

        print()

        needed_col_names_and_cols = pd.DataFrame({"depth_m": loaded_data_df["Depth [m]"],
                                                  "qc_mpa": loaded_data_df["Qc [MPa]"],
                                                  "fs_mpa": loaded_data_df["Fs [KPa]"]/1000, ### X/1000 to convert kPa to MPa
                                                  "u_mpa": loaded_data_df["U2 [KPa]"]})

    elif "CPT SUMMARY SHEET" in sheet_names:
        loaded_data_df = pd.read_excel(file_path, sheet_name=0, header=0) ## sheet names vary but its always the first sheet

        needed_col_names_and_cols = pd.DataFrame({"depth_m": loaded_data_df["Depth [m]"],
                                                  "qc_mpa": loaded_data_df["qc [MPa]"],
                                                  "fs_mpa": loaded_data_df["fs [MPa]"],
                                                  "u_mpa": loaded_data_df["u2 [MPa]"]})

    elif "-" in sheet_names[0]:

        loaded_data_df = pd.read_excel(file_path, sheet_name=0,
                                       header=0)  ## sheet names vary but its always the first sheet

        needed_col_names_and_cols = pd.DataFrame({"depth_m": loaded_data_df["Depth"],
                                                  "qc_mpa": loaded_data_df["Tip resistance"],
                                                  "fs_mpa": loaded_data_df["Local friction"],
                                                  "u_mpa": loaded_data_df["Pore shoulder"]})

    return needed_col_names_and_cols


def load_xls_file_brute_force(file_path, possible_xls_cols):

    possible_col1 = possible_xls_cols[0]
    possible_col2 = possible_xls_cols[1]
    possible_col3 = possible_xls_cols[2]
    possible_col4 = possible_xls_cols[3]

    # Load all sheet names in the file
    sheet_names = pd.ExcelFile(file_path).sheet_names

    # Iterate through each sheet
    for sheet in sheet_names:

        # Load the entire sheet without specifying headers
        df = pd.read_excel(file_path, sheet_name=sheet, header=None)

        for idx, row in df.iterrows():

            # Check if the current row contains valid column names
            for col1, col2, col3, col4 in product(possible_col1, possible_col2, possible_col3, possible_col4):
                if {col1, col2, col3, col4}.issubset(row):
                    # Set this row as the header and skip the rows above it
                    df.columns = row
                    df = df.iloc[idx + 1:]  # Skip the rows above the header

                    for col_name in [col2, col3, col4]:
                        ### Convert KPa to MPa
                        if "KPa" in col_name:
                            df.loc[:, col_name] /= 1000

                    # Return the DataFrame with only the relevant columns
                    return df[[col1, col2, col3, col4]].rename(
                        columns={col1: "depth_m", col2: "qc_mpa", col3: "fs_mpa", col4: "u_mpa"})

    # If no valid sheet/column combination was found raise an error
    raise ValueError(f"Failed to find valid columns in file: {file_path}")
