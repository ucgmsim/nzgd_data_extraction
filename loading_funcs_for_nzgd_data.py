"""
Functions for loading data from the New Zealand Geotechnical Database (NZGD).
"""

from python_ags4 import AGS4
from typing import Union
from pathlib import Path
import pandas as pd
from itertools import product
import numpy as np
import scipy
import copy
 #scipy.stats import mode
import xlrd
import pandas
import re

from scipy.special import special

import loading_helper_functions
import toml



def find_missing_cols_for_best_sheet(missing_cols_per_sheet: list[list]) -> list:

    """
    Find the sheet with the fewest missing columns.

    Parameters
    ----------
    missing_cols_per_sheet : list[list]
        A list of lists, where each inner list contains the missing columns for a sheet.

    Returns
    -------
    list
        The list of missing columns for the sheet with the fewest missing columns.
    """

    final_num_missing_cols = 5
    final_missing_cols = []
    for missing_cols in missing_cols_per_sheet:
        if len(missing_cols) < final_num_missing_cols:
            final_num_missing_cols = len(missing_cols)
            final_missing_cols = missing_cols
    return final_missing_cols




def find_col_name_from_substring(df:pd.DataFrame,
                                 substrings:list[str],
                                 remaining_cols_to_search:list[str],
                                 target_column_name:str) -> tuple[str, pd.DataFrame, list[str]]:

    """
    Find a column name containing a substring in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing the loaded xls file.
    substrings : list[str]
        A list of substrings to search for in the column names.
    remaining_cols_to_search : list[str]
        A list of column names to search for the substring in.
    target_column_name : str
        The name of the column being searched for.

    Returns
    -------
    tuple[str, pd.DataFrame, list[str]]
        The name of the column containing the substring, the updated DataFrame,
        and the updated list of columns to search.
    """

    candidate_col_names = []
    for col_name in remaining_cols_to_search:
        if isinstance(col_name, str):
            for substring in substrings:
                if substring in col_name.lower():
                    if col_name not in candidate_col_names:
                        candidate_col_names.append(col_name)

    ## Check that there are some candidate column names
    if len(candidate_col_names) >= 1:

        col = candidate_col_names[0]

        # check for "Clean" which is sometimes used for a cleaned version of the same data
        if len(candidate_col_names) > 1:
            for candidate_name in candidate_col_names:
                ## some "clean" columns are full of nans (no data) so also check that the number of nans
                ## in the "clean" column is less than or equal to the number of nans in the current column
                if (("clean" in candidate_name.lower()) and
                        (np.sum(pd.isnull(df[candidate_name])) <= np.sum(pd.isnull(df[col])))):
                    col = candidate_name
                    break

        df.attrs[f"candidate_{target_column_name}_column_names_in_original_file"] = candidate_col_names
        df.attrs[f"adopted_{target_column_name}_column_name_in_original_file"] = col
        remaining_cols_to_search.remove(col)

        if target_column_name != "depth":
            # if the column is in kPa, convert to MPa
            if "kpa" in col.lower():
                df.loc[:, col] /= 1000
        if target_column_name == "depth":
            # if the depth column is in cm, convert to m
            if "cm" in col:
                df.loc[:, col] /= 100

    # no relevant columns were found
    elif len(candidate_col_names) == 0:
        col = None
        if "missing_columns" not in df.attrs:
            df.attrs["missing_columns"] = [target_column_name]
        else:
            df.attrs["missing_columns"].append(target_column_name)

    return col, df, remaining_cols_to_search


def load_ags(file_path: Union[Path, str]) -> pd.DataFrame:
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

    try:
        tables, headings = AGS4.AGS4_to_dataframe(file_path)
    except(UnboundLocalError):
        # Found the meaning of this UnboundLocalError by uploading one of these files to the AGS file conversion tool on https://agsapi.bgs.ac.uk
        raise ValueError("ags_duplicate_headers - AGS file contains duplicate headers")

    if len(tables) == 0:
        raise ValueError("no_ags_data_tables - no data tables found in the AGS file")

    try:
        loaded_data_df = pd.DataFrame({
            "depth_m": tables["SCPT"]["SCPT_DPTH"],
            "cone_resistance_qc_mpa": tables["SCPT"]["SCPT_RES"],
            "sleeve_friction_fs_mpa": tables["SCPT"]["SCPT_FRES"],
            "pore_pressure_u2_mpa": tables["SCPT"]["SCPT_PWP2"]  ## Assuming dynamic pore pressure (u2) in MPa ???
        })
    except(KeyError):
        raise ValueError("ags_missing_columns - AGS file is missing at least one of the required columns")


    ### The first two rows are dropped as they contain header information from the ags file
    return loaded_data_df.apply(pd.to_numeric, errors='coerce').dropna()


##########################################################################################################
##########################################################################################################

# depth_col_substrings = ["depth", "length", "h ", "top", "w"]
#
# cone_resistance_col_substrings = [" q ", "qc", "q_c", "cone", "resistance", "res", "tip"]
#
# sleeve_friction_col_substrings = ["fs", "sleeve", "friction", "local"]
#
# porewater_pressure_col_substrings = ["dynamic", "u2", "pore", "u", "water"]
#
#
#

##################################################################################################################



def load_cpt_spreadsheet_file(file_path: Path) -> pd.DataFrame:
    """
    Load the results of a Cone Penetration Test (CPT) from an Excel file.

    Parameters
    ----------
    file_path : Path
        The path to the Excel file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the relevant CPT data columns:
            depth, cone resistance, sleeve friction, and porewater pressure.

    Raises
    ------
    ValueError
        If the required columns are not found in any sheet of the Excel file.
    """

    known_special_cases = toml.load("./resources/known_special_cases.toml")
    record_id = f"{file_path.name.split("_")[0]}_{file_path.name.split("_")[1]}"
    if record_id in known_special_cases.keys():
        raise ValueError(known_special_cases[record_id])

    known_note_labels = toml.load("./resources/known_note_labels.toml")



    if file_path.suffix.lower() in [".xls", ".xlsx"]:
        sheet_names, engine = loading_helper_functions.get_xls_sheet_names(file_path)
    else:
        # A dummy sheet name as .txt and .csv files do not have sheet names
        sheet_names = ["0"]

    missing_cols_per_sheet = []
    # Iterate through each sheet
    for sheet_idx, sheet in enumerate(sheet_names):
    #for sheet_idx, sheet in enumerate(["Test"]):

        if file_path.suffix.lower() in [".csv", ".txt"]:
            df = loading_helper_functions.load_csv_or_txt(file_path)

        else:
            df = pd.read_excel(file_path, sheet_name=sheet, header=None, engine=engine, parse_dates=False)

        ####################################################################################################################
        ####################################################################################################################

        # Now xls, csv and txt should all be in a dataframe so continue the same for all
        df.attrs["original_file_name"] = file_path.name
        df.attrs["source_sheet_in_original_file"] = sheet

        df_for_counting_str_per_row = df.map(lambda x: 1.0 if isinstance(x, (str)) else 0)
        num_str_per_row = np.nansum(df_for_counting_str_per_row, axis=1)

        # df_nan_to_str = df.fillna("nan")
        # df_for_counting_num_per_row = df_nan_to_str.map(lambda x:1.0 if isinstance(x, (int, float)) else 0)
        #
        # num_num_per_row = np.nansum(df_for_counting_num_per_row, axis=1)

        if df.shape == (0,0):
            if (sheet_idx == len(sheet_names) - 1):
                if len(missing_cols_per_sheet) > 0:
                    final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)
                    raise ValueError(f"missing_columns - sheet ({sheet.replace('-', '_')}) is missing [{' & '.join(final_missing_cols)}]")
                # no other sheets so raise error for this file
                raise ValueError(f"empty_file - sheet ({sheet.replace("-", "_")}) has size (0,0)")
            else:
                # There are more sheets to check so continue to next sheet
                missing_cols_per_sheet.append(["depth", "cone_resistance", "sleeve_friction", "porewater_pressure"])
                continue

        # check for a known note label
        if all((df.iloc[0] == known_note_labels["note_label_2a"]) & (df.iloc[1] == known_note_labels["note_label_2b"])):
            raise ValueError(f"note_file_without_data - {known_note_labels["note_label_2a"]} {known_note_labels["note_label_2b"]}")

        # can only find the header rows for xls or xlsx files after the data has been loaded with pandas so do that now

        ## Initial check of the two rows that are most likely to contain the header
        #first_check_rows = np.argmax(num_str_per_row)+np.array([0,1,2,3])
        #header_row_indices = loading_helper_functions.get_header_rows(df, first_check_rows)
        #header_row_indices = loading_helper_functions.find_all_header_rows(df)

        # Try to find a single header row from the column names
        initial_header_row_index = loading_helper_functions.find_one_header_row_from_column_names(df)
        if np.isfinite(initial_header_row_index):
            header_row_indices = loading_helper_functions.find_all_header_rows(df)
        else:
            header_row_indices = []

        if len(header_row_indices) == 0:
            if (sheet_idx == len(sheet_names) - 1):
                if len(missing_cols_per_sheet) > 0:
                    final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)
                    raise ValueError(f"missing_columns - sheet ({sheet.replace('-', '_')}) is missing [{' & '.join(final_missing_cols)}]")
                # no other sheets so raise error for this file
                raise ValueError(f"no_header_row - sheet ({sheet.replace("-", "_")}) has no header row")
            else:
                # There are more sheets to check so continue to next sheet
                missing_cols_per_sheet.append(["depth", "cone_resistance", "sleeve_friction", "porewater_pressure"])
                continue

        # if there are multiple header rows, combine them into the lowest row
        if len(header_row_indices) > 1:
            df, header_row_index = loading_helper_functions.combine_multiple_header_rows(df, header_row_indices)

        # If there is only one header row, take it as the header row
        else:
            header_row_index = header_row_indices[0]


        # set dataframe's headers/column names. Note that .values is used so that the row's index is not included in the header
        df.columns = df.iloc[header_row_index].values
        # Skip the rows that originally contained the column names as they are now stored as the dataframe header
        df = df.iloc[header_row_index+1:]
        df = df.apply(pd.to_numeric, errors='coerce').astype(float)

        # some cases have such as CPT_110939, CPT_110940, CPT_110941, CPT_110942
        # have three rows of headers where the third row
        # is a partial header that gives units for some of the columns
        # so the unit row below what is taken as the header is coerced to nan
        # so if there is a row of nan below the header, it is skipped

        # if all(~np.isfinite(df.iloc[0,:])):
        #     df = df.iloc[1:]

        if np.all(np.sum(np.isfinite(df), axis=0) == 0):
            if (sheet_idx == len(sheet_names) - 1):
                if len(missing_cols_per_sheet) > 0:
                    final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)
                    raise ValueError(f"missing_columns - sheet ({sheet.replace('-', '_')}) is missing [{' & '.join(final_missing_cols)}]")
                # no other sheets so raise error for this file
                raise ValueError(f"empty_file - sheet ({sheet.replace('-', '_')}) has no data")
            else:
                # There are more sheets to check so continue to next sheet
                continue

        if file_path.suffix.lower() in [".csv", ".txt"]:
            df.attrs["header_row_index_in_original_file"] = float(header_row_indices[0]) # this is the index of the header row which is the same as the preceeding number of rows to skip
        else:
            df.attrs["header_row_index_in_original_file"] = float(header_row_index)
        # reset the index so that the first row is index 0
        df.reset_index(inplace=True, drop=True)

        df, final_col_names = loading_helper_functions.check_for_clean_cols(df)
        df = loading_helper_functions.convert_to_m_and_mpa(df, final_col_names)

        col1 = final_col_names[0]
        col2 = final_col_names[1]
        col3 = final_col_names[2]
        col4 = final_col_names[3]

        ################################################
        if (col1 is not None) and (col2 is not None) and (col3 is not None) and (col4 is not None):

            # Return the DataFrame with only the relevant columns
            df = (df[[col1, col2, col3, col4]].rename(
                columns={col1:"depth_m",
                         col2:"cone_resistance_qc_mpa",
                         col3: "sleeve_friction_fs_mpa",
                         col4: "pore_pressure_u2_mpa"})).apply(pd.to_numeric, errors='coerce')

            # ensure that the depth column is defined as positive (some have depth as negative)
            df["depth_m"] = np.abs(df["depth_m"])

            return df

        else:
            missing_cols = []
            if col1 is None:
                missing_cols.append("depth")
            if col2 is None:
                missing_cols.append("cone_resistance")
            if col3 is None:
                missing_cols.append("sleeve_friction")
            if col4 is None:
                missing_cols.append("porewater_pressure")

            missing_cols_per_sheet.append(missing_cols)
            if sheet_idx < len(sheet_names) - 1:
                # There are more sheets to check so continue onto the next sheet
                continue

            else:
                # There are no more sheets to check so return the missing columns
                final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)

                raise ValueError(f"missing_columns - sheet ({sheet.replace("-", "_")}) is missing [{' & '.join(final_missing_cols)}]")