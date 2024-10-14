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
import loading_helper_functions

def convert_num_as_str_to_float(val):
    try:
        return float(val)
    except ValueError:
        return val

def can_convert_str_to_float(value: str) -> bool:

    """
    Check if a string can be converted to a float.

    Parameters
    ----------
    value : str
        The string to check.

    Returns
    -------
    bool
        True if the string can be converted to a float, False otherwise.
    """

    try:
        float(value)
        return True
    except ValueError:
        return False

def str_cannot_become_float(value: str) -> bool:

    """
    Check if a string cannot be converted to a float.

    Parameters
    ----------
    value : str
        The string to check.

    Returns
    -------
    bool
        True if the string cannot be converted to a float, False otherwise.
    """

    try:
        float(value)
        return False
    except ValueError:
        return True


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
            "qc_mpa": tables["SCPT"]["SCPT_RES"],
            "fs_mpa": tables["SCPT"]["SCPT_FRES"],
            "u_mpa": tables["SCPT"]["SCPT_PWP2"]  ## Assuming dynamic pore pressure (u2) in MPa ???
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

    min_num_numerical_rows_with_at_least_4_cols = 5

    if file_path.suffix.lower() in [".xls", ".xlsx"]:
        sheet_names, engine = loading_helper_functions.get_xls_sheet_names(file_path)

    else:
        # A dummy sheet name as .txt and .csv files do not have sheet names
        sheet_names = ["0"]

    missing_cols_per_sheet = []
    # Iterate through each sheet
    for sheet_idx, sheet in enumerate(sheet_names):

        # Load the entire sheet without specifying headers
        if file_path.suffix.lower() in [".csv", ".txt"]:

            sep = r"," if file_path.suffix == ".csv" else r"\s+"
            print()
            file_encoding = loading_helper_functions.find_encoding(file_path)

            split_readlines_iterable = loading_helper_functions.get_csv_or_txt_split_readlines(file_path, file_encoding)
            header_row_idx = loading_helper_functions.get_header_rows(split_readlines_iterable, range(len(split_readlines_iterable)-1))
            print()

            needed_col_indices = loading_helper_functions.search_line_for_all_needed_cells(split_readlines_iterable[header_row_idx])

            df = pd.read_csv(file_path, header=None, encoding=file_encoding, sep=sep,
                        skiprows=header_row_idx, usecols=needed_col_indices).map(convert_num_as_str_to_float)

        else:
            df = pd.read_excel(file_path, sheet_name=sheet, header=None, engine=engine)

        ####################################################################################################################
        ####################################################################################################################
        # Now xls, csv and txt should all be in a dataframe so continue the same for all
        df.attrs["original_file_name"] = file_path.name
        df.attrs["source_sheet_in_original_file"] = sheet

        df_for_counting_str_per_row = df.map(lambda x: 1.0 if isinstance(x, (str)) else 0)
        num_str_per_row = np.nansum(df_for_counting_str_per_row, axis=1)

        df_nan_to_str = df.fillna("nan")
        df_for_counting_num_per_row = df_nan_to_str.map(lambda x:1.0 if isinstance(x, (int, float)) else 0)

        num_num_per_row = np.nansum(df_for_counting_num_per_row, axis=1)
        num_data_rows_with_at_least_4_cols = np.sum(num_num_per_row >= 4)

        if df.shape == (0,0):
            if sheet_idx == len(sheet_names) - 1:
                # no other sheets so raise error for this file
                raise ValueError(f"empty_file - sheet ({sheet.replace("-", "_")}) has size (0,0)")
            else:
                # There are more sheets to check so continue to next sheet
                continue

        if num_data_rows_with_at_least_4_cols < min_num_numerical_rows_with_at_least_4_cols:
            # There are not enough rows with numerical data in all required columns

            if sheet_idx == len(sheet_names) - 1:
                # no other sheets so raise error for this file
                final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)
                if len(final_missing_cols) > 0:
                    raise ValueError(f"missing_columns - sheet ({sheet.replace("-", "_")}) is missing [{' & '.join(final_missing_cols)}]")
                raise ValueError(f"lacking_data_rows - sheet ({sheet.replace("-", "_")}) has fewer than {min_num_numerical_rows_with_at_least_4_cols} data rows with at least four numerical columns")

            else:
                # There are more sheets to check so continue to next sheet
                continue

        first_check_rows = np.argmax(num_str_per_row)+np.array([0,1])
        col_name_rows = loading_helper_functions.get_header_rows(df, first_check_rows)

        if len(col_name_rows) == 0:
            col_name_rows = loading_helper_functions.get_header_rows(df, np.arange(0, 50))

        ### If the file is a text file, skiprows is used so the header row is now the first row
        ### with zero numerical cells per row
        # else:
        #     col_name_rows = np.where(num_num_per_row == 0)[0]

        if len(col_name_rows) == 0:
            raise ValueError(f"no_header_row - sheet ({sheet.replace("-", "_")}) has no header row")

        # if there are multiple header rows, combine them into the lowest row
        if len(col_name_rows) > 1:
            col_name_row = np.max(col_name_rows)
            # copy the column names from the rows above the header row
            df2 = df.copy()
            for row_idx in col_name_rows:#[0:col_name_row]:
                for col_idx in range(df.shape[1]):
                    if row_idx != col_name_row:
                        df2.iloc[col_name_row, col_idx] = str(df.iloc[col_name_row,col_idx]) + " " + str(df.iloc[row_idx,col_idx])

            df = df2.copy()

        # If there is only one header row, take it as the header row
        else:
            col_name_row = col_name_rows[0]

        # set dataframe's headers/column names. Note that .values is used so that the row's index is not included in the header
        df.columns = df.iloc[col_name_row].values
        # Skip the rows that originally contained the column names as they are now stored as the dataframe header
        df = df.iloc[col_name_row+1:]
        df = df.apply(pd.to_numeric, errors='coerce').astype(float)

        if file_path.suffix.lower() in [".csv", ".txt"]:
            df.attrs["header_row_index_in_original_file"] = float(header_row_idx) # this is the index of the header row which is the same as the preceeding number of rows to skip
        else:
            df.attrs["header_row_index_in_original_file"] = float(col_name_row)
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
            # df = df[[col1, col2, col3, col4]].rename(
            #     columns={col1: "depth_m", col2: "qc_mpa", col3: "fs_mpa", col4: "u_mpa"})
            df = (df[[col1, col2, col3, col4]].rename(
                columns={col1: "depth_m", col2: "qc_mpa", col3: "fs_mpa", col4: "u_mpa"})).apply(pd.to_numeric, errors='coerce').astype({
                "depth_m": float,
                "qc_mpa": float,
                "fs_mpa": float,
                "u_mpa": float})

            # ensure that the depth column is defined as positive (some have depth as negative)
            df["depth_m"] = np.abs(df["depth_m"])

            return df

        else:
            print()
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