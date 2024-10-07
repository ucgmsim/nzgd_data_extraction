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

    if len(candidate_col_names) >= 1:

        col = candidate_col_names[0]

        # check for "Clean" which is sometimes used for a cleaned version of the same data
        if len(candidate_col_names) > 1:
            for candidate_name in candidate_col_names:
                if "clean" in candidate_name.lower():
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

    tables, headings = AGS4.AGS4_to_dataframe(file_path)
    loaded_data_df = pd.DataFrame({
        "depth_m": tables["SCPT"]["SCPT_DPTH"],
        "qc_mpa": tables["SCPT"]["SCPT_RES"],
        "fs_mpa": tables["SCPT"]["SCPT_FRES"],
        "u_mpa": tables["SCPT"]["SCPT_PWP2"]  ## Assuming dynamic pore pressure (u2) in MPa ???
    })

    ### The first two rows are dropped as they contain header information from the ags file
    return loaded_data_df.apply(pd.to_numeric, errors='coerce').dropna()


def load_cpt_xls_file(file_path: Path) -> pd.DataFrame:
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
    if file_path.suffix.lower() == ".xls":
        engine = "xlrd"
    elif file_path.suffix.lower() == ".xlsx":
        engine = "openpyxl"

    # Some .xls files are actually xlsx files and need to be opened with openpyxl
    try:
        sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names
    except(xlrd.biffh.XLRDError):
        if engine == "xlrd":
            other_engine = "openpyxl"
        if engine == "openpyxl":
            other_engine = "xlrd"

        engine = other_engine

        sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names
    missing_cols_per_sheet = []

    # Iterate through each sheet
    for sheet_idx, sheet in enumerate(sheet_names):

        # Load the entire sheet without specifying headers
        df = pd.read_excel(file_path, sheet_name=sheet, header=None, engine=engine)

        df.attrs["original_file_name"] = file_path.name
        df.attrs["source_sheet_in_original_file"] = sheet

        df_for_counting_str_per_row = df.map(lambda x: 1.0 if isinstance(x, (str)) else 0)
        num_str_per_row = np.nansum(df_for_counting_str_per_row, axis=1)

        #col_name_rows = np.where(num_str_per_row >= np.median(num_str_per_row) + 5*np.std(num_str_per_row))[0]

        df_nan_to_str = df.fillna("nan")
        df_for_counting_num_per_row = df_nan_to_str.map(lambda x:1.0 if isinstance(x, (int, float)) else 0)

        num_num_per_row = np.nansum(df_for_counting_num_per_row, axis=1)

        # The last available sheet
        if sheet_idx == len(sheet_names) - 1:
            if df.shape == (0,0):
                raise ValueError(f"No data found in file {file_path.name}")
            if ((df.shape[1]) > 0 and (df.shape[1]) < 4):
                final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)

                if len(final_missing_cols) > 0:
                    raise ValueError(f"Missing columns, {' - '.join(final_missing_cols)}")

                elif len(final_missing_cols) == 0:
                    raise ValueError(f"File {file_path.name} is missing {4-df.shape[1]} required columns")

        if (df.shape[1] < 4)  and sheet_idx < len(sheet_names) - 1:
            # There are not enough columns to contain the required data so continue to the next sheet
            continue
        if (df.shape[1] < 4) and sheet_idx == len(sheet_names) - 1:
            # There are not enough columns to contain the required data so return the missing columns
            final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)
            raise ValueError(f"Missing columns, {' - '.join(final_missing_cols)}")

        #first_data_row = np.where(num_num_per_row >= 4)[0][0]
        last_data_row = np.where(num_num_per_row >= 4)[0][-1]

        num_num_in_last_data_row = num_num_per_row[last_data_row]

        # some files have a lot of numbers in the rows to skip so now find the first row with at least the same number
        # of numbers as the last data row. This assumes that all extra information is before the data.
        first_data_row = np.where(num_num_per_row >= num_num_in_last_data_row)[0][0]

        col_name_rows = []
        #check_row = first_data_row - 1
        #prev_num_str_per_row = num_str_per_row[first_data_row]
        found_a_header_row = False

        max_num_rows_to_check_for_header = 4
        num_rows_checked_for_header = 0

        while num_rows_checked_for_header <= max_num_rows_to_check_for_header:
            check_row = first_data_row - num_rows_checked_for_header - 1
            # print(
            #     f"check row {check_row}, num_str_per_row {num_str_per_row[check_row]}, num_rows_checked_for_header {num_rows_checked_for_header}")
            if check_row < 0:
                break
            if num_str_per_row[check_row] >= 4:
                col_name_rows.append(check_row)
                found_a_header_row = True

            if found_a_header_row and (num_str_per_row[check_row] == 0):
                break

            num_rows_checked_for_header += 1

        # the header search algorithm finds in reverse order so sort to be in ascending order
        col_name_rows = np.sort(col_name_rows)

        if len(col_name_rows) == 0:
            raise ValueError(f"No header row found in file {file_path.name} sheet {sheet}")

        # if there are multiple header rows, combine them into one
        if len(col_name_rows) > 1:
            col_name_row = np.max(col_name_rows)
            # copy the column names from the rows above the header row
            df2 = df.copy()
            for row_idx in col_name_rows[0:col_name_row]:
                for col_idx in range(df.shape[1]):
                    df2.iloc[col_name_row, col_idx] = str(df.iloc[col_name_row,col_idx]) + " " + str(df.iloc[row_idx,col_idx])

            df = df2.copy()

        # If there is only one header row, take it as the header row
        else:
            col_name_row = col_name_rows[0]
        # set dataframe's headers/column names. Note that .values is used so that the row's index is not included in the header
        df.columns = df.iloc[col_name_row].values
        # Skip the rows that originally contained the column names as they are now stored as the dataframe header
        df = df.iloc[col_name_row+1:]

        # set the data types to float
        df.attrs["header_row_index_in_original_file"] = float(col_name_row)
        # reset the index so that the first row is index 0
        df.reset_index(inplace=True, drop=True)

        # Find the depth column
        # First search for a column name containing "m" or "cm"
        candidate_depth_col_names = []
        for col_name in df.columns:
            if isinstance(col_name, str):
                for letter_idx, letter in enumerate(col_name):
                    if letter.lower() == "m":

                        if (letter_idx == 0) and (col_name[letter_idx+1] in [" ", "]", ")"]) and (col_name not in candidate_depth_col_names):
                            candidate_depth_col_names.append(col_name)
                            break
                        # including "c" when checking preceeding characters to check for "cm"
                        elif (letter_idx == len(col_name)-1) and (col_name[letter_idx-1] in ["c", " ", "[", "("]) and (col_name not in candidate_depth_col_names):
                            candidate_depth_col_names.append(col_name)
                            break
                        # it's somewhere in the middle
                        else:
                            # including "c" when checking preceeding characters to check for "cm"
                            if (col_name[letter_idx-1] in ["c", " ", "[", "("]) and (col_name[letter_idx+1] in [" ", "]", ")"]) and (col_name not in candidate_depth_col_names):
                                candidate_depth_col_names.append(col_name)
                                break

        # The search for "m" identified the depth column
        if len(candidate_depth_col_names) >= 1:
            col1 = candidate_depth_col_names[0]
            df.attrs["candidate_depth_column_names_in_original_file"] = candidate_depth_col_names
            df.attrs["adopted_depth_column_name_in_original_file"] = col1
            remaining_cols_to_search = list(df.columns)
            remaining_cols_to_search.remove(col1)

        # If no columns with "m" or "cm" are found, look for columns with "depth" or "length" or "h "
        else:
            col1, df, remaining_cols_to_search = find_col_name_from_substring(df,
                                                                              ["depth", "length", "h "],
                                                                              list(df.columns),
                                                                              "depth")

        # Identify the cone resistance, q_c, column
        col2, df, remaining_cols_to_search = find_col_name_from_substring(df,
                                                                          [" q ", "qc","q_c", "cone",
                                                                                     "resistance", "res", "tip"],
                                                                          remaining_cols_to_search,
                                                                          "cone_resistance")

        # Identify the sleeve friction, f_s, column
        col3, df, remaining_cols_to_search = find_col_name_from_substring(df,
                                                                          ["fs", "sleeve", "friction",
                                                                                     "local"],
                                                                          remaining_cols_to_search,
                                                                          "sleeve_friction")
        # Identify the porewater pressure, u2, column
        col4, df, remaining_cols_to_search = find_col_name_from_substring(df,
                                                                          ["dynamic", "u2", "pore","u",
                                                                                    "water"],
                                                                          remaining_cols_to_search,
                                                                          "porewater_pressure")

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

                raise ValueError(f"Missing columns, {' - '.join(final_missing_cols)}")

def load_cpt_csv(file_path: Path) -> pd.DataFrame:
    """
    Load the results of a Cone Penetration Test (CPT) from a CSV file.

    Parameters
    ----------
    file_path : Path
        The path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the relevant CPT data columns:
            depth, cone resistance, sleeve friction, and porewater pressure.

    Raises
    ------
    ValueError
        If the required columns are not found in the CSV file.
    """
    df = pd.read_csv(file_path)
    df.attrs["original_file_name"] = file_path.name

    # Find the depth column
    # First search for a column name containing "m" or "cm"
    candidate_depth_col_names = []
    for col_name in df.columns:
        if isinstance(col_name, str):
            for letter_idx, letter in enumerate(col_name):
                if letter.lower() == "m":

                    if (letter_idx == 0) and (col_name[letter_idx+1] in [" ", "]", ")"]) and (col_name not in candidate_depth_col_names):
                        candidate_depth_col_names.append(col_name)
                        break
                    # including "c" when checking preceeding characters to check for "cm"
                    elif (letter_idx == len(col_name)-1) and (col_name[letter_idx-1] in ["c", " ", "[", "("]) and (col_name not in candidate_depth_col_names):
                        candidate_depth_col_names.append(col_name)
                        break
                    # it's somewhere in the middle
                    else:
                        # including "c" when checking preceeding characters to check for "cm"
                        if (col_name[letter_idx-1] in ["c", " ", "[", "("]) and (col_name[letter_idx+1] in [" ", "]", ")"]) and (col_name not in candidate_depth_col_names):
                            candidate_depth_col_names.append(col_name)
                            break

    # The search for "m" identified the depth column
    if len(candidate_depth_col_names) >= 1:
        col1 = candidate_depth_col_names[0]
        df.attrs["candidate_depth_column_names_in_original_file"] = candidate_depth_col_names
        df.attrs["adopted_depth_column_name_in_original_file"] = col1
        remaining_cols_to_search = list(df.columns)
        remaining_cols_to_search.remove(col1)

    # If no columns with "m" or "cm" are found, look for columns with "depth" or "length" or "h "
    else:
        col1, df, remaining_cols_to_search = find_col_name_from_substring(df,
                                                                          ["depth", "length", "h "],
                                                                          list(df.columns),
                                                                          "depth")

    # Identify the cone resistance, q_c, column
    col2, df, remaining_cols_to_search = find_col_name_from_substring(df,
                                                                      [" q ", "qc","q_c", "cone",
                                                                                 "resistance", "res", "tip"],
                                                                      remaining_cols_to_search,
                                                                      "cone_resistance")

    # Identify the sleeve friction, f_s, column
    col3, df, remaining_cols_to_search = find_col_name_from_substring(df,
                                                                      ["fs", "sleeve", "friction",
                                                                                 "local"],
                                                                      remaining_cols_to_search,
                                                                      "sleeve_friction")
    # Identify the porewater pressure, u2, column
    col4, df, remaining_cols_to_search = find_col_name_from_substring(df,
                                                                      ["dynamic", "u2", "pore","u",
                                                                                "water"],
                                                                      remaining_cols_to_search,
                                                                      "porewater_pressure")

    if (col1 is not None) and (col2 is not None) and (col3 is not None) and (col4 is not None):

            # Return the DataFrame with only the relevant columns
            df = df[[col1, col2, col3, col4]].rename(
                columns={col1: "depth_m", col2: "qc_mpa", col3: "fs_mpa", col4: "u_mpa"})

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

        raise ValueError(f"Missing columns, {' - '.join(missing_cols)}")




