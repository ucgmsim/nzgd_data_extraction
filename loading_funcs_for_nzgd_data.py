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

    try:
        tables, headings = AGS4.AGS4_to_dataframe(file_path)
    except(UnboundLocalError):
        # Found the meaning of this UnboundLocalError by uploading one of these files to the AGS file conversion tool on https://agsapi.bgs.ac.uk
        raise ValueError("ags_duplicate_headers - AGS file contains duplicate headers")

    if len(tables) == 0:
        raise ValueError("no_ags_data_tables - no data tables found in the AGS file")

    loaded_data_df = pd.DataFrame({
        "depth_m": tables["SCPT"]["SCPT_DPTH"],
        "qc_mpa": tables["SCPT"]["SCPT_RES"],
        "fs_mpa": tables["SCPT"]["SCPT_FRES"],
        "u_mpa": tables["SCPT"]["SCPT_PWP2"]  ## Assuming dynamic pore pressure (u2) in MPa ???
    })

    ### The first two rows are dropped as they contain header information from the ags file
    return loaded_data_df.apply(pd.to_numeric, errors='coerce').dropna()


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

    if file_path.suffix.lower() in [".csv", ".txt"]:
        # A dummy variable to allow the function to be called with a csv file which do not consist of multiple sheets
        sheet_names = [0]

    else:

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
        if file_path.suffix.lower() == ".csv":

            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

            for encoding in encodings:
                try:
                    # open the text file
                    with open(file_path, 'r', encoding=encoding) as file:
                        lines = file.readlines()

                    num_cols_per_line = [len(line.split(",")) for line in lines]
                    num_rows_to_skip = len(np.where(num_cols_per_line < np.max(num_cols_per_line))[0])

                    df = pd.read_csv(file_path, header=None, encoding=encoding, skiprows=num_rows_to_skip).map(convert_num_as_str_to_float)
                    break
                except UnicodeDecodeError:
                    continue


        elif file_path.suffix.lower() == ".txt":
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        lines = file.readlines()
                        break
                except:
                    continue

            split_lines = []

            for line in lines:
                split_lines.append(re.split(r"\s+", line))

            for line in split_lines:
                for string in line:
                    if string == "":
                        line.remove(string)


            split_lines_string_check = copy.deepcopy(split_lines)
            split_lines_float_check = copy.deepcopy(split_lines)

            for line in split_lines_string_check:
                for idx, string in enumerate(line):
                    line[idx] = str_cannot_become_float(string)

            for line in split_lines_float_check:
                for idx, string in enumerate(line):
                    line[idx] = can_convert_str_to_float(string)

            num_str_cell_per_row = np.array([np.sum(line) for line in split_lines_string_check])
            num_float_cell_per_row = np.array([np.sum(line) for line in split_lines_float_check])

            col_name_row_idx = np.argmax(num_str_cell_per_row)

            if len(num_float_cell_per_row[num_float_cell_per_row >= 4]) < min_num_numerical_rows_with_at_least_4_cols:
                raise ValueError(
                    f"lacking_data_rows - sheet ({sheet}) has fewer than {min_num_numerical_rows_with_at_least_4_cols} data rows with at least four numerical columns")

            if len(num_float_cell_per_row[num_float_cell_per_row >= 4]) == 0:
                raise ValueError(
                    f"no_data_rows - sheet ({sheet}) has 0 data rows with at least four numerical columns")

            if num_str_cell_per_row[col_name_row_idx] == num_str_cell_per_row[col_name_row_idx + 1]:
                col_name_row_idx += 1

            num_cols_to_use = len(split_lines[col_name_row_idx])

            df = pd.read_csv(file_path, header=None, encoding=encoding, sep=r"\s+",
                        skiprows=col_name_row_idx, usecols=np.arange(0, num_cols_to_use)).map(convert_num_as_str_to_float)

        ### This else statement is to load the xls and xlsx files
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
                raise ValueError(f"empty_file - sheet ({sheet}) has size (0,0)")
            else:
                # There are more sheets to check so continue to next sheet
                continue

        if num_data_rows_with_at_least_4_cols < min_num_numerical_rows_with_at_least_4_cols:
            # There are not enough rows with numerical data in all required columns

            if sheet_idx == len(sheet_names) - 1:
                # no other sheets so raise error for this file
                final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)
                if len(final_missing_cols) > 0:
                    raise ValueError(f"missing_columns - sheet ({sheet}) is missing [{' & '.join(final_missing_cols)}]")
                raise ValueError(f"lacking_data_rows - sheet ({sheet}) has fewer than {min_num_numerical_rows_with_at_least_4_cols} data rows with at least four numerical columns")

            else:
                # There are more sheets to check so continue to next sheet
                continue

        last_data_row = np.where(num_num_per_row >= 4)[0][-1]

        num_num_in_last_data_row = num_num_per_row[last_data_row]

        # some files have a lot of numbers in the rows to skip so now find the first row with at least the same number
        # of numbers as the last data row. This assumes that all extra information is before the data.
        first_data_row = np.where(num_num_per_row >= num_num_in_last_data_row)[0][0]

        col_name_rows = []
        found_a_header_row = False

        max_num_rows_to_check_for_header = 4
        num_rows_checked_for_header = 0

        if file_path.suffix.lower() != ".txt":

            while num_rows_checked_for_header <= max_num_rows_to_check_for_header:
                check_row = first_data_row - num_rows_checked_for_header - 1
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

        else:
            col_name_rows = np.where(num_num_per_row == 0)[0]

        if len(col_name_rows) == 0:
            raise ValueError(f"no_header_row - sheet ({sheet}) has no header row")

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
        if file_path.suffix.lower() == ".csv":
            df.attrs["header_row_index_in_original_file"] = float(num_rows_to_skip) # this is the index of the header row which is the same as the preceeding number of rows to skip
        if file_path.suffix.lower() == ".txt":
            df.attrs["header_row_index_in_original_file"] = float(col_name_row_idx) # this is the index of the header row which is the same as the preceeding number of rows to skip
        else:
            df.attrs["header_row_index_in_original_file"] = float(col_name_row)
        # reset the index so that the first row is index 0
        df.reset_index(inplace=True, drop=True)

        # Find the depth column
        # First search for a column name containing "m" or "cm"
        candidate_depth_col_names = []
        for col_name in df.columns:
            if isinstance(col_name, str):
                if len(col_name) == 1:
                    if col_name.lower() == "m":
                        candidate_depth_col_names.append(col_name)
                        break

                for letter_idx, letter in enumerate(col_name):
                    if letter.lower() == "m":

                        # if the letter is the first character
                        if (letter_idx == 0) and (col_name[letter_idx+1] in [" ", "]", ")"]) and (col_name not in candidate_depth_col_names):
                            candidate_depth_col_names.append(col_name)
                            break
                        # if the letter is the last character
                        # including "c" when checking preceeding characters to check for "cm"
                        elif (letter_idx == len(col_name)-1) and (col_name[letter_idx-1] in ["c", " ", "[", "("]) and (col_name not in candidate_depth_col_names):
                            candidate_depth_col_names.append(col_name)
                            break
                        # the letter is somewhere in the middle
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

                raise ValueError(f"missing_columns - sheet ({sheet}) is missing [{' & '.join(final_missing_cols)}]")