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
        # check for "Clean" which is sometimes used for a cleaned version of the same data
        if len(candidate_col_names) > 1:
            col = None
            for candidate_name in candidate_col_names:
                if "clean" in candidate_name.lower():
                    col = candidate_name
                    break
            if col is None:
                col = candidate_col_names[0]


        else:
            col = candidate_col_names[0]

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
    else:
        engine = "calamine"

    # Load all sheet names in the file
    sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names

    missing_cols_per_sheet = []

    # Iterate through each sheet
    for sheet_idx, sheet in enumerate(sheet_names):

        # Load the entire sheet without specifying headers
        df = pd.read_excel(file_path, sheet_name=sheet, header=None, engine=engine)
        if df.shape[1] < 4:
            # There are not enough columns to contain the required data so continue to the next sheet
            continue

        df.attrs["original_file_name"] = file_path.name
        df.attrs["source_sheet_in_original_file"] = sheet

        df_float_cell_to_1 = df.map(lambda x: np.nan if isinstance(x, (str)) else 1)
        num_data_cols = scipy.stats.mode(np.sum(df_float_cell_to_1, axis=1)).mode

        df_str_cell_to_1 = df.map(lambda x: np.nan if isinstance(x, (int, float)) else 1)

        num_text_cells_per_row = np.nansum(df_str_cell_to_1, axis=1)
        col_name_rows = np.where(num_text_cells_per_row >= num_data_cols)[0]

        # if there are multiple header rows, combine them into one
        if len(col_name_rows) > 1:
            col_name_row = np.max(col_name_rows)
            # copy the column names from the rows above the header row
            df2 = df.copy()
            for row_idx in col_name_rows[0:col_name_row]:
                for col_idx in range(df.shape[1]):

                    # up to but not including the header_row which is taken as the lowest
                    df2.iloc[col_name_row, col_idx] = str(df.iloc[col_name_row,col_idx]) + " " + str(df.iloc[row_idx,col_idx])

            df = df2.copy()

        # If there is only one header row, take it as the header row
        else:
            col_name_row = col_name_rows[0]



        # set dataframe's headers/column names. Note that .values is used so that the row's index is not included in the header
        df.columns = df.iloc[col_name_row].values
        # Skip the rows that originally contained the column names as they are now stored as the dataframe header
        df = df.iloc[col_name_row+1:]
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

            missing_cols_per_sheet.append(missing_cols)

            if sheet_idx < len(sheet_names) - 1:
                # There are more sheets to check so continue onto the next sheet
                continue

            else:
                # There are no more sheets to check so return the missing columns
                final_num_missing_cols = 5
                final_missing_cols = []
                for missing_cols in missing_cols_per_sheet:
                    if len(missing_cols) < final_num_missing_cols:
                        final_num_missing_cols = len(missing_cols)
                        final_missing_cols = missing_cols

                raise ValueError(f"Missing columns: {' ;'.join(final_missing_cols)}")
