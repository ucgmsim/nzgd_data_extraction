"""
Functions for loading data from the New Zealand Geotechnical Database (NZGD).
"""

from python_ags4 import AGS4
from typing import Union
from pathlib import Path
import pandas as pd
import numpy as np
import pandas

import download_nzgd_data.lib.processing_helpers as processing_helpers

import toml

from download_nzgd_data.lib.processing_helpers import FileProcessingError


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
    for missing_cols_per_sheet in missing_cols_per_sheet:
        if len(missing_cols_per_sheet) < final_num_missing_cols:
            final_num_missing_cols = len(missing_cols_per_sheet)
            final_missing_cols = missing_cols_per_sheet
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

        if target_column_name != "Depth":
            # if the column is in kPa, convert to MPa
            if "kpa" in col.lower():
                df.loc[:, col] /= 1000
        if target_column_name == "Depth":
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


def load_ags(file_path: Path, investigation_type: processing_helpers.InvestigationType) -> pd.DataFrame:
    """
    Load an AGS file.

    Parameters
    ----------
    file_path : Path
        The path to the AGS file.

    Returns
    -------
    pandas.DataFrame
        The CPT data from the AGS file.
    """

    with open(Path(__file__).parent.parent / "resources" / "cpt_column_name_descriptions.toml", "r") as toml_file:
        column_descriptions = toml.load(toml_file)

    try:
        tables, headings = AGS4.AGS4_to_dataframe(file_path)
    except UnboundLocalError:
        ## Found the meaning of this UnboundLocalError by uploading one of these files to the AGS file conversion tool on https://agsapi.bgs.ac.uk
        raise FileProcessingError("ags_duplicate_headers - AGS file contains duplicate headers")

    if len(tables) == 0:
        raise FileProcessingError("no_ags_data_tables - no data tables found in the AGS file")

    required_ags_column_names = ["SCPT_DPTH", "SCPT_RES","SCPT_FRES","SCPT_PWP2"]
    if investigation_type == processing_helpers.InvestigationType.scpt:
        #required_ags_column_names.extend(["SCPT_SWV","SCPT_PWV"])
        # assuming that only the s-wave velocity is required
        required_ags_column_names.extend(["SCPT_SWV"])

    ## Check if any required columns are completely missing from the ags file
    for required_column_name in required_ags_column_names:
        if required_column_name not in tables["SCPT"].columns:
            raise FileProcessingError(f"ags_missing_columns - AGS file is missing {required_column_name} (and possibly other) columns")

    loaded_data_df = pd.DataFrame({
        list(column_descriptions)[0]: tables["SCPT"]["SCPT_DPTH"],
        list(column_descriptions)[1]: tables["SCPT"]["SCPT_RES"],
        list(column_descriptions)[2]: tables["SCPT"]["SCPT_FRES"],
        list(column_descriptions)[3]: tables["SCPT"]["SCPT_PWP2"]
    })

    if ((investigation_type == processing_helpers.InvestigationType.scpt) &
        ("SCPT_SWV" in tables["SCPT"].columns)):
        loaded_data_df[list(column_descriptions)[4]] = tables["SCPT"]["SCPT_SWV"]

    if ((investigation_type == processing_helpers.InvestigationType.scpt) &
        ("SCPT_PWV" in tables["SCPT"].columns)):
        loaded_data_df[list(column_descriptions)[5]] = tables["SCPT"]["SCPT_PWV"]

    ## The first two data rows are skipped as they contain units and the number of decimal places for each column.
    ## For example:
    #     Depth      qc      fs    u
    # 0       m     MPa     MPa  MPa
    # 1     2DP     3DP     4DP  4DP
    loaded_data_df = loaded_data_df.iloc[2:]
    num_numerical_vals = loaded_data_df.map(processing_helpers.can_convert_str_to_float).sum()
    zero_value_columns = num_numerical_vals[num_numerical_vals == 0].index.tolist()
    if len(zero_value_columns) > 0:
        raise FileProcessingError(f"ags_lacking_numeric_data - AGS file has no numeric data in columns [{" ".join(zero_value_columns)}]")

    ## Convert all data to numeric values (dropping rows that contain non-numeric data)
    loaded_data_df = loaded_data_df.apply(pd.to_numeric, errors='coerce').dropna()

    ## Enusre that the depth column is defined as positive (some have depth as negative)
    loaded_data_df[list(column_descriptions)[0]] = np.abs(loaded_data_df[list(column_descriptions)[0]])

    return loaded_data_df


def load_scpt_ags(file_path: Path) -> pd.DataFrame:
    """
    Load an AGS file.

    Parameters
    ----------
    file_path : Path
        The path to the AGS file.

    Returns
    -------
    pandas.DataFrame
        The CPT data from the AGS file.
    """

    with open(Path(__file__).parent.parent / "resources" / "cpt_column_name_descriptions.toml", "r") as toml_file:
        column_descriptions = toml.load(toml_file)

    try:
        tables, headings = AGS4.AGS4_to_dataframe(file_path)
    except UnboundLocalError:
        # Found the meaning of this UnboundLocalError by uploading one of these files to the AGS file conversion tool on https://agsapi.bgs.ac.uk
        raise ValueError("ags_duplicate_headers - AGS file contains duplicate headers")

    if len(tables) == 0:
        raise ValueError("no_ags_data_tables - no data tables found in the AGS file")

    try:
        loaded_data_df = pd.DataFrame({
            list(column_descriptions)[0]: tables["SCPT"]["SCPT_DPTH"],
            list(column_descriptions)[1]: tables["SCPT"]["SCPT_RES"],
            list(column_descriptions)[2]: tables["SCPT"]["SCPT_FRES"],
            list(column_descriptions)[3]: tables["SCPT"]["SCPT_PWP2"]  ## Assuming dynamic pore pressure (u2) in MPa ???
        })
    except(KeyError):
        raise ValueError("ags_missing_columns - AGS file is missing at least one of the required columns")

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

    with open(Path(__file__).parent.parent / "resources" / "cpt_column_name_descriptions.toml", "r") as toml_file:
        column_descriptions = toml.load(toml_file)

    known_special_cases = toml.load(Path(__file__).parent.parent / "resources" / "cpt_column_name_descriptions.toml")
    record_id = f"{file_path.name.split("_")[0]}_{file_path.name.split("_")[1]}"
    if record_id in known_special_cases.keys():
        raise processing_helpers.FileProcessingError(known_special_cases[record_id])

    if file_path.suffix.lower() in [".xls", ".xlsx"]:
        sheet_names, engine = processing_helpers.get_xls_sheet_names(file_path)

        if len(sheet_names) == 0:
            raise processing_helpers.FileProcessingError(f"corrupt_file - cannot detect sheets in file {file_path.name}")

    else:
        # A dummy sheet name as .txt and .csv files do not have sheet names
        sheet_names = ["0"]

    # Iterate through each sheet

    missing_cols_per_sheet = []
    error_text = []
    dataframes_to_return = []

    for sheet_idx, sheet in enumerate(sheet_names):

        if file_path.suffix.lower() in [".csv", ".txt"]:
            df = processing_helpers.load_csv_or_txt(file_path)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet, header=None, engine=engine, parse_dates=False)

        ####################################################################################################################
        ####################################################################################################################

        # Now xls, csv and txt should all be in a dataframe so continue the same for all
        df.attrs["original_file_name"] = file_path.name
        df.attrs["sheet_in_original_file"] = sheet
        df.attrs["column_name_descriptions"] = column_descriptions

        df_for_counting_str_per_row = df.map(lambda x: 1.0 if isinstance(x, (str)) else 0)

        df_nan_to_str = df.fillna("nan")
        df_for_counting_num_of_num = df_nan_to_str.map(lambda x:1.0 if isinstance(x, (int, float)) else 0)

        numeric_surplus_per_col = np.nansum(df_for_counting_num_of_num, axis=0) - np.nansum(df_for_counting_str_per_row, axis=0)

        # Drop any columns that have more text than numeric data
        df = df.iloc[:, numeric_surplus_per_col >= 0]
        numeric_surplus_per_row = np.nansum(df_for_counting_num_of_num, axis=1) - np.nansum(df_for_counting_str_per_row, axis=1)

        header_row_indices = []
        if np.isfinite(processing_helpers.find_one_header_row_from_column_names(df)):
            header_row_indices = processing_helpers.find_all_header_rows(df)

        ## Check the dataframe for various issues
        if df.shape == (0,0):
            error_text.append(f"empty_file - sheet ({sheet.replace('-', '_')}) has size (0,0)")
            continue

        if df.shape[0] == 1:
            error_text.append(f"only_one_line - sheet ({sheet.replace('-', '_')}) has only one line with first cell of {df.iloc[0][0]}")
            continue

        if np.sum(df_for_counting_num_of_num.values) == 0:
            error_text.append(f"no_numeric_data - sheet ({sheet.replace('-', '_')}) has no numeric data")
            continue

        if all(numeric_surplus_per_col < 2):
            error_text.append(
                f"no_data_columns - all columns in sheet ({sheet.replace('-', '_')}) have more text cells than numeric cells")
            continue

        if all(numeric_surplus_per_row < 2):
            error_text.append(
                f"no_data_rows - all rows in sheet ({sheet.replace('-', '_')}) have more text cells than numeric cells")
            continue

        if len(header_row_indices) == 0:
            error_text.append(f"no_header_row - sheet ({sheet.replace('-', '_')}) has no header row")
            continue

        df, header_row_index = processing_helpers.combine_multiple_header_rows(df, header_row_indices)
        # set dataframe's headers/column names. Note that .values is used so that the row's index is not included in the header
        df.columns = df.iloc[header_row_index].values
        # Skip the rows that originally contained the column names as they are now stored as the dataframe header
        df = df.iloc[header_row_index+1:]
        df = df.apply(pd.to_numeric, errors='coerce').astype(float)

        header_row_index = header_row_indices[0] if file_path.suffix.lower() in [".csv", ".txt"] else header_row_index
        df.attrs["header_row_index_in_original_file"] = float(header_row_index)
        df.reset_index(inplace=True, drop=True)
        df, final_col_names = processing_helpers.get_column_names(df)
        df = processing_helpers.convert_to_m_and_mpa(df, final_col_names)

        final_col_names_without_none = [col for col in final_col_names if col is not None]
        if all(i is not None for i in final_col_names) & (len(np.unique(final_col_names_without_none)) == len(final_col_names_without_none)):

            # Return the DataFrame with only the relevant columns
            df = (df[[final_col_names[0], final_col_names[1], final_col_names[2], final_col_names[3]]].rename(
                columns={final_col_names[0]:list(column_descriptions)[0],
                         final_col_names[1]:list(column_descriptions)[1],
                         final_col_names[2]: list(column_descriptions)[2],
                         final_col_names[3]: list(column_descriptions)[3]})).apply(pd.to_numeric, errors='coerce')

            # ensure that the depth column is defined as positive (some have depth as negative)
            df[list(column_descriptions)[0]] = np.abs(df[list(column_descriptions)[0]])
            dataframes_to_return.append(df)

        else:

            if len(np.unique(final_col_names_without_none)) < len(final_col_names_without_none):
                error_text.append(f"non_unique_cols - in sheet ({sheet.replace('-', '_')}) some column names were selected more than once")
                continue

            else:
                missing_cols_per_sheet.append([list(column_descriptions)[idx] for idx, col in enumerate(final_col_names) if col is None])

    ##################################################
    if len(dataframes_to_return) > 0:
        return dataframes_to_return

    final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)
    if len(final_missing_cols) > 0:
        raise ValueError(f"missing_columns - sheet ({sheet.replace('-', '_')}) is missing [{' & '.join(final_missing_cols)}]")

    else:
        raise ValueError(error_text[0])
