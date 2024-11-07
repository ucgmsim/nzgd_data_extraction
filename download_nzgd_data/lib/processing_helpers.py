import enum

import numpy as np
import xlrd
import pandas as pd
import re
import toml
from typing import Union
import copy
import zipfile
from pathlib import Path



class FileProcessingError(Exception):
    """
    Custom exception class for file processing errors.

    This exception is raised when there is an error related to file processing,
    such as issues with reading or parsing files.

    Attributes:
        Inherits all attributes from the base Exception class.
    """
    pass

class InvestigationType(enum.StrEnum):
    """
    Enumeration for different types of investigations.

    Attributes
    ----------
    cpt : str
        Represents a Cone Penetration Test (CPT) investigation type.
    scpt : str
        Represents a Seismic Cone Penetration Test (SCPT) investigation type.
    """
    cpt = "cpt"
    scpt = "scpt"

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

def convert_num_as_str_to_float(val):
    try:
        return float(val)
    except ValueError:
        return val

def find_cell_with_exact_match_in_line(line, character):
    """Return the index of the first cell containing the given character in the given line."""

    for i, cell in enumerate(line):

        if isinstance(cell, str):
            if cell.lower() == character:
                return i


def find_cell_in_line_that_contains_string(line, string):
    """Return the index of the first cell containing the given string in the given line."""
    for i, cell in enumerate(line):

        if isinstance(cell, str):
            if string in cell.lower():
                return i

def find_cells_in_line_that_contains_string(line, string):
    """Return the index of the first cell containing the given string in the given line."""
    indices_to_return = []
    for i, cell in enumerate(line):
        if isinstance(cell, str):
            if string in cell.lower():
                indices_to_return.append(i)
    return indices_to_return


def search_line_for_cell(line, characters, substrings):

    candidates_idx = []

    for character in characters:
         candidates_idx.append(find_cell_with_exact_match_in_line(line, character))

    for substring in substrings:
        substring_cells = find_cells_in_line_that_contains_string(line, substring)
        candidates_idx.extend(substring_cells)

    ## remove None and duplicates
    candidates_idx = sorted(list(set([candidate for candidate in candidates_idx if candidate is not None])))

    return candidates_idx

def search_line_for_all_needed_cells(
        line,
        output_all_candidates=False,
        characters1=["m","w","h","r","cf"],
        substrings1=["depth", "length", "top", "h "],
        characters2=["q","mpa"],
        substrings2 = ["qc", "q_c", "cone", "resistance", "res", "tip"],
        characters3=['mpa'],
        substrings3=["fs", "sleeve", "friction","local"],
        characters4=["u","mpa"],
        substrings4=["u2", "u ", "pore","water","dynamic"]):


    col1_search = search_line_for_cell(line, characters1, substrings1)
    col2_search = search_line_for_cell(line, characters2, substrings2)
    col3_search = search_line_for_cell(line, characters3, substrings3)
    col4_search = search_line_for_cell(line, characters4, substrings4)

    if not output_all_candidates:

        col_idx = np.nan*np.ones(4)

        if len(col1_search) > 0:
            col_idx[0] = col1_search[0]
        if len(col2_search) > 0:
            col_idx[1] = col2_search[0]
        if len(col3_search) > 0:
            col_idx[2] = col3_search[0]
        if len(col4_search) > 0:
            col_idx[3] = col4_search[0]

        return col_idx

    else:
        return col1_search, col2_search, col3_search, col4_search


def check_if_line_is_header(line: Union[pd.Series, list], min_num_) -> bool:

    search_result = search_line_for_all_needed_cells(line)

    return bool(search_result)

def find_one_header_row_from_column_names(iterable):

    # make an array of row indices to check and roll the array such the first searched row 
    # is the one with the higest values of text cells

    if isinstance(iterable, pd.DataFrame):
        iterable = [iterable.iloc[i].to_list() for i in range(len(iterable))]

    num_text_cells_per_line = get_number_of_x_cells_per_line(iterable, NumOrText.TEXT)
    num_numerical_cells_per_line = get_number_of_x_cells_per_line(iterable, NumOrText.NUMERIC)

    text_surplus_per_line = num_text_cells_per_line - num_numerical_cells_per_line

    # roll the array such that the first row to be checked is the one with the highest number of text cells
    # as it is most likely to contain the column names. This will reduce the chance of accidentally choosing the
    # wrong row because it coincidentally contained the key words
    check_rows = np.roll(np.arange(0, len(iterable)-1), -np.argmax(text_surplus_per_line))

    best_partial_header_row = np.nan
    num_cols_in_best_possible_row = 0

    for check_row in check_rows:

        line_check = search_line_for_all_needed_cells(iterable[check_row])

        if (np.sum(np.isfinite(line_check)) >= 4):
            # Found all required columns
            return check_row

        elif (np.sum(np.isfinite(line_check)) >= 1):
            num_cols_in_check_row = np.sum(np.isfinite(line_check))
            if num_cols_in_check_row > num_cols_in_best_possible_row:
                best_partial_header_row = check_row
                num_cols_in_best_possible_row = num_cols_in_check_row

    if best_partial_header_row is not np.nan:
        return best_partial_header_row

    else:
        return np.nan


def convert_numerical_str_cells_to_float(iterable):

    if isinstance(iterable, pd.DataFrame):
        iterable = [iterable.iloc[i].to_list() for i in range(len(iterable))]

    iterable_no_numerical_str = copy.copy(iterable)

    for row_idx, line in enumerate(iterable_no_numerical_str):
        for col_idx, cell in enumerate(line):
            # some cells are read by pd.read_xls() as type datetime so need to convert them to str
            cell = str(cell)
            if can_convert_str_to_float(cell):
                iterable_no_numerical_str[row_idx][col_idx] = float(cell)

    return iterable_no_numerical_str


class NumOrText(enum.StrEnum):
    NUMERIC = enum.auto()
    TEXT = enum.auto()



def get_number_of_x_cells_per_line(iterable:Union[pd.Series, list], x: NumOrText):

    if isinstance(iterable, pd.DataFrame):
        iterable = [iterable.iloc[i].to_list() for i in range(len(iterable))]
    iterable = convert_numerical_str_cells_to_float(iterable)

    num_x_cells_per_line = np.zeros(len(iterable), dtype=int)

    for row_idx, line in enumerate(iterable):

        line = [x for x in line if "nan" not in str(x).lower()]

        if x == NumOrText.TEXT:
            num_x_cells_per_line[row_idx] = np.sum([isinstance(cell, str) for cell in line])
        elif x == NumOrText.NUMERIC:
            num_x_cells_per_line[row_idx] = np.sum([isinstance(cell, (int, float)) for cell in line])

    return num_x_cells_per_line


def find_all_header_rows(iterable):

    if isinstance(iterable, pd.DataFrame):
        iterable = [iterable.iloc[i].to_list() for i in range(len(iterable))]

    num_text_cells_per_line = get_number_of_x_cells_per_line(iterable, NumOrText.TEXT)
    num_numeric_cells_per_line = get_number_of_x_cells_per_line(iterable, NumOrText.NUMERIC)
    text_surplus_per_line = num_text_cells_per_line - num_numeric_cells_per_line

    header_rows = [find_one_header_row_from_column_names(iterable)]
    if (len(header_rows) == 1) & (np.isnan(header_rows[0])):
        return np.array([])

    # see if there are any header rows above the initially identified header row
    num_str_cells_in_header = num_numeric_cells_per_line[header_rows[0]]
    current_row_idx = header_rows[0] - 1
    while current_row_idx > 0:
        if len(iterable[current_row_idx]) != num_str_cells_in_header:
            break
        else:
            header_rows.append(current_row_idx)
        current_row_idx -= 1

    # check rows below
    current_row_idx = header_rows[0]+1
    while current_row_idx < len(iterable)-1:
        if text_surplus_per_line[current_row_idx] > 0:
            header_rows.append(current_row_idx)
        else:
            break
        current_row_idx += 1

    return np.array(sorted(header_rows))



def get_xls_sheet_names(file_path):

    if file_path.suffix.lower() == ".xls":
        engine = "xlrd"
    else:
        engine = "openpyxl"

    # Some .xls files are actually xlsx files and need to be opened with openpyxl
    try:
        sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names
        return sheet_names, engine

    except xlrd.biffh.XLRDError:
        if engine == "xlrd":
            other_engine = "openpyxl"
        else:
            other_engine = "xlrd"

        engine = other_engine
        sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names

        return sheet_names, engine

    except zipfile.BadZipFile:
        raise FileProcessingError(f"bad_zip_file - file {file_path.name} is not a valid xls or xlsx file")

    except xlrd.compdoc.CompDocError:
        raise FileProcessingError(f"corrupt_file - file {file_path.name} has MSAT extension corruption")



def find_encoding(file_path, encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']):

    for encoding in encodings:
        try:
            # open the text file
            with open(file_path, 'r', encoding=encoding) as file:
                # Check that the file can be read with this encoding
                lines = file.readlines()
            return encoding
        except UnicodeDecodeError:
            continue

def get_csv_or_txt_split_readlines(file_path, encoding):

    with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()

        if len(lines) == 1:
            raise FileProcessingError(f"only_one_line - sheet (0) has only one line with first cell of {lines[0]}")

    sep = r"," if file_path.suffix.lower() == ".csv" else r"\s+"

    split_lines = [re.split(sep, line) for line in lines]

    # remove empty strings
    split_lines = [[cell for cell in line if cell != ""] for line in split_lines]

    return split_lines

def get_column_names(df):

    known_false_positive_col_names = toml.load(Path(__file__).parent.parent / "resources" / "known_false_positive_column_names.toml")

    col_index_to_name = {0:"Depth",1:"qc",2:"fs", 3:"u"}

    all_possible_col_indices = search_line_for_all_needed_cells(df.columns,
                                                                output_all_candidates=True)
    final_col_names = []
    for possible_col_idx, possible_col_indices in enumerate(all_possible_col_indices):
        if len(possible_col_indices) == 0:
            final_col_names.append(None)

            if "missing_columns" not in df.attrs:
                df.attrs["missing_columns"] = [col_index_to_name[possible_col_idx]]
            else:
                df.attrs["missing_columns"].append(col_index_to_name[possible_col_idx])

        else:
            possible_col_names = [df.columns[int(idx)] for idx in possible_col_indices]
            candidate_col_name = possible_col_names[0]

            # see if the selected column name is used more than once in the original file
            if len(df[candidate_col_name].shape) > 1:
                raise FileProcessingError(f"repeated_col_names_in_source - sheet has multiple columns with the name {candidate_col_name}")

            num_finite_per_col = np.array([np.sum(np.isfinite(df[col_name])) for col_name in possible_col_names])
            valid_possible_col_names = np.array(possible_col_names)[num_finite_per_col > 0]

            ### Initially set the column name to the first valid column name
            col_name = valid_possible_col_names[0]

            for possible_col_name in valid_possible_col_names:
                ### If another valid column name does not include "clean" or "corrected" then use that column name
                ### instead as the "clean" or "corrected" columns may have been processed such that the
                ### correlations are no longer valid
                if ("clean" not in possible_col_name.lower()) & ("corrected" not in possible_col_name.lower()):
                    col_name = possible_col_name
                    break

            final_col_names.append(col_name)

            df.attrs[f"candidate_{col_index_to_name[possible_col_idx]}_column_names_in_original_file"] = list(valid_possible_col_names)
            df.attrs[f"adopted_{col_index_to_name[possible_col_idx]}_column_name_in_original_file"] = col_name
    ## Check if any of the identified column names are known false positives
    for col_name in final_col_names:
        if col_name in known_false_positive_col_names:
            raise FileProcessingError(f"false_positive_column_name - Using a column named [{col_name}] which is a known "
                                      f"false positive for column [{known_false_positive_col_names[col_name]}]")

    return df, final_col_names

def convert_to_m_and_mpa(df, col_names):

    for col_index, col_name in enumerate(col_names):

        if col_name is not None:

            if col_index == 0:
                # checking the depth column
                if "cm" in col_name.lower():
                    df.loc[:, col_name] /= 100

            else:
                # checking the other columns
                if "kpa" in col_name.lower():
                    df.loc[:, col_name] /= 1000
    return df

def load_csv_or_txt(file_path, sheet="0", col_data_types=np.array(["Depth",
                                                                 "qc",
                                                                 "fs",
                                                                 "u"])):

    sep = r"," if file_path.suffix.lower() == ".csv" else r"\s+"
    file_encoding = find_encoding(file_path)
    split_readlines_iterable = get_csv_or_txt_split_readlines(file_path, file_encoding)
    header_lines_in_csv_or_txt_file = find_all_header_rows(split_readlines_iterable)

    # csv and txt files do not have multiple sheets so just raise an error immediately if no header rows were found
    if len(header_lines_in_csv_or_txt_file) == 0:
        raise FileProcessingError(f"no_header_row - sheet ({sheet.replace("-", "_")}) has no header row")

    if len(header_lines_in_csv_or_txt_file) > 1:
        multi_row_header_array = np.zeros((len(header_lines_in_csv_or_txt_file), 4), dtype=float)
        multi_row_header_array[:] = np.nan
        for header_line_idx, header_line in enumerate(header_lines_in_csv_or_txt_file):
            multi_row_header_array[header_line_idx, :] = search_line_for_all_needed_cells(
                split_readlines_iterable[header_line])
        col_data_type_indices = np.nansum(multi_row_header_array, axis=0)
    else:
        col_data_type_indices = search_line_for_all_needed_cells(
            split_readlines_iterable[header_lines_in_csv_or_txt_file[0]])
    missing_cols = list(col_data_types[~np.isfinite(col_data_type_indices)])

    if len(missing_cols) > 0:
        raise FileProcessingError(
            f"missing_columns - sheet ({sheet.replace('-', '_')}) is missing [{' & '.join(missing_cols)}]")

    needed_col_indices_with_nans = search_line_for_all_needed_cells(
        split_readlines_iterable[header_lines_in_csv_or_txt_file[0]])
    needed_col_indices = [int(col_idx) for col_idx in needed_col_indices_with_nans if np.isfinite(col_idx)]

    df = pd.read_csv(file_path, header=None, encoding=file_encoding, sep=sep,
                     skiprows=header_lines_in_csv_or_txt_file[0], usecols=needed_col_indices).map(
        convert_num_as_str_to_float)

    return df

def combine_multiple_header_rows(df, header_row_indices):

    # take the header_row_index as the maximum of the header_row_indices
    # which is the lowest row in the spreadsheet
    header_row_index = np.max(header_row_indices)

    # copy the column names from the rows above the lowest header row
    df2 = df.copy()
    for row_idx in header_row_indices:
        for col_idx in range(df.shape[1]):
            if row_idx != header_row_index:
                df2.iloc[header_row_index, col_idx] = str(df.iloc[header_row_index, col_idx]) + " " + str(
                    df.iloc[row_idx, col_idx])

    return df2, header_row_index

def change_exception_for_last_sheet(error_category, description, sheet_idx, sheet, sheet_names, final_missing_cols):

    if ((sheet_idx == len(sheet_names) - 1) & len(final_missing_cols) > 0):
        raise FileProcessingError(
            f"missing_columns - sheet ({sheet.replace('-', '_')}) is missing [{' & '.join(final_missing_cols)}]")
    elif ((sheet_idx == len(sheet_names) - 1) & len(final_missing_cols) == 0):
        raise FileProcessingError(f"{error_category} - sheet ({sheet.replace("-", "_")}) {description}")

def make_summary_df(summary_df, record_dir_name, file_was_loaded, loaded_file_type,
                      loaded_file_name, pdf_file_list, cpt_file_list, ags_file_list, xls_file_list,
                      xlsx_file_list, csv_file_list, txt_file_list, unknown_list):
    if ((len(pdf_file_list) > 0) & (len(cpt_file_list) == 0) &
            (len(ags_file_list) == 0) & (len(xls_file_list) == 0) &
            (len(xlsx_file_list) == 0) & (len(csv_file_list) == 0) &
            (len(txt_file_list) == 0) & (len(unknown_list) == 0)):
        has_only_pdf = True
    else:
        has_only_pdf = False

    concat_df = pd.concat([summary_df,
                           pd.DataFrame({"record_name": [record_dir_name],
                                         "file_was_loaded": [file_was_loaded],
                                         "loaded_file_type": [loaded_file_type],
                                         "loaded_file_name": [loaded_file_name],
                                         "only_has_pdf" : [has_only_pdf],
                                         "num_pdf_files": [len(pdf_file_list)],
                                         "num_cpt_files": [len(cpt_file_list)],
                                         "num_ags_files": [len(ags_file_list)],
                                         "num_xls_files": [len(xls_file_list)],
                                         "num_xlsx_files": [len(xlsx_file_list)],
                                         "num_csv_files": [len(csv_file_list)],
                                         "num_txt_files": [len(txt_file_list)],
                                         "num_other_files": [len(unknown_list)]})],
                          ignore_index=True)
    return concat_df

def nth_highest_value(array, n):
    """
    Find the nth highest value in an array.

    Parameters
    ----------
    array : np.array
        The input array.
    n : int
        The value of n.

    Returns
    -------
    float
        The nth highest value in the array.
    """

    sorted_array = np.sort(array)

    return sorted_array[-n]

def final_check_for_wrong_units_and_negative_values(df, cm_threshold = 50, kpa_threshold = 100, nth_highest = 5):

    with open(Path(__file__).parent.parent / "resources" / "cpt_column_name_descriptions.toml", "r") as toml_file:
        column_descriptions = toml.load(toml_file)

    if nth_highest_value(df[list(column_descriptions)[0]].values, nth_highest) > cm_threshold:
        df[list(column_descriptions)[0]] /= 100

    if nth_highest_value(df[list(column_descriptions)[1]].values, nth_highest) > kpa_threshold:
        df[list(column_descriptions)[1]] /= 1000
    if nth_highest_value(df[list(column_descriptions)[2]].values, nth_highest) > kpa_threshold:
        df[list(column_descriptions)[2]] /= 1000
    if nth_highest_value(df[list(column_descriptions)[3]].values, nth_highest) > kpa_threshold:
        df[list(column_descriptions)[3]] /= 1000

    ## ensure that the depth column is defined as positive (some have depth as negative)
    df[list(column_descriptions)[0]] = np.abs(df[list(column_descriptions)[0]])

    ## Drop any rows of df that have negative values in columns qc or fs
    df = df[(df[list(column_descriptions)[1]] >= 0) & (df[list(column_descriptions)[2]] >= 0)]

    return df




