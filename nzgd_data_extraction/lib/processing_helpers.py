"""
Classes and functions to help process cone penetration test (CPT) data
from the New Zealand Geotechnical database (NZGD).
"""

import copy
import enum
import re
import zipfile
from pathlib import Path
from typing import Literal, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import toml
import xlrd


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


def convert_dataframe_to_list_of_lists(df: pd.DataFrame) -> list[list[str]]:
    """
    Convert a DataFrame to a list of lists.

    This function converts a DataFrame to a list of lists, where each list represents a row in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert.

    Returns
    -------
    list[list[str]]
        A list of lists, where each list represents a row in the DataFrame.
    """

    lines_and_cells_iterable = [df.iloc[i].to_list() for i in range(len(df))]

    return lines_and_cells_iterable


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


def convert_num_as_str_to_float(val: str) -> Union[float, str]:
    """
    Convert a numerical string to a float.

    This function attempts to convert a string to a float. If the conversion fails, it returns the original string.

    Parameters
    ----------
    val : str
        The string to convert.

    Returns
    -------
    Union[float, str]
        The converted float if the string can be converted, otherwise the original string.
    """

    try:
        return float(val)
    except ValueError:
        return val


def find_cell_with_exact_match_in_line(line: list[str], character: str) -> int:
    """
    Find the index of the first cell in a line that exactly matches a given character.

    This function iterates through a list of cells and returns the index of the first cell that exactly matches
    the specified character, ignoring case.

    Parameters
    ----------
    line : list[str]
        The line of cells to search through.
    character : str
        The character to match exactly.

    Returns
    -------
    int
        The index of the first cell that matches the given character, or None if no match is found.
    """
    for i, cell in enumerate(line):

        if isinstance(cell, str):
            if cell.lower() == character:
                return i


def find_cell_in_line_that_contains_string(line: list[str], string: str) -> int:
    """
    Find the index of the first cell in a line that contains a given string.

    This function iterates through a list of cells and returns the index of the first cell that contains
    the specified string, ignoring case.

    Parameters
    ----------
    line : list[str]
        The line of cells to search through.
    string : str
        The string to search for within the cells.

    Returns
    -------
    int
        The index of the first cell that contains the given string, or None if no match is found.
    """
    for i, cell in enumerate(line):

        if isinstance(cell, str):
            if string in cell.lower():
                return i


def find_cells_in_line_that_contains_string(line: list[str], string: str) -> list:
    """
    Find the indices of all cells in a line that contain a given string.

    This function iterates through a list of cells and returns the indices of all cells that contain
    the specified string, ignoring case.

    Parameters
    ----------
    line : list[str]
        The line of cells to search through.
    string : str
        The string to search for within the cells.

    Returns
    -------
    list
        A list of indices of cells that contain the given string.
    """
    indices_to_return = []
    for i, cell in enumerate(line):
        if isinstance(cell, str):
            if string in cell.lower():
                indices_to_return.append(i)
    return indices_to_return


def search_line_for_cell(
    line: list[str], characters: tuple[str], substrings: tuple[str]
) -> list[int]:
    """
    Search for cells in a line that match given characters or substrings.

    This function searches through the cells in a line and finds the indices of cells that either exactly match any of the
    given characters or contain any of the given substrings.

    Parameters
    ----------
    line : list[str]
        The line of cells to search through.
    characters : list[str]
        The list of characters to match exactly.
    substrings : list[str]
        The list of substrings to search for within the cells.

    Returns
    -------
    list[int]
        A list of indices of cells that match the given characters or contain the given substrings.
    """
    candidates_idx = []

    for character in characters:
        candidates_idx.append(find_cell_with_exact_match_in_line(line, character))

    for substring in substrings:
        substring_cells = find_cells_in_line_that_contains_string(line, substring)
        candidates_idx.extend(substring_cells)

    ## remove None and duplicates
    candidates_idx = sorted(
        list(set([candidate for candidate in candidates_idx if candidate is not None]))
    )

    return candidates_idx


def search_line_for_all_needed_cells(
    line: list,
    output_all_candidates: bool = False,
    characters1: tuple[str] = ("m", "w", "h", "r", "cf"),
    substrings1: tuple[str] = ("depth", "length", "top", "h "),
    characters2: tuple[str] = ("q", "mpa"),
    substrings2: tuple[str] = ("qc", "q_c", "cone", "resistance", "res", "tip"),
    characters3: tuple[str] = ("mpa",),
    substrings3: tuple[str] = ("fs", "sleeve", "friction", "local"),
    characters4: tuple[str] = ("u", "mpa"),
    substrings4: tuple[str] = ("u2", "u ", "pore", "water", "dynamic"),
) -> Union[npt.ArrayLike, tuple[npt.ArrayLike]]:
    """
    Search an iterable line for all needed cells.

    This function searches through the cells in a line and finds the indices of cells that either exactly match any
    of the given characters or contain any of the given substrings. The function returns the indices of the first
    cell that matches each of the given characters or substrings. If the output_all_candidates parameter is set to
    True, the function returns all the indices of cells that match the given characters or contain the given substrings.

    Parameters
    ----------
    line : list
        The line of cells to search through.
    output_all_candidates : bool, optional
        If True, return all candidate indices. Default is False.
    characters1 : tuple[str], optional
        The list of characters to match exactly for the first set of cells. Default is ("m","w","h","r","cf").
    substrings1 : tuple[str], optional
        The list of substrings to search for within the cells for the first set of cells. Default is ("depth", "length", "top", "h ").
    characters2 : tuple[str], optional
        The list of characters to match exactly for the second set of cells. Default is ("q","mpa").
    substrings2 : tuple[str], optional
        The list of substrings to search for within the cells for the second set of cells. Default is ("qc", "q_c", "cone", "resistance", "res", "tip").
    characters3 : tuple[str], optional
        The list of characters to match exactly for the third set of cells. Default is ('mpa',).
    substrings3 : tuple[str], optional
        The list of substrings to search for within the cells for the third set of cells. Default is ("fs", "sleeve", "friction","local").
    characters4 : tuple[str], optional
        The list of characters to match exactly for the fourth set of cells. Default is ("u","mpa").
    substrings4 : tuple[str], optional
        The list of substrings to search for within the cells for the fourth set of cells. Default is ("u2", "u ", "pore","water","dynamic").

    Returns
    -------
    Union[npt.ArrayLike, tuple[npt.ArrayLike]]
        The indices of the first cell that matches each of the given characters or substrings,
        or all candidate indices if output_all_candidates is True.

    Example output:
    ([0], [1, 8, 12, 14, 16, 17, 18], [2, 23, 29], [3, 4])
    which indicates that for Depth (position 0 in this list), column 0 (the first column) is the only
    possiblilty for the Depth column. For qc (position 1 in this list), columns 1, 8, 12, 14, 16, 17, and 18
    are all possible candidates for the qc column. For fs (position 2 in this list), columns 2, 23, and 29
    are all possible candidates for the fs column. For u2 (position 3 in this list), columns 3 and 4 are
    all possible candidates for the u2 column.
    """

    col1_search = search_line_for_cell(line, characters1, substrings1)
    col2_search = search_line_for_cell(line, characters2, substrings2)
    col3_search = search_line_for_cell(line, characters3, substrings3)
    col4_search = search_line_for_cell(line, characters4, substrings4)

    if not output_all_candidates:

        col_idx = np.nan * np.ones(4)

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


def find_one_header_row_index_from_column_names(
    lines_and_cells_iterable: Union[pd.Series, list]
) -> Union[int, float]:
    """
    Find the one header row index from column names in lines_and_cells_iterable.

    This function searches for the line (row index) that most likely contains the column names.

    Parameters
    ----------
    lines_and_cells_iterable : Union[pd.Series, list[list[str]]]
        The input data as a list of lists or a DataFrame.

    Returns
    -------
    float
        The index of the header row if found, otherwise NaN.
    """
    ## make an array of row indices to check and roll the array such the first searched row
    ## is the one with the highest values of text cells

    if isinstance(lines_and_cells_iterable, pd.DataFrame):
        lines_and_cells_iterable = convert_dataframe_to_list_of_lists(
            lines_and_cells_iterable
        )

    num_text_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        lines_and_cells_iterable, NumOrText.TEXT
    )
    num_numerical_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        lines_and_cells_iterable, NumOrText.NUMERIC
    )

    text_surplus_per_line = num_text_cells_per_line - num_numerical_cells_per_line

    ## roll the array such that the first row to be checked is the one with the highest number of text cells
    ## as it is most likely to contain the column names. This will reduce the chance of accidentally choosing the
    ## wrong row because it coincidentally contained the keywords
    check_rows = np.roll(
        np.arange(0, len(lines_and_cells_iterable) - 1),
        -np.argmax(text_surplus_per_line),
    )
    best_partial_header_row = np.nan
    num_cols_in_best_possible_row = 0

    for check_row in check_rows:

        line_check = search_line_for_all_needed_cells(
            lines_and_cells_iterable[check_row]
        )

        if np.sum(np.isfinite(line_check)) >= 4:
            # Found all required columns
            return check_row

        elif np.sum(np.isfinite(line_check)) >= 1:
            num_cols_in_check_row = np.sum(np.isfinite(line_check))
            if num_cols_in_check_row > num_cols_in_best_possible_row:
                best_partial_header_row = check_row
                num_cols_in_best_possible_row = num_cols_in_check_row

    if best_partial_header_row is not np.nan:
        return best_partial_header_row

    else:
        return np.nan


def convert_numerical_str_cells_to_float(
    iterable: Union[pd.Series, list]
) -> list[list]:
    """
    Convert numerical string cells to float in an iterable.

    This function iterates through each cell in the provided iterable and converts cells that are numerical strings to
    floats.

    Parameters
    ----------
    iterable : Union[pd.Series, list]
        The input data as a list of lists or a DataFrame.

    Returns
    -------
    list[list]
        The modified iterable with numerical strings converted to floats.
    """
    if isinstance(iterable, pd.DataFrame):
        iterable = [iterable.iloc[i].to_list() for i in range(len(iterable))]

    iterable_no_numerical_str = copy.copy(iterable)

    for row_idx, line in enumerate(iterable_no_numerical_str):
        for col_idx, cell in enumerate(line):
            ## some cells are read by pd.read_xls() as type datetime so they need to be converted to str
            cell = str(cell)
            if can_convert_str_to_float(cell):
                iterable_no_numerical_str[row_idx][col_idx] = float(cell)

    return iterable_no_numerical_str


class NumOrText(enum.StrEnum):
    """
    String enumeration for specifying whether cells are numeric (contain floats or ints) or text
    (contain strings).

    Attributes
    ----------
    NUMERIC : auto
        Represents numeric cells.
    TEXT : auto
        Represents text cells.
    """

    NUMERIC = enum.auto()
    TEXT = enum.auto()


def get_number_of_numeric_or_text_cells_per_line(
    iterable: Union[pd.Series, list], numeric_or_text: NumOrText
) -> npt.NDArray[np.int_]:
    """
    Get the number of numeric or text cells per line in an iterable.

    This function iterates through each line in the provided iterable and counts the number of cells that are either
    numeric or text, based on the specified type.

    Parameters
    ----------
    iterable : Union[pd.Series, list]
        The input data as a list of lists or a DataFrame.
    numeric_or_text : NumOrText
        The type of cells to count (numeric or text).

    Returns
    -------
    npt.NDArray[np.int_]
        An array containing the count of numeric or text cells for each line.
    """

    if isinstance(iterable, pd.DataFrame):
        iterable = [iterable.iloc[i].to_list() for i in range(len(iterable))]
    iterable = convert_numerical_str_cells_to_float(iterable)

    num_x_cells_per_line = np.zeros(len(iterable), dtype=int)

    for row_idx, line in enumerate(iterable):

        line = [
            numeric_or_text
            for numeric_or_text in line
            if "nan" not in str(numeric_or_text).lower()
        ]

        if numeric_or_text == NumOrText.TEXT:
            num_x_cells_per_line[row_idx] = np.sum(
                [isinstance(cell, str) for cell in line]
            )
        elif numeric_or_text == NumOrText.NUMERIC:
            num_x_cells_per_line[row_idx] = np.sum(
                [isinstance(cell, (int, float)) for cell in line]
            )

    return num_x_cells_per_line


def find_row_indices_of_header_lines(
    lines_and_cells_iterable: Union[pd.DataFrame, list[list[str]]]
) -> list:
    """
    Find the row indices of the header lines.

    Parameters
    ----------
    lines_and_cells_iterable : Union[pd.DataFrame, list]
        The input data as a list of lists containing strings or a DataFrame.

    Returns
    -------
    np.ndarray
        An array containing the row index of each header row.
    """

    if isinstance(lines_and_cells_iterable, pd.DataFrame):
        lines_and_cells_iterable = convert_dataframe_to_list_of_lists(
            lines_and_cells_iterable
        )

    num_text_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        lines_and_cells_iterable, NumOrText.TEXT
    )
    num_numeric_cells_per_line = get_number_of_numeric_or_text_cells_per_line(
        lines_and_cells_iterable, NumOrText.NUMERIC
    )
    text_surplus_per_line = num_text_cells_per_line - num_numeric_cells_per_line
    header_rows = [
        find_one_header_row_index_from_column_names(lines_and_cells_iterable)
    ]
    if (len(header_rows) == 1) & (np.isnan(header_rows[0])):
        return []

    ## Check for header rows above the one first identified
    num_str_cells_in_header = num_numeric_cells_per_line[header_rows[0]]
    current_row_idx = header_rows[0] - 1
    while current_row_idx > 0:
        ## break the while loop if the number of text cells in the current row is 0 or the number of text cells in
        ## the current row is not equal to the number of text cells in the header
        if (
            len(lines_and_cells_iterable[current_row_idx]) != num_str_cells_in_header
        ) or (text_surplus_per_line[current_row_idx] == 0):
            break
        else:
            header_rows.append(int(current_row_idx))
        current_row_idx -= 1

    ## Check for header rows below the one first identified
    current_row_idx = header_rows[0] + 1
    while current_row_idx < len(lines_and_cells_iterable) - 1:
        if text_surplus_per_line[current_row_idx] > 0:
            header_rows.append(int(current_row_idx))
        else:
            break
        current_row_idx += 1

    return sorted(header_rows)


def get_xls_sheet_names(
    file_path: Path,
) -> tuple[list[str], Literal["xlrd", "openpyxl"]]:
    """
    Get the sheet names from an Excel file and determine the engine used to read the file.

    This function attempts to read the sheet names from an Excel file using the xlrd and openpyxl engines.
    If the file is not a valid .xls or .xlsx file, it raises a FileProcessingError.

    Parameters
    ----------
    file_path : Path
        The path to the Excel file.

    Returns
    -------
    tuple[list[str], str]
        A tuple containing a list of sheet names and the engine used to read the file.

    Raises
    ------
    FileProcessingError
        If the file is not a valid .xls or .xlsx file.
    """
    if file_path.suffix.lower() == ".xls":
        engine: Literal["xlrd"] = "xlrd"
    else:
        engine: Literal["openpyxl"] = "openpyxl"

    # Some .xls files are actually xlsx files and need to be opened with openpyxl
    try:
        sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names
        return sheet_names, engine

    except (xlrd.biffh.XLRDError, TypeError):
        if engine == "xlrd":
            other_engine: Literal["openpyxl"] = "openpyxl"
        else:
            other_engine: Literal["xlrd"] = "xlrd"

        engine: Literal["xlrd", "openpyxl"] = other_engine
        try:
            sheet_names = pd.ExcelFile(file_path, engine=engine).sheet_names
        except:
            raise FileProcessingError(
                f"bad_xls_or_xlsx_file - file {file_path.name} is not a valid xls or xlsx file"
            )

        return sheet_names, engine

    except zipfile.BadZipFile:
        raise FileProcessingError(
            f"bad_xls_or_xlsx_file - file {file_path.name} is not a valid xls or xlsx file"
        )

    except xlrd.compdoc.CompDocError:
        raise FileProcessingError(
            f"corrupt_file - file {file_path.name} has MSAT extension corruption"
        )


def find_encoding(
    file_path: Path, encodings: tuple[str] = ("utf-8", "latin1", "iso-8859-1", "cp1252")
) -> str:
    """
    Determine the encoding of a text file from a list of possible encodings.

    This function attempts to open and read the file using each encoding in the provided list.
    If the file can be read without a UnicodeDecodeError, the encoding is returned.

    Parameters
    ----------
    file_path : Path
        The path to the text file whose encoding is to be determined.
    encodings : tuple[str], optional
        A list of possible encodings to try. Default is ('utf-8', 'latin1', 'iso-8859-1', 'cp1252').

    Returns
    -------
    str
        The encoding that successfully reads the file without errors.
    """

    for encoding in encodings:
        try:
            # open the text file
            with open(file_path, "r", encoding=encoding) as file:
                # Check that the file can be read with this encoding
                _ = file.readlines()
            return encoding
        except UnicodeDecodeError:
            continue


def get_csv_or_txt_split_readlines(file_path: Path, encoding: str) -> list[list[str]]:
    """

    Loads a csv or txt file as a list of lists of strings. Each list represents a line in the file. Each item in the
    list represents a cell in the line.

    Parameters
    ----------
    file_path : Path
        The path to the file to be read.
    encoding : str
        The encoding to use when reading the file.

    Returns
    -------
    list[list[str]]
        A list of lists containing strings. Each list represents a line in the file. Each item in the list represents a cell in
        the line.

    Raises
    ------
    FileProcessingError
        If the file contains only one line.
    """
    with open(file_path, "r", encoding=encoding) as file:
        lines = file.readlines()

        if len(lines) == 1:
            raise FileProcessingError(
                f"only_one_line - sheet (0) has only one line with first cell of {lines[0]}"
            )

    sep = r"," if file_path.suffix.lower() == ".csv" else r"\s+"

    split_lines = [re.split(sep, line) for line in lines]

    # remove empty strings
    split_lines = [[cell for cell in line if cell != ""] for line in split_lines]

    return split_lines


def get_column_names(loaded_data_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Finds names of the required columns in the DataFrame.

    Parameters
    ----------
    loaded_data_df : pd.DataFrame
        The DataFrame containing the data with potential column names.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A tuple containing the DataFrame with updated attributes and a list of final column names.

    Raises
    ------
    FileProcessingError
        If a column name is repeated or if a known false positive column name is used.
    """

    known_false_positive_col_names = toml.load(
        Path(__file__).parent.parent
        / "resources"
        / "known_false_positive_column_names.toml"
    )

    col_index_to_name = {0: "Depth", 1: "qc", 2: "fs", 3: "u"}

    all_possible_col_indices = search_line_for_all_needed_cells(
        loaded_data_df.columns, output_all_candidates=True
    )

    final_col_names = []
    for possible_col_idx, possible_col_indices in enumerate(all_possible_col_indices):
        if len(possible_col_indices) == 0:
            final_col_names.append(None)

            if "missing_columns" not in loaded_data_df.attrs:
                loaded_data_df.attrs["missing_columns"] = [
                    col_index_to_name[possible_col_idx]
                ]
            else:
                loaded_data_df.attrs["missing_columns"].append(
                    col_index_to_name[possible_col_idx]
                )

        else:
            possible_col_names = [
                loaded_data_df.columns[int(idx)] for idx in possible_col_indices
            ]
            candidate_col_name = possible_col_names[0]

            ## Check that the selected column name is used only once
            if len(loaded_data_df[candidate_col_name].shape) > 1:
                raise FileProcessingError(
                    f"repeated_col_names_in_source - sheet has multiple columns with the name {candidate_col_name}"
                )
            ## For every possible column name, check how many finite values are in the column
            num_finite_per_col_list = []
            for col_name in possible_col_names:
                data_col = loaded_data_df[col_name]
                ## If the shape of data col is like (x, y), instead of just (x,) then there are multiple columns
                # with the same name so raise an error
                if len(data_col.shape) > 1:
                    raise FileProcessingError(
                        f"repeated_col_names_in_source - sheet has multiple columns with the name {col_name}"
                    )
                finite_data_col = np.isfinite(data_col)
                num_finite = np.sum(finite_data_col)
                num_finite_per_col_list.append(num_finite)

            num_finite_per_col = np.array(num_finite_per_col_list)

            ## Valid possible column names will have at least one finite value
            valid_possible_col_names = np.array(possible_col_names)[
                num_finite_per_col > 0
            ]

            ## Initially set the column name to the first valid column name
            if len(valid_possible_col_names) == 0:
                raise FileProcessingError(
                    f"no_valid_column_names - sheet has no valid column names for column {col_index_to_name[possible_col_idx]}"
                )
            col_name = valid_possible_col_names[0]

            for possible_col_name in valid_possible_col_names:
                ## If another valid column name does not include "clean" or "corrected" then use that column name
                ## instead as the "clean" or "corrected" columns may have been processed such that the
                ## correlations are no longer valid
                if ("clean" not in possible_col_name.lower()) & (
                    "corrected" not in possible_col_name.lower()
                ):
                    col_name = possible_col_name
                    break

            final_col_names.append(col_name)

            loaded_data_df.attrs[
                f"candidate_{col_index_to_name[possible_col_idx]}_column_names_in_original_file"
            ] = list(valid_possible_col_names)
            loaded_data_df.attrs[
                f"adopted_{col_index_to_name[possible_col_idx]}_column_name_in_original_file"
            ] = col_name

    ## Check if any of the identified column names are known false positives
    for col_name in final_col_names:
        if col_name in known_false_positive_col_names:
            raise FileProcessingError(
                f"false_positive_column_name - Using a column named [{col_name}] which is a known "
                f"false positive for column [{known_false_positive_col_names[col_name]}]"
            )

    return loaded_data_df, final_col_names


def convert_explicit_indications_of_cm_and_kpa(
    loaded_data_df: pd.DataFrame, col_names: list
) -> pd.DataFrame:
    """
    Convert explicit indications of cm and kPa in column names to m and MPa, respectively.

    Parameters
    ----------
    loaded_data_df : pd.DataFrame
        The DataFrame containing the loaded data.
    col_names : list
        A list of column names to check for unit indications.

    Returns
    -------
    pd.DataFrame
        The DataFrame with converted units and updated attributes indicating the conversions performed.
    """

    explicit_unit_conversions = []

    for col_index, col_name in enumerate(col_names):

        if col_name is not None:

            if col_index == 0:
                ## checking the depth column
                if "cm" in col_name.lower():
                    loaded_data_df.loc[:, col_name] /= 100
                    explicit_unit_conversions.append(
                        f"{col_name} was converted from cm to m"
                    )

            else:
                ## checking the other columns
                if "kpa" in col_name.lower():
                    loaded_data_df.loc[:, col_name] /= 1000
                    explicit_unit_conversions.append(
                        f"{col_name} was converted from kPa to MPa"
                    )

    loaded_data_df.attrs["explicit_unit_conversions"] = ", ".join(
        explicit_unit_conversions
    )

    return loaded_data_df


def load_csv_or_txt(
    file_path: Path,
    sheet: str = "0",
    col_data_types: npt.NDArray[np.str_] = np.array(["Depth", "qc", "fs", "u"]),
) -> pd.DataFrame:
    """ "
    Load a .csv or .txt file and return a DataFrame with the required columns.

    Parameters
    ----------
    file_path : Path
        The path to the file to load.

    sheet : str, optional
        A placeholder value to output a consistent format with xls files which have multiple sheets per file.
        Default is "0".
    col_data_types : tuple, optional
        The required column types for the DataFrame. Default is ("Depth", "qc", "fs", "u").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the required columns from the file.
    """

    sep = r"," if file_path.suffix.lower() == ".csv" else r"\s+"
    file_encoding = find_encoding(file_path)
    lines_and_cells_iterable = get_csv_or_txt_split_readlines(file_path, file_encoding)

    if len(lines_and_cells_iterable) == 0:
        raise FileProcessingError(
            f"no_data - sheet ({sheet.replace('-', '_')}) has no data"
        )

    header_lines_in_csv_or_txt_file = find_row_indices_of_header_lines(
        lines_and_cells_iterable
    )
    # csv and txt files do not have multiple sheets so just raise an error immediately if no header rows were found
    if len(header_lines_in_csv_or_txt_file) == 0:
        raise FileProcessingError(
            f"no_header_row - sheet ({sheet.replace("-", "_")}) has no header row"
        )

    if len(header_lines_in_csv_or_txt_file) > 1:
        multi_row_header_array = np.zeros(
            (len(header_lines_in_csv_or_txt_file), 4), dtype=float
        )
        multi_row_header_array[:] = np.nan

        ## writing as a for loop with a counter instead of using enumerate because
        ## using enumerate seemed to confuse the type checker
        header_line_counter = 0
        for header_row_index in header_lines_in_csv_or_txt_file:
            multi_row_header_array[header_line_counter, :] = (
                search_line_for_all_needed_cells(
                    lines_and_cells_iterable[header_row_index]
                )
            )
            header_line_counter += 1
        col_data_type_indices = np.nansum(multi_row_header_array, axis=0)
    else:
        col_data_type_indices = search_line_for_all_needed_cells(
            lines_and_cells_iterable[header_lines_in_csv_or_txt_file[0]]
        )

    missing_cols = col_data_types[~np.isfinite(col_data_type_indices)]

    if len(missing_cols) > 0:
        raise FileProcessingError(
            f"missing_columns - sheet ({sheet.replace('-', '_')}) is missing [{' & '.join(missing_cols)}]"
        )

    needed_col_indices_with_nans = search_line_for_all_needed_cells(
        lines_and_cells_iterable[header_lines_in_csv_or_txt_file[0]]
    )
    needed_col_indices = [
        int(col_idx) for col_idx in needed_col_indices_with_nans if np.isfinite(col_idx)
    ]
    df = pd.read_csv(
        file_path,
        header=None,
        encoding=file_encoding,
        sep=sep,
        skiprows=int(header_lines_in_csv_or_txt_file[0]),
        usecols=needed_col_indices,
    )

    df = df.map(convert_num_as_str_to_float)

    return df


def combine_multiple_header_rows(
    loaded_data_df: pd.DataFrame, header_row_indices: npt.ArrayLike
) -> tuple[pd.DataFrame, int]:
    """
    Combine multiple header rows into a single header row in a DataFrame.

    Parameters
    ----------
    loaded_data_df : pd.DataFrame
        The DataFrame containing the data with multiple header rows.
    header_row_indices : npt.ArrayLike
        An array of indices representing the header rows to be combined.

    Returns
    -------
    tuple[pd.DataFrame, int]
        A tuple containing the DataFrame with combined header rows and the index of the final header row.
    """

    ## take the header_row_index as the maximum of the header_row_indices
    ## which is the lowest row in the spreadsheet
    header_row_index = np.max(header_row_indices)

    ## copy the column names from the rows above the lowest header row
    loaded_data_df_with_combined_header_rows = loaded_data_df.copy()
    for row_idx in header_row_indices:
        for col_idx in range(loaded_data_df.shape[1]):
            if row_idx != header_row_index:
                loaded_data_df_with_combined_header_rows.iloc[
                    header_row_index, col_idx
                ] = (
                    str(loaded_data_df.iloc[header_row_index, col_idx])
                    + " "
                    + str(loaded_data_df.iloc[row_idx, col_idx])
                )

    return loaded_data_df_with_combined_header_rows, header_row_index


def make_summary_df_per_record(
    record_dir_name: str,
    file_was_loaded: bool,
    loaded_file_type: str,
    loaded_file_name: str,
    pdf_file_list: list,
    cpt_file_list: list,
    ags_file_list: list,
    xls_file_list: list,
    xlsx_file_list: list,
    csv_file_list: list,
    txt_file_list: list,
    unknown_list: list,
):
    """
    Create a summary DataFrame with information about the loaded files.

    Parameters
    ----------
    record_dir_name : str
        The name of the record directory.
    file_was_loaded : bool
        A flag indicating whether a file was successfully loaded.
    loaded_file_type : str
        The type of the loaded file.
    loaded_file_name : str
        The name of the loaded file.
    pdf_file_list : list
        A list of PDF files.
    cpt_file_list : list
        A list of CPT files.
    ags_file_list : list
        A list of AGS files.
    xls_file_list : list
        A list of XLS files.
    xlsx_file_list : list
        A list of XLSX files.
    csv_file_list : list
        A list of CSV files.
    txt_file_list : list
        A list of TXT files.
    unknown_list : list
        A list of files with unknown types.

    Returns
    -------
    pd.DataFrame
        The concatenated summary DataFrame with the new information added.
    """

    if (
        (len(pdf_file_list) > 0)
        & (len(cpt_file_list) == 0)
        & (len(ags_file_list) == 0)
        & (len(xls_file_list) == 0)
        & (len(xlsx_file_list) == 0)
        & (len(csv_file_list) == 0)
        & (len(txt_file_list) == 0)
        & (len(unknown_list) == 0)
    ):
        has_only_pdf = True
    else:
        has_only_pdf = False

    loading_summary = pd.DataFrame(
        {
            "record_name": record_dir_name,
            "file_was_loaded": file_was_loaded,
            "loaded_file_type": loaded_file_type,
            "loaded_file_name": loaded_file_name,
            "only_has_pdf": has_only_pdf,
            "num_pdf_files": len(pdf_file_list),
            "num_cpt_files": len(cpt_file_list),
            "num_ags_files": len(ags_file_list),
            "num_xls_files": len(xls_file_list),
            "num_xlsx_files": len(xlsx_file_list),
            "num_csv_files": len(csv_file_list),
            "num_txt_files": len(txt_file_list),
            "num_other_files": len(unknown_list),
        },
        index=[0],
    )

    return loading_summary


def nth_highest_value(array: npt.NDArray, n: int) -> float:
    """
    Find the nth highest value in an array.

    Parameters
    ----------
    array : npt.ArrayLike
        The input array.
    n : int
        The value of n to select the nth highest value.

    Returns
    -------
    float
        The nth highest value in the array.
    """

    ## Filter out any nan depth values (such as at the end of a file) and sort the array
    sorted_array = np.sort(array[np.isfinite(array)])

    ## indexing a numpy array returns a numpy scalar which is not the same as a Python float so explicitly convert
    ## to satisfy type checking
    return float(sorted_array[-n])


def infer_wrong_units(
    loaded_data_df: pd.DataFrame,
    cm_threshold: float = 99,
    mm_threshold: float = 999,
    qc_kpa_threshold: float = 150,
    fs_kpa_threshold: float = 10,
    u_kpa_threshold: float = 3,
    nth_highest: int = 5,
) -> pd.DataFrame:
    """
    Infer the use of cm, mm, or kPa from the numerical values and convert to m and MPa where necessary.

    Parameters
    ----------
    loaded_data_df : pd.DataFrame
        The input DataFrame containing the data to be checked.
    cm_threshold : int, optional
        An nth highest value over this threshold indicates that depth is in cm. Default is 99.
    mm_threshold : int, optional
        An nth highest value over this threshold indicates that depth is in mm. Default is 999.
    qc_kpa_threshold : float, optional
        An nth highest value over this threshold indicates that qc is in kPa. Default is 150.
    fs_kpa_threshold : float, optional
        An nth highest value over this threshold indicates that fs is in kPa. Default is 10.
    u_kpa_threshold : float, optional
        An nth highest value over this threshold indicates that u is in kPa. Default is 3.
    nth_highest : int, optional
        The nth highest value to be checked in the columns. Default is 5.

    Returns
    -------
    pd.DataFrame
        The corrected DataFrame with inferred unit conversions and updated attributes.
    """

    with open(
        Path(__file__).parent.parent
        / "resources"
        / "cpt_column_name_descriptions.toml",
        "r",
    ) as toml_file:
        column_descriptions = toml.load(toml_file)

    inferred_unit_conversions = []

    if (
        cm_threshold
        < nth_highest_value(
            loaded_data_df[list(column_descriptions)[0]].values, nth_highest
        )
        < mm_threshold
    ):
        ## depth values are likely in cm
        loaded_data_df[list(column_descriptions)[0]] /= 100
        inferred_unit_conversions.append(
            f"{list(column_descriptions)[0]} was converted from cm to m"
        )
    elif (
        nth_highest_value(
            loaded_data_df[list(column_descriptions)[0]].values, nth_highest
        )
        > mm_threshold
    ):
        ## depth values are likely in mm
        loaded_data_df[list(column_descriptions)[0]] /= 1000
        inferred_unit_conversions.append(
            f"{list(column_descriptions)[0]} was converted from mm to m"
        )

    if (
        nth_highest_value(
            loaded_data_df[list(column_descriptions)[1]].values, nth_highest
        )
        > qc_kpa_threshold
    ):
        loaded_data_df[list(column_descriptions)[1]] /= 1000
        inferred_unit_conversions.append(
            f"{list(column_descriptions)[1]} was converted from kPa to MPa"
        )
    if (
        nth_highest_value(
            loaded_data_df[list(column_descriptions)[2]].values, nth_highest
        )
        > fs_kpa_threshold
    ):
        loaded_data_df[list(column_descriptions)[2]] /= 1000
        inferred_unit_conversions.append(
            f"{list(column_descriptions)[2]} was converted from kPa to MPa"
        )
    if (
        nth_highest_value(
            loaded_data_df[list(column_descriptions)[3]].values, nth_highest
        )
        > u_kpa_threshold
    ):
        loaded_data_df[list(column_descriptions)[3]] /= 1000
        inferred_unit_conversions.append(
            f"{list(column_descriptions)[3]} was converted from kPa to MPa"
        )

    loaded_data_df.attrs["inferred_unit_conversions"] = ", ".join(
        inferred_unit_conversions
    )

    return loaded_data_df


def ensure_positive_depth(loaded_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the depth column has positive values and remove rows with negative values in qc and fs.

    Parameters
    ----------
    loaded_data_df : pd.DataFrame
        The input DataFrame containing the data to be checked.

    Returns
    -------
    pd.DataFrame
        The corrected DataFrame with positive depth values and no negative values in qc and fs columns.
    """

    with open(
        Path(__file__).parent.parent
        / "resources"
        / "cpt_column_name_descriptions.toml",
        "r",
    ) as toml_file:
        column_descriptions = toml.load(toml_file)

    ## Ensure that the depth column is defined as positive (some have depth as negative)
    if loaded_data_df[list(column_descriptions)[0]].min() < 0:
        loaded_data_df[list(column_descriptions)[0]] = np.abs(
            loaded_data_df[list(column_descriptions)[0]]
        )
        loaded_data_df.attrs["depth_originally_defined_as_negative"] = True

    return loaded_data_df
