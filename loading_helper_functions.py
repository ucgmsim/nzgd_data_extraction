import numpy as np
import xlrd
import pandas as pd
import re

def find_cell_in_line_containing_single_character(line, character):
    """Return the index of the first cell containing the given character in the given line."""

    candidate_cells = []

    for i, cell in enumerate(line):

        if isinstance(cell, str):
            if len(cell) == 1:
                if cell.lower() == character:
                    return i


def find_cell_in_line_that_contains_string(line, string):
    """Return the index of the first cell containing the given string in the given line."""
    for i, cell in enumerate(line):

        if isinstance(cell, str):
            if string in cell.lower():
                return i




def search_line_for_cell(line, characters, substrings):

    candidates_idx = []

    for character in characters:
         candidates_idx.append(find_cell_in_line_containing_single_character(line, character))

    for substring in substrings:
        substring_cell = find_cell_in_line_that_contains_string(line, substring)
        if substring_cell not in candidates_idx:
            candidates_idx.append(substring_cell)


    ## remove None
    candidates_idx = [candidate for candidate in candidates_idx if candidate is not None]

    return candidates_idx

def search_line_for_all_needed_cells(
        line,
        output_all_candidates=False,
        characters1=["m","w","h"],
        substrings1=["depth", "length", "h ", "top"],
        characters2=["q"],
        substrings2 = [" q ", "q ", " q", "qc", "q_c", "cone", "resistance", "res", "tip"],
        characters3=[],
        substrings3=["fs", "sleeve", "friction","local"],
        characters4=["u"],
        substrings4=["u ", " u", "u2", "dynamic", "pore","water"]):

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


# def get_header_rows(iterable, check_rows):
#
#     partial_header_length = 1
#     header_rows = []
#
#     most_likely_partial_header_row = np.nan
#     most_columns_found = 0
#
#     for check_row in check_rows:
#
#         if isinstance(iterable, pd.DataFrame):
#             line1_check = search_line_for_all_needed_cells(iterable.iloc[check_row])
#             line2_check = search_line_for_all_needed_cells(iterable.iloc[check_row+1])
#         else:
#             line1_check = search_line_for_all_needed_cells(iterable[check_row])
#             line2_check = search_line_for_all_needed_cells(iterable[check_row+1])
#
#
#
#         if (np.sum(np.isfinite(line1_check)) >= 2) & (np.sum(np.isfinite(line1_check)) >= 1):
#             header_rows.extend([check_row, check_row+1])
#             return np.array(header_rows)
#
#
#
#
#
#
#         if np.sum(np.isfinite(line1_check)) >=1:
#             header_rows.append(check_row)
#             if np.sum(np.isfinite(line2_check)) >= 1:
#                 header_rows.append(check_row+1)
#
#             break
#
#         elif np.sum(np.isfinite(line2_check)) >= 1:
#             header_rows.append(check_row+1)
#
#
#     return np.array(header_rows)

def get_header_rows(iterable, check_rows):

    best_partial_header_row = np.nan
    num_cols_in_best_possible_row = 0

    header_rows = []

    for check_row in check_rows:

        if isinstance(iterable, pd.DataFrame):
            line1_check = search_line_for_all_needed_cells(iterable.iloc[check_row])
            line2_check = search_line_for_all_needed_cells(iterable.iloc[check_row + 1])
        else:
            line1_check = search_line_for_all_needed_cells(iterable[check_row])
            line2_check = search_line_for_all_needed_cells(iterable[check_row + 1])

        if (np.sum(np.isfinite(line1_check)) >= 4):
            header_rows.append(check_row)
            if np.sum(np.isfinite(line2_check)) >= 1:
                header_rows.append(check_row + 1)
            return np.array(header_rows)

        elif (np.sum(np.isfinite(line1_check)) >= 1):
            num_cols_in_check_row = np.sum(np.isfinite(line1_check))

            if num_cols_in_check_row > num_cols_in_best_possible_row:
                print()
                best_partial_header_row = check_row
                num_cols_in_best_possible_row = num_cols_in_check_row

    if best_partial_header_row is not np.nan:
        print()
        header_rows.append(best_partial_header_row)
        # check following line for a single column name
        print()
        if isinstance(iterable, pd.DataFrame):
            line2_check = search_line_for_all_needed_cells(iterable.iloc[best_partial_header_row + 1])
        else:
            line2_check = search_line_for_all_needed_cells(iterable[best_partial_header_row + 1])
        if np.sum(np.isfinite(line2_check)) >= 1:
            header_rows.append(best_partial_header_row + 1)
        print()
        return np.array(header_rows)

    else:
        print()
        return np.array([])


# best_partial_row_check =
#
#
#
#
#
#
#
#
#
#             header_rows.extend([check_row, check_row + 1])
#             return np.array(header_rows)
#
#         if np.sum(np.isfinite(line1_check)) >= 1:
#             header_rows.append(check_row)
#             if np.sum(np.isfinite(line2_check)) >= 1:
#                 header_rows.append(check_row + 1)
#
#             break
#
#         elif np.sum(np.isfinite(line2_check)) >= 1:
#             header_rows.append(check_row + 1)
#
#     return np.array(header_rows)



    #     if (np.sum(np.isfinite(line1_check)) > 4) & (np.unique(line1_check).size == 4):
    #         # found at least one header row so check the next row for a partial
    #         header_rows.append(check_row)
    #         print()
    #         if (np.sum(np.isfinite(line2_check)) == partial_header_length) & (np.unique(line2_check).size == partial_header_length):
    #             header_rows.append(check_row+1)
    #         return np.array(header_rows)
    #
    #     elif (np.sum(np.isfinite(line2_check)) == 4) & (np.unique(line2_check).size == 4):
    #         # found a full header row so check if the previous row was a partial
    #         header_rows.append(check_row+1)
    #         if (np.sum(np.isfinite(line1_check)) == partial_header_length) & (np.unique(line1_check).size == partial_header_length):
    #             header_rows.append(check_row)
    #         return np.array(header_rows)
    #
    #     elif np.sum(np.isfinite(np.unique(line1_check))) > most_columns_found:
    #             most_likely_partial_header_row = check_row
    #             most_columns_found = np.sum(np.isfinite(np.unique(line1_check)))
    #
    # if np.isfinite(most_likely_partial_header_row):
    #     print()
    #     header_rows.append(most_likely_partial_header_row)
    #     return np.array(header_rows)
    # else:
    #     # if no partial header rows were found after checking all check_rows, return an empty array
    #     return np.array([])

def get_xls_sheet_names(file_path):

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

    return sheet_names, engine

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

    sep = r"," if file_path.suffix == ".csv" else r"\s+"

    split_lines = [re.split(sep, line) for line in lines]

    # remove empty strings
    split_lines = [[cell for cell in line if cell != ""] for line in split_lines]

    return split_lines

def check_for_clean_cols(df):

    target_col_index_to_name = {0:"depth",1:"cone_resistance",2:"sleeve_friction",
                                3:"porewater_pressure"}

    all_target_col_candidate_indices = search_line_for_all_needed_cells(df.columns,
                                                                                     output_all_candidates=True)
    final_col_names = []
    for target_col_index, target_col_candidate_indices in enumerate(all_target_col_candidate_indices):
        print()
        #candidate_finite_indices = target_col_candidate_indices[np.isfinite(target_col_candidate_indices)]

        if len(target_col_candidate_indices) == 0:
            print()
            final_col_names.append(None)

            if "missing_columns" not in df.attrs:
                df.attrs["missing_columns"] = [target_col_index_to_name[target_col_index]]
            else:
                df.attrs["missing_columns"].append(target_col_index_to_name[target_col_index])

        else:
            print()
            #candidate_finite_indices = target_col_candidate_indices[np.isfinite(target_col_candidate_indices)]
            target_col_candidate_names = [df.columns[int(idx)] for idx in target_col_candidate_indices]

            candidate_col_name = target_col_candidate_names[0]
            num_finite_values_in_first_candidate_col = np.sum(np.isfinite(df[candidate_col_name]))

            for target_col_candidate_name in target_col_candidate_names:
                candidate_name = target_col_candidate_name
                if "clean" in target_col_candidate_name.lower():
                    if np.sum(np.isfinite(df[target_col_candidate_name])) <= num_finite_values_in_first_candidate_col:
                        candidate_name = target_col_candidate_name
                        break
            final_col_names.append(candidate_name)

            df.attrs[
                f"candidate_{target_col_index_to_name[target_col_index]}_column_names_in_original_file"] = target_col_candidate_names
            df.attrs[f"adopted_{target_col_index_to_name[target_col_index]}_column_name_in_original_file"] = candidate_name

    return df, final_col_names

def convert_to_m_and_mpa(df, col_names):

    for col_index, col_name in enumerate(col_names):

        if col_name is not None:

            if col_index == 0:
                # checking the depth column
                if "cm" in col_name:
                    df.loc[:, col_name] /= 100

            else:
                # checking the other columns
                if "kpa" in col_name:
                    df.loc[:, col_name] /= 1000

    return df

























