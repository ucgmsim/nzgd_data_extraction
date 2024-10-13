import numpy as np


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


# def test_for_header_row(line):
#
#     characters_to_test = ["m","w"]
#
#     depth_col_candidates = []
#     for character in characters_to_test:
#          depth_col_candidates.append(find_cell_of_single_character(line, character))
#
#     for substring in ["depth", "length", "h ", "top"]:
#         depth_col_candidates.append(find_cell_that_contains_string(line, substring))
#
#     return depth_col_candidates


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
        characters1=["m","w","h"],
        substrings1=["depth", "length", "h ", "top"],
        characters2=["q"],
        substrings2 = [" q ", "qc", "q_c", "cone", "resistance", "res", "tip"],
        characters3=[],
        substrings3=["fs", "sleeve", "friction","local"],
        characters4=[],
        substrings4=["dynamic", "u2", "pore","u","water"]):

    col1_search = search_line_for_cell(line, characters1, substrings1)
    col2_search = search_line_for_cell(line, characters2, substrings2)
    col3_search = search_line_for_cell(line, characters3, substrings3)
    col4_search = search_line_for_cell(line, characters4, substrings4)

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






    return col1, col2, col3, col4

def interpret_search(search_result):

    if len(search_result) == 0:
        return "No results found"
    elif len(search_result) == 1:
        return search_result[0]
    else:
        return search_result

# def get_header_rows_no_break(df, check_rows):
#
#     first_check_results = [search_line_for_all_needed_cells(line=df.iloc[x]) for x in check_rows]
#     print()
#     first_check_num_header_cols = np.array([np.sum(np.isfinite(x)) for x in first_check_results])
#
#     arg_4 = check_rows[np.where(first_check_num_header_cols == 4)[0]]
#
#     print()
#
#     if len(arg_4) > 0:
#         multi_row_header_check_array = check_rows[arg_4[0]-1:arg_4[0]+2]
#         header_rows = multi_row_header_check_array[np.where(first_check_num_header_cols >= 2)]
#
#     else:
#         header_rows = np.array([])
#     return header_rows
#
# def get_header_rows(df, check_row_idxs, second_call=False):
#
#     if not second_call:
#         search_result = search_line_for_all_needed_cells(df.iloc[check_row_idxs])
#         get_header_rows(df, check_row_idxs, second_call=True)
#
#
#     for check_row_idx in check_row_idxs:
#
#         print()
#
#
#
#         if np.sum(np.isfinite(search_result)) == 4:
#             print()
#             break
#
#         print()
#
#     if line_idx != check_rows[-1]:
#         header_rows = get_header_rows_no_break(df, check_rows[line_idx - 1:line_idx + 2])
#         print()
#
#     else:
#         header_rows = np.array([])
#
#     return header_rows


def get_header_rows(df, check_rows):

    partial_header_length = 2
    header_rows = []

    for check_row in check_rows:
        line1_check = search_line_for_all_needed_cells(df.iloc[check_row])
        line2_check = search_line_for_all_needed_cells(df.iloc[check_row+1])

        if (np.sum(np.isfinite(line1_check)) == 4) & (np.unique(line1_check).size == 4):
            # found at least one header row so check the next row for a partial
            header_rows.append(check_row)
            if (np.sum(np.isfinite(line2_check)) == partial_header_length) & (np.unique(line2_check).size == partial_header_length):
                header_rows.append(check_row+1)

            return np.array(header_rows)

        elif (np.sum(np.isfinite(line2_check)) == 4) & (np.unique(line2_check).size == 4):
            # found a full header row so check if the previous row was a partial
            header_rows.append(check_row+1)
            if (np.sum(np.isfinite(line1_check)) == partial_header_length) & (np.unique(line1_check).size == partial_header_length):
                header_rows.append(check_row)

            return np.array(header_rows)

    # if no header rows were found after checking all check_rows, return an empty array
    return np.array([])







