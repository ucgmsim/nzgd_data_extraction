"""
Functions for loading data from the New Zealand Geotechnical Database (NZGD).
"""
from dataclasses import dataclass
from pathlib import Path
import functools

import numpy as np
import pandas
import pandas as pd
import toml
from python_ags4 import AGS4
import xlrd

import nzgd_data_extraction.lib.processing_helpers as processing_helpers
from nzgd_data_extraction.lib.processing_helpers import (
    FileProcessingError
)


def find_missing_cols_for_best_sheet(missing_columns_per_sheet: list[list]) -> list:
    """
    Find the sheet with the fewest missing columns.

    Parameters
    ----------
    missing_columns_per_sheet : list[list]
        A list of lists, where each inner list contains the missing columns for a sheet.

    Returns
    -------
    list
        The list of missing columns for the sheet with the fewest missing columns.
    """

    final_num_missing_cols = 5
    final_missing_cols = []
    for missing_cols in missing_columns_per_sheet:
        if len(missing_cols) < final_num_missing_cols:
            final_num_missing_cols = len(missing_cols)
            final_missing_cols = missing_cols
    return final_missing_cols


def find_col_name_from_substring(
    df: pd.DataFrame,
    substrings: list[str],
    remaining_cols_to_search: list[str],
    target_column_name: str,
) -> tuple[str, pd.DataFrame, list[str]]:
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

    # no relevant columns were found
    if len(candidate_col_names) == 0:
        col = None
        if "missing_columns" not in df.attrs:
            df.attrs["missing_columns"] = [target_column_name]
        else:
            df.attrs["missing_columns"].append(target_column_name)

    ## Check that there are some candidate column names
    else:
        col = candidate_col_names[0]

        # check for "Clean" which is sometimes used for a cleaned version of the same data
        if len(candidate_col_names) > 1:
            for candidate_name in candidate_col_names:
                ## some "clean" columns are full of nans (no data) so also check that the number of nans
                ## in the "clean" column is less than or equal to the number of nans in the current column
                if ("clean" in candidate_name.lower()) and (
                    np.sum(pd.isnull(df[candidate_name])) <= np.sum(pd.isnull(df[col]))
                ):
                    col = candidate_name
                    break

        df.attrs[f"candidate_{target_column_name}_column_names_in_original_file"] = (
            candidate_col_names
        )
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

    return col, df, remaining_cols_to_search


def load_ags(
    file_path: Path, investigation_type: processing_helpers.InvestigationType
) -> pd.DataFrame:
    """
    Load an AGS file.

    Parameters
    ----------
    file_path : Path
        The path to the AGS file.

    investigation_type : processing_helpers.InvestigationType
        The type of investigation being processed.

    Returns
    -------
    pandas.DataFrame
        The CPT data from the AGS file.
    """

    with open(
        Path(__file__).parent.parent
        / "resources"
        / "cpt_column_name_descriptions.toml",
        "r",
    ) as toml_file:
        column_descriptions = toml.load(toml_file)

    try:
        tables, headings = AGS4.AGS4_to_dataframe(file_path)
    except UnboundLocalError:
        ## Found the meaning of this UnboundLocalError by uploading one of these files to the AGS file conversion tool on https://agsapi.bgs.ac.uk
        raise FileProcessingError(
            "ags_duplicate_headers - AGS file contains duplicate headers"
        )

    if len(tables) == 0:
        raise FileProcessingError(
            "no_ags_data_tables - no data tables found in the AGS file"
        )

    required_ags_column_names = ["SCPT_DPTH", "SCPT_RES", "SCPT_FRES", "SCPT_PWP2"]

    ## Check if any required columns are completely missing from the ags file
    for required_column_name in required_ags_column_names:
        if required_column_name not in tables["SCPT"].columns:
            raise FileProcessingError(
                f"ags_missing_columns - AGS file is missing {required_column_name} (and possibly other) columns"
            )

    loaded_data_df = pd.DataFrame(
        {
            list(column_descriptions)[0]: tables["SCPT"]["SCPT_DPTH"],
            list(column_descriptions)[1]: tables["SCPT"]["SCPT_RES"],
            list(column_descriptions)[2]: tables["SCPT"]["SCPT_FRES"],
            list(column_descriptions)[3]: tables["SCPT"]["SCPT_PWP2"],
        }
    )

    if (investigation_type == processing_helpers.InvestigationType.scpt) & (
        "SCPT_SWV" in tables["SCPT"].columns
    ):
        loaded_data_df[list(column_descriptions)[4]] = tables["SCPT"]["SCPT_SWV"]

    if (investigation_type == processing_helpers.InvestigationType.scpt) & (
        "SCPT_PWV" in tables["SCPT"].columns
    ):
        loaded_data_df[list(column_descriptions)[5]] = tables["SCPT"]["SCPT_PWV"]

    ## The first two data rows are skipped as they contain units and the number of decimal places for each column.
    ## For example:
    #     Depth      qc      fs    u
    # 0       m     MPa     MPa  MPa
    # 1     2DP     3DP     4DP  4DP
    loaded_data_df = loaded_data_df.iloc[2:]
    num_numerical_vals = loaded_data_df.map(
        processing_helpers.can_convert_str_to_float
    ).sum()
    zero_value_columns = num_numerical_vals[num_numerical_vals == 0].index.tolist()
    if len(zero_value_columns) > 0:
        raise FileProcessingError(
            f"ags_lacking_numeric_data - AGS file has no numeric data in columns [{" ".join(zero_value_columns)}]"
        )

    ## Convert all data to numeric values (dropping rows that contain non-numeric data)
    loaded_data_df = loaded_data_df.apply(pd.to_numeric, errors="coerce").dropna()

    ### If the values are unrealistically large in MPa, they are likely in kPa so convert to MPa.
    ### Similarly, unrealistically large depth values may be in cm so convert to m.
    loaded_data_df = processing_helpers.infer_wrong_units(loaded_data_df)

    ### Ensure that the depth column has positive values and that qc and fs are greater than 0
    loaded_data_df = processing_helpers.ensure_positive_depth_and_qc_fs_gtr_0(
        loaded_data_df
    )

    if loaded_data_df.empty:
        raise FileProcessingError(
            "ags_tried_to_save_empty - Tried to save an empty DataFrame"
        )
    return loaded_data_df


def load_cpt_spreadsheet_file(file_path: Path) -> list[pd.DataFrame]:
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

    with open(
        Path(__file__).parent.parent
        / "resources"
        / "cpt_column_name_descriptions.toml",
        "r",
    ) as toml_file:
        column_descriptions = toml.load(toml_file)

    known_special_cases = toml.load(
        Path(__file__).parent.parent / "resources" / "cpt_column_name_descriptions.toml"
    )

    if "_" not in file_path.name:
        raise processing_helpers.FileProcessingError(
            f"bad_file_name_format - file name {file_path.name} does not contain an NZGD record name")

    record_id = f"{file_path.name.split("_")[0]}_{file_path.name.split("_")[1]}"
    if record_id in known_special_cases.keys():
        raise processing_helpers.FileProcessingError(known_special_cases[record_id])

    ## Set an initial value for the sheet names that will be used if the file is a csv or txt file
    ## as these do not have multiple sheets per file like xls
    sheet_names = ["0"]
    if file_path.suffix.lower() in [".xls", ".xlsx"]:
        ## Get the actual sheet names
        sheet_names, _ = processing_helpers.get_xls_sheet_names(file_path)

        if len(sheet_names) == 0:
            raise processing_helpers.FileProcessingError(
                f"corrupt_file - cannot detect sheets in file {file_path.name}"
            )

    # Iterate through each sheet
    missing_cols_per_sheet = []
    error_text = []
    dataframes_to_return = []

    for sheet in sheet_names:

        if file_path.suffix.lower() in [".csv", ".txt"]:
            df = processing_helpers.load_csv_or_txt(file_path)
        else:
            _, engine = processing_helpers.get_xls_sheet_names(file_path)
            df = pd.read_excel(
                file_path,
                sheet_name=sheet,
                header=None,
                engine=engine,
                parse_dates=False,
            )

        ####################################################################################################################
        ####################################################################################################################
        # Now xls, csv and txt should all be in a dataframe so continue the same for all
        ## Check the dataframe for various issues
        if df.shape == (0, 0):
            error_text.append(
                f"empty_file - sheet ({sheet.replace('-', '_')}) has size (0,0)"
            )
            continue

        if df.shape[0] == 1:
            error_text.append(
                f"only_one_line - sheet ({sheet.replace('-', '_')}) has only one line with first cell of {df.iloc[0][0]}"
            )
            continue

        ## Add some attributes to the dataframe to store information about the original file
        df.attrs["original_file_name"] = file_path.name
        df.attrs["sheet_in_original_file"] = sheet
        df.attrs["column_name_descriptions"] = column_descriptions
        df.attrs["explicit_unit_conversions"] = []
        df.attrs["inferred_unit_conversions"] = []
        df.attrs["depth_originally_defined_as_negative"] = False

        df_for_counting_str_per_row = df.map(
            lambda x: 1.0 if isinstance(x, str) else 0
        )

        df_nan_to_str = df.fillna("nan")
        df_for_counting_num_of_num = df_nan_to_str.map(
            lambda x: 1.0 if isinstance(x, (int, float)) else 0
        )

        numeric_surplus_per_col = np.nansum(
            df_for_counting_num_of_num, axis=0
        ) - np.nansum(df_for_counting_str_per_row, axis=0)

        # Drop any columns that have more text than numeric data
        df = df.iloc[:, numeric_surplus_per_col >= 0]
        numeric_surplus_per_row = np.nansum(
            df_for_counting_num_of_num, axis=1
        ) - np.nansum(df_for_counting_str_per_row, axis=1)
        header_row_indices = []
        header_row_from_col_names = (
            processing_helpers.find_one_header_row_index_from_column_names(df)
        )
        if np.isfinite(header_row_from_col_names):
            header_row_indices = processing_helpers.find_row_indices_of_header_lines(df)

        ## Check if the dataframe has any numeric data
        if np.sum(df_for_counting_num_of_num.values) == 0:
            error_text.append(
                f"no_numeric_data - sheet ({sheet.replace('-', '_')}) has no numeric data"
            )
            continue

        if all(numeric_surplus_per_col < 2):
            error_text.append(
                f"no_data_columns - all columns in sheet ({sheet.replace('-', '_')}) have more text cells than numeric cells"
            )
            continue

        if all(numeric_surplus_per_row < 2):
            error_text.append(
                f"no_data_rows - all rows in sheet ({sheet.replace('-', '_')}) have more text cells than numeric cells"
            )
            continue

        if len(header_row_indices) == 0:
            error_text.append(
                f"no_header_row - sheet ({sheet.replace('-', '_')}) has no header row"
            )
            continue

        df, header_row_index = processing_helpers.combine_multiple_header_rows(
            df, header_row_indices
        )
        # set dataframe's headers/column names. Note that .values is used so that the row's index is not included in the header
        df.columns = df.iloc[header_row_index].values
        # Skip the rows that originally contained the column names as they are now stored as the dataframe header
        df = df.iloc[header_row_index + 1 :]
        df = df.apply(pd.to_numeric, errors="coerce").astype(float)

        header_row_index = (
            header_row_indices[0]
            if file_path.suffix.lower() in [".csv", ".txt"]
            else header_row_index
        )
        df.attrs["header_row_index_in_original_file"] = float(header_row_index)
        df.reset_index(inplace=True, drop=True)
        df, final_col_names = processing_helpers.get_column_names(df)

        ## Check if the identified "Depth" column is actually an index rather than a measurement in metres.
        if final_col_names[0] is not None:
            if (df[final_col_names[0]] == df[final_col_names[0]].astype(int)).all():
                raise processing_helpers.FileProcessingError(
                    f"depth_is_index - sheet ({sheet.replace('-', '_')}) has its depth column as an index "
                    f"rather than a depth measurement"
                )

        df = processing_helpers.convert_explicit_indications_of_cm_and_kpa(
            df, final_col_names
        )

        final_col_names_without_none = [
            col for col in final_col_names if col is not None
        ]
        if all(i is not None for i in final_col_names) & (
            len(np.unique(final_col_names_without_none))
            == len(final_col_names_without_none)
        ):

            # Return the DataFrame with only the relevant columns
            df = (
                df[
                    [
                        final_col_names[0],
                        final_col_names[1],
                        final_col_names[2],
                        final_col_names[3],
                    ]
                ].rename(
                    columns={
                        final_col_names[0]: list(column_descriptions)[0],
                        final_col_names[1]: list(column_descriptions)[1],
                        final_col_names[2]: list(column_descriptions)[2],
                        final_col_names[3]: list(column_descriptions)[3],
                    }
                )
            ).apply(pd.to_numeric, errors="coerce")

            ### If the values are unrealistically large in MPa, they are likely in kPa so convert to MPa.
            ### Similarly, unrealistically large depth values may be in cm so convert to m.
            ### Also make sure that depth is positive and drop rows that have negative values of qc and fs
            df = processing_helpers.infer_wrong_units(df)

            ### Ensure that the depth column has positive values and that qc and fs are greater than 0
            df = processing_helpers.ensure_positive_depth_and_qc_fs_gtr_0(df)
            dataframes_to_return.append(df)

        else:
            if len(np.unique(final_col_names_without_none)) < len(
                final_col_names_without_none
            ):
                error_text.append(
                    f"non_unique_cols - in sheet ({sheet.replace('-', '_')}) some column names were selected more than once"
                )
                continue

            else:
                missing_cols_per_sheet.append(
                    [
                        list(column_descriptions)[idx]
                        for idx, col in enumerate(final_col_names)
                        if col is None
                    ]
                )

    ##################################################
    if len(dataframes_to_return) > 0:
        return dataframes_to_return

    final_missing_cols = find_missing_cols_for_best_sheet(missing_cols_per_sheet)
    if len(final_missing_cols) > 0:
        raise processing_helpers.FileProcessingError(
            f"missing_columns - sheet ({sheet_names[0].replace('-', '_')}) is missing [{' & '.join(final_missing_cols)}]"
        )

    else:
        raise processing_helpers.FileProcessingError(error_text[0])

@dataclass
class CptProcessingMetadata:
    """
    A dataclass for storing
    metadata about the processing of CPT data.
    """

    spreadsheet_format_description_per_record : pd.DataFrame
    loading_summary_df : pd.DataFrame
    all_failed_loads_df : pd.DataFrame


def process_one_record(record_dir: Path,
                       parquet_output_dir: Path,
                       nzgd_index_df: pd.DataFrame,
                       investigation_type: processing_helpers.InvestigationType) -> CptProcessingMetadata:

    ags_file_list = list(record_dir.glob("*.ags")) + list(record_dir.glob("*.AGS"))
    xls_file_list = list(record_dir.glob("*.xls")) + list(record_dir.glob("*.XLS"))
    xlsx_file_list = list(record_dir.glob("*.xlsx")) + list(record_dir.glob("*.XLSX"))
    csv_file_list = list(record_dir.glob("*.csv")) + list(record_dir.glob("*.CSV"))
    txt_file_list = list(record_dir.glob("*.txt")) + list(record_dir.glob("*.Txt")) + list(record_dir.glob("*.TXT"))
    cpt_file_list = list(record_dir.glob("*.cpt")) + list(record_dir.glob("*.CPT"))
    pdf_file_list = list(record_dir.glob("*.pdf")) + list(record_dir.glob("*.PDF")) + list(record_dir.glob("*.Pdf"))

    known_file_set = set(
        ags_file_list + xls_file_list + xlsx_file_list + csv_file_list + txt_file_list + cpt_file_list + pdf_file_list)
    unknown_list = list(set(list(record_dir.glob("*"))) - known_file_set)

    partial_summary_df_helper = functools.partial(processing_helpers.make_summary_df_per_record, record_dir_name=record_dir.name,
                                                  pdf_file_list=pdf_file_list, cpt_file_list=cpt_file_list, ags_file_list=ags_file_list,
                                                  xls_file_list=xls_file_list, xlsx_file_list=xlsx_file_list, csv_file_list=csv_file_list,
                                                  txt_file_list=txt_file_list, unknown_list=unknown_list)
    if (
            (len(ags_file_list) == 0) &
            (len(xls_file_list) == 0) &
            (len(xlsx_file_list) == 0) &
            (len(csv_file_list) == 0) &
            (len(txt_file_list) == 0)):
        if (len(pdf_file_list) == 0) & (len(cpt_file_list) == 0):
            error_as_string = "no_files - no files in the record directory"
        elif (len(pdf_file_list) > 0) & (len(cpt_file_list) == 0):
            error_as_string = "only_pdf_files - only pdf files in the record directory"
        else:
            error_as_string = "only_unknown_cpt_type_files - only unknown .cpt type files in the record directory"

        loading_summary_df = partial_summary_df_helper(file_was_loaded=False,
                                                       loaded_file_type="N/A",
                                                       loaded_file_name="N/A")

        all_failed_loads_df = pd.DataFrame({"record_name": record_dir.name,
                                                       "file_type": "N/A",
                                                       "file_name": "N/A",
                                                       "category": error_as_string.split("-")[0].strip(),
                                                       "details": error_as_string.split("-")[1].strip()},
                                                 index = [0])

        return CptProcessingMetadata(pd.DataFrame(), loading_summary_df, all_failed_loads_df)

    nzgd_meta_data_record = nzgd_index_df[nzgd_index_df["ID"]==record_dir.name].to_dict(orient="records")[0]

    ## This dataframe will store the details of all the files that failed to load over the loops below
    all_failed_loads_df = pd.DataFrame(columns=["record_name", "file_type", "file_name", "category", "details"])

    ### ags files
    ags_files_to_try = list(record_dir.glob("*.ags")) + list(record_dir.glob("*.AGS"))

    ########
    ## spreadsheet files
    files_to_try = list(record_dir.glob("*.xls")) + list(record_dir.glob("*.XLS")) + \
                   list(record_dir.glob("*.xlsx")) + list(record_dir.glob("*.XLSX")) + \
                   list(record_dir.glob("*.csv")) + list(record_dir.glob("*.CSV")) +\
                   list(record_dir.glob("*.txt")) + list(record_dir.glob("*.Txt")) +\
                   list(record_dir.glob("*.TXT"))

    for file_to_try_index, file_to_try in enumerate(files_to_try):
        try:
            record_df_list = load_cpt_spreadsheet_file(file_to_try)
            record_df_copy_for_attrs = record_df_list[0].copy()

            record_df = pd.DataFrame()
            for record_df_idx in range(len(record_df_list)):
                record_df_list[record_df_idx].insert(0, "multiple_measurements", record_df_idx)
                if record_df_idx == 0:
                    record_df = record_df_list[record_df_idx]
                else:
                    record_df = pd.concat([record_df, record_df_list[record_df_idx]], ignore_index=True)

            # If some attributes were lost by the concatenation, add them back
            if len(record_df.attrs.keys()) != record_df_copy_for_attrs.attrs.keys():
                for i in record_df_copy_for_attrs.attrs.keys():
                    if i not in record_df.attrs.keys():
                        record_df.attrs[i] = record_df_copy_for_attrs.attrs[i]

            record_df.attrs["max_depth"] = record_df["Depth"].max()
            record_df.attrs["min_depth"] = record_df["Depth"].min()
            record_df.attrs["original_file_name"] = file_to_try.name
            record_df.attrs["nzgd_meta_data"] = nzgd_meta_data_record
            record_df.insert(0, "record_name", record_dir.name)
            record_df.insert(1, "latitude", nzgd_meta_data_record["Latitude"])
            record_df.insert(2, "longitude", nzgd_meta_data_record["Longitude"])

            record_df.reset_index(inplace=True, drop=True)
            if record_df.empty:
                raise processing_helpers.FileProcessingError("spreadsheet_dataframe_empty - while loading from a spreadsheet, tried to save an empty dataframe")

            record_df.to_parquet(parquet_output_dir / f"{record_dir.name}.parquet")

            spreadsheet_format_description_per_record = pd.DataFrame([{"record_name":record_dir.name,
                                                               "header_row_index":record_df_copy_for_attrs.attrs["header_row_index_in_original_file"],
                                                               "depth_col_name_in_original_file": record_df_copy_for_attrs.attrs[
                                                               "adopted_Depth_column_name_in_original_file"],
                                                               "adopted_qc_column_name_in_original_file": record_df_copy_for_attrs.attrs["adopted_qc_column_name_in_original_file"],
                                                               "adopted_fs_column_name_in_original_file":record_df_copy_for_attrs.attrs["adopted_fs_column_name_in_original_file"],
                                                               "adopted_u_column_name_in_original_file":
                                                                   record_df_copy_for_attrs.attrs[
                                                                "adopted_u_column_name_in_original_file"],
                                                              "file_name":file_to_try.name}])

            loading_summary_df = partial_summary_df_helper(file_was_loaded=True,
                                                           loaded_file_type=file_to_try.suffix.lower(),
                                                           loaded_file_name=record_dir.name)
            return CptProcessingMetadata(spreadsheet_format_description_per_record,
                                         loading_summary_df,
                                         all_failed_loads_df)

        except(processing_helpers.FileProcessingError, ValueError, xlrd.compdoc.CompDocError, Exception) as e:

            error_as_string = str(e)

            if "-" not in error_as_string:
                error_as_string = "unknown_category - " + error_as_string

            all_failed_loads_df = pd.concat([all_failed_loads_df,
                    pd.DataFrame({"record_name": [record_dir.name],
                                 "file_type": [file_to_try.suffix.lower()],
                                 "file_name": [file_to_try.name],
                                 "category": [error_as_string.split("-")[0].strip()],
                                 "details": [error_as_string.split("-")[1].strip()]})], ignore_index=True)

            if file_to_try_index == len(files_to_try) - 1:
                ## No more files to try for this record
                loading_summary_df = partial_summary_df_helper(file_was_loaded=False,
                                                               loaded_file_type="N/A",
                                                               loaded_file_name="N/A")

                if len(ags_files_to_try) > 0:
                    continue
                else:
                    return CptProcessingMetadata(pd.DataFrame(), loading_summary_df, all_failed_loads_df)

    if len(ags_files_to_try) > 0:
        for file_to_try_index, file_to_try in enumerate(ags_files_to_try):
            try:
                record_df = load_ags(file_to_try, investigation_type)

                # record original name and location as attributes and columns
                record_df.attrs["original_file_name"] = file_to_try.name
                record_df.attrs["nzgd_meta_data"] = nzgd_meta_data_record
                record_df.attrs["max_depth"] = record_df["Depth"].max()
                record_df.attrs["min_depth"] = record_df["Depth"].min()

                record_df.insert(0, "multiple_measurements", 0)
                record_df.insert(0,"record_name",record_dir.name)
                record_df.insert(1, "latitude", nzgd_meta_data_record["Latitude"])
                record_df.insert(2, "longitude", nzgd_meta_data_record["Longitude"])

                record_df.reset_index(inplace=True, drop=True)
                record_df.to_parquet(parquet_output_dir / f"{record_dir.name}.parquet")

                loading_summary_df = partial_summary_df_helper(file_was_loaded=True,
                                                               loaded_file_type=file_to_try.suffix.lower(),
                                                               loaded_file_name=record_dir.name)

                ## Returning an empty dataframe in place of the spreadsheet_format_description_per_record
                ## (because ags file does not have a spreadsheet format)
                return CptProcessingMetadata(pd.DataFrame(), loading_summary_df, all_failed_loads_df)

            ## If the ags file is missing data, KeyError or UnboundLocalError will be raised
            except(FileProcessingError, KeyError, Exception) as e:

                error_as_string = str(e)

                if "-" not in error_as_string:
                    error_as_string = "unknown_category - " + error_as_string

                all_failed_loads_df = pd.concat([all_failed_loads_df,
                        pd.DataFrame({"record_name": [record_dir.name],
                                     "file_type": [file_to_try.suffix.lower()],
                                     "file_name": [file_to_try.name],
                                     "category": [error_as_string.split("-")[0].strip()],
                                     "details": [error_as_string.split("-")[1].strip()]})], ignore_index=True)

                if file_to_try_index == len(ags_files_to_try) - 1:
                    ## No more files to try for this record
                    loading_summary_df = partial_summary_df_helper(file_was_loaded=False,
                                                                   loaded_file_type="N/A",
                                                                   loaded_file_name="N/A")
                    return CptProcessingMetadata(pd.DataFrame(), loading_summary_df, all_failed_loads_df)