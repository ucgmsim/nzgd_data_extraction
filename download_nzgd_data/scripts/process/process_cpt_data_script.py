"""
Script to load NZGD data and convert to a standard format
"""

from pathlib import Path
import natsort
import pandas as pd
import functools
from tqdm import tqdm
import xlrd

from download_nzgd_data.lib import process_cpt_data, processing_helpers

investigation_type = processing_helpers.InvestigationType.cpt

nzgd_index_df = pd.read_csv(Path("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_23102024_1042.csv"))
output_dir = Path(f"/home/arr65/data/nzgd/processed_data_test/{investigation_type}")

# if output_dir.exists():
#     raise ValueError("Output directory already exists.")

### !!! GO HERE
parquet_output_dir = output_dir / "data"
metadata_output_dir = output_dir / "metadata"

parquet_output_dir.mkdir(exist_ok=True, parents=True)
metadata_output_dir.mkdir(exist_ok=True, parents=True)

downloaded_files = Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt")

#previous_loading_summary = pd.read_csv(Path("/home/arr65/data/nzgd/standard_format_batch30/cpt/metadata") / "loading_summary.csv")

### !!! GO HERE !!!
#records_to_skip = list(previous_loading_summary[previous_loading_summary["file_was_loaded"]==True]["record_name"])
#records_to_skip = pd.read_csv("/home/arr65/src/download_nzgd_data/download_nzgd_data/resources/cpt_loaded_from_spreadsheet_in_23102024_1042.csv")["record_name"].to_list()
records_to_skip = []

records_to_process = []
for record_dir in natsort.natsorted(list(downloaded_files.glob("*"))):
    if record_dir.name not in records_to_skip:
        records_to_process.append(record_dir)

downloaded_record_names = set([record_dir.name for record_dir in records_to_process])

# A small number of records have been removed from the NZGD after they were downloaded.
# These records were likely removed for a reason such data quality or permission issues, so they are not considered.
records_currently_in_nzgd = set(nzgd_index_df["ID"].values)

records_that_have_been_removed = downloaded_record_names - records_currently_in_nzgd

if len(records_that_have_been_removed) > 0:
    print("The following records have been removed from the NZGD and will not be processed:")
    for removed_record in records_that_have_been_removed:
        print(removed_record)

    ## Remove the records that have been removed from the list of records to process
    records_to_process = [record_dir for record_dir in records_to_process if record_dir.name not in records_that_have_been_removed]


## Create dataframes to store metadata
spreadsheet_format_description = pd.DataFrame()

all_failed_loads_df = pd.DataFrame(columns=["record_name", "file_type", "file_name", "category", "details"])

loading_summary_df = pd.DataFrame(columns=["record_name", "file_was_loaded", "loaded_file_type", "loaded_file_name",
                                           "only_has_pdf","num_pdf_files", "num_cpt_files", "num_ags_files",
                                           "num_xls_files", "num_xlsx_files", "num_csv_files", "num_txt_files",
                                           "num_other_files"])

### !!! GO HERE
record_counter = 0
#for record_dir in tqdm(records_to_process):
#for record_dir in [Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt/CPT_22400")]:
for record_dir in [Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/scpt/SCPT_14539")]:

#for record_dir in [Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt/CPT_158096")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt/CPT_223176")]:

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

    partial_summary_df_helper = functools.partial(processing_helpers.make_summary_df, record_dir_name=record_dir.name,
                                                  pdf_file_list=pdf_file_list, cpt_file_list=cpt_file_list, ags_file_list=ags_file_list,
                                                  xls_file_list=xls_file_list, xlsx_file_list=xlsx_file_list, csv_file_list=csv_file_list,
                                                  txt_file_list=txt_file_list, unknown_list=unknown_list)

    nzgd_meta_data_record = nzgd_index_df[nzgd_index_df["ID"]==record_dir.name].to_dict(orient="records")[0]

    has_loaded_a_file_for_this_record = False

    ags_file_load_attempted = False
    xls_file_load_attempted = False

    ags_load_failed = False
    xls_load_failed = False

    record_counter += 1
    if record_counter % 100 == 0:

        spreadsheet_format_description.to_csv(metadata_output_dir / "spreadsheet_format_description.csv", index=False)
        all_failed_loads_df.to_csv(metadata_output_dir / "all_failed_loads.csv", index=False)
        loading_summary_df.to_csv(metadata_output_dir / "loading_summary.csv", index=False)

    ### Skip this record if the only available files are pdf
    if len(list(record_dir.glob("*.pdf"))) == len(list(record_dir.glob("*"))):
        loading_summary_df = partial_summary_df_helper(loading_summary_df, file_was_loaded=False,
                                                       loaded_file_type="N/A", loaded_file_name="N/A")
        continue

    ### ags files
    files_to_try = list(record_dir.glob("*.ags")) + list(record_dir.glob("*.AGS"))

    if len(files_to_try) > 0:
        for file_to_try in files_to_try:
            try:
                ags_file_load_attempted = True
                record_df = process_cpt_data.load_ags(file_to_try, investigation_type)
                print()


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

                has_loaded_a_file_for_this_record = True

                loading_summary_df = partial_summary_df_helper(loading_summary_df, file_was_loaded=True,
                                                               loaded_file_type=file_to_try.suffix.lower(),
                                                               loaded_file_name=file_to_try.name)
                continue

            ## If the ags file is missing data, KeyError or UnboundLocalError will be raised
            except(ValueError) as e:

                error_as_string = str(e)

                if "-" not in error_as_string:
                    error_as_string = "unknown_category - " + error_as_string

                all_failed_loads_df = pd.concat([all_failed_loads_df,
                        pd.DataFrame({"record_name": [record_dir.name],
                                     "file_type": [file_to_try.suffix.lower()],
                                     "file_name": [file_to_try.name],
                                     "category": [error_as_string.split("-")[0].strip()],
                                     "details": [error_as_string.split("-")[1].strip()]})], ignore_index=True)

                loading_summary_df = partial_summary_df_helper(loading_summary_df, file_was_loaded=False,
                                                               loaded_file_type="N/A", loaded_file_name="N/A")

                ags_load_failed = True


    if has_loaded_a_file_for_this_record:
        continue

    # ### spreadsheet files
    files_to_try = list(record_dir.glob("*.xls")) + list(record_dir.glob("*.XLS")) + \
                   list(record_dir.glob("*.xlsx")) + list(record_dir.glob("*.XLSX")) + \
                   list(record_dir.glob("*.csv")) + list(record_dir.glob("*.CSV")) +\
                   list(record_dir.glob("*.txt")) + list(record_dir.glob("*.Txt")) +\
                   list(record_dir.glob("*.TXT"))

    for file_to_try_index, file_to_try in enumerate(files_to_try):
        try:
            xls_file_load_attempted = True
            record_df_list = process_cpt_data.load_cpt_spreadsheet_file(file_to_try)
            record_df_copy_for_attrs = record_df_list[0].copy()

            record_df = pd.DataFrame()
            for record_df_idx in range(len(record_df_list)):
                record_df_list[record_df_idx].insert(0, "multiple_measurements", record_df_idx)
                if record_df_idx == 0:
                    record_df = record_df_list[record_df_idx]
                else:
                    record_df = pd.concat([record_df, record_df_list[record_df_idx]], ignore_index=True)

            record_df.attrs["max_depth"] = record_df["Depth"].max()
            record_df.attrs["min_depth"] = record_df["Depth"].min()
            # record original name and location as attributes and columns
            record_df.attrs["original_file_name"] = file_to_try.name
            record_df.attrs["nzgd_meta_data"] = nzgd_meta_data_record
            record_df.insert(0, "record_name", record_dir.name)
            record_df.insert(1, "latitude", nzgd_meta_data_record["Latitude"])
            record_df.insert(2, "longitude", nzgd_meta_data_record["Longitude"])

            record_df.reset_index(inplace=True, drop=True)

            record_df.to_parquet(parquet_output_dir / f"{record_dir.name}.parquet")
            has_loaded_a_file_for_this_record = True
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

            spreadsheet_format_description = pd.concat([spreadsheet_format_description,
                                                        spreadsheet_format_description_per_record],ignore_index=True)

            loading_summary_df = partial_summary_df_helper(loading_summary_df, file_was_loaded=True,
                                                           loaded_file_type=file_to_try.suffix.lower(),
                                                           loaded_file_name=record_dir.name)

            break

        except(processing_helpers.FileProcessingError, ValueError, xlrd.compdoc.CompDocError, Exception) as e:

            loading_summary_df = partial_summary_df_helper(loading_summary_df, file_was_loaded=False,
                                                           loaded_file_type="N/A",
                                                           loaded_file_name="N/A")
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
                # it's the last file to try
                xls_load_failed = True
            else:
                # there are other files to try so continue to the next file
                continue

spreadsheet_format_description.to_csv(metadata_output_dir / "spreadsheet_format_description.csv", index=False)
all_failed_loads_df.to_csv(metadata_output_dir / "all_failed_loads.csv", index=False)
loading_summary_df.to_csv(metadata_output_dir / "loading_summary.csv", index=False)