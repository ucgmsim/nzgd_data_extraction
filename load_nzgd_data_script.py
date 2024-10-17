"""
Script to load NZGD data and convert to a standard format
"""
import numpy as np
from pathlib import Path
import toml
import loading_funcs_for_nzgd_data
import xlrd
from tqdm import tqdm
import natsort
import pandas as pd
import functools

def summary_df_helper(summary_df, record_dir_name, file_was_loaded, loaded_file_type,
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

nzgd_index_df = pd.read_csv(Path("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_25092024_1043.csv"))
output_dir = Path("/home/arr65/data/nzgd/standard_format_batch50/cpt")
### !!! GO HERE

parquet_output_dir = output_dir / "data"
metadata_output_dir = output_dir / "metadata"

parquet_output_dir.mkdir(exist_ok=True, parents=True)
metadata_output_dir.mkdir(exist_ok=True, parents=True)

downloaded_files = Path("/home/arr65/data/nzgd/downloaded_files/cpt")

#previous_loading_summary = pd.read_csv(Path("/home/arr65/data/nzgd/standard_format_batch30/cpt/metadata") / "loading_summary.csv")

### !!! GO HERE !!!
#previously_converted_filenames = list(previous_loading_summary[previous_loading_summary["file_was_loaded"]==True]["record_name"])
previously_converted_filenames = []

previously_converted_records = []
for filename in previously_converted_filenames:
    file_name_parts = filename.split("_")
    previously_converted_records.append(f"{file_name_parts[0]}_{file_name_parts[1]}")

records_to_convert = []
for record_dir in natsort.natsorted(list(downloaded_files.glob("*"))):
    if record_dir.name not in previously_converted_records:
        records_to_convert.append(record_dir)

spreadsheet_format_description = pd.DataFrame()

all_failed_loads_df = pd.DataFrame(columns=["record_name", "file_type", "file_name", "category", "details"])

loading_summary_df = pd.DataFrame(columns=["record_name", "file_was_loaded", "loaded_file_type", "loaded_file_name",
                                           "only_has_pdf","num_pdf_files", "num_cpt_files", "num_ags_files",
                                           "num_xls_files", "num_xlsx_files", "num_csv_files", "num_txt_files",
                                           "num_other_files"])
### !!! GO HERE
record_counter = 0
for record_dir in tqdm(records_to_convert):

#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_212092")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_125853")]:

#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_2497")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_110939")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_9326")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_60575")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_17546")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_2497")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_24230")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_72538")]:
#for record_dir in [Path("/home/arr65/data/nzgd/downloaded_files/cpt/CPT_187908")]:

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

    partial_summary_df_helper = functools.partial(summary_df_helper, record_dir_name=record_dir.name,
                                                  pdf_file_list=pdf_file_list, cpt_file_list=cpt_file_list, ags_file_list=ags_file_list,
                                                  xls_file_list=xls_file_list, xlsx_file_list=xlsx_file_list, csv_file_list=csv_file_list,
                                                  txt_file_list=txt_file_list, unknown_list=unknown_list)

    if record_dir.name not in nzgd_index_df["ID"].values:
        # it has been removed from the NZGD so just skip it and don't try to load any files
        continue

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
                record_df = loading_funcs_for_nzgd_data.load_ags(file_to_try)

                # record original name and location as attributes and columns
                record_df.attrs["original_file_name"] = file_to_try.name
                record_df.attrs["nzgd_meta_data"] = nzgd_meta_data_record
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

    ### spreadsheet files
    files_to_try = list(record_dir.glob("*.xls")) + list(record_dir.glob("*.XLS")) + \
                   list(record_dir.glob("*.xlsx")) + list(record_dir.glob("*.XLSX")) + \
                   list(record_dir.glob("*.csv")) + list(record_dir.glob("*.CSV")) +\
                   list(record_dir.glob("*.txt")) + list(record_dir.glob("*.Txt")) +\
                   list(record_dir.glob("*.TXT"))

    for file_to_try_index, file_to_try in enumerate(files_to_try):
        try:
            xls_file_load_attempted = True
            record_df = loading_funcs_for_nzgd_data.load_cpt_spreadsheet_file(file_to_try)

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
                                                               "header_row_index":record_df.attrs["header_row_index_in_original_file"],
                                                               "depth_col_name_in_original_file": record_df.attrs[
                                                               "adopted_depth_column_name_in_original_file"],
                                                               "adopted_cone_resistance_column_name_in_original_file": record_df.attrs["adopted_cone_resistance_column_name_in_original_file"],
                                                               "adopted_sleeve_friction_column_name_in_original_file":record_df.attrs["adopted_sleeve_friction_column_name_in_original_file"],
                                                               "adopted_porewater_pressure_column_name_in_original_file":
                                                                   record_df.attrs[
                                                                "adopted_porewater_pressure_column_name_in_original_file"],
                                                              "file_name":file_to_try.name}])

            spreadsheet_format_description = pd.concat([spreadsheet_format_description,
                                                        spreadsheet_format_description_per_record],ignore_index=True)

            loading_summary_df = partial_summary_df_helper(loading_summary_df, file_was_loaded=True,
                                                           loaded_file_type=file_to_try.suffix.lower(),
                                                           loaded_file_name=record_dir.name)

            break

        except(ValueError, xlrd.compdoc.CompDocError, Exception) as e:

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
                #meta_xls_failed_to_load.append(f"{record_dir.name}, {file_to_try.name}, {e}")
                xls_load_failed = True
            else:
                # there are other files to try so continue to the next file
                continue

spreadsheet_format_description.to_csv(metadata_output_dir / "spreadsheet_format_description.csv", index=False)
all_failed_loads_df.to_csv(metadata_output_dir / "all_failed_loads.csv", index=False)
loading_summary_df.to_csv(metadata_output_dir / "loading_summary.csv", index=False)