"""
Script to load NZGD data and convert to a standard format
"""

from pathlib import Path
import natsort
import pandas as pd
import functools
from tqdm import tqdm
import multiprocessing as mp


from nzgd_data_extraction.lib import process_cpt_data, processing_helpers

if __name__ == "__main__":

    for investigation_type in [processing_helpers.InvestigationType.cpt, processing_helpers.InvestigationType.scpt]:
    #for investigation_type in [processing_helpers.InvestigationType.cpt]:

        nzgd_index_df = pd.read_csv(Path("/home/arr65/data/nzgd/resources/nzgd_index_files/csv_files/"
                                         "NZGD_Investigation_Report_08112024_1017.csv"))

        output_dir = Path(f"/home/arr65/data/nzgd/processed_data/{investigation_type}")

        if output_dir.exists():
            raise ValueError("Output directory already exists. Delete or rename previous output and try again.")

        parquet_output_path = output_dir / "data"
        metadata_output_dir = output_dir / "metadata"

        parquet_output_path.mkdir(exist_ok=True, parents=True)
        metadata_output_dir.mkdir(exist_ok=True, parents=True)

        downloaded_files = Path(f"/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/{investigation_type}")

        ## If records should be skipped, enter them here
        records_to_skip = []

        records_to_process = []
        for record_dir in natsort.natsorted(list(downloaded_files.glob("*"))):
            if record_dir.name not in records_to_skip:
                records_to_process.append(record_dir)

        downloaded_record_names = set([record_dir.name for record_dir in records_to_process])

        ## A small number of records have been removed from the NZGD after they were downloaded.
        ## These records were likely removed for a reason such data quality or permission issues, so they are not considered.
        records_currently_in_nzgd = set(nzgd_index_df["ID"].values)
        records_that_have_been_removed = downloaded_record_names - records_currently_in_nzgd

        if len(records_that_have_been_removed) > 0:
            print("The following records have been removed from the NZGD and will not be processed:")
            for removed_record in records_that_have_been_removed:
                print(removed_record)

            ## Remove the records that have been removed from the list of records to process
            records_to_process = [record_dir for record_dir in records_to_process if record_dir.name not in records_that_have_been_removed]

        # previous_failed_loads_df = pd.read_csv(Path("/home/arr65/data/nzgd/processed_data/cpt/metadata") / "all_failed_loads.csv")
        # actual_records_to_process = previous_failed_loads_df[previous_failed_loads_df["category"] == "unknown_category"]["record_name"].values

        #records_to_process = [Path(f"/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt/{x}") for x in actual_records_to_process]
        # records_to_process = [Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt/CPT_26432")]



        process_one_record_partial = functools.partial(process_cpt_data.process_one_record,
                                                       parquet_output_dir=parquet_output_path,
                                                       nzgd_index_df=nzgd_index_df,
                                                       investigation_type=investigation_type)
        results = []
        num_workers = 8
        with mp.Pool(processes=num_workers) as pool:
            results.extend(list(tqdm(pool.imap(process_one_record_partial, records_to_process),
                                     total=len(records_to_process))))


        ### concatenate all the metadata dataframes
        print("Concatenating the metadata dataframes")

        spreadsheet_format_descriptions_dfs = []
        all_failed_loads_dfs = []
        loading_summary_dfs = []

        for result in tqdm(results):
            spreadsheet_format_descriptions_dfs.append(result.spreadsheet_format_description_per_record)
            all_failed_loads_dfs.append(result.all_failed_loads_df)
            loading_summary_dfs.append(result.loading_summary_df)

        ## If processing a small number of CPT records, some dataframes will be empty
        ## which will cause an error when concatenating, so only concatenate if the dataframes are not empty
        if all(x.size == 0 for x in spreadsheet_format_descriptions_dfs):
            spreadsheet_format_descriptions_df = pd.DataFrame()
        else:
            spreadsheet_format_descriptions_df = pd.concat(spreadsheet_format_descriptions_dfs, ignore_index=True)

        if all(x.size == 0 for x in all_failed_loads_dfs):
            all_failed_loads_df = pd.DataFrame()
        else:
            all_failed_loads_df = pd.concat(all_failed_loads_dfs, ignore_index=True)

        if all(x.size == 0 for x in loading_summary_dfs):
            loading_summary_df = pd.DataFrame()
        else:
            loading_summary_df = pd.concat(loading_summary_dfs, ignore_index=True)

        spreadsheet_format_descriptions_df.to_csv(metadata_output_dir / "spreadsheet_format_description.csv", index=False)
        all_failed_loads_df.to_csv(metadata_output_dir / "all_failed_loads.csv", index=False)
        loading_summary_df.to_csv(metadata_output_dir / "loading_summary.csv", index=False)

        if len(spreadsheet_format_descriptions_df) > 0:

            unique_spreadsheet_format_descriptions_df = spreadsheet_format_descriptions_df.drop(columns=["header_row_index", "record_name", "file_name"])

            # get number of unique rows in spreadsheet_format_description_df
            unique_spreadsheet_format_descriptions_df = unique_spreadsheet_format_descriptions_df.drop_duplicates()
            unique_spreadsheet_format_descriptions_df.to_csv(metadata_output_dir / "spreadsheet_format_description_unique.csv",
                                                            index=False)

