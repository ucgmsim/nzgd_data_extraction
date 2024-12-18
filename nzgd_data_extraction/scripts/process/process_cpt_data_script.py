"""
Script to load NZGD data and convert to a standard format
"""

import functools
import multiprocessing as mp
from pathlib import Path

import natsort
import pandas as pd
from tqdm import tqdm

from nzgd_data_extraction.lib import (
    process_cpt_data,
    processing_helpers,
)

if __name__ == "__main__":

    nzgd_index_df = pd.read_csv(
        Path(
            "/home/arr65/data/nzgd/resources/nzgd_index_files/csv_files/"
            "NZGD_Investigation_Report_08112024_1017.csv"
        )
    )

    for investigation_type in [
        processing_helpers.InvestigationType.cpt,
        processing_helpers.InvestigationType.scpt,
    ]:

        print(f"Extracting data from {investigation_type} records...")

        output_dir = Path(f"/home/arr65/data/nzgd/extracted_data/{investigation_type}")

        if output_dir.exists():
            raise ValueError(
                "Output directory already exists. Delete or rename previous output and try again."
            )

        extracted_data_per_record_output_path = output_dir / "extracted_data_per_record"
        extraction_failures_per_record_output_path = (
            output_dir / "extraction_failures_per_record_per_record"
        )

        extracted_data_per_record_output_path.mkdir(exist_ok=True, parents=True)
        extraction_failures_per_record_output_path.mkdir(exist_ok=True, parents=True)

        downloaded_files = Path(
            f"/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/{investigation_type}"
        )

        ## If records should be skipped, enter them here
        records_to_skip = []

        records_to_process = []
        for record_dir in natsort.natsorted(list(downloaded_files.glob("*"))):
            if record_dir.name not in records_to_skip:
                records_to_process.append(record_dir)

        downloaded_record_names = set(
            [record_dir.name for record_dir in records_to_process]
        )

        ## A small number of records have been removed from the NZGD after they were downloaded.
        ## These records were likely removed for a reason such data quality or permission issues, so they are not considered.
        records_currently_in_nzgd = set(nzgd_index_df["ID"].values)
        records_that_have_been_removed = (
            downloaded_record_names - records_currently_in_nzgd
        )

        if len(records_that_have_been_removed) > 0:
            print(
                "The following records have been removed from the NZGD and will not be processed:"
            )
            for removed_record in records_that_have_been_removed:
                print(removed_record)

            ## Remove the records that have been removed from the list of records to process
            records_to_process = [
                record_dir
                for record_dir in records_to_process
                if record_dir.name not in records_that_have_been_removed
            ]

        process_one_record_partial = functools.partial(
            process_cpt_data.extract_all_data_for_one_record,
            investigation_type=investigation_type,
            extracted_data_per_record_output_path=extracted_data_per_record_output_path,
            extraction_failures_per_record_output_path=extraction_failures_per_record_output_path,
        )
        results = []
        num_workers = 8
        with mp.Pool(processes=num_workers) as pool:
            results.extend(
                list(
                    tqdm(
                        pool.imap(process_one_record_partial, records_to_process),
                        total=len(records_to_process),
                    )
                )
            )
