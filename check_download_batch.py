"""
This script identifies irregularities with the downloaded files that might
indicate that some downloads were not successful. It does this by comparing the
downloaded file names against the list of available files for each record.
"""

from pathlib import Path

import toml
from tqdm import tqdm

import combine_download_notes_dicts

batch_num = 7

high_level_download_dir = Path(
    f"/home/arr65/data/nzgd/downloaded_files/download_run_{batch_num}"
)
combined_dict_dir = Path("/home/arr65/data/nzgd/combined_dicts")

################################################################################################################
### If the dictionaries need to be assembled from individual files

name_to_files_dict = combine_download_notes_dicts.combine_dicts(
    list(Path("/home/arr65/data/nzgd/name_to_files_dicts_per_record").iterdir())
)
name_to_link_strs_dict = combine_download_notes_dicts.combine_dicts(
    list(Path("/home/arr65/data/nzgd/name_to_link_str_dicts_per_record").iterdir())
)

with open(
    combined_dict_dir / f"combined_name_to_files_dict_batch{batch_num}.toml", "w"
) as toml_file:
    toml.dump(name_to_files_dict, toml_file)
with open(
    combined_dict_dir / f"combined_name_to_link_strs_dict_batch{batch_num}.toml", "w"
) as toml_file:
    toml.dump(name_to_link_strs_dict, toml_file)

## End dictionary assembly
################################################################################################################

name_to_files_dict = toml.load(
    Path("/home/arr65/data/nzgd/combined_dicts")
    / f"combined_name_to_files_dict_batch{batch_num}.toml"
)
name_to_link_strs_dict = toml.load(
    Path("/home/arr65/data/nzgd/combined_dicts")
    / f"combined_name_to_link_strs_dict_batch{batch_num}.toml"
)

assert len(name_to_files_dict) == len(
    name_to_link_strs_dict
), "Should have the same number of keys"

print()

id_not_in_dict = []
file_mismatch = []
files_and_links_mismatch = []


with tqdm(total=len(list(high_level_download_dir.iterdir()))) as pbar:

    for downloaded_record_folder in high_level_download_dir.iterdir():

        if downloaded_record_folder.is_dir():
            record_id = downloaded_record_folder.name
            if (record_id not in name_to_files_dict.keys()) or (
                record_id not in name_to_link_strs_dict.keys()
            ):
                if record_id not in id_not_in_dict:
                    print()
                    id_not_in_dict.append(record_id)
                    continue

            if (record_id in name_to_link_strs_dict.keys()) and (
                record_id in name_to_files_dict.keys()
            ):
                if len(name_to_link_strs_dict[record_id]) != len(
                    name_to_files_dict[record_id]
                ):
                    files_and_links_mismatch.append(record_id)
                    continue

            for downloaded_file in downloaded_record_folder.iterdir():

                if downloaded_file.is_file():
                    downloaded_file_name = downloaded_file.name

                    if record_id in name_to_files_dict.keys():
                        if downloaded_file_name not in name_to_files_dict[record_id]:
                            ## Check if the file was uploaded twice so will be downloaded twice with (1) appended to the second file
                            ## by removing the number of characters that would correpond to this pattern and checking again
                            dot_index = downloaded_file_name.rfind(".")
                            download_file_name_no_num = (
                                downloaded_file_name[: dot_index - 4]
                                + downloaded_file_name[dot_index:]
                            )
                            if (
                                download_file_name_no_num
                                not in name_to_files_dict[record_id]
                            ):
                                file_mismatch.append(record_id)

        pbar.update(1)


print(f"Number of records not in the dictionary: {len(id_not_in_dict)}")
print(f"Number of records with different file names and : {len(file_mismatch)}")
print(
    f"Number of records with mismatch in files and links: {len(files_and_links_mismatch)}"
)

ids_with_download_issue = set(id_not_in_dict + file_mismatch + files_and_links_mismatch)

print(f"Total number of records with download issues: {len(ids_with_download_issue)}")

## Save a list of the records with download issues
with open(
    Path("/home/arr65/data/nzgd/combined_dicts")
    / f"records_with_download_issues_in_batch_{batch_num}.txt",
    "w",
) as file:
    for record in ids_with_download_issue:
        file.write(f"{record}\n")

print()
