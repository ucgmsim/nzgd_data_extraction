"""
This script identifies irregularities with the downloaded files that might
indicate that some downloads were not successful. It does this by comparing the
downloaded file names against the list of available files for each record.
"""

from pathlib import Path

from tqdm import tqdm

import combine_download_notes_dicts


high_level_download_dir = Path("/home/arr65/data/nzgd/downloads_and_metadata/21102024")

################################################################################################################
### If the dictionaries need to be assembled from individual files

name_to_files_dict = combine_download_notes_dicts.combine_dicts((high_level_download_dir / "downloads_metadata" / "name_to_files_dicts_per_record").glob("*.toml"))

name_to_link_strs_dict = combine_download_notes_dicts.combine_dicts((high_level_download_dir / "downloads_metadata" / "name_to_files_dicts_per_record").glob("*.toml"))

# with open(
#     combined_dict_dir / f"combined_name_to_files_dict_batch{batch_num}.toml", "w"
# ) as toml_file:
#     toml.dump(name_to_files_dict, toml_file)
# with open(
#     combined_dict_dir / f"combined_name_to_link_strs_dict_batch{batch_num}.toml", "w"
# ) as toml_file:
#     toml.dump(name_to_link_strs_dict, toml_file)

## End dictionary assembly
################################################################################################################

# name_to_files_dict = toml.load(
#     Path("/home/arr65/data/nzgd/combined_dicts")
#     / f"combined_name_to_files_dict_batch{batch_num}.toml"
# )
# name_to_link_strs_dict = toml.load(
#     Path("/home/arr65/data/nzgd/combined_dicts")
#     / f"combined_name_to_link_strs_dict_batch{batch_num}.toml"
# )

if len(name_to_files_dict) != len(name_to_link_strs_dict):
    raise ValueError("Should have the same number of keys")

print()

id_not_in_dict = []
file_mismatch = []
files_and_links_mismatch = []

data_path = high_level_download_dir/"downloaded_files"
#downloaded_records = [record.name for record in data_path.glob("*")]

print()


with tqdm(total=len(list(data_path.glob("*")))) as pbar:

    for record_id_path in list(data_path.glob("*")):

        if record_id_path.is_dir():
            if (record_id_path.name not in name_to_files_dict.keys()) or (
                record_id_path.name not in name_to_link_strs_dict.keys()
            ):
                if record_id_path.name not in id_not_in_dict:
                    id_not_in_dict.append(record_id_path.name)
                    continue

            if (record_id_path.name in name_to_link_strs_dict.keys()) and (
                record_id_path.name in name_to_files_dict.keys()
            ):
                if len(name_to_link_strs_dict[record_id_path.name]) != len(
                    name_to_files_dict[record_id_path.name]
                ):
                    files_and_links_mismatch.append(record_id_path.name)
                    continue

        for downloaded_file in record_id_path.glob("*"):

            if downloaded_file.is_file():
                downloaded_file_name = downloaded_file.name

                if downloaded_file.parent.name in name_to_files_dict.keys():
                    if downloaded_file_name not in name_to_files_dict[downloaded_file.parent.name]:
                        ## Check if the file was uploaded twice so will be downloaded twice with (1) appended to the second file
                        ## by removing the number of characters that would correpond to this pattern and checking again
                        dot_index = downloaded_file_name.rfind(".")
                        download_file_name_no_num = (
                            downloaded_file_name[: dot_index - 4]
                            + downloaded_file_name[dot_index:]
                        )
                        if (
                            download_file_name_no_num
                            not in name_to_files_dict[downloaded_file.parent.name]
                        ):
                            file_mismatch.append(downloaded_file.parent.name)

        pbar.update(1)


print(f"Number of records not in the dictionary: {len(id_not_in_dict)}")
print(f"Number of records with different file names and : {len(file_mismatch)}")
print(
    f"Number of records with mismatch in files and links: {len(files_and_links_mismatch)}"
)

ids_with_download_issue = set(id_not_in_dict + file_mismatch + files_and_links_mismatch)

print(f"Total number of records with download issues: {len(ids_with_download_issue)}")

## Save a list of the records with download issues
with open(Path(high_level_download_dir/"downloads_metadata"/ f"records_with_download_issues.txt"), "w") as file:
        for record in ids_with_download_issue:
            file.write(f"{record}\n")


