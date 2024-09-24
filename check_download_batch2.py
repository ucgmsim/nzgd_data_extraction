from pathlib import Path
import toml
from tqdm import tqdm

high_level_download_dir = Path("/home/arr65/data/nzgd/downloaded_files/download_run_4")

name_to_files_dicts_dir = toml.load(Path("/home/arr65/data/nzgd/combined_dicts") / "combined_name_to_files_dict.toml")

name_to_link_strs_dict = toml.load(Path("/home/arr65/data/nzgd/combined_dicts") / "combined_name_to_link_strs_dict.toml")

assert len(name_to_files_dict) == len(name_to_link_strs_dict), "Should have the same number of keys"

print()

id_not_in_dict = []
file_mismatch = []
files_and_links_mismatch = []


with tqdm(total=len(list(high_level_download_dir.iterdir()))) as pbar:

    for downloaded_record_folder in high_level_download_dir.iterdir():

        if downloaded_record_folder.is_dir():

            for downloaded_file in downloaded_record_folder.iterdir():

                if downloaded_file.is_file():

                    record_id = downloaded_record_folder.name
                    downloaded_file_name = downloaded_file.name

                    if record_id not in name_to_files_dict.keys():
                        id_not_in_dict.append(record_id)
                    if record_id in name_to_files_dict.keys():
                        if downloaded_file_name not in name_to_files_dict[record_id]:
                            file_mismatch.append(record_id)

                    if (record_id in name_to_link_strs_dict.keys()) and (record_id in name_to_files_dict.keys()):
                        if len(name_to_link_strs_dict[record_id]) != len(name_to_files_dict[record_id]):
                            files_and_links_mismatch.append(record_id)
        pbar.update(1)


print(f"Number of records not in the dictionary: {len(id_not_in_dict)}")
print(f"Number of records with mismatch in files: {len(file_mismatch)}")
print(f"Number of records with mismatch in files and links: {len(files_and_links_mismatch)}")

ids_with_download_issue = set(id_not_in_dict + file_mismatch + files_and_links_mismatch)

print(f"Total number of records with download issues: {len(ids_with_download_issue)}")

## Save a list of the records with download issues
# with open(Path("/home/arr65/data/nzgd/combined_dicts")/"records_with_download_issues.txt", "w") as file:
#     for record in ids_with_download_issue:
#         file.write(f"{record}\n")

print()

