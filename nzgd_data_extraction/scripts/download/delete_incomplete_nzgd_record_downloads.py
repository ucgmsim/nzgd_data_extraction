from pathlib import Path
import shutil


def delete_partial_downloads(partial_download_ids: list[str], high_level_download_dir: Path):

    num_deletions = 0

    for partial_download_id in partial_download_ids:

        partial_download_dir = high_level_download_dir / partial_download_id

        if partial_download_dir.exists():
            print(f"Deleting {partial_download_dir}")
            shutil.rmtree(partial_download_dir)
            num_deletions += 1

    print("Number of deletions: ", num_deletions)
    print()

def remove_partial_download_ids_from_dict(partial_download_ids: list[str], name_to_files_dict: dict):

    num_deletions = 0

    for partial_download_id in partial_download_ids:

        if partial_download_id in name_to_files_dict.keys():
            del name_to_files_dict[partial_download_id]
            num_deletions += 1

    print("Number of deletions: ", num_deletions)

    return name_to_files_dict

if __name__ == "__main__":

    high_level_download_dir = Path("/home/arr65/data/nzgd/downloads_and_metadata/21102024")

    downloaded_record_ids = [record.name for record in high_level_download_dir.glob("*")]

    print()

    # batch_num = 7
    #
    # file_path = Path(f"/home/arr65/data/nzgd/combined_dicts/records_with_download_issues_in_batch_{batch_num}.txt")
    # print()
    # with open(file_path, "r") as file:
    #     partial_download_ids = file.readlines()
    # partial_download_ids = [line.strip() for line in partial_download_ids]
    #
    # high_level_download_dir = Path(f"/home/arr65/data/nzgd/downloaded_files/download_run_{batch_num}")
    # delete_partial_downloads(partial_download_ids, high_level_download_dir)
    #
    # name_to_files_dict = toml.load(Path("/home/arr65/data/nzgd/combined_dicts") / f"combined_name_to_files_dict_batch{batch_num}.toml")
    # name_to_files_dict = remove_partial_download_ids_from_dict(partial_download_ids, name_to_files_dict)
    #
    # name_to_link_strs_dict = toml.load(Path("/home/arr65/data/nzgd/combined_dicts") / f"combined_name_to_link_strs_dict_batch{batch_num}.toml")
    # name_to_link_strs_dict = remove_partial_download_ids_from_dict(partial_download_ids, name_to_link_strs_dict)
    #
    # with open(Path("/home/arr65/data/nzgd/combined_dicts")/f"combined_name_to_files_dict_batch{batch_num}.toml", "w") as toml_file:
    #     toml.dump(name_to_files_dict, toml_file)
    #
    # with open(Path("/home/arr65/data/nzgd/combined_dicts")/f"combined_name_to_link_strs_dict_batch{batch_num}.toml", "w") as toml_file:
    #     toml.dump(name_to_link_strs_dict, toml_file)
    #
    # print("Done")