"""
Combine several toml files containing dictionaries into a single dictionary.
"""

from pathlib import Path
import toml


def combine_dicts(toml_files:list[Path]):

    """
    Combine several toml files containing dictionaries into a single dictionary.

    Parameters
    ----------
    toml_files : list[Path]
        Paths to the toml files to be combined.

    Returns
    -------
    dict
        A single dictionary containing the combined information from the
        input toml files.
    """

    combined_dict = {}

    for toml_file in toml_files:
        combined_dict = {**combined_dict, **toml.load(toml_file)}

    return combined_dict


if __name__ == "__main__":

    toml_file_paths = []
    for batch_num in range(3,8,1):
        #toml_file_paths.append(Path("/home/arr65/data/nzgd/combined_dicts")/f"combined_name_to_files_dict_batch{batch_num}.toml")
        toml_file_paths.append(Path("/home/arr65/data/nzgd/combined_dicts")/f"combined_name_to_link_strs_dict_batch{batch_num}.toml")

    combined_files_dict = combine_dicts(toml_file_paths)

    #with open(Path("/home/arr65/data/nzgd/combined_dicts")/"record_id_to_file_name_dict_25_Sept_2024.toml", "w") as toml_file:
    with open(Path("/home/arr65/data/nzgd/combined_dicts") / "record_id_to_link_str_dict_25_Sept_2024.toml",
              "w") as toml_file:
        toml.dump(combined_files_dict, toml_file)