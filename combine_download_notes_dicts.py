from pathlib import Path
import toml


def combine_dicts(toml_files:list[Path]):

    combined_dict = {}

    for toml_file in toml_files:
        combined_dict = {**combined_dict, **toml.load(toml_file)}

    return combined_dict


if __name__ == "__main__":

    # dict_dirs = [Path("/home/arr65/data/nzgd/name_to_files_dicts"),
    #              Path("/home/arr65/data/nzgd/nzgd_from_laptop/nzgd/name_to_files_dicts")]

    dict_dirs = [Path("/home/arr65/data/nzgd/name_to_link_str_dicts"),
                 Path("/home/arr65/data/nzgd/nzgd_from_laptop/nzgd/name_to_link_str_dicts")]

    toml_files = [list(x.iterdir()) for x in dict_dirs]

    toml_files = [item for sublist in toml_files for item in sublist]

    combined_files_dict = combine_dicts(toml_files)

    # with open(Path("/home/arr65/data/nzgd/combined_dicts")/"combined_name_to_files_dict.toml", "w") as toml_file:
    #     toml.dump(combined_files_dict, toml_file)

    with open(Path("/home/arr65/data/nzgd/combined_dicts")/"combined_name_to_link_strs_dict.toml", "w") as toml_file:
        toml.dump(combined_files_dict, toml_file)