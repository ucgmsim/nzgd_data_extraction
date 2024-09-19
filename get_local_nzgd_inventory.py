import os
import pandas as pd
from pathlib import Path
import toml
from tqdm import tqdm

url_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/qgis_export_index_18_sep_2024.csv")

url_df = url_df[url_df["TypeofInvestigation"] == "CPT"]

url_df = url_df[0:5]

name_to_files_dict = {}

local_nzgd_data_dir = Path("/home/arr65/data/nzgd/downloaded_files/organized_downloads")

downloaded_files = []

ags = os.listdir("/home/arr65/data/nzgd/downloaded_files/organized_downloads/ags")
pdf = os.listdir("/home/arr65/data/nzgd/downloaded_files/organized_downloads/pdf")
txt = os.listdir("/home/arr65/data/nzgd/downloaded_files/organized_downloads/txt")
xls = os.listdir("/home/arr65/data/nzgd/downloaded_files/organized_downloads/xls")

combined_list = [x for x in ags if "CPT" in x] + [x for x in pdf if "CPT" in x] + [x for x in txt if "CPT" in x] + [x for x in xls if "CPT" in x]
#combined_list = combined_list[0:5]

with tqdm(total=len(combined_list)) as pbar:

    for file_name in combined_list:

        right_find_index = file_name.rfind("_")

        cpt_name = file_name[0:right_find_index]

        if cpt_name not in name_to_files_dict.keys():
            name_to_files_dict[cpt_name] = [file_name]

        else:
            name_to_files_dict[cpt_name].append(file_name)

        pbar.update(1)

### Dump the dictionary to the TOML file
with open("/home/arr65/data/nzgd/inventory_files/cpt_inventory_19_Sept_2024.toml", 'w') as toml_file:
    toml.dump(name_to_files_dict, toml_file)

with tqdm(total=len(combined_list)) as pbar:

    num_cpts_with_only_pdf = 0
    for key, value in name_to_files_dict.items():

        if len(value) == 1:

            if "pdf" in value[0]:
                num_cpts_with_only_pdf += 1

        pbar.update(1)

print(f"Total number of downloaded CPTs: {len(name_to_files_dict.keys())}")
print(f"Number of CPTs with only a pdf file: {num_cpts_with_only_pdf}")
print(f"Number of CPTs with a data file: {len(name_to_files_dict.keys()) - num_cpts_with_only_pdf}")


