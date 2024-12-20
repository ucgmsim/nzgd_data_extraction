from pathlib import Path
from natsort import natsorted

cpt_dir = Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt")

num_cpt_records = len(list(cpt_dir.glob("*")))

with open("/home/arr65/data/nzgd/resources/example_spreadsheets/cpt_records_with_area_ratio.txt", "r") as file:
    lines = file.readlines()

record_names = []
record_names_and_files = []

for line in lines:
    record_names.append(Path(line.strip("\n")).parent.name)
    record_names_and_files.append(Path(line.strip()).parent.name + "/" + Path(line.strip()).name)

cpt_records_with_area_ratio = natsorted(list(set(record_names)))

num_cpt_records_with_area_ratio = len(cpt_records_with_area_ratio)

print(f"Number of CPT records with area ratio: {num_cpt_records_with_area_ratio}")
print(f"Percent of CPT records with area ratio: {100*num_cpt_records_with_area_ratio / num_cpt_records:.2f}%")

with open("/home/arr65/data/nzgd/resources/example_spreadsheets/cpt_records_with_area_ratio.txt", "w") as file:
    for line in record_names_and_files:
        file.write(line + "\n")


print()