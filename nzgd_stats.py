import toml
import enum
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import natsort
import numpy as np


class InvestigationType(enum.StrEnum):
    """
    Values for the investigation type of the record
    """

    CPT = "CPT"
    SCPT = "SCPT"
    BH = "BH"
    VsVp = "VsVp"


def find_records_with_only_these_file_types(record_id_to_files: dict[str, list[str]], files_types: list[str],
                                            investigation_type: InvestigationType):
    """
    Identifies records that contain only the specified file types for a given investigation type.

    Parameters
    ----------
    record_id_to_files : dict[str, list[str]]
        A dictionary where keys are record IDs. For each key, the corresponding value is a list of file names for that record.
    files_types : list[str]
        File types to identify (e.g., "pdf").
    investigation_type : InvestigationType
        The type of investigation to filter records by.

    Returns
    -------
    list
        Record IDs that contain only the specified file types.
    """

    identified_records = []
    for record_id, file_list in record_id_to_files.items():
        if investigation_type in record_id:
            num_files_for_record = len(file_list)
            num_non_data_files_for_record = 0
            for file in file_list:
                file_type = file.split(".")[-1]
                if file_type in files_types:
                    num_non_data_files_for_record += 1

            if num_files_for_record == num_non_data_files_for_record:
                identified_records.append(record_id)

    return identified_records

record_id_to_files = toml.load("/home/arr65/data/nzgd/record_name_to_file_dicts/record_id_to_file_name_dict_25_Sept_2024.toml")
record_id_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_25092024_1043.csv")


## For every record, the file names are stored in a dictionary
## Count the number of files of each file type for each record

record_file_count_dict = {}
for record_id, file_list in record_id_to_files.items():
    record_file_count_dict[record_id] = {}
    for file in file_list:
        file_type = file.split(".")[-1]
        if file_type in record_file_count_dict[record_id]:
            record_file_count_dict[record_id][file_type] += 1
        else:
            record_file_count_dict[record_id][file_type] = 1

cpt_ids = [record_id for record_id in record_file_count_dict.keys() if "CPT" in record_id]
scpt_ids = [record_id for record_id in record_file_count_dict.keys() if "SCPT" in record_id]
borehole_ids = [record_id for record_id in record_file_count_dict.keys() if "BH" in record_id]
vsvp_ids = [record_id for record_id in record_file_count_dict.keys() if "VsVp" in record_id]

cpt_file_types = []
for record_id in cpt_ids:
    cpt_file_types.extend(record_file_count_dict[record_id].keys())
cpt_file_types = set(cpt_file_types)

scpt_file_types = []
for record_id in scpt_ids:
    scpt_file_types.extend(record_file_count_dict[record_id].keys())
scpt_file_types = set(scpt_file_types)

borehole_file_types = []
for record_id in borehole_ids:
    borehole_file_types.extend(record_file_count_dict[record_id].keys())
borehole_file_types = set(borehole_file_types)

vsvp_file_types = []
for record_id in vsvp_ids:
    vsvp_file_types.extend(record_file_count_dict[record_id].keys())
vsvp_file_types = set(vsvp_file_types)


non_data_file_types = ["pdf", "PDF", "kmz", "jpeg", "docx", "Pdf"]



non_data_cpts = find_records_with_only_these_file_types(record_id_to_files, non_data_file_types, InvestigationType.CPT)

non_data_scpts = find_records_with_only_these_file_types(record_id_to_files, non_data_file_types, InvestigationType.SCPT)

non_data_borehole = find_records_with_only_these_file_types(record_id_to_files, non_data_file_types, InvestigationType.BH)

non_data_vsvp = find_records_with_only_these_file_types(record_id_to_files, non_data_file_types, InvestigationType.VsVp)

sung_cpt_names = np.loadtxt("/home/arr65/data/nzgd/stats_plots/sung_cpt_names.txt", dtype=str)

digitized_data_cpts = list(set(cpt_ids) - set(non_data_cpts))
digitized_data_scpts = list(set(scpt_ids) - set(non_data_scpts))
digitized_data_borehole = list(set(borehole_ids) - set(non_data_borehole))
digitized_data_vsvp = list(set(vsvp_ids) - set(non_data_vsvp))

print()

categorized_record_ids = {
    "CPT": natsort.natsorted(cpt_ids),
    "SCPT": natsort.natsorted(scpt_ids),
    "Borehole": natsort.natsorted(borehole_ids),
    "VsVp": natsort.natsorted(vsvp_ids),
    "Non-data CPT": natsort.natsorted(non_data_cpts),
    "Non-data SCPT": natsort.natsorted(non_data_scpts),
    "Non-data Borehole": natsort.natsorted(non_data_borehole),
    "Non-data VsVp": natsort.natsorted(non_data_vsvp),
    "Digitized data CPT": natsort.natsorted(digitized_data_cpts),
    "Digitized data SCPT": natsort.natsorted(digitized_data_scpts),
    "Digitized data Borehole": natsort.natsorted(digitized_data_borehole),
    "Digitized data VsVp": natsort.natsorted(digitized_data_vsvp)
}

## Write the categorized record IDs to a file using toml
# categorized_record_ids_file = Path("/home/arr65/data/nzgd/stats_plots/categorized_record_ids.toml")
# with open(categorized_record_ids_file, "w") as f:
#     toml.dump(categorized_record_ids, f)

print()


stats_plots_dir = Path("/home/arr65/data/nzgd/stats_plots")
stats_plots_dir.mkdir(exist_ok=True)

# ### Borehole records that are not pdfs
# fig, ax = plt.subplots()
# sizes = [len(non_data_borehole), len(borehole_ids) - len(non_data_borehole)]
# labels = [f"only pdf\n({sizes[0]})", f"digitized data\n({sizes[1]})"]
# ax.pie(sizes,
#        labels=labels,
#        autopct='%1.1f%%',
#        textprops={'fontsize': 16})
# total_num = sum(sizes)
# plt.title(f"Borehole (Total: {total_num} records)", fontsize=25)
# plt.savefig(stats_plots_dir / "borehole_non_pdf.png",dpi=500)
# plt.show()
#
#
# new_cpt_ids = natsort.natsorted(list((set(digitized_data_cpts) - set(sung_cpt_names))))
#
# ### Comparison of the old and new CPT data
# fig, ax = plt.subplots()
# sizes = [len(sung_cpt_names), len(new_cpt_ids)]
# labels = [f"old CPT dataset \n({sizes[0]})", f"new CPT dataset\n({sizes[1]})"]
# ax.pie(sizes,
#        labels=labels,
#        autopct='%1.1f%%',
#        textprops={'fontsize': 16})
# total_num = sum(sizes)
# plt.title(f"CPT (Total: {total_num} records)", fontsize=25)
# plt.savefig(stats_plots_dir / "digitized_CPT_data.png",dpi=500)
# plt.show()

### All NZGD digitized records
fig, ax = plt.subplots()
sizes = [len(digitized_data_cpts), len(digitized_data_scpts), len(digitized_data_borehole), len(digitized_data_vsvp)]
labels = [f"CPT\n({sizes[0]})", f"SCPT\n({sizes[1]})", f"Borehole\n({sizes[2]})", f"VsVp\n({sizes[3]})"]
ax.pie(sizes,
       labels=labels,
       autopct='%1.1f%%',
       textprops={'fontsize': 16})
total_num = sum(sizes)
plt.title(f"NZGD records with data\n(Total: {total_num} records)", fontsize=15)
plt.savefig(stats_plots_dir / "all_NZGD_digitized_records.png",dpi=500)
plt.show()

### All NZGD records
fig, ax = plt.subplots()
sizes = [len(cpt_ids), len(scpt_ids), len(borehole_ids), len(vsvp_ids)]
labels = [f"CPT\n({sizes[0]})", f"SCPT\n({sizes[1]})", f"Borehole\n({sizes[2]})", f"VsVp\n({sizes[3]})"]
ax.pie(sizes,
       labels=labels,
       autopct='%1.1f%%',
       textprops={'fontsize': 16})
total_num = sum(sizes)
plt.title(f"All NZGD records including only pdf\n(Total: {total_num} records)", fontsize=15)
plt.savefig(stats_plots_dir / "all_NZGD_records_including_only_pdf.png",dpi=500)
plt.show()

print()

record_id_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_25092024_1043.csv")

## Borehole, CPT, HandAuger, HandAugerScala, Other, Scala, SCPT, SDMT, SWS, TestPit, VsVp
#other_investigation_types = ["HandAuger", "HandAugerScala", "Other", "Scala", "SDMT", "SWS", "TestPit"]
other_investigation_types = ["HandAuger", "HandAugerScala", "Other", "Scala", "TestPit"]

sizes = []
for investigation_type in other_investigation_types:
    sizes.append(len(record_id_df[record_id_df["Type"] == investigation_type]))

### All NZGD records
fig, ax = plt.subplots()

labels = []
for investigation_type, size in zip(other_investigation_types, sizes):
    labels.append(f"{investigation_type}\n({size})")

ax.pie(sizes,
       labels=labels,
       autopct='%1.1f%%',
       textprops={'fontsize': 14})
total_num = sum(sizes)
plt.text(-2.0, 1.0, 'Also SDMT (181)\nand SWS (65)', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))


plt.title(f"Other investigation types\n(Total: {total_num} records)", fontsize=15)
plt.savefig(stats_plots_dir / "other_NZGD_records.png",dpi=500)
plt.show()

print()