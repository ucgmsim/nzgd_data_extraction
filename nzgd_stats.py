from collections import Counter

import toml
import enum
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import natsort
import numpy as np
import itertools

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
        if investigation_type == record_id.split("_")[0]:
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


# cpt_file_type_set = set()
# for record_id, file_list in record_id_to_files.items():
#     if "CPT" in record_id:
#         for file in file_list:
#             file_type = file.split(".")[-1]
#             cpt_file_type_set.add(file_type)

# cpt_file_type_count = Counter()
# for record_id, file_list in record_id_to_files.items():
#     if "CPT" in record_id:
#         for file in file_list:
#             file_type = file.split(".")[-1]
#             if file_type in cpt_file_type_set:
#                 cpt_file_type_count.update([file_type])

file_type_count = {}
for record_id, file_list in record_id_to_files.items():
    if "CPT" in record_id:
        for file in file_list:
            file_type = file.split(".")[-1]
            if file_type in file_type_count:
                file_type_count[file_type] += 1
            else:
                file_type_count[file_type] = 1
# sort file_type_count by value
cpt_file_type_count = sorted(file_type_count.items(), key=lambda x: x[1], reverse=True)

## delete the key value pair with the key "pdf"
del cpt_file_type_count[0]

# save cpt_file_type_count to a csv file with columns "file type" and "count"
cpt_file_type_count_df = pd.DataFrame(cpt_file_type_count, columns=["file type", "count"])
cpt_file_type_count_df.to_csv("/home/arr65/data/nzgd/stats_plots/cpt_file_type_count.csv", index=False)


print()

## for each in cpt_file_type_count, print the file type and the number of files of that type, the cumulative number of files, and the cumulative percentage of files

#cpt_file_type_count = cpt_file_type_count.most_common()

## for every object, count pair in cpt_file_type_count, print the object, and count

cpt_file_types = []
cpt_file_type_count = []

for file_type, count in cpt_file_type_count:
    cpt_file_types.append(file_type)
    cpt_file_type_count.append(count)

cumulative_count = np.sum(cpt_file_type_count)

# for i in range(len(cpt_file_types)):
#     print(f"{cpt_file_types[i]}: {cpt_file_type_count[i]} ({100 * cpt_file_type_count[i] / cumulative_count:.2f}%)")



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

print()

cpt_ids = [record_id for record_id in record_file_count_dict.keys() if "CPT" == record_id.split("_")[0]]
scpt_ids = [record_id for record_id in record_file_count_dict.keys() if "SCPT" == record_id.split("_")[0]]
borehole_ids = [record_id for record_id in record_file_count_dict.keys() if "BH" == record_id.split("_")[0]]
vsvp_ids = [record_id for record_id in record_file_count_dict.keys() if "VsVp" == record_id.split("_")[0]]

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


stats_plots_dir = Path("/home/arr65/data/nzgd/stats_plots")
stats_plots_dir.mkdir(exist_ok=True)

# ### Borehole records that are not pdfs
# fig, ax = plt.subplots()
# sizes = [len(non_data_borehole), len(borehole_ids) - len(non_data_borehole)]
# labels = [f"only pdf\n({sizes[0]})", f"digitized data\n({sizes[1]})"]
# colors = ["#2ca02c", "#ff7f0e"]
# ax.pie(sizes,
#        labels=labels,
#        colors=colors,
#        autopct='%1.1f%%',
#        textprops={'fontsize': 16})
# total_num = sum(sizes)
# plt.title(f"Borehole (Total: {total_num} records)", fontsize=25)
# plt.savefig(stats_plots_dir / "borehole_non_pdf.png",dpi=500)

# fig, ax = plt.subplots()
# sizes = [len(borehole_ids) - len(non_data_borehole), len(non_data_borehole)]
# labels = [f"digitized data\n({sizes[0]})", f"only pdf\n({sizes[1]})"]
# colors = ["#ff7f0e", "#2ca02c"]
# ax.pie(sizes,
#        labels=labels,
#        colors=colors,
#        autopct='%1.1f%%',
#        textprops={'fontsize': 16})
# total_num = sum(sizes)
# plt.title(f"Borehole (Total: {total_num} records)", fontsize=25)
# plt.savefig(stats_plots_dir / "borehole_non_pdf.png",dpi=500)


# fig, ax = plt.subplots()
# sizes = [len(scpt_ids) - len(non_data_scpts), len(non_data_scpts)]
# labels = [f"digitized data\n({sizes[0]})", f"only pdf\n({sizes[1]})"]
# colors = ["#ff7f0e", "#2ca02c"]
# ax.pie(sizes,
#        labels=labels,
#        colors=colors,
#        autopct='%1.1f%%',
#        textprops={'fontsize': 16})
# total_num = sum(sizes)
# plt.title(f"SCPT (Total: {total_num} records)", fontsize=25)
# plt.savefig(stats_plots_dir / "scpt_only_pdf_pie_chart.png",dpi=500)
#
fig, ax = plt.subplots()
sizes = [len(cpt_ids) - len(non_data_cpts), len(non_data_cpts)]
labels = [f"digitized data\n({sizes[0]})", f"only pdf\n({sizes[1]})"]
colors = ["#ff7f0e", "#2ca02c"]
ax.pie(sizes,
       labels=labels,
       colors=colors,
       autopct='%1.1f%%',
       textprops={'fontsize': 16})
total_num = sum(sizes)
plt.title(f"CPT (Total: {total_num} records)", fontsize=25)
plt.savefig(stats_plots_dir / "cpt_only_pdf_pie_chart.png",dpi=500)

# fig, ax = plt.subplots()
# sizes = [len(vsvp_ids)]
# labels = [f"digitized data\n({sizes[0]})"]
# colors = ["tab:orange"]
# ax.pie(sizes,
#        labels=labels,
#        autopct='%1.1f%%',
#        colors=colors,
#        textprops={'fontsize': 16})
# total_num = sum(sizes)
# plt.title(f"velocity profiles (Total: {total_num} records)", fontsize=25)
# plt.savefig(stats_plots_dir / "vsvp_pie_chart.png",dpi=500)

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
# fig, ax = plt.subplots()
# sizes = [len(digitized_data_cpts), len(digitized_data_scpts), len(digitized_data_borehole), len(digitized_data_vsvp)]
# labels = [f"CPT\n({sizes[0]})", f"SCPT\n({sizes[1]})", f"Borehole\n({sizes[2]})", f"VsVp\n({sizes[3]})"]
# ax.pie(sizes,
#        labels=labels,
#        autopct='%1.1f%%',
#        textprops={'fontsize': 16})
# total_num = sum(sizes)
# plt.title(f"NZGD records with data\n(Total: {total_num} records)", fontsize=15)
# plt.savefig(stats_plots_dir / "all_NZGD_digitized_records.png",dpi=500)
# plt.show()

### All NZGD records
# fig, ax = plt.subplots()
# sizes = [len(cpt_ids), len(scpt_ids), len(borehole_ids), len(vsvp_ids)]
# labels = [f"CPT\n({sizes[0]})", f"SCPT\n({sizes[1]})", f"Borehole\n({sizes[2]})", f"VsVp\n({sizes[3]})"]
# ax.pie(sizes,
#        labels=labels,
#        autopct='%1.1f%%',
#        textprops={'fontsize': 16})
# total_num = sum(sizes)
# plt.title(f"All NZGD records including only pdf\n(Total: {total_num} records)", fontsize=15)
# plt.savefig(stats_plots_dir / "all_NZGD_records_including_only_pdf.png",dpi=500)
# plt.show()
#
# print()



# ##### Make the pie chart of other data types
#
# ## Borehole, CPT, HandAuger, HandAugerScala, Other, Scala, SCPT, SDMT, SWS, TestPit, VsVp
# #other_investigation_types = ["HandAuger", "HandAugerScala", "Other", "Scala", "SDMT", "SWS", "TestPit"]
# other_investigation_types = ["HandAuger", "HandAugerScala", "Other", "Scala", "TestPit"]
#
# sizes = []
# for investigation_type in other_investigation_types:
#     sizes.append(len(record_id_df[record_id_df["Type"] == investigation_type]))

### All NZGD records
# fig, ax = plt.subplots()
# labels = []
# for investigation_type, size in zip(other_investigation_types, sizes):
#     labels.append(f"{investigation_type}\n({size})")
# ### make a bar plot of sizes vs labels
# bars = ax.bar(labels, sizes, width=0.5)  # Adjust the width parameter to increase spacing
# ax.set_ylabel("Number of Records")
# plt.savefig(stats_plots_dir / "other_NZGD_records_bar_plot.png", dpi=500)


#
# ax.pie(sizes,
#        labels=labels,
#        autopct='%1.1f%%',
#        textprops={'fontsize': 14})
# total_num = sum(sizes)
# plt.text(-2.0, 1.0, 'Also SDMT (181)\nand SWS (65)', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
#
#
# plt.title(f"Other investigation types\n(Total: {total_num} records)", fontsize=15)
# plt.savefig(stats_plots_dir / "other_NZGD_records.png",dpi=500)
# plt.show()
#
# print()

# nums = np.array([49124, len(digitized_data_cpts), len(record_id_df[record_id_df["Type"] == "CPT"])])
# labels = ["old CPT dataset", "new CPT dataset\nexcluding records\nwith only pdfs",
#           "new CPT dataset\nincluding records\nwith only pdfs"]
#
# fig, ax = plt.subplots()
# bars = ax.bar(labels, nums, color=["#228B22", "#8B008B", "#8B008B"])
# ax.set_ylabel("Number of Records")
# bars[2].set_alpha(0.5)
#
# #ax.set_title("Comparison of Old and New CPT Datasets")
# plt.savefig(stats_plots_dir / "cpt_dataset_comparison_bar_plot.png", dpi=500)

