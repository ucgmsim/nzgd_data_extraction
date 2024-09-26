import toml
import enum
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

record_id_to_files = toml.load("/home/arr65/data/nzgd/record_name_to_file_dicts/record_id_to_file_name_dict_25_Sept_2024.toml")

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


non_data_file_types = ["pdf", "PDF", "kmz", "jpeg", "docx", "Pdf"]

class InvestigationType(enum.StrEnum):

    """
    Values for the investigation type of the record
    """

    CPT = "CPT"
    SCPT = "SCPT"
    BH = "BH"

def find_records_with_only_these_file_types(record_id_to_files: dict[str, list[str]], files_types: list[str], investigation_type: InvestigationType):

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

non_data_cpts = find_records_with_only_these_file_types(record_id_to_files, non_data_file_types, InvestigationType.CPT)

non_data_scpts = find_records_with_only_these_file_types(record_id_to_files, non_data_file_types, InvestigationType.SCPT)

non_data_borehole = find_records_with_only_these_file_types(record_id_to_files, non_data_file_types, InvestigationType.BH)

stats_plots_dir = Path("/home/arr65/data/nzgd/stats_plots")
stats_plots_dir.mkdir(exist_ok=True)

### Borehole records that are not pdfs
fig, ax = plt.subplots()
sizes = [len(non_data_borehole), len(borehole_ids) - len(non_data_borehole)]
labels = [f"only pdf\n({sizes[0]})", f"digitized data\n({sizes[1]})"]
ax.pie(sizes,
       labels=labels,
       autopct='%1.1f%%',
       textprops={'fontsize': 16})
total_num = sum(sizes)
plt.title(f"Borehole (Total: {total_num} records)", fontsize=25)
plt.savefig(stats_plots_dir / "borehole_non_pdf.png",dpi=500)
plt.show()

print()

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

info_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_25092024_1043.csv")

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 10))

# Set up the Basemap instance for New Zealand
m = Basemap(projection='merc', llcrnrlat=-48, urcrnrlat=-34, llcrnrlon=166, urcrnrlon=179, resolution='i', ax=ax)

# Draw coastlines and country boundaries
m.drawcoastlines()
m.drawcountries()

# Draw parallels and meridians
m.drawparallels(range(-50, -30, 2), labels=[1,0,0,0])
m.drawmeridians(range(160, 180, 2), labels=[0,0,0,1])

# List of coordinates (latitude, longitude) for markers
coordinates = [(-36.8485, 174.7633),  # Auckland
               (-41.2865, 174.7762),  # Wellington
               (-43.5321, 172.6362)]  # Christchurch

# Plot markers at the specified coordinates
for lat, lon in coordinates:
    x, y = m(lon, lat)
    m.plot(x, y, 'bo', markersize=10)

# Add a title
plt.title('Markers at Specific Coordinates in New Zealand')

# Show the plot
plt.show()

print()