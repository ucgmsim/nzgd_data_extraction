# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_csv("/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_region_classification.csv")
#
#
# # count the number of times each value appears in the "district" column
# district_counts = df["district"].value_counts()
#
# # make a pie chart
# district_counts.plot.pie(autopct='%1.1f%%')
# plt.show()
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the data
# df = pd.read_csv("/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_region_classification.csv")
#
# df_canterbury = df[df["district"]=="Canterbury"]
#
# print()
#
#
# # Count the number of times each value appears in the "district" column
# #district_counts = df["district"].value_counts()
# district_counts = df["name"].value_counts()
# district_counts.name = ''
#
#
# # Define the threshold percentage
# threshold = 5.0
#
# # Calculate the total number of counts
# total_counts = district_counts.sum()
#
# # Separate the values above and below the threshold
# above_threshold = district_counts[district_counts / total_counts * 100 > threshold]
# below_threshold = district_counts[district_counts / total_counts * 100 <= threshold]
#
# # Add the "Other" category
# if not below_threshold.empty:
#     above_threshold['Other'] = below_threshold.sum()
#
# # Define a function to format the percentages and counts
# def autopct_format(values):
#     def my_format(pct):
#         total = sum(values)
#         val = int(round(pct * total / 100.0))
#         return '{:.1f}%\n({:d})'.format(pct, val) if pct > threshold else ''
#     return my_format
#
# # Make the main pie chart
# above_threshold.plot.pie(autopct=autopct_format(above_threshold.values), title='Main Districts')
# #plt.savefig("/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_main_districts_region_classification.png",dpi=400)
# plt.savefig("/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_canterbury_suburb.png",dpi=400)
# plt.show()
#
# # Make the pie chart for the "Other" category if it exists
# if 'Other' in above_threshold.index:
#     below_threshold.plot.pie(autopct=autopct_format(below_threshold.values), title='Other Districts')
#
#     #plt.savefig("/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_other_districts_region_classification.png",dpi=400)
#     plt.savefig(
#         "/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_other_canterbury_suburb.png",
#         dpi=400)

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_region_classification.csv')

# Filter the DataFrame to only include rows where the district is "Canterbury"
df_canterbury = df[df['district'] == 'Canterbury']

# Count the number of times each value appears in the "name" column
name_counts = df_canterbury['name'].value_counts()

# Separate the values above and below the threshold
threshold = 500
above_threshold = name_counts[name_counts >= threshold]
below_threshold = name_counts[name_counts < threshold]

# Create a bar plot for counts greater than or equal to 500
#plt.figure(figsize=(10, 6))
above_threshold.plot.bar(title='Count of suburb in Canterbury (>= 500)')
plt.xlabel('Name')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_canterbury_name_counts_above_500.png', dpi=400)
plt.show()

# Create a bar plot for counts less than 500
# plt.figure(figsize=(10, 6))
# below_threshold.plot.bar(title='Count of Names in Canterbury District (< 500)')
# plt.xlabel('Name')
# plt.ylabel('Count')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_canterbury_name_counts_below_500.png', dpi=400)
# plt.show()

# print()
#
# # make a pie chart of "name" column counts in df_canterbury
# canterbury_suburb_counts = df_canterbury["name"].value_counts()
# canterbury_suburb_counts.name = ''
#
# canterbury_suburb_counts.plot.pie(autopct='%1.1f%%')
# plt.savefig("/home/arr65/data/nzgd/cpt_region_classiciation/cpt_25092024_1043_canterbury_suburb_classification.png",dpi=400)
#
#
