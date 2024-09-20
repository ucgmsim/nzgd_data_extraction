import pandas as pd

url_df = pd.read_csv("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_19092024_1045.csv")

num_cpt = len(url_df[url_df["Type"] == "CPT"])

num_scpt = len(url_df[url_df["Type"] == "SCPT"])

num_borehole = len(url_df[url_df["Type"] == "Borehole"])

print(f"Number of CPTs: {num_cpt}")
print(f"Number of SCPTs: {num_scpt}")
print(f"Number of Boreholes: {num_borehole}")
print(f"Total number of useful investigations: {num_cpt + num_scpt + num_borehole}")


print()