import pandas as pd

df = pd.read_parquet("/home/arr65/src/nzgd_map_from_webplate/instance/website_database.parquet")

vs30_correlation = "boore_2011"
spt_vs_correlation = "brandenberg_2010"
cpt_vs_correlation = "andrus_2007_pleistocene"

# Filter the dataframe for the selected spt_vs_correlation_and_vs30_correlation
df = df[df["vs30_correlation"] == vs30_correlation]

spt_bool = (df["spt_vs_correlation"] == spt_vs_correlation) & (df["spt_hammer_type"] == "Auto")
cpt_bool = df["cpt_vs_correlation"] == cpt_vs_correlation

df = df[spt_bool | cpt_bool]

num_records = df["record_name"].nunique()

#df.drop_duplicates(subset=["record_name"], keep="first", inplace=True)



print()