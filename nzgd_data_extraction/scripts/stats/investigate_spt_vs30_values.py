import pandas as pd

df = pd.read_csv("/home/arr65/data/nzgd/processed_data/spt/spt_vs30.csv")

problematic_records = []

problematic_records.extend(df[df["num_depth_levels"] == 1]["record_name"].values)

problematic_records.extend(df[df["depth_span"] == 0.0]["record_name"].values)

problematic_records.extend(df[df["min_depth"] < 0.0]["record_name"].values)

problematic_records.extend(df[df["max_depth"] > 100.0]["record_name"].values)


problematic_records = list(set(problematic_records))


print()