import pandas as pd

df = pd.read_csv("/home/arr65/data/nzgd/processed_data/cpt/metadata/all_failed_loads.csv")

num_depth_is_index = df[df["category"] == "depth_is_index"].shape[0]


print()