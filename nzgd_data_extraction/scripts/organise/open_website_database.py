import pandas as pd

df = pd.read_parquet("/home/arr65/src/nzgd_map_from_webplate/instance/website_database.parquet")

df.drop_duplicates(subset=["record_name"], keep="first", inplace=True)




print()