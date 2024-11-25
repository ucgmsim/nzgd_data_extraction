import pandas as pd
from pathlib  import Path

data_files = list(Path("/home/arr65/data/nzgd/standard_format_batch50/cpt/data").glob("*.parquet"))
csv_dir = Path("/home/arr65/data/nzgd/for_vsviewer")

for data_file in data_files:
    df = pd.read_parquet(data_file)

    multiple_measurements = df["multiple_measurements"].unique()


    for mm in multiple_measurements:
        mm_df = df[df["multiple_measurements"] == mm]

        mm_df = mm_df.drop(columns=["record_name", "latitude",
                              "longitude", "multiple_measurements"])

        mm_csv_file = csv_dir / (data_file.stem + f"measurement_num_{mm}.csv")
        mm_df.to_csv(mm_csv_file, index=False)

        #print(f"Converted {data_file} to {mm_csv_file}")

