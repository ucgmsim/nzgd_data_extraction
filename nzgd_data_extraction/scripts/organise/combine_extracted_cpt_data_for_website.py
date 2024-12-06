""""
This script combines the extracted CPT and SCPT data into a single parquet file.
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

cpt_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")
cpt_parquet_files = list(cpt_data_dir.rglob("*.parquet"))

spt_data_dir = Path("/home/arr65/data/nzgd/processed_data/spt/data")
spt_parquet_files = list(cpt_data_dir.rglob("*.parquet"))

parquet_files = cpt_parquet_files + spt_parquet_files

cpt_df = pd.concat([pd.read_parquet(file) for file in tqdm(parquet_files)])

cpt_df.to_parquet(Path("/home/arr65/src/nzgd_map_from_webplate/instance/extracted_cpt_and_scpt_data.parquet"))
