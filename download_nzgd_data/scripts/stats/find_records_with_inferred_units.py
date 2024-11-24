import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import natsort
from tqdm import tqdm


processed_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")
parquet_files = natsort.natsorted(list(processed_data_dir.rglob("*.parquet")))



#cpt_data = pd.read_parquet(parquet_files[0])

record_names_with_inferred_units = []
inferred_unit_conversions = []



for parquet_file in tqdm(parquet_files):
    cpt_data = pd.read_parquet(parquet_file)

    if len(cpt_data.attrs["inferred_unit_conversions"]):
        record_names_with_inferred_units.append(parquet_file.stem)
        inferred_unit_conversions.append(cpt_data.attrs["inferred_unit_conversions"])

inferred_cm_to_m_record_names = []
inferred_cm_to_m_messages = []

for record_name, inferred_unit_conversion in zip(record_names_with_inferred_units, inferred_unit_conversions):

    if "from cm to m" in inferred_unit_conversion:
        inferred_cm_to_m_record_names.append(record_name)
        inferred_cm_to_m_messages.append(inferred_unit_conversion)

print()

np.savetxt("/home/arr65/data/nzgd/resources/inferred_cm_to_m_names.txt",np.array(inferred_cm_to_m_record_names),fmt="%s")







print()
