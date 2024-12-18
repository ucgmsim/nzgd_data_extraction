import pandas as pd
import matplotlib.pyplot as plt

# CPT_22400



#test = pd.read_parquet("/home/arr65/data/nzgd/processed_data/cpt/data/CPT_1.parquet")
#test = pd.read_parquet("/home/arr65/data/nzgd/processed_data/cpt/data/CPT_203701.parquet")
#/home/arr65/data/nzgd/processed_data/cpt/data/CPT_203701.parquet

test = pd.read_parquet("/home/arr65/data/nzgd/extracted_data/cpt/extracted_data_per_record/CPT_1.parquet")
print()
plt.plot(test["Depth"],test["qc"])
plt.plot(test["Depth"],test["fs"])
plt.plot(test["Depth"],test["u"])
plt.show()
