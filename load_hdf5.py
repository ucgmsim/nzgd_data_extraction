import pandas as pd
import h5py

file_path = "/home/arr65/data/nzgd/bulk_loading/cpt_V0.h5"


hdf = h5py.File(file_path, 'r')
keys = list(hdf.keys())
hdf.close()



for key in keys:

    #print(key)

    pd.read_hdf(file_path, key=key)

print()