import toml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import enum
import pickle

path = Path("/home/arr65/data/nzgd/processed_data/cpt/metadata/residual_plots")

with open(path / 'all_residuals.pkl', 'rb') as pickle_file:
    resid = pickle.load(pickle_file)

hist_labels = []
hist_data = []

class CPTCorrelation(enum.StrEnum):
    andrus_2007 = "andrus_2007"
    robertson_2009 = "robertson_2009"
    hegazy_2006 = "hegazy_2006"
    mcgann_2015 = "mcgann_2015"
    mcgann_2018 = "mcgann_2018"

class Vs30Correlation(enum.StrEnum):
    boore_2011 = "boore_2011"
    boore_2004 = "boore_2004"

class DataSubset(enum.StrEnum):
    new_and_old = "new_and_old"
    only_old = "only_old"
    only_new = "only_new"


#vs30_correlation = Vs30Correlation.boore_2011
vs30_correlation = Vs30Correlation.boore_2004


for key in resid.keys():
    if (DataSubset.only_new in key) & (vs30_correlation in key):
        hist_labels.append(key)
        hist_data.append(resid[key])
histtype='bar'
plt.figure()
# plt.hist(hist_data, bins=20, label=hist_labels, histtype='bar',
#          stacked=False, fill=True)

plt.hist(hist_data, bins=10, label=hist_labels, histtype='step',
         stacked=False, fill=False)


plt.legend()
plt.xlabel("log residual")
plt.ylabel("count")
plt.savefig(path / f"combined_hist_{vs30_correlation}.png", dpi=500)
plt.close()



print(resid)