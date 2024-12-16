import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


cpt_data_path = Path("/home/arr65/src/Ancillary-tools/CPT_Vsz_Vs30/cptVsDATAfromChris/smsCPTdata")
cpt_data_files = list(cpt_data_path.glob("*"))
cpt_data_files = sorted([file for file in cpt_data_files if file.suffix != ".txt"])



print()


