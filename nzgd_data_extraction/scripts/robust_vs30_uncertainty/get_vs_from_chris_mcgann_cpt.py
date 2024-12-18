import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

file_list = list(
    Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/"
         "reformatted_for_easier_loading/smsCPTdata").rglob("*.xls")
              )

print()

