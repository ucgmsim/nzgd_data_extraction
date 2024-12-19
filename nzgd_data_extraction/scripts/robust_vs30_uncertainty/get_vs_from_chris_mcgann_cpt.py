import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

vs_calc_path = Path("/home/arr65/src/vs30/VsViewer")
sys.path.append(str(vs_calc_path))

import vs_calc

cpt_vs_correlation = "mcgann_2015"
vs30_correlation = "boore_2004"

vs_output_dir = Path(f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/chris_mcgann_vs_from_cpt/{cpt_vs_correlation}_vs_profiles")
vs_output_dir.mkdir(exist_ok=True, parents=True)


file_list = list(
    Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/"
         "reformatted_for_easier_loading/smsCPTdata").rglob("*.xls")
              )

for file_path in file_list:

    id = file_path.parent.name

    sheet_names = pd.ExcelFile(file_path).sheet_names

    vs_df = pd.DataFrame()

    for sheet_index, sheet_name in enumerate(sheet_names):

        cpt_df = pd.read_excel(file_path)

        # Create a CPT object from the DataFrame
        cpt = vs_calc.CPT(
            id,
            cpt_df["Depth (m)"].values,
            cpt_df["qc (MPa)"].values,
            cpt_df["fs (MPa)"].values,
            cpt_df["u (MPa)"].values)

        # Create a VsProfile from the CPT object using the specified correlation
        cpt_vs_profile = vs_calc.VsProfile.from_cpt(cpt, cpt_vs_correlation)
        cpt_vs_profile.vs30_correlation = vs30_correlation

        vs_df_for_sheet = cpt_vs_profile.to_dataframe()
        vs_df_for_sheet["investigation_number"] = sheet_index

        vs_df = pd.concat([vs_df, vs_df_for_sheet])

    vs_df.to_csv(vs_output_dir / f"{id}_vs_profiles.csv", index=False)




