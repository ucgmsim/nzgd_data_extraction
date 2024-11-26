import pandas as pd
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

vs_calc_path = Path("/home/arr65/src/vs30/VsViewer")
sys.path.append(str(vs_calc_path))

import vs_calc

spt_vs_correlations = vs_calc.spt_vs_correlations.SPT_CORRELATIONS
vs30_correlations = list(vs_calc.vs30_correlations.VS30_CORRELATIONS.keys())

hammer_types = [vs_calc.constants.HammerType.Auto, vs_calc.constants.HammerType.Safety,
                vs_calc.constants.HammerType.Standard]
borehole_diameter = 150

all_spt_df = pd.read_parquet("/home/arr65/data/nzgd/processed_data/spt/spts.parquet")
output_dir = Path("/home/arr65/data/nzgd/processed_data/spt")

## Get unique values of the column "NZGD_ID" in all_spt_df
unique_nzgd_ids = all_spt_df["NZGD_ID"].unique()

spt_vs30_df = pd.DataFrame(columns=["record_name", "error", "Vs30", "Vs30_sd", "spt_vs_correlation", "vs30_correlation",
                                    "used_soil_info", "hammer_type","borehole_diameter"])

progress_bar = tqdm(total=len(unique_nzgd_ids)*len(spt_vs_correlations)*len(vs30_correlations)*len(hammer_types))

for spt_vs_correlation in spt_vs_correlations:
    for vs30_correlation in vs30_correlations:
        for hammer_type in hammer_types:

            for nzgd_id in unique_nzgd_ids:

                spt_df = all_spt_df[all_spt_df["NZGD_ID"] == nzgd_id]

                soil_type_list_str = []
                for x in spt_df["Soil Type"]:
                    if len(x) > 0:
                        soil_type_list_str.append(x[0].capitalize())
                    else:
                        soil_type_list_str.append(np.nan)

                soil_type_list_enum = []
                for x in soil_type_list_str:
                    try:
                        soil_type_enum = vs_calc.constants.SoilType[x]
                    except KeyError:
                        ## Skip the soil type at this depth level if it is not in the SoilType enum
                        continue
                    soil_type_list_enum.append(soil_type_enum)

                spt = vs_calc.SPT(
                    name=spt_df["NZGD_ID"].values[0],
                    depth=spt_df["Depth"].values,
                    n=spt_df["N"].values,
                    hammer_type=hammer_type,
                    borehole_diameter=borehole_diameter)

                used_soil_info = False
                if len(soil_type_list_enum) == len(spt_df):
                    ## each depth level has a valid soil type
                    spt.soil_type = np.array(soil_type_list_enum)
                    used_soil_info = True

                try:
                    spt_vs_profile = vs_calc.VsProfile.from_spt(spt, spt_vs_correlation)
                    spt_vs_profile.vs30_correlation = vs30_correlation
                    vs30 = spt_vs_profile.vs30
                    vs30_sd = spt_vs_profile.vs30_sd
                    error = np.nan

                except Exception as e:
                    vs30 = np.nan
                    vs30_sd = np.nan
                    error = e

                spt_vs30_df = pd.concat([spt_vs30_df,
                                         pd.DataFrame(
                                             {"record_name": f"BH_{nzgd_id}",
                                                    "error": error,
                                                    "Vs30": vs30,
                                                    "Vs30_sd": vs30_sd,
                                                    "spt_vs_correlation": spt_vs_correlation,
                                                    "vs30_correlation": vs30_correlation,
                                                    "used_soil_info" : used_soil_info,
                                                    "hammer_type": hammer_type.name,
                                                    "borehole_diameter": borehole_diameter},
                                             index=[0])], ignore_index=True)
                progress_bar.update(1)

progress_bar.close()

spt_vs30_df.to_csv(output_dir/ "spt_vs30.csv", index=False)

