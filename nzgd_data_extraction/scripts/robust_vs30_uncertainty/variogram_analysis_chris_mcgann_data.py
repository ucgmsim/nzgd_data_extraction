import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skgstat import Variogram
import time
import toml

def make_semivariogram_and_output(base_output_dir:Path, name:str, input_df:pd.DataFrame):

    combined_path = base_output_dir / name
    combined_path.mkdir(exist_ok=True, parents=True)

    plt.figure()
    plt.plot(input_df["depth_m"],input_df["ln_measured_minus_ln_inferred"], '.')
    plt.xlabel("depth (m)")
    plt.ylabel("log(measured_vs) - log(cpt_inferred_vs)")
    plt.ylim(-3.1,3.1)
    plt.savefig(combined_path / f"{name}_ln_residual.png", dpi=500)
    plt.close()

    log_resid_df = input_df.sort_values(by="depth_m")

    print("Doing Variogram...")

    selected_n_lags = 100

    variogram_calc_start_time = time.time()

    semivar = Variogram(log_resid_df["depth_m"].values,
                      log_resid_df["ln_measured_minus_ln_inferred"].values,
                      normalize=False,
                      n_lags = selected_n_lags,
                      model = "exponential")

    fitted_semivariogram_df = Variogram.to_DataFrame(semivar, n=selected_n_lags)

    fitted_semivariogram_df["bins"] = semivar.bins
    fitted_semivariogram_df["bin_count"] = semivar.bin_count

    empirical_semivar = Variogram.get_empirical(semivar)
    fitted_semivariogram_df["empirical_bins"] = empirical_semivar[0]
    fitted_semivariogram_df["empirical_semivariance"] = empirical_semivar[1]

    fitted_semivariogram_df.to_csv(base_output_dir / name / f"{name}_semivar.csv",index=False)

    semivar_plot = semivar.plot(show=False)
    semivar_plot.savefig(base_output_dir / name / f"{name}_semivariogram.png",dpi=500)


    dist_diff_plot = semivar.distance_difference_plot(show=False)
    dist_diff_plot.savefig(base_output_dir / name / f"{name}_distance_difference_plot.png",dpi=500)


    describe_dict = Variogram.describe(semivar)
    with open(base_output_dir / name / "description.toml", "w") as toml_file:
        toml.dump(describe_dict, toml_file)

    print(f"Variogram calculation took {(time.time() - variogram_calc_start_time)/60} mins")

measured_vs_profile_path = Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/updated_format_for_nzgd_code/vsProf042015")
vs_profile_from_cpt_path = Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/McGann2015_velocity_profiles_from_cpt")

base_output_dir = Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/semivariance_analysis")
base_output_dir.mkdir(exist_ok=True, parents=True)

measured_vs_profile_files = list(measured_vs_profile_path.glob("*.dat"))

custom_header = ['depth', 'vs']

cpt_dir = Path("/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/updated_format_for_nzgd_code/smsCPTdata")
cpt_dirs = list(cpt_dir.glob("*"))

ids_with_cpt = [cpt_dir.name.split("_")[1] for cpt_dir in cpt_dirs]
#ids_with_cpt = ["NNBS"]
#ids_with_cpt = ["SHLC"]
#ids_with_cpt = ["CHHC"]


log_resid_df = pd.DataFrame(columns=["record_name","depth_m","measured_vs","inferred_vs","ln_measured_minus_ln_inferred"])

for file in measured_vs_profile_files:

    cpt_output_dir = base_output_dir / file.stem
    cpt_output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Doing file: {file.name}")
    id_code = file.stem[0:4]

    if id_code not in ids_with_cpt:
        print(f"{id_code} is missing CPT data")
        continue

    from_cpt = pd.read_csv(
        f"/home/arr65/data/nzgd/resources/chris_mcgann_cpt_vs_data/McGann2015_velocity_profiles_from_cpt/CPT_{id_code}_vs_profiles.csv")
    #from_cpt = from_cpt[from_cpt["Depth"]>1.5]

    measured_vs_depth_boundaries_df = pd.read_csv(file,sep=r"\s+",header=None, names=custom_header)

    measured_vs = np.copy(from_cpt["Depth"].values)
    measured_vs[:] = np.nan

    for i in range(len(measured_vs_depth_boundaries_df["depth"])-1):

        depth_1 = measured_vs_depth_boundaries_df["depth"].iloc[i]
        depth_2 = measured_vs_depth_boundaries_df["depth"].iloc[i+1]

        #print(f"between index {i} ({depth_1}) and {i+1} ({depth_2}), adopting vs = {measured_vs_depth_boundaries_df['vs'].iloc[i]}")

        measured_depth_indices = np.where((from_cpt["Depth"] >= depth_1) & (from_cpt["Depth"] <= depth_2))

        measured_vs[measured_depth_indices] = measured_vs_depth_boundaries_df["vs"].iloc[i]



    log_resid = np.log(measured_vs) - np.log(from_cpt["Vs"])

    current_cpt_df = pd.DataFrame({"record_name":id_code,
                                                "depth_m":from_cpt["Depth"].values,
                                                "measured_vs":measured_vs,
                                                "inferred_vs":from_cpt["Vs"],
                                                "ln_measured_minus_ln_inferred":log_resid})

    make_semivariogram_and_output(base_output_dir, name=file.stem, input_df=current_cpt_df)

    log_resid_df = pd.concat([log_resid_df,
                                current_cpt_df],
                                 ignore_index=True)

make_semivariogram_and_output(base_output_dir, name="combined", input_df=log_resid_df)
