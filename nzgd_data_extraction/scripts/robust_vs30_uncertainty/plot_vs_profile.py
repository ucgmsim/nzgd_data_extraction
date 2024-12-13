import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import natsort
import pandas as pd

correlation_flag = "partial"
velocity_profile_path = Path(f"/home/arr65/data/nzgd/robust_vs30/cpt/randomly_sampled_velocity_profiles")

file_list = natsort.natsorted(list(velocity_profile_path.glob("*")))

for file in file_list:

    data_df = pd.read_parquet(file)

    num_sampled_profiles = int(data_df.at[0, "num_sampled_profiles"])
    depth = data_df.at[0, "depth"]
    vs_at_depth_level_list = data_df.at[0, "vs"]
    record_name = data_df.at[0, "record_name"]

    vs_array = np.zeros((len(vs_at_depth_level_list), num_sampled_profiles))

    for depth_level_idx in range(len(vs_at_depth_level_list)):
        vs_array[depth_level_idx,:] = vs_at_depth_level_list[depth_level_idx]

    mean_vs = np.mean(vs_array, axis=1)
    std_vs = np.std(vs_array, axis=1)

    profiles_to_plot = [0,1,2]

    for profile_idx in profiles_to_plot:
        plt.plot(vs_array[:,profile_idx],depth,"-")
    plt.ylabel("depth (m)")
    plt.xlabel("Vs (m/s)")
    plt.title(record_name)
    plt.gca().invert_yaxis()
    plt.show()

    plt.figure()
    plt.plot(mean_vs, depth)
    plt.fill_betweenx(depth, mean_vs - std_vs, mean_vs + std_vs, alpha=0.5)

    plt.ylabel("depth (m)")
    plt.xlabel("Vs (m/s)")
    plt.title(record_name)
    plt.gca().invert_yaxis()
    plt.show()

    print()


