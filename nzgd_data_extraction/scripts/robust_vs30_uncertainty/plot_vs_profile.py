import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import natsort

velocity_profile_path = Path("/home/arr65/data/nzgd/robust_vs30/randomly_sampled_velocity_profiles")

file_list = natsort.natsorted(list(velocity_profile_path.glob("*.csv")))

for file in file_list:
    data = np.load(file)

    plt.plot(velocity, depth, label=file.name)
