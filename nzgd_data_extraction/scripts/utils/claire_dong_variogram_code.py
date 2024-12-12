import numpy as np
import matplotlib.pyplot as plt

rho_0 = 0.02
h_0 = 5
a = np.log(rho_0) / (-1*h_0)
# Find the lag distance between each pair of values
#h = abs(np.subtract.outer(z, z))
h = np.linspace(0.1,30,10000)

# (in the range from 0 to 1)
rho = np.exp(-1 * a * h)

plt.plot(h,rho)
plt.xlabel("h")
plt.ylabel("rho")

plt.show()
print()