import numpy as np
import matplotlib.pyplot as plt


## The range parameter, r, is the distance at which the semivariogram, gamma(h), is equal to 95% of the sill
## (i.e., the distance at which 95% of the correlation is lost).
range_m = 3 # metres, where

## The sill parameter, s, is equal to the variance of Z = delta W  (the residual of the within event term)
sill = 0.8

h = np.linspace(0, 10, 1000)

gamma = sill*(1 - np.exp(-3*h/range_m))

plt.plot(h, gamma)
plt.xlabel("separation distance, h (m)")
plt.ylabel("semivariance, gamma(h)")
plt.show()
print()


## The covariance, C(h), is equal to C(0) - gamma(h) (where C(0) = sill)
covariance = sill - gamma

plt.figure()
plt.plot(h, covariance)
plt.xlabel("separation distance, h (m)")
plt.ylabel("covariance, C(h)")
plt.show()



## The correlation coefficient, rho(h), is equal to C(h)/C(0) (where C(0) = sill)
correlation_coefficient = covariance/sill
rho_andrew = correlation_coefficient




#############################
### Claire Dong's correlation coefficient

rho_0 = 0.02
h_0 = 5
a = np.log(rho_0) / (-1*h_0)
# Find the lag distance between each pair of values
#h = abs(np.subtract.outer(z, z))
#h = np.linspace(0.1,30,10000)

# (in the range from 0 to 1)
rho_claire = np.exp(-1 * a * h)

plt.figure()
plt.plot(h, rho_andrew, label="Andrew rho")
plt.plot(h,rho_claire, label="Claire rho")
plt.legend()
plt.xlabel("separation distance, h (m)")
plt.ylabel("rho (h)")
plt.show()
print()
