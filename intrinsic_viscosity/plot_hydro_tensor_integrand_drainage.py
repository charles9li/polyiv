import matplotlib.pyplot as plt
import numpy as np

from intrinsic_viscosity_general import *


# parameters
N = 190
sigma = 0.334
a = 0.1
b = 1.0
Rg2 = N/6.

# compute rho function
rho = rho_gaussian(N, Rg2)

# plot
plt.figure("hydro_tensor")
plt.figure("integrand_drainage")
r0_list = np.linspace(0, 10, 5)
for r0 in r0_list:
    z_list = np.linspace(-5, 15, 1000)
    T_list = np.empty(len(z_list))
    integrand_list = np.empty(len(z_list))
    for i, z in enumerate(z_list):
        T = hydro_tensor([0, 0, z], [0, 0, r0], rho, b)
        T_list[i] = T
        integrand_list[i] = rho(np.linalg.norm([0, 0, z])) * T
    plt.figure("hydro_tensor")
    plt.plot(z_list, T_list, label="{}".format(r0))
    plt.figure("integrand_drainage")
    plt.plot(z_list, integrand_list, label="{}".format(r0))
plt.figure("hydro_tensor")
plt.legend(title="r0")
plt.figure("integrand_drainage")
plt.legend(title="r0")
plt.show()
