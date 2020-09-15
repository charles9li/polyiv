import matplotlib.pyplot as plt

from compute_rho import *
from intrinsic_viscosity_general import *


# Parameters
N = 190
sigma = 0.2
a = 0.1
b = 1.0
Rg2 = N/6.

# Compute drag
rho = rho_gaussian(N, Rg2)
# rho = compute_rho("plma_50mA12_NPT_293K_4500bar_3wt_1map_traj.dcd",
#                   "plma_50mA12_NPT_293K_4500bar_3wt_1map_equilibrated.pdb",
#                   np.arange(50)*3)
drag = compute_drag_function(rho, sigma, a)

# Compute drainage
drainage = compute_drainage_function(rho, drag, sigma, b, truncate=np.inf)

# Plot
plt.figure()
r_list = np.linspace(0, 20, 10)

drag_list = []
for r in r_list:
    drag_list.append(drag(r))
plt.plot(r_list, drag_list, label=r"$\xi$")

for bound in [50, 100]:
    drainage = compute_drainage_function(rho, drag, sigma, b, truncate=bound)
    drainage_list = []
    for r in r_list:
        print(r)
        val = drainage(r)
        print(val)
        print
        drainage_list.append(val)
    plt.plot(r_list, drainage_list, label=r"$\kappa$, bound={}$\sigma$".format(bound))
plt.legend()
plt.show()
