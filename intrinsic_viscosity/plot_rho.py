import matplotlib.pyplot as plt
import numpy as np

from compute_rho import *


plot_radial = False

dcd_filename = "plma_50mA12_NPT_293K_4500bar_3wt_1map_traj.dcd"
pdb_filename = "plma_50mA12_NPT_293K_4500bar_3wt_1map_equilibrated.pdb"
rho = compute_rho(dcd_filename, pdb_filename, np.arange(50)*3, bin_width=0.10)

plt.figure()
r_list = np.linspace(0, 10)
rho_list = np.empty(len(r_list))
for i, r in enumerate(r_list):
    rho_list[i] = rho(r)
if plot_radial:
    plt.plot(r_list, 4*np.pi*r_list**2*rho_list)
    plt.plot(r_list, 4*np.pi*r_list**2*50*(3/(2*np.pi*2.4**2))**1.5*np.exp(-3*r_list**2/(2*2.4**2)))
else:
    plt.plot(r_list, rho_list)
    plt.plot(r_list, 50*(3/(2*np.pi*2.4**2))**1.5*np.exp(-3*r_list**2/(2*2.4**2)))
plt.show()
