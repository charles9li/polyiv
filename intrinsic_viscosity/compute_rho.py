from numba import njit
from scipy.interpolate import interp1d
import mdtraj as md
import numpy as np
import pandas as pd


def compute_rho(dcd_filename, top_filename, backbone_indices, r_max=10., bin_width=0.05, output_filename="rho.csv"):

    # convert backbone_indices to numpy array
    backbone_indices = np.array(backbone_indices)

    # import traj
    traj = md.load_dcd(dcd_filename, top=top_filename)

    # initialize arrays
    r_list = np.arange(0, r_max+bin_width, bin_width)
    histogram = np.zeros(len(r_list))

    # create histogram
    for frame in traj.xyz:
        backbone_positions = np.array(frame[backbone_indices])
        r_cm = np.mean(backbone_positions, axis=0)
        relative_positions = backbone_positions - r_cm
        relative_dist = np.sqrt(np.sum(relative_positions*relative_positions, axis=1))
        for dist in relative_dist:
            index = int(dist/bin_width)
            histogram[index] += 1

    # compute volume of each bin
    v_list = np.zeros(len(r_list))
    for i in range(len(r_list)):
        v_list[i] = 4./3.*np.pi*bin_width**3*((i+1)**3 - i**3)

    # compute rho
    rho_list = histogram / len(traj.xyz) / v_list

    @njit
    def rho_function(r):
        if r <= np.max(r_list):
            return np.interp(r, r_list, rho_list)
        return 0

    return rho_function
