from __future__ import absolute_import, division

import warnings

from scipy.constants import N_A, R
import mdtraj as md
import numpy as np

from polyiv.utils import *


class Zimm(object):

    def __init__(self, trajectory, chain_index=0):

        # slice trajectory so that only the chain of interest is kept
        self._slice_trajectory(trajectory, chain_index)

        # determine the type of each bead and number of beads
        self._determine_bead_types()
        self._n_beads = len(self._bead_types)

        # dictionaries for radii of each bead type and force constants between
        # bead type pairs
        self._bead_type_radii = {}
        self._force_constants = PairKeyDict()

        # dictionaries for bonds
        self._bonds = Pairs()

        # initialize A matrix and reference force constant
        self._A = None
        self._k_ref = None

        # initialize matrix storing r^2 for each pair
        self._r2 = None

        # initialize H matrix and reference radius and friction constant
        self._H = None
        self._radius_ref = None
        self._friction_ref = None
        
        # initialize intrinsic viscosity
        self._eta0 = None

    def _slice_trajectory(self, trajectory, chain_index):
        topology = trajectory.topology
        chain = list(topology.chains)[chain_index]
        bead_indices = [atom.index for atom in chain.atoms]
        self._trajectory = trajectory.atom_slice(bead_indices)
        self._topology = self._trajectory.topology

    def _determine_bead_types(self):
        self._bead_types = []
        for atom in self._topology.atoms:
            self._bead_types.append(atom.name)

    @classmethod
    def from_dcd(cls, dcd, top, stride=1, chain_index=0):
        trajectory = md.load_dcd(dcd, top, stride=stride)
        return cls(trajectory, chain_index=chain_index)

    @property
    def trajectory(self):
        """Trajectory of the system.

        Returns
        -------
        n_frames : md.Trajectory
            the trajectory of the system
        """
        return self._trajectory

    @property
    def topology(self):
        """Topology of the system, describing the organization of atoms into
        residues, bonds, etc

        Returns
        -------
        topology : md.Topology
            the topology object, describing the organization of atoms into
            residues, bonds, etc
        """
        return self._topology

    @property
    def bead_types(self):
        return self._bead_types

    @property
    def n_beads(self):
        return self._n_beads

    def set_bead_type_radius(self, bead_type, radius):
        self._bead_type_radii[bead_type] = radius

    def get_bead_type_radius(self, bead_type):
        return self._bead_type_radii[bead_type]
        
    def set_force_constant(self, bead_type_1, bead_type_2, force_constant):
        self._force_constants.set_pair_value(bead_type_1, bead_type_2, force_constant)

    def get_force_constant(self, bead_type_1, bead_type_2):
        return self._force_constants.get_pair_value(bead_type_1, bead_type_2)

    def add_bond(self, bead_index_1, bead_index_2):
        if bead_index_1 == bead_index_2:
            raise ValueError("bead_index_1 is same as bead_index_2; bead cannot be bonded to itself")
        self._bonds.add_pair(bead_index_1, bead_index_2)

    @property
    def bonds(self):
        for bond in self._bonds:
            yield bond

    def compute_A(self):
        self._A = np.zeros((self._n_beads, self._n_beads))
        for i, j in self.bonds:
            k_ij = self.get_force_constant(self._bead_types[i], self._bead_types[j])
            if self._k_ref is None:
                self._k_ref = k_ij
            self._A[i][j] = -k_ij
            self._A[j][i] = -k_ij
        di = np.diag_indices(self._n_beads)
        self._A[di] = -np.sum(self._A, axis=1)
        self._A /= self._k_ref

    @property
    def A(self):
        return self._A

    def compute_r2(self):
        xyz = self._trajectory.xyz[:, None, :]
        self._r2 = np.sum((xyz - np.transpose(xyz, [0, 2, 1, 3]))**2, axis=3)

    @property
    def r2(self):
        return self._r2

    def compute_H(self):

        # get bead radii and determine reference radius and friction coefficient
        bead_radii = np.array([self.get_bead_type_radius(bead_type) for bead_type in self._bead_types])
        self._radius_ref = bead_radii[0]
        self._friction_ref = 6. * np.pi * self._radius_ref

        # compute rms separation distances
        rms_r = np.sqrt(np.mean(self._r2, axis=0))
        np.fill_diagonal(rms_r, 1.)     # fill diagonals with 1 to prevent divide by 0 error
        
        # compute H matrix
        self._H = 1. / rms_r
        for i, ri in enumerate(bead_radii):
            for j, rj in enumerate(bead_radii):
                if i != j and 1. / self._H[i][j] < ri + rj:
                    self._H[i][j] = 1. / (self._H[i][j] * (ri + rj)**2)
        di = np.diag_indices(self._n_beads)
        self._H[di] = 1. / bead_radii
        self._H *= self._radius_ref

    @property
    def H(self):
        return self._H
    
    def compute_intrinsic_viscosity(self, T, M):
        
        # compute eigenvalues of HA
        HA = np.matmul(self._H, self._A)
        w = np.linalg.eigvals(HA)
        w = np.sort(w)
        
        if np.any(w[1:] < 0):
            warnings.warn("HA matrix has negative eigenvalues")
        
        # compute intrinsic viscosity
        kT = R * 1.e-3 * T
        self._eta0 = 3. * np.pi * N_A * kT * self._radius_ref / (M * self._k_ref) * np.sum(1. / w[1:]) * 1.e-21
        
    @property
    def eta0(self):
        return self._eta0
