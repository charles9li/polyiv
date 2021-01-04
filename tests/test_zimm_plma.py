from __future__ import absolute_import

import unittest

from polyiv import Zimm


class MyTestCase(unittest.TestCase):

    def test_something(self):
        # create Zimm object
        zimm = Zimm.from_dcd("data/plma_50mA12_NPT_313K_3866bar_3wt_1map_traj.dcd", "data/plma_50mA12_NPT_313K_3866bar_3wt_1map_equilibrated.pdb")

        # set radius of each bead type
        zimm.set_bead_type_radius('B', 0.19)
        zimm.set_bead_type_radius('C', 0.25)
        zimm.set_bead_type_radius('D', 0.25)

        # set force constant of each bond type
        zimm.set_force_constant('B', 'B', 2.5512e3)
        zimm.set_force_constant('B', 'C', 6.6185e2)
        zimm.set_force_constant('C', 'D', 3.9857e2)

        # add bonds
        for i in range(50):
            zimm.add_bond(i*3, i*3 + 1)
            zimm.add_bond(i*3+1, i*3+2)
        for i in range(50 - 1):
            zimm.add_bond(i*3, (i+1)*3)

        # compute A matrix
        zimm.compute_A()

        # compute square distances between each bead
        zimm.compute_r2()

        # compute H matrix
        zimm.compute_H()

        # compute intrinsic viscosity
        zimm.compute_intrinsic_viscosity(313.15, 50*254.41)
        self.assertEqual(0.12451113916710885, zimm.eta0)


if __name__ == '__main__':
    unittest.main()
