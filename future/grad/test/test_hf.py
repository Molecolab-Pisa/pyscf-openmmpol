#!/usr/bin/env python

import unittest
from pyscf import scf
from pyscf import gto
from pyscf import grad

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'

mol.atom.extend([
    [1   , (0. , 0.1, .817)],
    ["F" , (0. , 0. , 0.)], ])
mol.basis = {"H": '6-31g',
             "F": '6-31g',}
mol.build()

def finger(mat):
    return abs(mat).sum()

class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        rhf = scf.RHF(mol)
        rhf.scf()
        g = grad.hf.RHF(rhf)
        self.assertAlmostEqual(finger(g.grad_elec()), 7.9210392362911595, 7)
        self.assertAlmostEqual(finger(g.grad()), 0.367743084803, 7)

    def test_r_uhf(self):
        uhf = scf.dhf.UHF(mol)
        uhf.scf()
        g = grad.dhf.UHF(uhf)
        self.assertAlmostEqual(finger(g.grad_elec()), 7.9216825870803245, 7)
        g.level = 'LLLL'
        self.assertAlmostEqual(finger(g.grad_elec()), 7.924684281032623, 7)

    def test_energy_nuc(self):
        rhf = scf.RHF(mol)
        rhf.scf()
        g = grad.hf.RHF(rhf)
        self.assertAlmostEqual(finger(g.grad_nuc()), 8.2887823210941249, 9)


if __name__ == "__main__":
    print("Full Tests for HF")
    unittest.main()
