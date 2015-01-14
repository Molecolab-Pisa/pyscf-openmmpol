#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.scf import dfhf

mol = gto.Mole()
mol.build(
    verbose = 5,
    output = '/dev/null',
    atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)


class KnowValues(unittest.TestCase):
    def test_rhf(self):
        mf = scf.dfhf.RHF(mol)
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 9)

    def test_uhf(self):
        mf = scf.dfhf.UHF(mol)
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 9)

    def test_rohf(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.dfhf.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -75.626515724371814, 9)

    def test_rhf_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,nao,nao))
        mf = scf.dfhf.RHF(mol)
        vhf1 = mf.get_veff(mol, dm, hermi=0)
        naux = mf._cderi.shape[0]
        cderi = numpy.empty((naux,nao,nao))
        for i in range(naux):
            cderi[i] = lib.unpack_tril(mf._cderi[i])
        vj0 = []
        vk0 = []
        for dmi in dm:
            v1 = numpy.einsum('kij,ij->k', cderi, dmi)
            vj0.append(numpy.einsum('kij,k->ij', cderi, v1))
            v1 = numpy.einsum('pij,jk->pki', cderi, dmi.T)
            vk0.append(numpy.einsum('pki,pkj->ij', cderi, v1))
        vj1 = scf.dfhf._make_j(mf, dm, 0)
        self.assertTrue(numpy.allclose(vj0, vj1))
        vk1 = scf.dfhf._make_k(mf, dm, 0)
        self.assertTrue(numpy.allclose(numpy.array(vk0), vk1))
        vhf0 = vj1 - vk1 * .5
        print abs(vhf0-vhf1).sum()
        self.assertTrue(numpy.allclose(vhf0, vhf1), 9)

    def test_uhf_veff(self):
        mf = scf.dfhf.UHF(mol)
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((4,nao,nao))
        vhf = mf.get_veff(mol, dm, hermi=0)
        self.assertAlmostEqual(numpy.linalg.norm(vhf), 188.34081056589872, 9)

if __name__ == "__main__":
    print("Full Tests for df")
    unittest.main()

