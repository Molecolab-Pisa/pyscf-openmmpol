import numpy as np
import sys
from pyscf import scf, df
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp

INPUT_AMOEBA = sys.argv[1]

with open(sys.argv[2], 'r') as f:
    molstr = f.read()

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)

env = ommp.OMMPSystem(INPUT_AMOEBA)
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)
myscf_qmmmpol.kernel()
myscf_grad = myscf_qmmmpol.nuc_grad_method()
ana_grad_QM = myscf_grad.grad()
ana_grad_MM = myscf_grad.MM_atoms_grad()

print("Analytical Forces on QM atoms")
print(ana_grad_QM)
print("Analytical Forces on MM atoms")
print(ana_grad_MM)
