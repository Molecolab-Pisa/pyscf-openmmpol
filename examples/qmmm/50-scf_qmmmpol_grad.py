import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp

INPUT_AMOEBA = 'NMA_AMOEBA_norot_noquad.mmp'

molstr = \
'''C                 -1.57450000    1.55590000    0.07110000
  H                 -1.45050000    0.47240000    0.13850000
  H                 -0.72600000    2.03180000    0.56740000
  H                 -2.49140000    1.83090000    0.59740000
  C                 -1.64480000    2.00110000   -1.37200000
  O                 -2.30430000    2.95790000   -1.72690000
  C                 -0.87328726    1.24775882   -2.26707939
  H                  0.13930735    1.21260318   -1.92310810
  H                 -1.26463077    0.25364426   -2.32612588
  H                 -0.90547150    1.70183658   -3.23541691'''
molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)

env = ommp.OMMPSystem(INPUT_AMOEBA)
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)
myscf_qmmmpol.kernel()
print(myscf_qmmmpol.get_mmpol_induced_dipoles())
myscf_grad = myscf_qmmmpol.nuc_grad_method()
print(myscf_grad.MM_atoms_grad())

