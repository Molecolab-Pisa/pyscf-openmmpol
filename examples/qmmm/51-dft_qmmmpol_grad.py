import numpy as np
from pyscf import scf, dft
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
from os import path

basedir = './test_ommp/N-methylacetamide/'
INPUT_MMP = path.join(basedir, 'input_AMOEBA.mmp')

with open(path.join(basedir, 'QM.xyz')) as f:
    molstr = f.read()

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = dft.RKS(molQM)
myscf.xc = 'B3LYPG'

env = ommp.OMMPSystem(INPUT_MMP)
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)
myscf_qmmmpol.kernel()
myscf_grad = myscf_qmmmpol.nuc_grad_method()
myscf_grad.verbose = 0

qmg, mmg = myscf_grad.kernel(domm=True)
print("GRAD ON QM ATOMS")
print(qmg)
print("GRAD ON MM ATOMS")
print(mmg)
