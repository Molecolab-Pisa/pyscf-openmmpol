import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
from os import path

basedir = './test_ommp/N-methylacetamide/'
INPUT_XYZ = path.join(basedir, 'input.xyz')
INPUT_PRM = path.join(basedir, '../amoeba09.prm')

with open(path.join(basedir, 'QM.xyz')) as f:
    molstr = f.read()

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)

env = ommp.OMMPSystem(INPUT_XYZ, INPUT_PRM)
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)

myscf_qmmmpol.ommp_qm_helper.set_attype([133, 134, 134, 134, 127, 128, 129, 130, 131, 132, 132, 132])
myscf_qmmmpol.ommp_qm_helper.init_vdw_prm(INPUT_PRM)
myscf_qmmmpol.kernel()
myscf_grad = myscf_qmmmpol.nuc_grad_method()
myscf_grad.verbose = 0

qmg, mmg = myscf_grad.kernel(domm=True)
print("GRAD ON QM ATOMS")
print(qmg)
print("GRAD ON MM ATOMS")
print(mmg)
