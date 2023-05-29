import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
from os import path

basedir = './test_ommp/N-methylacetamide/'
INPUT_AMOEBA = path.join(basedir, 'input_AMOEBA.mmp')

with open(path.join(basedir, 'QM.xyz')) as f:
    molstr = f.read()

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)
env = ommp.OMMPSystem(INPUT_AMOEBA)
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)
myscf_qmmmpol.kernel()
myscf_qmmmpol.energy_analysis()
