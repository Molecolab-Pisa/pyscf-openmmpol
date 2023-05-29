import numpy as np
from pyscf import scf, tdscf
from pyscf import dft
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
              basis='sto-3g')

myscf = scf.RHF(molQM)

env = ommp.OMMPSystem(INPUT_AMOEBA)
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)

myscf_qmmmpol.kernel()

tdscf = tdscf.TDHF(myscf_qmmmpol)
tdscf.nroots = 5
tdscf.max_cycle = 1000
tdscf.run()
tdscf.analyze()
