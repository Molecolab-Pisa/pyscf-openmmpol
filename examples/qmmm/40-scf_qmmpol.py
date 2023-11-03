import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
from os import path

with open('NMA_QM.xyz') as f:
    molstr = f.read()

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)
myscf_qmmmpol = qmmm.add_mmpol(myscf, 'tests/NMA_amoeba_xyz.json')
myscf_qmmmpol.kernel()
myscf_qmmmpol.energy_analysis()
