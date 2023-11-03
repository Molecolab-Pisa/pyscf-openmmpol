import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
from pyscf.geomopt.geometric_solver import optimize
from os import path

molstr = ''
with open('tests/alanine_cap//input_LA_QM_AMOEBA.xyz') as f:
    for i, l in enumerate(f):
        if i > 0:
            molstr += ' '.join(l.split()[1:5])
            molstr += '\n'
print(molstr)
molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)
myscf_qmmmpol = qmmm.add_mmpol(myscf, 'tests/alacap_amoeba_xyz.json')
myscf_grad = myscf_qmmmpol.nuc_grad_method()
qmmm_scanner = qmmm.qmmmpol_grad_as_qmmm_scanner(myscf_grad)

mol_eq = optimize(qmmm_scanner, maxsteps=1000)

