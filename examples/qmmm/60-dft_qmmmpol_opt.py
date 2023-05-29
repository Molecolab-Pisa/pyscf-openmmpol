import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
from pyscf.geomopt.geometric_solver import optimize
from os import path

basedir = './test_ommp/four_waters/'
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
myscf_qmmmpol.ommp_qm_helper.set_attype([39, 40, 40])
myscf_qmmmpol.ommp_qm_helper.init_vdw_prm(INPUT_PRM)
myscf_grad = myscf_qmmmpol.nuc_grad_method()
qmmm_scanner = qmmm.qmmmpol_grad_as_qmmm_scanner(myscf_grad)

mol_eq = optimize(qmmm_scanner, maxsteps=1000)

