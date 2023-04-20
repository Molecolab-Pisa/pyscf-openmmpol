import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
from pyscf.geomopt.geometric_solver import optimize

with open('water_opt_QM.xyz', 'r') as f:
    molstr = f.read()

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)
ommp.set_verbose(0)
env = ommp.OMMPSystem('water_opt_MM.xyz', 'amoeba09.prm')
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)
myscf_qmmmpol.ommp_qm_helper.init_vdw_prm([39, 40, 40], 'amoeba09.prm')

mol_eq = optimize(myscf_qmmmpol, maxsteps=100)

