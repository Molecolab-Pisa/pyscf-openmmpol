import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
from pyscf import md
import pyopenmmpol as ommp

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
env.set_frozen_atoms([0,1,2])
print(env.frozen)
#exit()
myscf_grad = myscf_qmmmpol.nuc_grad_method()
myscf_grad.atmlst = [1,2]
qmmm_scanner = qmmm.qmmmpol_grad_as_qmmm_scanner(myscf_grad)

au2fs = 0.02418884254
myintegrator = md.NVE(qmmm_scanner,
                      dt=.5 / au2fs,
                      steps=100,
                      energy_output="energies.dat",
                      trajectory_output="pyscf_md.xyz")

myintegrator.run()
myintegrator.energy_output.close()
myintegrator.trajectory_output.close()
