import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
from pyscf import md
import pyopenmmpol as ommp
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
env.set_frozen_atoms([0,1,2])

myscf_grad = myscf_qmmmpol.nuc_grad_method()
qmmm_scanner = qmmm.qmmmpol_grad_as_qmmm_scanner(myscf_grad)

au2fs = 0.02418884254
myintegrator = md.NVE(qmmm_scanner,
                      dt=.5 / au2fs,
                      steps=100,
                      energy_output="62-energies.dat",
                      trajectory_output="62-pyscf_md.xyz")

myintegrator.run()
myintegrator.energy_output.close()
myintegrator.trajectory_output.close()
