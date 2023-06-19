import numpy as np
from pyscf import md
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
import os.path as path
import scipy.constants as pc

basedir = './test_ommp/alanine_cap/'
INPUT_XYZ = path.join(basedir, 'input.xyz')
INPUT_PRM = path.join(basedir, '../amoebabio18.prm')

with open(path.join(basedir, 'QM.xyz')) as f:
    molstr = f.read()

au2kcalmol = pc.physical_constants['Hartree energy'][0]/(1000.0 * pc.calorie / pc.N_A )

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)

env = ommp.OMMPSystem(INPUT_XYZ, INPUT_PRM)
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)
myscf_qmmmpol.ommp_qm_helper.set_attype([13,14,14,14,14])
myscf_qmmmpol.ommp_qm_helper.init_vdw_prm(INPUT_PRM)
myscf_qmmmpol.create_link_atom(2, 1, 5, INPUT_PRM)

myscf_grad = myscf_qmmmpol.nuc_grad_method()
myscf_grad.verbose = 0

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
