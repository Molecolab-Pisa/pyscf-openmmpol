import numpy as np
from pyscf import md
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
import os.path as path
import scipy.constants as pc
import json

au2kcalmol = pc.physical_constants['Hartree energy'][0]/(1000.0 * pc.calorie / pc.N_A )

si_file = 'tests/tyr2cap_amoeba_xyz.json'

# Try to get QM info directly from JSON
with open(si_file, 'r') as f:
    si_data = json.loads(f.read())

molstr = ''
if 'qm' in si_data:
    atc = si_data['qm']['qm_coords']
    atn = si_data['qm']['qm_atoms']
    for z, c in zip(atn, atc):
        molstr += "{:3s} {:f} {:f} {:f}\n".format(z, c[0], c[1], c[2])

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)

myscf_qmmmpol = qmmm.add_mmpol(myscf, si_file)
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
