import numpy as np
import sys
from pyscf import scf, df
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp

def numgrad_scf(myscf):
    molQM = myscf.mol
    c0 = molQM.atom_coords()
    grd = np.zeros(c0.shape)
    delta = 1e-4
    for i in range(molQM.natm):
        for j in range(3):
            c0[i,j] += delta
            molQM.set_geom_(c0, unit='AU')
            newscf = scf.RHF(molQM)
            newscf.verbose=0
            env = ommp.OMMPSystem(INPUT_AMOEBA)
            newscf = qmmm.add_mmpol(newscf, env)
            newscf.kernel()
            ep = newscf.e_tot

            c0[i,j] -= 2*delta
            molQM.set_geom_(c0, unit='AU')
            newscf = scf.RHF(molQM)
            newscf.verbose=0
            env = ommp.OMMPSystem(INPUT_AMOEBA)
            newscf = qmmm.add_mmpol(newscf, env)
            newscf.kernel()
            em = newscf.e_tot

            grd[i,j] = (ep-em)/(2*delta)

            c0[i,j] += delta
            molQM.set_geom_(c0, unit='AU')
    return grd

def numgrad_scf_MM(myscf):
    molQM = myscf.mol
    c0 = myscf.ommp_obj.cmm
    grd = np.zeros(c0.shape)
    env = ommp.OMMPSystem(INPUT_AMOEBA)
    delta = 1e-4
    for i in range(c0.shape[0]):
        for j in range(3):
            c0[i,j] += delta
            newscf = scf.RHF(molQM)
            newscf.verbose=0
            env.update_coordinates(c0)
            newscf = qmmm.add_mmpol(newscf, env)
            newscf.kernel()
            ep = newscf.e_tot

            c0[i,j] -= 2*delta
            newscf = scf.RHF(molQM)
            newscf.verbose=0
            env.update_coordinates(c0)
            newscf = qmmm.add_mmpol(newscf, env)
            newscf.kernel()
            em = newscf.e_tot

            grd[i,j] = (ep-em)/(2*delta)

            c0[i,j] += delta
    return grd

INPUT_AMOEBA = sys.argv[1]

with open(sys.argv[2], 'r') as f:
    molstr = f.read()

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)

env = ommp.OMMPSystem(INPUT_AMOEBA)
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)
myscf_qmmmpol.kernel()
myscf_grad = myscf_qmmmpol.nuc_grad_method()
ana_grad_QM = myscf_grad.grad()
ana_grad_MM = myscf_grad.MM_atoms_grad()
num_grad_MM = numgrad_scf_MM(myscf_qmmmpol)
num_grad_QM = numgrad_scf(myscf)

if np.allclose(ana_grad_QM, num_grad_QM, rtol=1e-4, atol=1e-6):
    print("Forces on QM atoms are OK")
else:
    print("Analytical Forces on QM atoms")
    print(ana_grad_QM)
    print("Numerical Forces on QM atoms")
    print(num_grad_QM)
    print("Max absolute diff: {}".format(np.max(np.abs(ana_grad_QM-num_grad_QM))))
    print("Max absolute diff: {}".format(np.max(np.abs(ana_grad_QM-num_grad_QM)/num_grad_QM)))

if np.allclose(ana_grad_MM, num_grad_MM, rtol=1e-4, atol=1e-6):
    print("Forces on MM atoms are OK")
else:
    print("Analytical Forces on MM atoms")
    print(ana_grad_MM)
    print("Numerical Forces on MM atoms")
    print(num_grad_MM)
    print("Max absolute diff: {}".format(np.max(np.abs(ana_grad_MM-num_grad_MM))))
    print("Max absolute diff: {}".format(np.max(np.abs(ana_grad_MM-num_grad_MM)/num_grad_MM)))
