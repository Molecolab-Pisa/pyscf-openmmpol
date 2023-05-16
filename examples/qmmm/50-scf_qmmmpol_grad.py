import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp

molstr = \
''' C               -1.574500     1.555900     0.071100
 H               -1.450500     0.472400     0.138500
 H               -0.726000     2.031800     0.567400
 H               -2.491400     1.830900     0.597400
 C               -1.644800     2.001100    -1.372000
 O               -2.304300     2.957900    -1.726900
 N               -0.901800     1.275600    -2.234000
 H               -0.369300     0.491600    -1.882500
 C               -0.819600     1.544900    -3.670900
 H               -0.445600     2.559600    -3.831500
 H               -0.137400     0.826800    -4.133100
 H               -1.812300     1.445700    -4.118000'''

au2k = 627.50960803

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

myscf = scf.RHF(molQM)

env = ommp.OMMPSystem('input.xyz', 'amoeba09.prm')
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)
myscf_qmmmpol.ommp_qm_helper.init_vdw_prm([133, 134, 134, 134, 127, 128, 129, 130, 131, 132, 132, 132], 'amoeba09.prm')
print(myscf_qmmmpol.ommp_qm_helper.vdw_energy(env)*au2k)
print(myscf_qmmmpol.ommp_qm_helper.vdw_geomgrad(env)['QM']*au2k*1.8897161646320724)
print(env.get_vdw_energy()*au2k)
myscf_qmmmpol.kernel()
myscf_grad = myscf_qmmmpol.nuc_grad_method()
myscf_grad.verbose = 0

print("GRAD ON MM ATOMS")
print(myscf_grad.MM_atoms_grad())
print("GRAD ON QM ATOMS")
print(myscf_grad.grad())