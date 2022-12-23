import numpy as np
from pyscf import scf
from pyscf import dft
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp

INPUT_AMOEBA = 'NMA_AMOEBA.mmp'

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

molQM = gto.M(verbose=3,
              atom = molstr,
              basis='3-21g')

#myscf = scf.RHF(molQM)
#myscf.conv_tol = 1e-10
#myscf.init_guess = 'huckel'
myscf = dft.RKS(molQM)
myscf.xc = 'B3LYPG'

env = ommp.OMMPSystem(INPUT_AMOEBA)
myscf_qmmmpol = qmmm.add_mmpol(myscf, env)
myscf_qmmmpol.kernel()

au2k = 627.50960803
scf_ene = myscf_qmmmpol.e_tot
smm_ene = env.get_fixedelec_energy()
pmm_ene = env.get_polelec_energy()
nuc_mm_ene = myscf_qmmmpol.nuc_static_mm

dm  = myscf_qmmmpol.make_rdm1()
ele_mm_ene = np.einsum('nm,nm', myscf_qmmmpol.h1e_mmpol, dm)
ele_p_ene = myscf_qmmmpol.get_veff(dm=dm).e_mmpol - pmm_ene
qm_mm_ene = nuc_mm_ene + ele_mm_ene
etot = scf_ene + smm_ene + pmm_ene

print("SCF e-tot: {:20.10f} ({:20.10f})".format(scf_ene, scf_ene*au2k))
print("MM-MM:     {:20.10f} ({:20.10f})".format(smm_ene, smm_ene*au2k))
print("IPD-MM:    {:20.10f} ({:20.10f})".format(pmm_ene, pmm_ene*au2k))
print("NUC-MM:    {:20.10f} ({:20.10f})".format(nuc_mm_ene,
                                                nuc_mm_ene*au2k))
print("ELE-MM:    {:20.10f} ({:20.10f})".format(ele_mm_ene,
                                                ele_mm_ene*au2k))
print("ELE-IPD:   {:20.10f} ({:20.10f})".format(ele_p_ene,
                                                ele_p_ene*au2k))
print("QM-MM:     {:20.10f} ({:20.10f})".format(qm_mm_ene,
                                                qm_mm_ene*au2k))

print("E TOT:     {:20.10f} ({:20.10f})".format(etot,
                                                etot*au2k))
