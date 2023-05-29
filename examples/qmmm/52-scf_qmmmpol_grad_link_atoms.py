import numpy as np
from pyscf import scf
from pyscf import gto
from pyscf import qmmm
import pyopenmmpol as ommp
import os.path as path
import scipy.constants as pc
from os import path

basedir = './test_ommp/tyrosine_cap/'
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
myscf_qmmmpol.ommp_qm_helper.set_attype([70,72,73,73,75,75,77,78,71,71,74,74,76,76,79,71])
myscf_qmmmpol.ommp_qm_helper.init_vdw_prm(INPUT_PRM)
myscf_qmmmpol.create_link_atom(2, 1, 16, INPUT_PRM)

myscf_qmmmpol.kernel()

vdw_energy = env.get_vdw_energy() + myscf_qmmmpol.ommp_qm_helper.vdw_energy(myscf_qmmmpol.ommp_obj)
em =   env.get_fixedelec_energy() * au2kcalmol
ep =   env.get_polelec_energy() * au2kcalmol
ev =   vdw_energy * au2kcalmol
eb =   env.get_bond_energy() * au2kcalmol
ea =   env.get_angle_energy() * au2kcalmol
eba =  env.get_strbnd_energy() * au2kcalmol
eub =  env.get_urey_energy() * au2kcalmol
eopb = env.get_opb_energy() * au2kcalmol
ept =  env.get_pitors_energy() * au2kcalmol
et =   env.get_torsion_energy() * au2kcalmol
ett =  env.get_tortor_energy() * au2kcalmol
eat =  env.get_angtor_energy() * au2kcalmol
ebt =  env.get_strtor_energy() * au2kcalmol
eit =  env.get_imptorsion_energy() * au2kcalmol
eaa = 0.0
eopd = 0.0
eid = 0.0
er = 0.0
edsp = 0.0
ec = 0.0
ecd = 0.0
ed = 0.0
ect = 0.0
erxf = 0.0
es = 0.0
elf = 0.0
eg = 0.0
ex = 0.0

print("EM      {:20.12e}".format(em))
print("EP      {:20.12e}".format(ep))
print("EV      {:20.12e}".format(ev))
print("EB      {:20.12e}".format(eb))
print("EA      {:20.12e}".format(ea))
print("EBA     {:20.12e}".format(eba))
print("EUB     {:20.12e}".format(eub))
print("EOPB    {:20.12e}".format(eopb))
print("EPT     {:20.12e}".format(ept))
print("ET      {:20.12e}".format(et))
print("ETT     {:20.12e}".format(ett))

print("EAA     {:20.12e}".format(eaa))
print("EOPD    {:20.12e}".format(eopd))
print("EID     {:20.12e}".format(eid))
print("EIT     {:20.12e}".format(eit))
print("EBT     {:20.12e}".format(ebt))
print("EAT     {:20.12e}".format(eat))
print("ER      {:20.12e}".format(er))
print("EDSP    {:20.12e}".format(edsp))
print("EC      {:20.12e}".format(ec))
print("ECD     {:20.12e}".format(ecd))
print("ED      {:20.12e}".format(ed))
print("ECT     {:20.12e}".format(ect))
print("ERXF    {:20.12e}".format(erxf))
print("ES      {:20.12e}".format(es))
print("ELF     {:20.12e}".format(elf))
print("EG      {:20.12e}".format(eg))
print("EX      {:20.12e}".format(ex))

myscf_grad = myscf_qmmmpol.nuc_grad_method()
myscf_grad.verbose = 0

qmg, mmg = myscf_grad.kernel(domm=True)
print("GRAD ON MM ATOMS")
print(mmg)
print(np.sum(mmg, axis=0))
print("GRAD ON QM ATOMS")
print(qmg)
