#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Mattia Bondanza <mattia.bondanza@unipi.it>
#

'''
QM/MMPol and QM/AMOEBA helper functions that modify the QM methods.
'''

import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf import scf
from pyscf import mcscf
from pyscf import grad
from pyscf.lib import logger
from pyscf.qmmm.itrf import _QMMM, _QMMMGrad
import pyopenmmpol as ommp


def add_mmpol(scf_method, ommp_obj):
    if isinstance(ommp_obj, ommp.OMMPSystem):
        return qmmmpol_for_scf(scf_method, ommp_obj)
    else:
        raise RuntimeError("Initialize OMMP library before adding "
                           "mmpol environment to a method.")

def qmmmpol_for_scf(scf_method, ommp_obj):
    assert(isinstance(scf_method, (scf.hf.SCF, mcscf.casci.CASCI)))
    assert(isinstance(ommp_obj, ommp.OMMPSystem))

    if isinstance(scf_method, scf.hf.SCF):
        # Avoid to initialize _QMMM twice
        if isinstance(scf_method, _QMMMPOL):
            # TODO Insert a log message here
            return scf_method
        method_class = scf_method.__class__
    else:
        raise NotImplementedError("Only SCF-method are currently "
                                  "supported in QM/MMPol")

    class QMMMPOL(_QMMMPOL, method_class):
        def __init__(self, scf_method, ommp_obj):
            self.__dict__.update(scf_method.__dict__)
            self.ommp_obj = ommp_obj
            self._keys.update(['ommp_obj'])

        @property
        def do_pol(self):
            return self.ommp_obj.pol_atoms > 0

        def get_mmpol_induced_dipoles(self):
            if not self.ommp_obj.is_amoeba:
                mu = self.ommp_obj.ipd[0,:,:]
                return mu
            else:
                mu_d = self.ommp_obj.ipd[0,:,:]
                mu_p = self.ommp_obj.ipd[1,:,:]
                return mu_d, mu_p

        @property
        def fakemol_static(self):
            if not hasattr(self, '_fakemol_static'):
                self._fakemol_static = gto.fakemol_for_charges(self.ommp_obj.cmm)
            return self._fakemol_static

        @property
        def fakemol_pol(self):
            if self.do_pol:
                if not hasattr(self, '_fakemol_pol'):
                    self._fakemol_pol = gto.fakemol_for_charges(self.ommp_obj.cpol)
            else:
                self._fakemol_pol = None

            return self._fakemol_pol

        @property
        def v_integrals_at_static(self):
            if not hasattr(self, '_v_int_at_cmm'):
                self._v_int_at_cmm = df.incore.aux_e2(self.mol,
                                                      self.fakemol_static,
                                                      intor='int3c2e')
            return self._v_int_at_cmm

        @property
        def ef_integrals_at_static(self):
            if not hasattr(self, '_ef_int_at_cmm'):
                self._ef_int_at_cmm = df.incore.aux_e2(self.mol,
                                                       self.fakemol_static,
                                                       intor='int3c2e_ip1')
                self._ef_int_at_cmm += numpy.einsum('imnj->inmj', self._ef_int_at_cmm)
            return self._ef_int_at_cmm

        @property
        def gef_integrals_at_static(self):
            if not hasattr(self, '_gef_int_at_cmm'):
                # PySCF order for field gradient tensor
                #  0  1  2  3  4  5  6  7  8
                # xx xy xz xy yy yz xz yz zz
                #
                # 1 3
                # 2 6
                # 5 7
                #
                # OMMP order for field gradient tensor
                #  0  1  2  3  4  5
                # xx xy yy xz yz zz

                pyscf2ommp_idx = [0,1,4,2,5,8]

                int1 = df.incore.aux_e2(self.mol,
                                       self.fakemol_static,
                                        intor='int3c2e_ipip1')
                # Make symmetric, (double the out-of diagonal), and compress
                int1[[1,2,5]] += int1[[3,6,7]]
                int1 = int1[pyscf2ommp_idx]

                int2 = df.incore.aux_e2(self.mol,
                                        self.fakemol_static,
                                        intor='int3c2e_ipvip1')
                # Make symmetric, (double the out-of diagonal), and compress
                int2[[1,2,5]] += int2[[3,6,7]]
                int2 = int2[pyscf2ommp_idx]

                self._gef_int_at_cmm = int1 + numpy.einsum('inmj->imnj', int1) + 2 * int2

            return self._gef_int_at_cmm

        @property
        def ef_integrals_at_pol(self):
            if self.fakemol_pol is None:
                return np.zeros([3, self.mol.nbas, self.mol.nbas, 0])

            if not hasattr(self, '_ef_int_at_cpol'):
                self._ef_int_at_cpol = df.incore.aux_e2(self.mol,
                                                        self.fakemol_pol,
                                                        intor='int3c2e_ip1')
                self._ef_int_at_cpol += numpy.einsum('imnj->inmj', self._ef_int_at_cpol)
            return self._ef_int_at_cpol

        @property
        def gef_integrals_at_fixed(self):
            if not hasattr(self, '_gef_int_at_mm'):
                # PySCF order for field gradient tensor
                #  0  1  2  3  4  5  6  7  8
                # xx xy xz xy yy yz xz yz zz
                #
                # 1 3
                # 2 6
                # 5 7
                #
                # OMMP order for field gradient tensor
                #  0  1  2  3  4  5
                # xx xy yy xz yz zz

                pyscf2ommp_idx = [0,1,4,2,5,8]

                int1 = df.incore.aux_e2(self.mol,
                                        self.fakemol_static,
                                        intor='int3c2e_ipip1')
                # Make symmetric, (double the out-of diagonal), and compress
                int1[[1,2,5]] += int1[[3,6,7]]
                int1[[1,2,5]] /= 2.0
                int1 = int1[pyscf2ommp_idx]

                int2 = df.incore.aux_e2(self.mol,
                                        self.fakemol_static,
                                        intor='int3c2e_ipvip1')
                # Make symmetric, (double the out-of diagonal), and compress
                int2[[1,2,5]] += int2[[3,6,7]]
                int2[[1,2,5]] /= 2.0
                int2 = int2[pyscf2ommp_idx]

                self._gef_int_at_cmm = int1 + numpy.einsum('inmj->imnj', int1) + 2 * int2

            return self._gef_int_at_cmm

        @property
        def gef_integrals_at_pol(self):
            if not hasattr(self, '_gef_int_at_cpol'):
                # PySCF order for field gradient tensor
                #  0  1  2  3  4  5  6  7  8
                # xx xy xz xy yy yz xz yz zz
                #
                # 1 3
                # 2 6
                # 5 7
                #
                # OMMP order for field gradient tensor
                #  0  1  2  3  4  5
                # xx xy yy xz yz zz

                pyscf2ommp_idx = [0,1,4,2,5,8]

                int1 = df.incore.aux_e2(self.mol,
                                        self.fakemol_pol,
                                        intor='int3c2e_ipip1')
                # Make symmetric, (double the out-of diagonal), and compress
                int1[[1,2,5]] += int1[[3,6,7]]
                int1[[1,2,5]] /= 2.0
                int1 = int1[pyscf2ommp_idx]

                int2 = df.incore.aux_e2(self.mol,
                                        self.fakemol_pol,
                                        intor='int3c2e_ipvip1')
                # Make symmetric, (double the out-of diagonal), and compress
                int2[[1,2,5]] += int2[[3,6,7]]
                int2[[1,2,5]] /= 2.0
                int2 = int2[pyscf2ommp_idx]

                self._gef_int_at_cpol = int1 + numpy.einsum('inmj->imnj', int1) + 2 * int2

            return self._gef_int_at_cpol

        @property
        def ef_nucl_at_pol(self):
            # nuclear component of EF should only be computed once
            if not hasattr(self, '_nuclear_ef'):
                if self.fakemol_pol is None:
                    self._nuclear_ef_pol = numpy.zeros([0,3])
                else:
                    c = self.ommp_obj.cpol
                    qmat_q = self.mol.atom_charges()
                    qmat_c = self.mol.atom_coords()
                    self._nuclear_ef_pol = ommp.charges_elec_prop(qmat_c,
                                                                  qmat_q,
                                                                  c,
                                                                  False,
                                                                  True,
                                                                  False,
                                                                  False)['E']

            return self._nuclear_ef_pol
        @property
        def gef_nucl_at_pol(self):
            # nuclear component of EF should only be computed once
            if self.fakemol_pol is None:
                self._nuclear_gef_pol = numpy.zeros([0,6])
            else:
                c = self.ommp_obj.cpol
                qmat_q = self.mol.atom_charges()
                qmat_c = self.mol.atom_coords()

                self._nuclear_gef_pol = ommp.charges_elec_prop(qmat_c,
                                                               qmat_q,
                                                               c,
                                                               False,
                                                               False,
                                                               True,
                                                               False)['Egrad']
                self._nuclear_gef_pol *= -1 #Those are positive charges!
            return self._nuclear_gef_pol

        @property
        def gef_nucl_at_fixed(self):
            # nuclear component of EF should only be computed once
            c = self.ommp_obj.cmm
            qmat_q = self.mol.atom_charges()
            qmat_c = self.mol.atom_coords()

            self._nuclear_gef_fixed = ommp.charges_elec_prop(qmat_c,
                                                             qmat_q,
                                                             c,
                                                             False,
                                                             False,
                                                             True,
                                                             False)['Egrad']
            self._nuclear_gef_fixed *= -1 #Those are positive charges!
            return self._nuclear_gef_fixed

        def ef_at_pol_sites(self, dm, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at polarizable sites"""

            ef = numpy.einsum('inmj,nm->ji',
                              self.ef_integrals_at_pol,
                              dm, dtype="f8")

            if not exclude_nuclei:
                ef += self.ef_nucl_at_pol
            return ef

        @property
        def ef_nucl_at_fixed(self):
            # nuclear component of EF should only be computed once
            c = self.ommp_obj.cmm
            qmat_q = self.mol.atom_charges()
            qmat_c = self.mol.atom_coords()

            self._nuclear_ef_fixed = ommp.charges_elec_prop(qmat_c,
                                                            qmat_q,
                                                            c,
                                                            False,
                                                            True,
                                                            False,
                                                            False)['E']

            return self._nuclear_ef_fixed

        def ef_at_fixed_sites(self, dm, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at fixed electrostatic sites"""

            ef = numpy.einsum('inmj,nm->ji',
                              self.ef_integrals_at_static,
                              dm, dtype="f8")
            if not exclude_nuclei:
                ef += self.ef_nucl_at_fixed
            return ef

        def gef_at_pol_sites(self, dm, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at fixed electrostatic sites"""

            gef = numpy.einsum('inmj,nm->ji',
                              self.gef_integrals_at_pol,
                              dm, dtype="f8")
            if not exclude_nuclei:
                gef += self.gef_nucl_at_pol
            return gef

        def gef_at_fixed_sites(self, dm, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at fixed electrostatic sites"""

            gef = numpy.einsum('inmj,nm->ji',
                              self.gef_integrals_at_fixed,
                              dm, dtype="f8")
            if not exclude_nuclei:
                gef += self.gef_nucl_at_fixed
            return gef

        @property
        def Hef_integrals_at_fixed(self):
            if not hasattr(self, '_Hef_int_at_cmm'):
                nnni_j = df.incore.aux_e2(self.mol,
                                          self.fakemol_static,
                                          intor='int3c2e_ipipip1')
                nni_nj = df.incore.aux_e2(self.mol,
                                          self.fakemol_static,
                                          intor='int3c2e_ipipvip1')
                self._Hef_int_at_cmm = nnni_j + numpy.einsum('inmj->imnj', nnni_j) + \
                                       3 * (nni_nj + numpy.einsum('inmj->imnj', nni_nj))

            return self._Hef_int_at_cmm

        @property
        def Hef_nucl_at_fixed(self):
            # nuclear component of EF should only be computed once
            if not hasattr(self, '_nuclear_Hef_fixed'):
                c = self.ommp_obj.cmm
                qmat_q = self.mol.atom_charges()
                qmat_c = self.mol.atom_coords()
                self._nuclear_Hef_fixed = ommp.charges_elec_prop(qmat_c,
                                                              qmat_q,
                                                              c,
                                                              False,
                                                              False,
                                                              False,
                                                              True)['EHess']

            return self._nuclear_Hef_fixed

        def Hef_at_fixed_sites(self, dm, exclude_nuclei=False):
            # 0   1   2   3   4   5   6   7   8   9  10  11  12  13
            #xxx xxy xxz xyx xyy xyz xzx xzy xzz yxx yxy yxz yyx yyy
            #14  15  16  17  18  19  20  21  22  23  24  25  26
            #yyz yzx yzy yzz zxx zxy zxz zyx zyy zyz zzx zzy zzz
            # OpenMMPol order:
            # 0   1   2   3   4   5   6   7   8   9
            # xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz
            # 0   1   2   4   5   8   13  14  17  26
            #     3   6  10   7  20       16  23
            #     9  18  12  11  24       22  25
            #                15
            #                19
            #                21
            Hef = numpy.einsum('inmj,nm->ij',
                              self.Hef_integrals_at_fixed,
                              dm, dtype="f8")
            Hef[1] = (Hef[1] + Hef[3] + Hef[9]) / 3
            Hef[2] = (Hef[2] + Hef[6] + Hef[18]) / 3
            Hef[4] = (Hef[4] + Hef[10] + Hef[12]) / 3
            Hef[5] = (Hef[5] + Hef[7] + Hef[11] + Hef[15] + Hef[19] + Hef[21]) / 6
            Hef[8] = (Hef[8] + Hef[20] + Hef[24]) / 3
            Hef[14] = (Hef[14] + Hef[16] + Hef[22]) / 3
            Hef[17] = (Hef[17] + Hef[23] + Hef[25]) / 3
            Hef = numpy.einsum('ij->ji', Hef[[0,1,2,4,5,8,13,14,17,26]])
            if not exclude_nuclei:
                Hef += self.Hef_nucl_at_fixed
            return Hef

        def get_veff(self, mol=None, dm=None, *args, **kwargs):
            vhf = method_class.get_veff(self, mol, dm, *args, **kwargs)

            if self.fakemol_pol is not None:
                # NOTE: v_solvent should not be added to vhf in this place. This is
                # because vhf is used as the reference for direct_scf in the next
                # iteration. If v_solvent is added here, it may break direct SCF.

                # 1. compute the EF generated by the current DM
                current_ef = self.ef_at_pol_sites(dm)

                # 2. update the induced dipoles
                self.ommp_obj.set_external_field(current_ef)

                # 3. get the induced dipoles
                if not self.ommp_obj.is_amoeba:
                    current_ipds = self.get_mmpol_induced_dipoles()
                else:
                    current_ipds_d, current_ipds_p = self.get_mmpol_induced_dipoles()
                    current_ipds = (current_ipds_d+current_ipds_p) / 2

                # 4. compute the induced dipoles term in the Hamiltonian
                v_mmpol = -numpy.einsum('inmj,ji->nm',
                                        self.ef_integrals_at_pol, current_ipds)

                # 5. Compute the MMPol contribution to energy
                if not self.ommp_obj.is_amoeba:
                    e_mmpol = -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds)
                else:
                    e_mmpol = -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds_d)
                e_mmpol += self.ommp_obj.get_polelec_energy()

            else:
                # If there are no polarizabilities, there are no contribution to the Fock Matrix
                v_mmpol = numpy.zeros(dm.shape)
                e_mmpol = 0.0

            return lib.tag_array(vhf, e_mmpol=e_mmpol, v_mmpol=v_mmpol)

        def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                     diis=None, diis_start_cycle=None,
                     level_shift_factor=None, damp_factor=None):

            if getattr(vhf, 'v_mmpol', None) is None:
                vhf = self.get_veff(self.mol, dm)

            return method_class.get_fock(self, h1e, s1e, vhf+vhf.v_mmpol, dm, cycle, diis,
                                         diis_start_cycle, level_shift_factor, damp_factor)


        def get_hcore(self, mol=None):
            if mol is None:
                mol = self.mol

            if getattr(method_class, 'get_hcore', None):
                h1e = method_class.get_hcore(self, mol)
            else:
                # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise RuntimeError('mm_charge function cannot be applied on post-HF methods')

            q = self.ommp_obj.static_charges
            self.h1e_mmpol = - numpy.einsum('nmi,i->nm', self.v_integrals_at_static, q)

            if self.ommp_obj.is_amoeba:
                mu = self.ommp_obj.static_dipoles
                self.h1e_mmpol += - numpy.einsum('inmj,ji->nm', self.ef_integrals_at_static, mu)
                quad = self.ommp_obj.static_quadrupoles
                self.h1e_mmpol += -numpy.einsum('inmj,ji->nm', self.gef_integrals_at_static, quad)
            return h1e + self.h1e_mmpol

        def energy_nuc(self):
            nuc = self.mol.energy_nuc()

            # interactions between QM nuclei and MM particles
            vmm = self.ommp_obj.mm_potential_at_external(self.mol.atom_coords())
            self.nuc_static_mm = numpy.dot(vmm, self.mol.atom_charges())
            nuc += self.nuc_static_mm

            return nuc

        def energy_elec(self, dm=None, h1e=None, vhf=None):
            # TODO check
            ene_el = method_class.energy_elec(self, dm, h1e, vhf)

            if getattr(vhf, 'e_mmpol', None):
                e_mmpol_2e = vhf.e_mmpol - self.ommp_obj.get_polelec_energy()
            else:
                e_mmpol_2e = self.get_veff(dm=dm).e_mmpol - self.ommp_obj.get_polelec_energy()

            ene_el = (ene_el[0]+e_mmpol_2e,
                      ene_el[1]+e_mmpol_2e)

            return ene_el

        def energy_tot(self, dm=None, h1e=None, vhf=None):
            e_tot = method_class.energy_tot(self, dm, h1e, vhf)
            e_tot += self.ommp_obj.get_fixedelec_energy()
            e_tot += self.ommp_obj.get_polelec_energy()
            return e_tot

        def gen_response(self, *args, **kwargs):
            vind = method_class.gen_response(self, *args, **kwargs)
            is_uhf = isinstance(self, scf.uhf.UHF)

            singlet = kwargs.get('singlet', True)
            singlet = singlet or singlet is None

            def vind_mmpol(dm1):
                v = vind(dm1)

                if is_uhf:
                    raise NotImplementedError("UHF is currently unsupported")
                elif singlet:
                    v_mmpol = []

                    for d in dm1:
                        current_ef = self.ef_at_pol_sites(d, exclude_nuclei=True)
                        self.ommp_obj.set_external_field(current_ef, nomm=False)

                        if not self.ommp_obj.is_amoeba:
                            current_ipds = self.get_mmpol_induced_dipoles()
                        else:
                            current_ipds, empty = \
                                    self.get_mmpol_induced_dipoles()

                        vd = -numpy.einsum('inmj,ji->nm',
                                           self.ef_integrals_at_pol,
                                           current_ipds)
                        v_mmpol += [vd]

                    v_mmpol = numpy.array(v_mmpol)

                    v += v_mmpol

                return v

            return vind_mmpol

        def nuc_grad_method(self):
            scf_grad = method_class.nuc_grad_method(self)
            return qmmmpol_grad_for_scf(scf_grad)

        Gradients = nuc_grad_method

    if isinstance(scf_method, scf.hf.SCF):
        return QMMMPOL(scf_method, ommp_obj)

def qmmmpol_grad_for_scf(scf_grad):
    # Why? TODO
    if getattr(scf_grad.base, 'with_x2c', None):
        raise NotImplementedError('X2C with QM/MM charges')

    if isinstance(scf_grad, _QMMMPOLGrad):
        return scf_grad

    assert(isinstance(scf_grad.base, scf.hf.SCF) and
           isinstance(scf_grad.base, _QMMMPOL))

    grad_class = scf_grad.__class__
    class QMMMPOLG(_QMMMPOLGrad, grad_class):
        def __init__(self, scf_grad):
            self.__dict__.update(scf_grad.__dict__)

        def dump_flags(self, verbose=None):
            grad_class.dump_flags(self, verbose)
            logger.info(self, 'MMPol system with {:d} sites ({:d} polarizable)'.format(self.base.ommp_obj.mm_atoms, self.base.ommp_obj.pol_atoms))
            return self

        def MM_atoms_grad(self):
            dm = self.base.make_rdm1()

            ef_QMatMM = self.base.ef_at_fixed_sites(dm)
            force = -numpy.einsum('ij,i->ij', ef_QMatMM, self.base.ommp_obj.static_charges)
            if self.base.ommp_obj.is_amoeba:
                mu = self.base.ommp_obj.static_dipoles
                gef_QMatMM = self.base.gef_at_fixed_sites(dm)
                force += -numpy.einsum('ij,i->ij', gef_QMatMM[:,[0,1,3]], mu[:,0])
                force += -numpy.einsum('ij,i->ij', gef_QMatMM[:,[1,2,4]], mu[:,1])
                force += -numpy.einsum('ij,i->ij', gef_QMatMM[:,[3,4,5]], mu[:,2])

                quad = self.base.ommp_obj.static_quadrupoles
                Hef_QMatMM = self.base.Hef_at_fixed_sites(dm)
                force += -numpy.einsum('ij,i->ij', Hef_QMatMM[:,[0,1,2]], quad[:,0]) #xx
                force += -2.0*numpy.einsum('ij,i->ij', Hef_QMatMM[:,[1,3,4]], quad[:,1]) #xy
                force += -numpy.einsum('ij,i->ij', Hef_QMatMM[:,[3,6,7]], quad[:,2]) #yy
                force += -2.0*numpy.einsum('ij,i->ij', Hef_QMatMM[:,[2,4,5]], quad[:,3]) #xz
                force += -2.0*numpy.einsum('ij,i->ij', Hef_QMatMM[:,[1,7,8]], quad[:,4]) #yz
                force += -numpy.einsum('ij,i->ij', Hef_QMatMM[:,[5,8,9]], quad[:,5]) #zz

            if self.base.do_pol:
                gef_QMatPOL = self.base.gef_at_pol_sites(dm)
                if not self.base.ommp_obj.is_amoeba:
                    mu = self.base.get_mmpol_induced_dipoles()
                else:
                    mu_d, mu_p = self.base.get_mmpol_induced_dipoles()
                    mu = 0.5 * (mu_d + mu_p)

                force += -numpy.einsum('ij,i->ij', gef_QMatPOL[:,[0,1,3]], mu[:,0])
                force += -numpy.einsum('ij,i->ij', gef_QMatPOL[:,[1,2,4]], mu[:,1])
                force += -numpy.einsum('ij,i->ij', gef_QMatPOL[:,[3,4,5]], mu[:,2])

            force += self.base.ommp_obj.do_rotation_grad(ef_QMatMM, -gef_QMatMM)

            force += self.base.ommp_obj.do_polelec_grad()
            force += self.base.ommp_obj.do_fixedelec_grad()

            return force

        def get_hcore(self, mol=None):
            if mol is None:
                mol = self.mol

            if getattr(grad_class, 'get_hcore', None):
                g_qm = grad_class.get_hcore(self, mol)
            else:
                # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise NotImplementedError("For some reason QM method does not have get_hcore func.")

            q = self.base.ommp_obj.static_charges
            #TODO Why the symmetrized version is not ok? QWhy????
            if mol.cart:
                intor = 'int3c2e_ip1_cart'
            else:
                intor = 'int3c2e_ip1_sph'
            j3c = df.incore.aux_e2(mol, self.base.fakemol_static, intor, aosym='s1',
                                   comp=3)
            g_mm = numpy.einsum('ipqk,k->ipq', j3c, q)
            if self.base.ommp_obj.is_amoeba:
                ints = df.incore.aux_e2(self.base.mol,
                                        self.base.fakemol_static,
                                        intor='int3c2e_ipip1')

                ints += df.incore.aux_e2(self.base.mol,
                                        self.base.fakemol_static,
                                        intor='int3c2e_ipvip1')

                mu = self.base.ommp_obj.static_dipoles

                g_mm[0] += numpy.einsum('ipqk,ki->pq', ints[0:3], mu)
                g_mm[1] += numpy.einsum('ipqk,ki->pq', ints[3:6], mu)
                g_mm[2] += numpy.einsum('ipqk,ki->pq', ints[6:9], mu)

                A = df.incore.aux_e2(self.base.mol,
                                        self.base.fakemol_static,
                                        intor='int3c2e_ipipip1')
                B =   df.incore.aux_e2(self.base.mol,
                                       self.base.fakemol_static,
                                       intor='int3c2e_ipipvip1')
                Bt = numpy.einsum('ipqk->iqpk',
                                  numpy.concatenate((B[::3],
                                                     B[1::3],
                                                     B[2::3]), axis=0))
                ints = A + 2*B + Bt

                quad = self.base.ommp_obj.static_quadrupoles

                g_mm[0] += numpy.einsum('ipqk,ki->pq', ints[0:9],   quad[:,[0,1,3,1,2,4,3,4,5]])
                g_mm[1] += numpy.einsum('ipqk,ki->pq', ints[9:18],  quad[:,[0,1,3,1,2,4,3,4,5]])
                g_mm[2] += numpy.einsum('ipqk,ki->pq', ints[18:27], quad[:,[0,1,3,1,2,4,3,4,5]])

            if self.base.do_pol:
                ints = df.incore.aux_e2(self.base.mol,
                                        self.base.fakemol_pol,
                                        intor='int3c2e_ipip1')

                ints += df.incore.aux_e2(self.base.mol,
                                        self.base.fakemol_pol,
                                        intor='int3c2e_ipvip1')

                if not self.base.ommp_obj.is_amoeba:
                    mu = self.base.get_mmpol_induced_dipoles()
                else:
                    mu_d, mu_p = self.base.get_mmpol_induced_dipoles()
                    mu = 0.5 * (mu_p + mu_d)

                g_pol = numpy.zeros(g_mm.shape)
                g_pol[0] = numpy.einsum('ipqk,ki->pq', ints[0:3], mu)
                g_pol[1] = numpy.einsum('ipqk,ki->pq', ints[3:6], mu)
                g_pol[2] = numpy.einsum('ipqk,ki->pq', ints[6:9], mu)
                return  g_qm + g_mm + g_pol
            else:
                return g_qm + g_mm

        def grad_nuc(self, mol=None, atmlst=None):
            if mol is None:
                mol = self.mol

            if getattr(grad_class, 'grad_nuc', None):
                g_qm = grad_class.grad_nuc(self, mol, atmlst)
            else:
                # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise NotImplementedError("For some reason QM method does not have grad_nuc func.")

            g_mm = -numpy.einsum('i,ij->ij',
                                 self.mol.atom_charges(),
                                 scf_grad.base.ommp_obj.mmpol_field_at_external(self.mol.atom_coords()))
            if atmlst is not None:
                g_mm = g_mm[atmlst]

            return g_qm + g_mm

    return QMMMPOLG(scf_grad)

class _QMMMPOLGrad:
    pass

class _QMMMPOL:
    pass
