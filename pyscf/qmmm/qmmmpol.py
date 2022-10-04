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


def add_mmpol(scf_method):
    if ommp.is_init():
        return qmmmpol_for_scf(scf_method)
    else:
        raise RuntimeError("Initialize OMMP library before adding "
                           "mmpol environment to a method.")

def qmmmpol_for_scf(scf_method):
    assert(isinstance(scf_method, (scf.hf.SCF, mcscf.casci.CASCI)))

    if isinstance(scf_method, scf.hf.SCF):
        # Avoid to initialize _QMMM twice
        if isinstance(scf_method, _QMMM):
            # TODO Insert a log message here
            return scf_method
        method_class = scf_method.__class__
    else:
        raise NotImplementedError("Only SCF-method are currently "
                                  "supported in QM/MMPol")

    class QMMMPOL(_QMMM, method_class):
        def __init__(self, scf_method):
            self.__dict__.update(scf_method.__dict__)

        @property
        def do_pol(self):
            return ommp.get_pol_atoms() > 0

        def get_mmpol_static_charges(self):
            q = ommp.get_q()[:,0]
            return q

        def get_mmpol_static_dipoles(self):
            mu = ommp.get_q()[:,1:4]
            return mu

        def get_mmpol_static_quadrupoles(self):
            quad = ommp.get_q()[:,4:10]
            return quad

        def get_mmpol_induced_dipoles(self):
            if not ommp.ff_is_amoeba():
                mu = ommp.get_ipd()[0,:,:]
                return mu
            else:
                mu_d = ommp.get_ipd()[0,:,:]
                mu_p = ommp.get_ipd()[1,:,:]
                return mu_d, mu_p

        @property
        def fakemol_static(self):
            if not hasattr(self, '_fakemol_static'):
                self._fakemol_static = gto.fakemol_for_charges(ommp.get_cmm())
            return self._fakemol_static

        @property
        def fakemol_pol(self):
            if self.do_pol:
                if not hasattr(self, '_fakemol_pol'):
                    self._fakemol_pol = gto.fakemol_for_charges(ommp.get_cpol())
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
        def ef_nucl_at_pol(self):
            # nuclear component of EF should only be computed once
            if not hasattr(self, '_nuclear_ef'):
                if self.fakemol_pol is None:
                    self._nuclear_ef = numpy.zeros([0,3])
                else:
                    c = ommp.get_cpol() #TODO
                    qmat_q = self.mol.atom_charges()
                    qmat_c = self.mol.atom_coords()
                    self._nuclear_ef = numpy.zeros(c.shape, dtype="f8")
                    for qc, qq in zip(qmat_c, qmat_q):
                        # Compute eletric field from QM nuclei
                        dr = c - qc
                        self._nuclear_ef += (qq * dr.T / (numpy.linalg.norm(dr,axis=1) ** 3)).T

            return self._nuclear_ef

        def ef_at_pol_sites(self, dm):
            """Compute the electric field generated by the QM system with density dm
            at polarizable sites"""

            ef = numpy.einsum('inmj,nm->ji',
                              self.ef_integrals_at_pol,
                              dm, dtype="f8")
            ef += self.ef_nucl_at_pol

            return ef

        def get_veff(self, mol=None, dm=None, *args, **kwargs):
            vhf = method_class.get_veff(self, mol, dm, *args, **kwargs)

            if self.fakemol_pol is not None:
                # NOTE: v_solvent should not be added to vhf in this place. This is
                # because vhf is used as the reference for direct_scf in the next
                # iteration. If v_solvent is added here, it may break direct SCF.

                # 1. compute the EF generated by the current DM
                current_ef = self.ef_at_pol_sites(dm)
                #print("EF")
                #print(current_ef)

                # 2. update the induced dipoles
                ommp.set_external_field(current_ef, 'inversion')
                # 3. get the induced dipoles
                if not ommp.ff_is_amoeba():
                    current_ipds = self.get_mmpol_induced_dipoles()
                else:
                    current_ipds_d, current_ipds_p = self.get_mmpol_induced_dipoles()
                    current_ipds = (current_ipds_d+current_ipds_p) / 2

                #print("IPDS")
                #print(current_ipds)
                # 4. compute the induced dipoles term in the Hamiltonian
                v_mmpol = -numpy.einsum('inmj,ji->nm',
                                        self.ef_integrals_at_pol, current_ipds)

                # 5. Compute the MMPol contribution to energy
                if not ommp.ff_is_amoeba():
                    e_mmpol = -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds)
                else:
                    e_mmpol = -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds_d)

                e_mmpol += ommp.get_polelec_energy()
                #e_mmpol += numpy.einsum('nm,nm', self.h1e_mmpol, dm)
                #if ommp.ff_is_amoeba():
                #    print("QM-IND", -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds_d) * 627.5096080306)
                #else:
                #    print("QM-IND", -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds) * 627.5096080306)
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

            q = self.get_mmpol_static_charges()
            self.h1e_mmpol = - numpy.einsum('nmi,i->nm', self.v_integrals_at_static, q)

            if ommp.ff_is_amoeba():
                mu = self.get_mmpol_static_dipoles()
                self.h1e_mmpol += - numpy.einsum('inmj,ji->nm', self.ef_integrals_at_static, mu)
                quad = self.get_mmpol_static_quadrupoles()
                self.h1e_mmpol += -numpy.einsum('inmj,ji->nm', self.gef_integrals_at_static, quad)
            return h1e + self.h1e_mmpol

        def energy_nuc(self):
            nuc = self.mol.energy_nuc()

            # interactions between QM nuclei and MM particles
            vmm = numpy.zeros(self.mol.natm)
            ommp.potential_mm2ext(self.mol.atom_coords(),
                                  vmm)
            self.nuc_static_mm = numpy.dot(vmm, self.mol.atom_charges())
            nuc += self.nuc_static_mm

            return nuc

        def nuc_grad_method(self):
            scf_grad = method_class.nuc_grad_method(self)
            return qmmm_grad_for_scf(scf_grad)
        Gradients = nuc_grad_method

    if isinstance(scf_method, scf.hf.SCF):
        return QMMMPOL(scf_method)

