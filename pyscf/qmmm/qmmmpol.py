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

        def get_mmpol_static_charges(self):
            q = ommp.get_q()[:,0]
            c = ommp.get_cmm()
            return q, c

        def get_mmpol_static_dipoles(self):
            mu = ommp.get_q()[:,1:4]
            c = ommp.get_cmm()
            return mu, c

        def get_mmpol_static_quadrupoles(self):
            quad = ommp.get_q()[:,4:10]
            c = ommp.get_cmm()
            return quad, c

        def get_mmpol_induced_dipoles(self):
            if not ommp.ff_is_amoeba():
                mu = ommp.get_ipd()[0,:,:]
                return mu
            else:
                raise NotImplementedError("AMOEBA is still not supported")

        @property
        def v_integrals_at_static(self):
            if not hasattr(self, '_v_int_at_cmm'):
                fakemol = gto.fakemol_for_charges(ommp.get_cmm())
                self._v_int_at_cmm = df.incore.aux_e2(self.mol,
                                                       fakemol,
                                                       intor='int3c2e')
            return self._v_int_at_cmm

        @property
        def ef_integrals_at_static(self):
            if not hasattr(self, '_ef_int_at_cmm'):
                fakemol = gto.fakemol_for_charges(ommp.get_cmm())
                self._ef_int_at_cmm = df.incore.aux_e2(self.mol,
                                                       fakemol,
                                                       intor='int3c2e_ip1')
                self._ef_int_at_mm += numpy.einsum('imnj->inmj', self.ef_int_at_cmm)
            return self._ef_int_at_cmm

        @property
        def ef_integrals_at_pol(self):
            if not hasattr(self, '_ef_int_at_cpol'):
                fakemol = gto.fakemol_for_charges(ommp.get_cpol())
                self._ef_int_at_cpol = df.incore.aux_e2(self.mol,
                                                       fakemol,
                                                       intor='int3c2e_ip1')
                self._ef_int_at_cpol += numpy.einsum('imnj->inmj', self._ef_int_at_cpol)
            return self._ef_int_at_cpol

        @property
        def nuclear_ef(self):
            # nuclear component of EF should only be computed once
            if not hasattr(self, '_nuclear_ef'):
                c = ommp.get_cpol()
                qmat_q = self.mol.atom_charges()
                qmat_c = self.mol.atom_coords()
                self._nuclear_ef = numpy.zeros(c.shape, dtype="f8")
                for qc, qq in zip(qmat_c, qmat_q):
                    # Compute eletric field from QM nuclei
                    dr = c - qc
                    self._nuclear_ef += (qq * dr.T / (numpy.linalg.norm(dr,axis=1) ** 3)).T
            return self._nuclear_ef

        def ef_at_pol_sites(self, dm):
            ef = numpy.einsum('inmj,nm->ji',
                              self.ef_integrals_at_pol,
                              dm, dtype="f8")
            ef += self.nuclear_ef

            return ef

        def get_veff(self, mol=None, dm=None, *args, **kwargs):
            vhf = method_class.get_veff(self, mol, dm, *args, **kwargs)

            # 1. compute the EF generated by the current DM
            current_ef = self.ef_at_pol_sites(dm)

            # 2. update the induced dipoles
            ommp.set_external_field(current_ef, 'inversion')
            # 3. get the induced dipoles
            current_ipds = self.get_mmpol_induced_dipoles()
            # 4. compute the induced dipoles term in the Hamiltonian
            v_mmpol = -numpy.einsum('inmj,ji->nm',
                                     self.ef_integrals_at_pol, current_ipds)
            e_mmpol = -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds)
            e_mmpol += ommp.get_polelec_energy()

            # NOTE: v_solvent should not be added to vhf in this place. This is
            # because vhf is used as the reference for direct_scf in the next
            # iteration. If v_solvent is added here, it may break direct SCF.

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
                # TODO ?
                raise RuntimeError('mm_charge function cannot be applied on post-HF methods')

            charges, coords = self.get_mmpol_static_charges()

            if pyscf.DEBUG:
                v = 0
                for i, q in enumerate(charges):
                    mol.set_rinv_origin(coords[i])
                    v += mol.intor('int1e_rinv') * -q
            else:
                if mol.cart:
                    intor = 'int3c2e_cart'
                else:
                    intor = 'int3c2e_sph'
                nao = mol.nao
                max_memory = self.max_memory - lib.current_memory()[0]
                blksize = int(min(max_memory*1e6/8/nao**2, 200))
                if max_memory <= 0:
                    blksize = 1
                    logger.warn(self, 'Memory estimate for reading point charges is negative. '
                                'Trying to read point charges one by one.')
                cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                                     mol._env, intor)
                v = 0
                for i0, i1 in lib.prange(0, charges.size, blksize):
                    fakemol = gto.fakemol_for_charges(coords[i0:i1])
                    j3c = df.incore.aux_e2(mol, fakemol, intor=intor,
                                           aosym='s2ij', cintopt=cintopt)
                    v += numpy.einsum('xk,k->x', j3c, -charges[i0:i1])
                v = lib.unpack_tril(v)

                if ommp.ff_is_amoeba():
                    raise NotImplementedError("AMOEBA is still not supported")
                    mu, c = self.get_mmpol_static_charges()
                    quad, c = self.get_mmpol_static_quadrupoles()

            return h1e + v

        def energy_nuc(self):
            # interactions between QM nuclei and MM particles
            nuc = self.mol.energy_nuc()
            charges, coords = self.get_mmpol_static_charges()
            for j in range(self.mol.natm):
                q2, r2 = self.mol.atom_charge(j), self.mol.atom_coord(j)
                r = lib.norm(r2-coords, axis=1)
                nuc += q2*(charges/r).sum()
            return nuc

        def nuc_grad_method(self):
            scf_grad = method_class.nuc_grad_method(self)
            return qmmm_grad_for_scf(scf_grad)
        Gradients = nuc_grad_method

    if isinstance(scf_method, scf.hf.SCF):
        return QMMMPOL(scf_method)

