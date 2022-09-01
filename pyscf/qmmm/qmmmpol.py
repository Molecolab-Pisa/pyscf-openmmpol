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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
QM/MM helper functions that modify the QM methods.
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
    return qmmmpol_for_scf(scf_method)

def qmmmpol_for_scf(scf_method):
    assert(isinstance(scf_method, (scf.hf.SCF, mcscf.casci.CASCI)))

    if isinstance(scf_method, scf.hf.SCF):
        # Avoid to initialize _QMMM twice
        if isinstance(scf_method, _QMMM):
            # TODO Insert a log message here
            return scf_method
        method_class = scf_method.__class__

    else:
        if isinstance(scf_method._scf, _QMMM):
            # TODO Insert a log message here
            return scf_method
        method_class = scf_method._scf.__class__

    class QMMMPOL(_QMMM, method_class):
        def __init__(self, scf_method):
            self.__dict__.update(scf_method.__dict__)

        def get_charges(self):
            q = ommp.get_q()[:,0]
            c = ommp.get_cmm()
            return q, c

        def dump_flags(self, verbose=None):
            method_class.dump_flags(self, verbose)
            logger.info(self, '** Add background charges for %s **',
                        method_class)
            if self.verbose >= logger.DEBUG:
                logger.debug(self, 'Charge      Location')
                charges, coords = self.get_charges()
                for i, z in enumerate(charges):
                    logger.debug(self, '%.9g    %s', z, coords[i])
            return self

        def get_hcore(self, mol=None):
            if mol is None: mol = self.mol
            if getattr(method_class, 'get_hcore', None):
                h1e = method_class.get_hcore(self, mol)
            else:  # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise RuntimeError('mm_charge function cannot be applied on post-HF methods')

            charges, coords = self.get_charges()
            if pyscf.DEBUG:
                v = 0
                for i,q in enumerate(charges):
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
            return h1e + v

        def energy_nuc(self):
            # interactions between QM nuclei and MM particles
            nuc = self.mol.energy_nuc()
            charges, coords = self.get_charges()
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
    else:  # post-HF methods
        # TODO
        #scf_method._scf = QMMM(scf_method._scf, mm_mol).run()
        #scf_method.mo_coeff = scf_method._scf.mo_coeff
        #scf_method.mo_energy = scf_method._scf.mo_energy
        return scf_method

# Inject QMMM interface wrapper to other modules
# scf.hf.SCF.QMMMPOL =
#mcscf.casci.CASCI.QMMM = mm_charge
#grad.rhf.Gradients.QMMM = mm_charge_grad
