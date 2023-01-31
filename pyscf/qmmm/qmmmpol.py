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
        raise RuntimeError("""Use an instance of OMMPSystem to
                           initalize the environment.""")

def qmmmpol_for_scf(scf_method, ommp_obj):
    assert(isinstance(ommp_obj, ommp.OMMPSystem))

    if isinstance(scf_method, scf.hf.SCF):
        # Avoid to initialize _QMMM twice
        if isinstance(scf_method, _QMMM):
            logger.info(scf_method, "The SCF method passed to qmmmpol_for_scf "
                                    "already contains an environment. Doing nothing.")
            return scf_method
        method_class = scf_method.__class__
    else:
        raise NotImplementedError("Only SCF-method are currently "
                                  "supported in QM/MMPol")

    class QMMMPOL(_QMMMPOL, method_class):
        """Class overload over an SCF method, to add the contribution
        from QMMMPol environment through OMMP"""

        def __init__(self, scf_method, ommp_obj):
            self.__dict__.update(scf_method.__dict__)
            self.ommp_obj = ommp_obj
            self._keys.update(['ommp_obj'])

        @property
        def do_pol(self):
            """Flag to check if the MM system has non-zero polarizability
            and enable/disable certain part of the calculation."""
            return self.ommp_obj.pol_atoms > 0

        def get_mmpol_induced_dipoles(self):
            """Return the last induced dipoles computed by OMMP removing
            the index on the dipole set. For non-amoeba FF, a single
            set is returned; for AMOEBA D and P dipoles are returned as
            a touple of numpy array."""

            if not self.ommp_obj.is_amoeba:
                mu = self.ommp_obj.ipd[0,:,:]
                return mu
            else:
                mu_d = self.ommp_obj.ipd[0,:,:]
                mu_p = self.ommp_obj.ipd[1,:,:]
                return mu_d, mu_p

        @property
        def fakemol_static(self):
            """Fakemol for computing 3-centers-2-electron integrals
            at polarizable sites coordinates"""
            if not hasattr(self, '_fakemol_static'):
                self._fakemol_static = gto.fakemol_for_charges(self.ommp_obj.cmm)
            return self._fakemol_static

        @property
        def fakemol_pol(self):
            """Fakemol for computing 3-centers-2-electron integrals
            at static sites coordinates"""
            if self.do_pol:
                if not hasattr(self, '_fakemol_pol'):
                    self._fakemol_pol = gto.fakemol_for_charges(self.ommp_obj.cpol)
            else:
                self._fakemol_pol = None

            return self._fakemol_pol

        @property
        def ommp_qm_helper(self):
            if not hasattr(self, '_qmhelper'):

                qmat_q = self.mol.atom_charges()
                qmat_c = self.mol.atom_coords()
                self._qmhelper = ommp.OMMPQmHelper(qmat_c, qmat_q)

            return self._qmhelper

        def v_integrals_ommp(self, pol=False):
            """Electrostatic potential integrals <\mu|r^{-1}|\\nu> = (\mu,\\nu|\delta)
            at coordinates of MM atoms.
            For a reference on how 1-electron integrals can be computed as
            3-center-2-electron electron integrals see Chem. Phys. Lett vol. 206,
            pp. 239-246."""

            if pol:
                fm = self.fakemol_pol
            else:
                fm = self.fakemol_static

            return df.incore.aux_e2(self.mol, fm,
                                    intor='int3c2e')

        def ef_integrals_ommp(self, pol=False):
            """Electric field integrals
            <mu|\hat(E)|nu> = <\mu|\\nabla r^{-1}|\\nu> =
                            = (\\nabla\mu\\nu|\delta) + (\mu\\nabla\\nu|\delta)
                            = (\\nabla\mu\\nu|\delta) + (\\nabla\mu\\nu|\delta)^\dagger
            at coordinates of MM atoms."""

            if pol:
                fm = self.fakemol_pol
            else:
                fm = self.fakemol_static

            Ef = df.incore.aux_e2(self.mol, fm,
                                  intor='int3c2e_ip1')
            Ef += numpy.einsum('imnj->inmj', Ef)
            return Ef

        def gef_integrals_ommp(self, pol=False):
            """Electric field gradients integrals
            <mu|\hat(G)|nu> = <\mu|\\nabla\\nabla r^{-1}|\\nu> =
                            = ... =
                            = (\\nabla\\nabla\mu\\nu|\delta) +
                              (\\nabla\\nabla\mu\\nu|\delta)^\dagger +
                              2 (\\nabla\mu\\nabla\\nu|\delta)
            at coordinates of MM atoms. Those integrals have formally
            9 different components, but we use it symmetrized and
            compressed."""

            if pol:
                fm = self.fakemol_pol
            else:
                fm = self.fakemol_static

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

            nni_j = df.incore.aux_e2(self.mol, fm,
                                    intor='int3c2e_ipip1')

            ni_nj = df.incore.aux_e2(self.mol, fm,
                                    intor='int3c2e_ipvip1')

            Gef = nni_j + numpy.einsum('inmj->imnj', nni_j) + 2 * ni_nj

            Gef[[1,2,5]] += Gef[[3,6,7]]
            Gef[[1,2,5]] /= 2.0

            return Gef[[0,1,4,2,5,8]]


        def Hef_integrals_ommp(self, pol=False):
            """Electric field Hessian integrals
            <mu|\hat(G)|nu> = <\mu|\\nabla \\nabla\\nabla r^{-1}|\\nu> =
                            = ... =
                            = (\\nabla\\nabla\\nabla\mu\\nu|\delta) +
                              (\\nabla\\nabla\\nabla\mu\\nu|\delta)^\dagger +
                              3 (\\nabla\\nabla\mu\\nabla\\nu|\delta) +
                              3 (\\nabla\\nabla\mu\\nabla\\nu|\delta)^\dagger
            at coordinates of MM atoms. Those integrals have formally
            27 different components, but we use it symmetrized and
            compressed."""

            if pol:
                fm = self.fakemol_pol
            else:
                fm = self.fakemol_static

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
            nnni_j = df.incore.aux_e2(self.mol, fm,
                                      intor='int3c2e_ipipip1')
            nni_nj = df.incore.aux_e2(self.mol, fm,
                                      intor='int3c2e_ipipvip1')
            Hef = nnni_j + numpy.einsum('inmj->imnj', nnni_j) + \
                  3 * (nni_nj + numpy.einsum('inmj->imnj', nni_nj))

            # Compress and make symmetric
            Hef[1] = (Hef[1] + Hef[3] + Hef[9]) / 3
            Hef[2] = (Hef[2] + Hef[6] + Hef[18]) / 3
            Hef[4] = (Hef[4] + Hef[10] + Hef[12]) / 3
            Hef[5] = (Hef[5] + Hef[7] + Hef[11] + Hef[15] + Hef[19] + Hef[21]) / 6
            Hef[8] = (Hef[8] + Hef[20] + Hef[24]) / 3
            Hef[14] = (Hef[14] + Hef[16] + Hef[22]) / 3
            Hef[17] = (Hef[17] + Hef[23] + Hef[25]) / 3

            return Hef[[0,1,2,4,5,8,13,14,17,26]]

        @property
        def V_at_nucl(self):
            try:
                return self.ommp_qm_helper.V_m2n
            except AttributeError:
                self.ommp_qm_helper.prepare_energy(self.ommp_obj)
                return self.ommp_qm_helper.V_m2n

        @property
        def E_at_nucl(self):
            try:
                return self.ommp_qm_helper.E_m2n
            except AttributeError:
                self.ommp_qm_helper.prepare_geomgrad(self.ommp_obj)
                return self.ommp_qm_helper.E_m2n

        @property
        def ef_nucl_at_pol(self):
            try:
                return self.ommp_qm_helper.E_n2p
            except AttributeError:
                self.ommp_qm_helper.prepare_energy(self.ommp_obj)
                return self.ommp_qm_helper.E_n2p

        @property
        def gef_nucl_at_pol(self):
            if not self.do_pol:
                return numpy.zeros([0,6])
            else:
                try:
                    return self.ommp_qm_helper.G_n2p
                except AttributeError:
                    self.ommp_qm_helper.prepare_geomgrad(self.ommp_obj)
                    return self.ommp_qm_helper.G_n2p

        @property
        def ef_nucl_at_static(self):
            try:
                print(self.ommp_qm_helper.E_n2m)
                return self.ommp_qm_helper.E_n2m
            except AttributeError:
                self.ommp_qm_helper.prepare_geomgrad(self.ommp_obj)
                return self.ommp_qm_helper.E_n2m

        @property
        def gef_nucl_at_static(self):
            try:
                return self.ommp_qm_helper.G_n2m
            except AttributeError:
                self.ommp_qm_helper.prepare_geomgrad(self.ommp_obj)
                return self.ommp_qm_helper.G_n2m

        @property
        def Hef_nucl_at_static(self):
            try:
                return self.ommp_qm_helper.H_n2m
            except AttributeError:
                self.ommp_qm_helper.prepare_geomgrad(self.ommp_obj)
                return self.ommp_qm_helper.H_n2m

        def ef_at_static(self, dm, exclude_nuclei=False):
            """Computes the electric field generated by the QM system with density dm
            at fixed electrostatic sites"""

            ef = numpy.einsum('inmj,nm->ji',
                              self.ef_integrals_ommp(),
                              dm, dtype="f8")
            if not exclude_nuclei:
                ef += self.ef_nucl_at_static
            return ef

        def gef_at_static(self, dm, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at fixed electrostatic sites"""

            gef = numpy.einsum('inmj,nm->ji',
                              self.gef_integrals_ommp(),
                              dm, dtype="f8")
            if not exclude_nuclei:
                gef -= self.gef_nucl_at_static
            return gef

        def Hef_at_static(self, dm, exclude_nuclei=False):
            Hef = numpy.einsum('inmj,nm->ji',
                              self.Hef_integrals_ommp(),
                              dm, dtype="f8")

            if not exclude_nuclei:
                Hef += self.Hef_nucl_at_static
            return Hef


        def ef_at_pol(self, dm, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at polarizable sites"""

            ef = numpy.einsum('inmj,nm->ji',
                              self.ef_integrals_ommp(pol=True),
                              dm, dtype="f8")

            if not exclude_nuclei:
                ef += self.ef_nucl_at_pol
            return ef

        def gef_at_pol(self, dm, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at fixed electrostatic sites"""

            gef = numpy.einsum('inmj,nm->ji',
                              self.gef_integrals_ommp(pol=True),
                              dm, dtype="f8")
            if not exclude_nuclei:
                gef -= self.gef_nucl_at_pol
            return gef

        def get_veff(self, mol=None, dm=None, *args, **kwargs):
            """Function to add the contributions from polarizable sites
            to the Fock matrix. To do so:
            (1) the electric field at polarizable sites is computed
            (2) this field is used to solve the linear system and
                    compute the induced dipoles.
            (3) the induced dipoles are read and contracted with the
                    electric field operator to provide the contribution
                    to the fock matrix.
            (4) the contribution to the energy is also computed."""

            vhf = method_class.get_veff(self, mol, dm, *args, **kwargs)

            if self.fakemol_pol is not None:
                # NOTE: v_solvent should not be added to vhf in this place. This is
                # because vhf is used as the reference for direct_scf in the next
                # iteration. If v_solvent is added here, it may break direct SCF.

                # 1. compute the EF generated by the current DM
                current_ef = self.ef_at_pol(dm)

                # 2. update the induced dipoles
                self.ommp_obj.set_external_field(current_ef)

                # 3.1. get the induced dipoles
                if not self.ommp_obj.is_amoeba:
                    current_ipds = self.get_mmpol_induced_dipoles()
                else:
                    current_ipds_d, current_ipds_p = self.get_mmpol_induced_dipoles()
                    current_ipds = (current_ipds_d+current_ipds_p) / 2

                # 3.2. compute the induced dipoles term in the Hamiltonian
                v_mmpol = -numpy.einsum('inmj,ji->nm',
                                        self.ef_integrals_ommp(pol=True), current_ipds)

                # 4. Compute the MMPol contribution to energy
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
            """Assemble the Fock matrix for SCF with MMPol environments."""

            if getattr(vhf, 'v_mmpol', None) is None:
                vhf = self.get_veff(self.mol, dm)

            return method_class.get_fock(self, h1e, s1e, vhf+vhf.v_mmpol, dm, cycle, diis,
                                         diis_start_cycle, level_shift_factor, damp_factor)


        def get_hcore(self, mol=None):
            """Compute the core Hamiltonian for MMPol-SCF"""
            if mol is None:
                mol = self.mol

            if getattr(method_class, 'get_hcore', None):
                h1e = method_class.get_hcore(self, mol)
            else:
                # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise RuntimeError('openMMPol cannot be applied here in post-HF methods')

            q = self.ommp_obj.static_charges
            self.h1e_mmpol = - numpy.einsum('nmi,i->nm', self.v_integrals_ommp(), q)

            if self.ommp_obj.is_amoeba:
                mu = self.ommp_obj.static_dipoles
                self.h1e_mmpol += - numpy.einsum('inmj,ji->nm', self.ef_integrals_ommp(), mu)
                quad = self.ommp_obj.static_quadrupoles

                # Off diagonal components are multiplied by two
                quad[:,[1,3,4]] *= 2.0
                self.h1e_mmpol += -numpy.einsum('inmj,ji->nm', self.gef_integrals_ommp(), quad)
            return h1e + self.h1e_mmpol

        def energy_nuc(self):
            """Computes the interaction between nuclei and nuclei and
            between nuclei and external MM centers"""
            # interactions between QM nuclei and QM nuclei
            nuc = self.mol.energy_nuc()

            # interactions between QM nuclei and MM particles
            self.nuc_static_mm = numpy.dot(self.V_at_nucl, self.mol.atom_charges())
            nuc += self.nuc_static_mm

            return nuc

        def energy_elec(self, dm=None, h1e=None, vhf=None):
            ene_el = method_class.energy_elec(self, dm, h1e, vhf)

            if getattr(vhf, 'e_mmpol', None):
                e_mmpol_2e = vhf.e_mmpol - self.ommp_obj.get_polelec_energy()
            else:
                e_mmpol_2e = self.get_veff(dm=dm).e_mmpol - self.ommp_obj.get_polelec_energy()

            ene_el = (ene_el[0]+e_mmpol_2e,
                      ene_el[1]+e_mmpol_2e)

            return ene_el

        def energy_tot(self, dm=None, h1e=None, vhf=None):
            """Compute total SCF energy, that also includes the energy
            of the MM part."""
            e_tot = method_class.energy_tot(self, dm, h1e, vhf)
            e_tot += self.ommp_obj.get_fixedelec_energy()
            e_tot += self.ommp_obj.get_polelec_energy()
            # Todo add also bonded / vdw terms
            return e_tot

        def gen_response(self, *args, **kwargs):
            """Compute the response function accounting for the MMPol terms"""
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
                        current_ef = self.ef_at_pol(d, exclude_nuclei=True)
                        self.ommp_obj.set_external_field(current_ef, nomm=False)

                        if not self.ommp_obj.is_amoeba:
                            current_ipds = self.get_mmpol_induced_dipoles()
                        else:
                            current_ipds, empty = \
                                    self.get_mmpol_induced_dipoles()

                        vd = -numpy.einsum('inmj,ji->nm',
                                           self.ef_integrals_ommp(),
                                           current_ipds)
                        v_mmpol += [vd]

                    v_mmpol = numpy.array(v_mmpol)

                    v += v_mmpol

                return v

            return vind_mmpol

        def nuc_grad_method(self):
            """Return a method for computing nuclear gradients."""
            scf_grad = method_class.nuc_grad_method(self)
            return qmmmpol_grad_for_scf(scf_grad)

        Gradients = nuc_grad_method

    if isinstance(scf_method, scf.hf.SCF):
        return QMMMPOL(scf_method, ommp_obj)

def qmmmpol_grad_for_scf(scf_grad):
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
            logger.info(self,
                        'MMPol system with {:d} sites ({:d} polarizable)'.format(self.base.ommp_obj.mm_atoms,
                                                                                 self.base.ommp_obj.pol_atoms))
            return self

        def MM_atoms_grad(self):
            """Computes the energy gradients on MM atoms of the system"""
            dm = self.base.make_rdm1()

            # Charges
            ef_QMatMM = self.base.ef_at_static(dm)
            force = -numpy.einsum('ij,i->ij', ef_QMatMM, self.base.ommp_obj.static_charges)
            if self.base.ommp_obj.is_amoeba:
                # Dipoles
                mu = self.base.ommp_obj.static_dipoles
                gef_QMatMM = self.base.gef_at_static(dm)
                force += -numpy.einsum('ij,i->ij', gef_QMatMM[:,[0,1,3]], mu[:,0])
                force += -numpy.einsum('ij,i->ij', gef_QMatMM[:,[1,2,4]], mu[:,1])
                force += -numpy.einsum('ij,i->ij', gef_QMatMM[:,[3,4,5]], mu[:,2])

                # Quadrupoles
                quad = self.base.ommp_obj.static_quadrupoles
                Hef_QMatMM = self.base.Hef_at_static(dm)
                force += -numpy.einsum('ij,i->ij', Hef_QMatMM[:,[0,1,2]], quad[:,0]) #xx
                force += -2.0*numpy.einsum('ij,i->ij', Hef_QMatMM[:,[1,3,4]], quad[:,1]) #xy
                force += -numpy.einsum('ij,i->ij', Hef_QMatMM[:,[3,6,7]], quad[:,2]) #yy
                force += -2.0*numpy.einsum('ij,i->ij', Hef_QMatMM[:,[2,4,5]], quad[:,3]) #xz
                force += -2.0*numpy.einsum('ij,i->ij', Hef_QMatMM[:,[4,7,8]], quad[:,4]) #yz
                force += -numpy.einsum('ij,i->ij', Hef_QMatMM[:,[5,8,9]], quad[:,5]) #zz

                # Contribution for the multipoles rotation
                force += self.base.ommp_obj.do_rotation_grad(ef_QMatMM, -gef_QMatMM)

            if self.base.do_pol:
                # Induced dipoles
                gef_QMatPOL = self.base.gef_at_pol(dm)
                if not self.base.ommp_obj.is_amoeba:
                    mu = self.base.get_mmpol_induced_dipoles()
                else:
                    mu_d, mu_p = self.base.get_mmpol_induced_dipoles()
                    mu = 0.5 * (mu_d + mu_p)

                force_pol = -numpy.einsum('ij,i->ij', gef_QMatPOL[:,[0,1,3]], mu[:,0])
                force_pol += -numpy.einsum('ij,i->ij', gef_QMatPOL[:,[1,2,4]], mu[:,1])
                force_pol += -numpy.einsum('ij,i->ij', gef_QMatPOL[:,[3,4,5]], mu[:,2])
                force[self.base.ommp_obj.polar_mm] += force_pol

            force += self.base.ommp_obj.do_polelec_grad()
            force += self.base.ommp_obj.do_fixedelec_grad()

            return force

        def get_hcore(self, mol=None):
            """Computes QM/MMPol contribution to the derivative
            of core Hamiltonian"""

            if mol is None:
                mol = self.mol

            if getattr(grad_class, 'get_hcore', None):
                g_qm = grad_class.get_hcore(self, mol)
            else:
                # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise NotImplementedError("For some reason QM method does not have get_hcore func.")

            q = self.base.ommp_obj.static_charges

            ints = df.incore.aux_e2(mol, self.base.fakemol_static,
                                   intor='int3c2e_ip1')
            g_mm = numpy.einsum('ipqk,k->ipq', ints, q)

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
                # Contribution of the converged induced dipoles
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
            """Compute gradients (on QM atoms) due to the interaction of nuclear
            charges with the MM multipoles and induced dipoles"""

            if mol is None:
                mol = self.mol

            if getattr(grad_class, 'grad_nuc', None):
                g_qm = grad_class.grad_nuc(self, mol, atmlst)
            else:
                # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise NotImplementedError("For some reason QM method does not have grad_nuc func.")

            g_mm = -numpy.einsum('i,ij->ij',
                                 self.mol.atom_charges(),
                                 scf_grad.base.E_at_nucl)
            if atmlst is not None:
                g_mm = g_mm[atmlst]

            return g_qm + g_mm

    return QMMMPOLG(scf_grad)


class _QMMMPOL(_QMMM):
    pass
class _QMMMPOLGrad(_QMMMGrad):
    pass
