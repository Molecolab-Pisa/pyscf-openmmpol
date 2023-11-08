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
from pyscf.data.elements import NUC as Symbol2Z, _symbol
from pyscf.qmmm.itrf import _QMMM, _QMMMGrad
import pyopenmmpol as ommp

class QMMMPolMole(gto.Mole):
    def __init__(self, molQM, ommp_obj, ommp_qm_helper, remove_frozen_atoms=False):
        self.__dict__.update(molQM.__dict__)
        self.molQM = molQM
        self.ommp_obj = ommp_obj
        self.ommp_qm_helper = ommp_qm_helper

        if remove_frozen_atoms:
            # Only for MM atoms
            self.frozenMM = []
            self.unfrozenMM = []
            for i, f in enumerate(self.ommp_obj.frozen):
                if f:
                    self.frozenMM += [i]
                else:
                    self.unfrozenMM += [i]
            self.frozenMM = numpy.array(self.frozenMM, dtype=numpy.int64)
            self.unfrozenMM = numpy.array(self.unfrozenMM, dtype=numpy.int64)
            self.nfrozen = self.frozenMM.shape[0]
        else:
            self.frozenMM = numpy.array([], dtype=numpy.int64)
            self.unfrozenMM = numpy.arange(0, self.ommp_obj.mm_atoms, 1, dtype=numpy.int64)
            self.nfrozen = 0

        self.natm_QM = self.molQM.natm
        self.QM_atm_lst = numpy.arange(0, self.natm_QM, 1, dtype=numpy.int64)

        self.natm_MM = self.ommp_obj.mm_atoms - self.nfrozen
        self.MM_atm_lst = numpy.arange(self.natm_QM, self.natm, 1, dtype=numpy.int64)

        self._atom = molQM._atom.copy()
        for i in range(self.ommp_obj.mm_atoms):
            if i not in self.frozenMM:
                atmstr = "{:s} {:f} {:f} {:f}".format(_symbol(self.ommp_obj.zmm[i]),
                                                      self.ommp_obj.cmm[i,0],
                                                      self.ommp_obj.cmm[i,1],
                                                      self.ommp_obj.cmm[i,2])
                self._atom += self.molQM.format_atom(atmstr, unit=self.unit)

    @property
    def natm(self):
        return self.natm_QM + self.natm_MM

    def set_geom_(self, atoms_or_coords, unit=None, symmetry=None,
                  inplace=True):
        if inplace:
            mol = self
        else:
            raise NotImplementedError

        if self.nfrozen > 0:
            stored_c = mol.ommp_obj.cmm.copy()
            stored_c[self.unfrozenMM] = atoms_or_coords[self.MM_atm_lst]
            mol.ommp_obj.update_coordinates(stored_c)
        else:
            mol.ommp_obj.update_coordinates(atoms_or_coords[self.MM_atm_lst])
        self.ommp_qm_helper.update_coord(atoms_or_coords[self.QM_atm_lst])
        self.ommp_qm_helper.update_link_atoms_position(self.ommp_obj)
        mol.molQM.set_geom_(self.ommp_qm_helper.cqm, 'B', inplace)

        return mol

    def build(self):
        self.molQM.build()
        pass

    def atom_coords(self):
        full_coords = numpy.empty([self.natm, 3])
        full_coords[self.QM_atm_lst] = self.molQM.atom_coords()
        full_coords[self.MM_atm_lst] = self.ommp_obj.cmm[self.unfrozenMM]
        return full_coords

class _QMMM_GradScanner(lib.GradScanner):
    def __init__(self, gs, remove_frozen_atoms = False):
        self.mol = QMMMPolMole(gs.mol, gs.base.ommp_obj, gs.base._qmhelper, remove_frozen_atoms)
        self.qm_scanner = gs
        self.base = self.qm_scanner.base
        self.verbose = self.base.verbose
        self.stdout = self.base.stdout
        self.atmlst = self.qm_scanner.atmlst

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.Mole):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        if self.mol.nfrozen > 0:
            mm_coords = self.mol.ommp_obj.cmm.copy()
            mm_coords[self.mol.unfrozenMM] = mol.atom_coords()[mol.MM_atm_lst,:]
        else:
            mm_coords = mol.atom_coords()[mol.MM_atm_lst,:]
        qm_mol = mol.molQM

        mf_scanner = self.qm_scanner.base
        e_tot_qm, de_qm, de_mm = self.qm_scanner(qm_mol,
                                                 mm_coords=mm_coords,
                                                 do_mm_grad=True)

        e_tot = e_tot_qm
        de = numpy.zeros((mol.natm, 3))
        de[mol.QM_atm_lst[self.atmlst], :] = de_qm
        de[mol.MM_atm_lst, :] = de_mm[mol.unfrozenMM, :]

        return e_tot, de

def add_mmpol(scf_method, smartinput_file, use_si_qm_coord = None):
    ommp_system, ommp_qmhelper = ommp.smartinput(smartinput_file)
    scfmmpol = qmmmpol_for_scf(scf_method, ommp_system)

    if ommp_qmhelper is not None:
        # Do some checks here...
        # Atomic number
        # Coordinates
        if scf_method.mol.atom_coords().shape != ommp_qmhelper.cqm.shape:
            logger.info(scfmmpol, "Coordinates in smartinput file have a different shape from the ones in mol object.")
            raise RuntimeError("smartinput file and pyscf input should be consistent")
        if not numpy.allclose(scf_method.mol.atom_coords(), ommp_qmhelper.cqm):
            logger.info(scfmmpol, "Coordinates in smartinput file differ from the ones in mol object, those last will be used.")
            if use_si_qm_coord is None:
                # Default option: if coordinates are different stop, unless the different coords are on LA that
                # are handled by OMMP
                if ommp_system.use_linkatoms:
                    # TODO Improve this, it should be for link atoms only
                    scf_method.mol.set_geom_(ommp_qmhelper.cqm, unit='B', inplace=True)
                    #raise RuntimeError("I Gelati sono buoni ma costano milioni")
                else:
                    raise RuntimeError("Coordinates in smartinput file differ from the ones in mol objecti, set a policy with use_si_qm_coord option.")
            elif use_si_qm_coord == True:
                scf_method.mol.set_geom_(ommp_qmhelper.cqm, unit='B', inplace=True)
            elif use_si_qm_coord == False:
                ommp_qmhelper.update_coord(scf_method.mol.atom_coords())
            else:
                raise RuntimeError("use_si_coord should be either Ture, False or None")
        scfmmpol._qmhelper = ommp_qmhelper
    return scfmmpol

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

        def __init__(self, scf_method, ommp_obj, use_qmmm_mol=False):
            self.__dict__.update(scf_method.__dict__)
            self.ommp_obj = ommp_obj
            self._keys.update(['ommp_obj'])
            self.mol = self.mol

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
            elif not numpy.allclose(self.ommp_obj.cmm, self._fakemol_static.atom_coords()):
                self._fakemol_static.set_geom_(self.ommp_obj.cmm, unit='B')
            return self._fakemol_static

        @property
        def fakemol_pol(self):
            """Fakemol for computing 3-centers-2-electron integrals
            at static sites coordinates"""
            if self.do_pol:
                if not hasattr(self, '_fakemol_pol'):
                    self._fakemol_pol = gto.fakemol_for_charges(self.ommp_obj.cpol)
                elif not numpy.allclose(self.ommp_obj.cpol, self._fakemol_pol.atom_coords()):
                    self._fakemol_pol.set_geom_(self.ommp_obj.cpol, unit='B')
            else:
                self._fakemol_pol = None

            return self._fakemol_pol

        @property
        def fakeget_mem(self):
            self._fakeget_mem = self.max_memory - lib.current_memory()[0]
            return self._fakeget_mem

        @property
        def ommp_qm_helper(self, inmol=None):
            if inmol is None:
                mol = self.mol
            else:
                mol = inmol
            if not hasattr(self, '_qmhelper'):
                qmat_q = mol.atom_charges()
                qmat_c = mol.atom_coords()
                qmat_z = [Symbol2Z[a[0]] for a in mol._atom]
                self._qmhelper = ommp.OMMPQmHelper(qmat_c, qmat_q, qmat_z)
            elif not numpy.allclose(mol.atom_coords(), self._qmhelper.cqm):
                self._qmhelper.update_coord(mol.atom_coords())
            return self._qmhelper

        def v_integrals_ommp(self, pol=False, mol=None, i0=None, i1=None):
            """Electrostatic potential integrals <\mu|r^{-1}|\\nu> = (\mu,\\nu|\delta)
            at coordinates of MM atoms.
            For a reference on how 1-electron integrals can be computed as
            3-center-2-electron electron integrals see Chem. Phys. Lett vol. 206,
            pp. 239-246."""

            if pol:
                if i0 is not None:
                    fm = gto.fakemol_for_charges(self.ommp_obj.cpol[i0:i1])
                else:
                    fm = self.fakemol_pol
            else:
                if i0 is not None:
                    fm = gto.fakemol_for_charges(self.ommp_obj.cmm[i0:i1])
                else:
                    fm = self.fakemol_static

            if mol is None:
                mol = self.mol

            return df.incore.aux_e2(mol, fm,
                                    intor='int3c2e')

        def ef_integrals_ommp(self, pol=False, mol=None,
                              dm=None, dipoles=None):
            """Electric field integrals
            <mu|\hat(E)|nu> = <\mu|\\nabla r^{-1}|\\nu> =
                            = (\\nabla\mu\\nu|\delta) + (\mu\\nabla\\nu|\delta)
                            = (\\nabla\mu\\nu|\delta) + (\\nabla\mu\\nu|\delta)^\dagger
            at coordinates of MM atoms."""

            if pol:
                fm = self.fakemol_pol
            else:
                fm = self.fakemol_static

            if mol is None:
                mol = self.mol

            nao = mol.nao
            nmmp = fm.natm
            blksize = int(min(self.fakeget_mem*1e6/8/nao**2, 500))

            if dipoles is None and dm is None:
                Ef = df.incore.aux_e2(mol, fm,
                                      intor='int3c2e_ip1')
                Ef += numpy.einsum('imnj->inmj', Ef)
                return Ef
            elif dipoles is not None and dm is None:
                Ef_dc = numpy.zeros([nao,nao])
                for j0, j1, in lib.prange(0, fm.natm, blksize):
                    _fm = gto.fakemol_for_charges(fm.atom_coords()[j0:j1])
                    Ef_dc -= numpy.einsum('imnj,ji->mn',
                                          df.incore.aux_e2(mol, _fm, intor='int3c2e_ip1'),
                                          dipoles[j0:j1])
                return Ef_dc + Ef_dc.T
            elif dm is not None:
                Ef_dc = numpy.empty([nmmp,3])

                for j0, j1 in lib.prange(0, fm.natm, blksize):
                    _fm = gto.fakemol_for_charges(fm.atom_coords()[j0:j1])
                    Ef_dc[j0:j1] = numpy.einsum('imnj,mn->ji',
                                                df.incore.aux_e2(mol, _fm, intor='int3c2e_ip1'),
                                                dm)
                Ef_dc *= 2.
                if dipoles is None:
                    return Ef_dc
                else:
                    return -numpy.einsum('ji,ji->', Ef_dc, dipoles)
            return None

        def gef_integrals_ommp(self, pol=False, mol=None,
                               dm=None, quadrupoles=None):
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

            if mol is None:
                mol = self.mol

            nao = mol.nao
            nmmp = fm.natm
            blksize = int(min(self.fakeget_mem*1e6/8/nao**2, 500))

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

            if quadrupoles is None and dm is None:
                nni_j = df.incore.aux_e2(mol, fm,
                                        intor='int3c2e_ipip1')

                ni_nj = df.incore.aux_e2(mol, fm,
                                        intor='int3c2e_ipvip1')

                Gef = nni_j + numpy.einsum('inmj->imnj', nni_j) + 2 * ni_nj

                Gef[[1,2,5]] += Gef[[3,6,7]]
                Gef[[1,2,5]] /= 2.0
                #TODO TRANSPOSE
                return Gef[[0,1,4,2,5,8]]

            elif quadrupoles is not None and dm is None:
                Gef = numpy.zeros([nao, nao])
                for j0, j1, in lib.prange(0, fm.natm, blksize):
                    _fm = gto.fakemol_for_charges(fm.atom_coords()[j0:j1])
                    tmp_quad = numpy.zeros([j1-j0,9])
                    tmp_quad[:,[1,2,5]] = quadrupoles[j0:j1,[1,3,4]]
                    tmp_quad[:,[3,6,7]] = tmp_quad[:,[1,2,5]]
                    tmp_quad[:,[0,4,8]] = quadrupoles[j0:j1,[0,2,5]]
                    nni_j = df.incore.aux_e2(mol, _fm, intor='int3c2e_ipip1')
                    Gef += numpy.einsum('inmj,ji->nm', nni_j, tmp_quad)
                    ni_nj = df.incore.aux_e2(mol, _fm, intor='int3c2e_ipvip1')
                    Gef += numpy.einsum('inmj,ji->nm', ni_nj, tmp_quad)
                return Gef + Gef.T
            elif dm is not None:
                Gef = numpy.empty([nmmp,9])

                for j0, j1 in lib.prange(0, fm.natm, blksize):
                    _fm = gto.fakemol_for_charges(fm.atom_coords()[j0:j1])
                    Gef[j0:j1] = numpy.einsum('inmj,nm->ji',
                                              df.incore.aux_e2(mol, _fm, intor='int3c2e_ipip1') + \
                                              df.incore.aux_e2(mol, _fm, intor='int3c2e_ipvip1'),
                                              dm)
                Gef[:,[1,2,5]] += Gef[:,[3,6,7]]
                Gef[:,[0,4,8]] *= 2
                if quadrupoles is None:
                    return Gef[:,[0,1,4,2,5,8]]
                else:
                    return numpy.einsum('ij,ij->', Gef, quadrupoles)
            return None


        def Hef_integrals_ommp(self, pol=False, mol=None, dm=None):
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

            if mol is None:
                mol = self.mol

            nao = mol.nao
            nmmp = fm.natm
            blksize = int(min(self.fakeget_mem*1e6/8/nao**2, 500))

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

            if dm is None:
                nnni_j = df.incore.aux_e2(mol, fm,
                                          intor='int3c2e_ipipip1')
                nni_nj = df.incore.aux_e2(mol, fm,
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

                #TODO Transpose
                return Hef[[0,1,2,4,5,8,13,14,17,26]]
            else:
                Hef = numpy.empty([nmmp,27])

                for j0, j1 in lib.prange(0, fm.natm, blksize):
                    _fm = gto.fakemol_for_charges(fm.atom_coords()[j0:j1])
                    Hef[j0:j1] = numpy.einsum('inmj,nm->ji',
                                              df.incore.aux_e2(mol, _fm, intor='int3c2e_ipipip1') + \
                                              3 * df.incore.aux_e2(mol, _fm, intor='int3c2e_ipipvip1'),
                                              dm)

                # Compress and make symmetric
                Hef[:,1] =  (Hef[:,1] +  Hef[:,3] +  Hef[:,9]) / 3
                Hef[:,2] =  (Hef[:,2] +  Hef[:,6] +  Hef[:,18]) / 3
                Hef[:,4] =  (Hef[:,4] +  Hef[:,10] + Hef[:,12]) / 3
                Hef[:,5] =  (Hef[:,5] +  Hef[:,7] +  Hef[:,11] + Hef[:,15] + Hef[:,19] + Hef[:,21]) / 6
                Hef[:,8] =  (Hef[:,8] +  Hef[:,20] + Hef[:,24]) / 3
                Hef[:,14] = (Hef[:,14] + Hef[:,16] + Hef[:,22]) / 3
                Hef[:,17] = (Hef[:,17] + Hef[:,23] + Hef[:,25]) / 3
                return Hef[:,[0,1,2,4,5,8,13,14,17,26]] * 2.
            return None


        @property
        def V_mm_at_nucl(self):
            try:
                return self.ommp_qm_helper.V_m2n
            except AttributeError:
                self.ommp_qm_helper.prepare_qm_ele_ene(self.ommp_obj)
                return self.ommp_qm_helper.V_m2n

        @property
        def E_mm_at_nucl(self):
            try:
                return self.ommp_qm_helper.E_m2n
            except AttributeError:
                self.ommp_qm_helper.prepare_qm_ele_grd(self.ommp_obj)
                return self.ommp_qm_helper.E_m2n

        @property
        def V_pol_at_nucl(self):
            try:
                return self.ommp_qm_helper.V_p2n
            except AttributeError:
                self.ommp_qm_helper.prepare_qm_ele_ene(self.ommp_obj)
                return self.ommp_qm_helper.V_p2n

        @property
        def E_pol_at_nucl(self):
            try:
                return self.ommp_qm_helper.E_p2n
            except AttributeError:
                self.ommp_qm_helper.prepare_qm_ele_grd(self.ommp_obj)
                return self.ommp_qm_helper.E_p2n

        @property
        def ef_nucl_at_pol(self):
            try:
                return self.ommp_qm_helper.E_n2p
            except AttributeError:
                self.ommp_qm_helper.prepare_qm_ele_ene(self.ommp_obj)
                return self.ommp_qm_helper.E_n2p

        @property
        def gef_nucl_at_pol(self):
            if not self.do_pol:
                return numpy.zeros([0,6])
            else:
                try:
                    return self.ommp_qm_helper.G_n2p
                except AttributeError:
                    self.ommp_qm_helper.prepare_qm_ele_grd(self.ommp_obj)
                    return self.ommp_qm_helper.G_n2p

        @property
        def ef_nucl_at_static(self):
            try:
                return self.ommp_qm_helper.E_n2m
            except AttributeError:
                self.ommp_qm_helper.prepare_qm_ele_grd(self.ommp_obj)
                return self.ommp_qm_helper.E_n2m

        @property
        def gef_nucl_at_static(self):
            try:
                return self.ommp_qm_helper.G_n2m
            except AttributeError:
                self.ommp_qm_helper.prepare_qm_ele_grd(self.ommp_obj)
                return self.ommp_qm_helper.G_n2m

        @property
        def Hef_nucl_at_static(self):
            try:
                return self.ommp_qm_helper.H_n2m
            except AttributeError:
                self.ommp_qm_helper.prepare_qm_ele_grd(self.ommp_obj)
                return self.ommp_qm_helper.H_n2m

        def ef_at_static(self, dm, mol=None, exclude_nuclei=False):
            """Computes the electric field generated by the QM system with density dm
            at fixed electrostatic sites"""

            ef = self.ef_integrals_ommp(mol=mol, dm=dm)
            if not exclude_nuclei:
                ef += self.ef_nucl_at_static
            return ef

        def gef_at_static(self, dm, mol=None, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at fixed electrostatic sites"""

            gef = self.gef_integrals_ommp(mol=mol, dm=dm)
            if not exclude_nuclei:
                gef -= self.gef_nucl_at_static
            return gef

        def Hef_at_static(self, dm, mol=None, exclude_nuclei=False):
            Hef = self.Hef_integrals_ommp(mol=mol, dm=dm)
            if not exclude_nuclei:
                Hef += self.Hef_nucl_at_static
            return Hef


        def ef_at_pol(self, dm, mol=None, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at polarizable sites"""

            ef = self.ef_integrals_ommp(pol=True, dm=dm, mol=mol)

            if not exclude_nuclei:
                ef += self.ef_nucl_at_pol
            return ef

        def gef_at_pol(self, dm, mol=None, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at fixed electrostatic sites"""

            gef = self.gef_integrals_ommp(pol=True, dm=dm)
            if not exclude_nuclei:
                gef -= self.gef_nucl_at_pol
            return gef

        def kernel(self, conv_tol=1e-10, conv_tol_grad=None,
                   dump_chk=True, dm0=None, callback=None,
                   conv_check=True, **kwargs):

            mol = self.mol
            if dm0 is None:
                dm = method_class.get_init_guess(self, mol, self.init_guess)
            else:
                dm = dm0

            return method_class.kernel(self, dm, **kwargs)

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
            is_uhf = isinstance(self, scf.uhf.UHF)
            if is_uhf:
                dm_tot = dm[0] + dm[1]
            else:
                dm_tot = dm


            if self.fakemol_pol is not None:
                # NOTE: v_solvent should not be added to vhf in this place. This is
                # because vhf is used as the reference for direct_scf in the next
                # iteration. If v_solvent is added here, it may break direct SCF.
                # 1. compute the EF generated by the current DM
                current_ef = self.ef_at_pol(dm_tot, mol=mol)

                # 2. update the induced dipoles
                self.ommp_obj.set_external_field(current_ef)

                # 3.1. get the induced dipoles
                if not self.ommp_obj.is_amoeba:
                    current_ipds = self.get_mmpol_induced_dipoles()
                else:
                    current_ipds_d, current_ipds_p = self.get_mmpol_induced_dipoles()
                    current_ipds = (current_ipds_d+current_ipds_p) / 2

                # 3.2. compute the induced dipoles term in the Hamiltonian
                v_mmpol = self.ef_integrals_ommp(pol=True, mol=mol, dipoles=current_ipds)

                # 4. Compute the MMPol contribution to energy
                if not self.ommp_obj.is_amoeba:
                    e_mmpol = -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds)
                else:
                    e_mmpol = -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds_d)
                e_mmpol += self.ommp_obj.get_polelec_energy()

            else:
                # If there are no polarizabilities, there are no contribution to the Fock Matrix
                v_mmpol = numpy.zeros(dm_tot.shape)
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

            nao = mol.nao
            q = self.ommp_obj.static_charges
            blksize = int(min(self.fakeget_mem*1e6/8/nao**2, 200))

            self.h1e_mmpol = 0.0
            for i0, i1 in lib.prange(0, q.size, blksize):
                self.h1e_mmpol -= numpy.einsum('nmi,i->nm', self.v_integrals_ommp(mol=mol, i0=i0, i1=i1), q[i0:i1])

            if self.ommp_obj.is_amoeba:
                self.h1e_mmpol += self.ef_integrals_ommp(mol=mol, dipoles=self.ommp_obj.static_dipoles)

                quad = self.ommp_obj.static_quadrupoles

                self.h1e_mmpol -= self.gef_integrals_ommp(mol=mol, quadrupoles=quad)
            return h1e + self.h1e_mmpol

        def energy_nuc(self):
            """Computes the interaction between nuclei and nuclei and
            between nuclei and external MM centers; also compute MM energy
            terms"""

            mol = self.mol
            # interactions between QM nuclei and QM nuclei
            nuc = mol.energy_nuc()

            # interactions between QM nuclei and MM particles (static part)
            # the polarizable part is computed when the QM electric field, which
            # includes nuclei, is contracted with induced dipoles. This is because
            # when this function is called IPD are still unavailable.
            self.nuc_static_mm = numpy.dot(self.V_mm_at_nucl, self.mol.atom_charges())
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

            if hasattr(self, 'e_mmpol_crdhash'):
                tmp_hash = hash(self.mol.atom_coords().tostring()+\
                                self.ommp_obj.cmm.tostring())
                if tmp_hash != self.e_mmpol_crdhash:
                    delattr(self, e_mmpol_crdhash)
                    delattr(self, e_mmpol)

            if not hasattr(self, 'e_mmpol'):
                self.e_mmpol = self.ommp_obj.get_full_bnd_energy()
                self.e_mmpol += self.ommp_obj.get_full_ele_energy()
                self.e_mmpol += self.ommp_obj.get_vdw_energy()
                # QM-MM VdW interaction
                if(self.ommp_qm_helper.use_nonbonded):
                    self.e_mmpol += self.ommp_qm_helper.vdw_energy(self.ommp_obj)
                # Hash the coordinates of the system to avoid useless calculations
                # of energy components.
                self.e_mmpol_crdhash = hash(self.mol.atom_coords().tostring()+\
                                            self.ommp_obj.cmm.tostring())
            e_tot += self.e_mmpol

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

                        vd = self.ef_integrals_ommp(dipoles=current_ipds)
                        v_mmpol += [vd]

                    v_mmpol = numpy.array(v_mmpol)

                    v += v_mmpol

                return v

            return vind_mmpol

        def energy_analysis(self):
            # TODO Improve this analysis
            au2k = 627.50960803
            smm_ene = self.ommp_obj.get_fixedelec_energy()
            pmm_ene = self.ommp_obj.get_polelec_energy()
            scf_ene = self.e_tot - smm_ene - pmm_ene
            nuc_mm_ene = self.nuc_static_mm

            dm  = self.make_rdm1()
            ele_mm_ene = numpy.einsum('nm,nm', self.h1e_mmpol, dm)
            ele_p_ene = self.get_veff(dm=dm).e_mmpol - pmm_ene
            qm_mm_ene = nuc_mm_ene + ele_mm_ene
            e_vdw = self.ommp_obj.get_vdw_energy()
            # QM-MM VdW interaction
            if(self.ommp_qm_helper.use_nonbonded):
                e_vdw += self.ommp_qm_helper.vdw_energy(self.ommp_obj)
            e_bnd = self.ommp_obj.get_full_bnd_energy()
            etot = self.e_tot

            print("==== QM-MMPOL ENERGY ANALYSIS ====")
            print("SCF e-tot: {:20.10f} ({:20.10f})".format(scf_ene, scf_ene*au2k))
            print("MM-MM:     {:20.10f} ({:20.10f})".format(smm_ene, smm_ene*au2k))
            print("MM-IPD:    {:20.10f} ({:20.10f})".format(pmm_ene, pmm_ene*au2k))
            print("QM-MM:     {:20.10f} ({:20.10f})".format(qm_mm_ene,
                                                            qm_mm_ene*au2k))
            print("QM-IPD:    {:20.10f} ({:20.10f})".format(ele_p_ene,
                                                            ele_p_ene*au2k))
            print("NUC-MM:    {:20.10f} ({:20.10f})".format(nuc_mm_ene,
                                                            nuc_mm_ene*au2k))
            print("ELE-MM:    {:20.10f} ({:20.10f})".format(ele_mm_ene,
                                                            ele_mm_ene*au2k))
            print("E QM+EEL:  {:20.10f} ({:20.10f})".format(etot-e_vdw-e_bnd,
                                                            (etot-e_vdw-e_bnd)*au2k))
            print("E VDW:     {:20.10f} ({:20.10f})".format(e_vdw,
                                                            e_vdw*au2k))
            print("E BND:     {:20.10f} ({:20.10f})".format(e_bnd,
                                                            e_bnd*au2k))

            print("E TOT:     {:20.10f} ({:20.10f})".format(etot,
                                                            etot*au2k))
            print("==================================")

        def create_link_atom(self, imm, iqm, ila, prmfile):
            """Set ila to be a link atom between iqm and imm. This function
            changes the coordinates of ila"""
            self.ommp_qm_helper
            idxla = self.ommp_obj.create_link_atom(self._qmhelper, imm, iqm, ila, prmfile)

            # Put link atom in the correct position
            self.mol.set_geom_(self._qmhelper.cqm, unit='B')

        def nuc_grad_method(self):
            """Return a method for computing nuclear gradients."""
            scf_grad = method_class.nuc_grad_method(self)
            return qmmmpol_grad_for_scf(scf_grad)

        Gradients = nuc_grad_method

    if isinstance(scf_method, scf.hf.SCF):
        return QMMMPOL(scf_method, ommp_obj)

def qmmmpol_grad_as_qmmm_scanner(qmmmpol_grad, remove_frozen_atoms=False):

    if not isinstance(qmmmpol_grad, _QMMMPOLGrad):
        raise TypeError("Only _QMMMPOLGrad objects can be transformed in _QMMM_GradScanner")

    if isinstance(qmmmpol_grad, _QMMM_GradScanner):
        return qmmmpol_grad

    if not isinstance(qmmmpol_grad, lib.GradScanner):
        qmmmpol_grad_scanner = qmmmpol_grad.as_scanner()
    else:
        qmmmpol_grad_scanner = qmmmpol_grad

    return _QMMM_GradScanner(qmmmpol_grad_scanner, remove_frozen_atoms)


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
            self.scf_obj = scf_grad
            self.de_mm = None

            if self.atmlst is None and numpy.any(self.base.ommp_qm_helper.frozen):
                self.atmlst = numpy.where(numpy.logical_not(self.base.ommp_qm_helper.frozen))
            elif self.atmlst is not None:
                self.base.ommp_qm_helper.set_frozen_atoms(self.atmlst)

        @property
        def fakeget_mem(self):
            self._fakeget_mem = self.max_memory - lib.current_memory()[0]
            return self._fakeget_mem

        def set_qm_frozen_atoms(self, value):
            self.atmlst = value
            self.base.ommp_qm_helper.set_frozen_atoms(self.atmlst)

        def dump_flags(self, verbose=None):
            grad_class.dump_flags(self, verbose)
            logger.info(self,
                        'MMPol system with {:d} sites ({:d} polarizable)'.format(self.base.ommp_obj.mm_atoms,
                                                                                 self.base.ommp_obj.pol_atoms))
            return self

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

            nao = mol.nao
            q = self.base.ommp_obj.static_charges
            blksize = int(min(self.fakeget_mem*1e6/8/nao**2, 200))

            g_mm = numpy.zeros(g_qm.shape)
            for i0, i1 in lib.prange(0, q.size, blksize):
                fm = gto.fakemol_for_charges(self.base.ommp_obj.cmm[i0:i1])
                ints = df.incore.aux_e2(mol, fm,
                                       intor='int3c2e_ip1')
                g_mm += numpy.einsum('ipqk,k->ipq', ints, q[i0:i1])

            if self.base.ommp_obj.is_amoeba:
                mu = self.base.ommp_obj.static_dipoles
                quad = self.base.ommp_obj.static_quadrupoles

                for i0, i1 in lib.prange(0, q.size, blksize):
                    fm = gto.fakemol_for_charges(self.base.ommp_obj.cmm[i0:i1])
                    ints = df.incore.aux_e2(self.base.mol,
                                            fm,
                                            intor='int3c2e_ipip1')

                    ints += df.incore.aux_e2(self.base.mol,
                                            fm,
                                            intor='int3c2e_ipvip1')

                    g_mm[0] += numpy.einsum('ipqk,ki->pq', ints[0:3], mu[i0:i1])
                    g_mm[1] += numpy.einsum('ipqk,ki->pq', ints[3:6], mu[i0:i1])
                    g_mm[2] += numpy.einsum('ipqk,ki->pq', ints[6:9], mu[i0:i1])

                    A = df.incore.aux_e2(self.base.mol,
                                            fm,
                                            intor='int3c2e_ipipip1')
                    B =   df.incore.aux_e2(self.base.mol,
                                           fm,
                                           intor='int3c2e_ipipvip1')
                    Bt = numpy.einsum('ipqk->iqpk',
                                      numpy.concatenate((B[::3],
                                                         B[1::3],
                                                         B[2::3]), axis=0))
                    ints = A + 2*B + Bt

                    g_mm[0] += numpy.einsum('ipqk,ki->pq', ints[0:9],   quad[i0:i1,[0,1,3,1,2,4,3,4,5]])
                    g_mm[1] += numpy.einsum('ipqk,ki->pq', ints[9:18],  quad[i0:i1,[0,1,3,1,2,4,3,4,5]])
                    g_mm[2] += numpy.einsum('ipqk,ki->pq', ints[18:27], quad[i0:i1,[0,1,3,1,2,4,3,4,5]])

            if self.base.do_pol:
                # Contribution of the converged induced dipoles
                if not self.base.ommp_obj.is_amoeba:
                    mu = self.base.get_mmpol_induced_dipoles()
                else:
                    mu_d, mu_p = self.base.get_mmpol_induced_dipoles()
                    mu = 0.5 * (mu_p + mu_d)

                g_pol = numpy.zeros(g_mm.shape)

                for i0, i1 in lib.prange(0, q.size, blksize):
                    fm = gto.fakemol_for_charges(self.base.ommp_obj.cpol[i0:i1])
                    ints = df.incore.aux_e2(self.base.mol,
                                            fm,
                                            intor='int3c2e_ipip1')

                    ints += df.incore.aux_e2(self.base.mol,
                                            fm,
                                            intor='int3c2e_ipvip1')

                    g_pol[0] += numpy.einsum('ipqk,ki->pq', ints[0:3], mu[i0:i1])
                    g_pol[1] += numpy.einsum('ipqk,ki->pq', ints[3:6], mu[i0:i1])
                    g_pol[2] += numpy.einsum('ipqk,ki->pq', ints[6:9], mu[i0:i1])

                return  g_qm + g_mm + g_pol
            else:
                return g_qm + g_mm

        def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
            if isinstance(self.mol, QMMMPolMole):
                return grad_class.grad_elec(self, mo_energy, mo_coeff, mo_occ, self.mol.QM_atm_lst[atmlst])
            else:
                return grad_class.grad_elec(self, mo_energy, mo_coeff, mo_occ, atmlst)

        def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, domm=False):
            cput0 = (logger.process_clock(), logger.perf_counter())
            if mo_energy is None: mo_energy = self.base.mo_energy
            if mo_coeff is None: mo_coeff = self.base.mo_coeff
            if mo_occ is None: mo_occ = self.base.mo_occ
            if atmlst is None:
                atmlst = self.atmlst
            else:
                self.atmlst = atmlst

            if self.verbose >= logger.WARN:
                self.check_sanity()
            if self.verbose >= logger.INFO:
                self.dump_flags()

            de_elec_qm = self.grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
            de_nuc_qm = self.grad_nuc(atmlst=atmlst)
            if isinstance(self.mol, QMMMPolMole):
                self.de = numpy.zeros((self.mol.natm,3))
                self.de[self.mol.QM_atm_lst[atmlst],:] = de_nuc_qm + de_elec_qm
            else:
                self.de = de_nuc_qm + de_elec_qm

            if self.base.ommp_obj.use_linkatoms:
                la_contrib = self.base.ommp_qm_helper.link_atom_geomgrad(self.base.ommp_obj, self.de)
                self.de += la_contrib['QM']

            if domm:
                dm = self.base.make_rdm1()
                # Charges
                ef_QMatMM = self.base.ef_at_static(dm, mol=self.base.mol)
                force = -numpy.einsum('ij,i->ij', ef_QMatMM, self.base.ommp_obj.static_charges)
                if self.base.ommp_obj.is_amoeba:
                    # Dipoles
                    mu = self.base.ommp_obj.static_dipoles
                    gef_QMatMM = self.base.gef_at_static(dm, mol=self.base.mol)
                    force += -numpy.einsum('ij,i->ij', gef_QMatMM[:,[0,1,3]], mu[:,0])
                    force += -numpy.einsum('ij,i->ij', gef_QMatMM[:,[1,2,4]], mu[:,1])
                    force += -numpy.einsum('ij,i->ij', gef_QMatMM[:,[3,4,5]], mu[:,2])

                    # Quadrupoles
                    quad = self.base.ommp_obj.static_quadrupoles
                    Hef_QMatMM = self.base.Hef_at_static(dm, mol=self.base.mol)
                    force += -numpy.einsum('ij,i->ij', Hef_QMatMM[:,[0,1,2]], quad[:,0]) #xx
                    force += -2.0*numpy.einsum('ij,i->ij', Hef_QMatMM[:,[1,3,4]], quad[:,1]) #xy
                    force += -numpy.einsum('ij,i->ij', Hef_QMatMM[:,[3,6,7]], quad[:,2]) #yy
                    force += -2.0*numpy.einsum('ij,i->ij', Hef_QMatMM[:,[2,4,5]], quad[:,3]) #xz
                    force += -2.0*numpy.einsum('ij,i->ij', Hef_QMatMM[:,[4,7,8]], quad[:,4]) #yz
                    force += -numpy.einsum('ij,i->ij', Hef_QMatMM[:,[5,8,9]], quad[:,5]) #zz

                    # Contribution for the multipoles rotation
                    force += self.base.ommp_obj.rotation_geomgrad(ef_QMatMM, -gef_QMatMM)
                if self.base.do_pol:
                    # Induced dipoles
                    gef_QMatPOL = self.base.gef_at_pol(dm, mol=self.base.mol)
                    if not self.base.ommp_obj.is_amoeba:
                        mu = self.base.get_mmpol_induced_dipoles()
                    else:
                        mu_d, mu_p = self.base.get_mmpol_induced_dipoles()
                        mu = 0.5 * (mu_d + mu_p)

                    force_pol = -numpy.einsum('ij,i->ij', gef_QMatPOL[:,[0,1,3]], mu[:,0])
                    force_pol += -numpy.einsum('ij,i->ij', gef_QMatPOL[:,[1,2,4]], mu[:,1])
                    force_pol += -numpy.einsum('ij,i->ij', gef_QMatPOL[:,[3,4,5]], mu[:,2])
                    force[self.base.ommp_obj.polar_mm] += force_pol

                force += self.base.ommp_obj.polelec_geomgrad()
                force += self.base.ommp_obj.fixedelec_geomgrad()
                force += self.base.ommp_obj.vdw_geomgrad()
                force += self.base.ommp_obj.full_bnd_geomgrad()

                # QM-MM VdW interaction
                if self.base.ommp_qm_helper.use_nonbonded:
                    force += self.base.ommp_qm_helper.vdw_geomgrad(self.base.ommp_obj)['MM']

                if self.base.ommp_obj.use_linkatoms:
                    force += la_contrib['MM']

                self.de_mm = force

            logger.timer(self, 'SCF/MMPol gradients', *cput0)
            self._finalize()
            if domm:
                return self.de, self.de_mm
            else:
                return self.de

        def grad_nuc(self, mol=None, atmlst=None):
            """Compute gradients (on QM atoms) due to the interaction of nuclear
            charges with the MM multipoles and induced dipoles, and from
            molecular-mechanics like term on QM atoms (eg. VdW interactions with
            MM atoms)"""

            if mol is None:
                mol = self.mol

            if getattr(grad_class, 'grad_nuc', None):
                g_qm = grad_class.grad_nuc(self, mol, atmlst)
            else:
                # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise NotImplementedError("For some reason QM method does not have grad_nuc func.")

            g_mm = -numpy.einsum('i,ij->ij',
                                 self.mol.atom_charges(),
                                 self.base.E_mm_at_nucl + self.base.E_pol_at_nucl)

            if(self.base.ommp_qm_helper.use_nonbonded):
                g_vdw = self.base.ommp_qm_helper.vdw_geomgrad(self.base.ommp_obj)['QM']
            else:
                g_vdw = numpy.zeros(g_mm.shape)

            if atmlst is not None:
                g_mm = g_mm[atmlst]
                g_vdw = g_vdw[atmlst]

            return g_qm + g_mm + g_vdw

        def as_scanner(self):
            class QMMMPOL_GradScanner(self.__class__, lib.GradScanner):
                def __init__(self, g):
                    lib.GradScanner.__init__(self, g)

                def __call__(self, mol_or_geom, mm_coords=None, do_mm_grad=False, **kwargs):
                    if isinstance(mol_or_geom, gto.Mole):
                        mol = mol_or_geom
                    else:
                        mol = self.mol.set_geom_(mol_or_geom, inplace=False)

                    if mm_coords is not None:
                        old_coords = self.base.ommp_obj.cmm
                        self.base.ommp_obj.update_coordinates(mm_coords)

                    self.base._qmhelper.update_coord(mol.atom_coords())
                    if self.base.ommp_obj.use_linkatoms:
                        self.base._qmhelper.update_link_atoms_position(self.base.ommp_obj)
                        mol.set_geom_(self.base._qmhelper.cqm, 'B')

                    mf_scanner = self.base

                    e_tot = mf_scanner(mol)
                    if do_mm_grad:
                        de, de_mm = self.kernel(domm=True)
                    else:
                        de = self.kernel()

                    if mm_coords is not None:
                        self.base.ommp_obj.update_coordinates(old_coords)

                    if do_mm_grad:
                        return e_tot, de, de_mm
                    else:
                        return e_tot, de

            return QMMMPOL_GradScanner(self)

    return QMMMPOLG(scf_grad)


class _QMMMPOL(_QMMM):
    pass
class _QMMMPOLGrad(_QMMMGrad):
    pass
