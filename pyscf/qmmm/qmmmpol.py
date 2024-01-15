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

def ommp_get_3c2eint(mol, auxmol_or_auxbasis, intor='int3c2e', aosym='s1', comp=None, out=None,
                     cintopt=None):
    '''3-center AO integrals (ij|L), where L is the auxiliary basis.

    Kwargs:
        cintopt : Libcint-3.14 and newer version support to compute int3c2e
            without the opt for the 3rd index.  It can be precomputed to
            reduce the overhead of cintopt initialization repeatedly.

            cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e')
    '''
    from pyscf import gto
    from pyscf.gto.moleintor import _get_intor_and_comp, ANG_OF, make_loc, libcgto, make_cintopt
    import ctypes

    if not intor.startswith('int3c2e'):
        raise NotImplementedError("Only int3c2e integrals can be computed in ommp_get_3c2eints; not {:s}.".format(intor))

    if isinstance(auxmol_or_auxbasis, gto.MoleBase):
        auxmol = auxmol_or_auxbasis
    else:
        auxbasis = auxmol_or_auxbasis
        auxmol = addons.make_auxmol(mol, auxbasis)
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)

    # Extract the call of the two lines below
    #  pmol = gto.mole.conc_mol(mol, auxmol)
    #  return pmol.intor(intor, comp, aosym=aosym, shls_slice=shls_slice, out=out)
    intor = mol._add_suffix(intor)
    hermi = 0
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)

    intor, comp = _get_intor_and_comp(intor, comp)
    if any(bas[:,ANG_OF] > 12):
        raise NotImplementedError('cint library does not support high angular (l>12) GTOs')

    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    assert (shls_slice[1] <= nbas and
           shls_slice[3] <= nbas and
           shls_slice[5] <= nbas)

    i0, i1, j0, j1, k0, k1 = shls_slice[:6]
    ao_loc = make_loc(bas, intor)
    if 'ssc' in intor:
        ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'cart')
    elif 'spinor' in intor:
        # The auxbasis for electron-2 is in real spherical representation
        ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'sph')

    naok = ao_loc[k1] - ao_loc[k0]

    if aosym in ('s1',):
        naoi = ao_loc[i1] - ao_loc[i0]
        naoj = ao_loc[j1] - ao_loc[j0]
        shape = (naoi, naoj, naok, comp)
    else:
        aosym = 's2ij'
        nij = ao_loc[i1]*(ao_loc[i1]+1)//2 - ao_loc[i0]*(ao_loc[i0]+1)//2
        shape = (nij, naok, comp)

    if 'spinor' in intor:
        mat = numpy.ndarray(shape, numpy.complex128, out, order='F')
        drv = libcgto.GTOr3c_drv
        fill = getattr(libcgto, 'GTOr3c_fill_'+aosym)
    else:
        mat = numpy.ndarray(shape, numpy.double, out, order='F')
        drv = libcgto.GTOnr3c_drv
        fill = getattr(libcgto, 'GTOnr3c_fill_'+aosym)

    if mat.size > 0:
        # Generating opt for all indices leads to large overhead and poor OMP
        # speedup for solvent model and COSX functions. In these methods,
        # the third index of the three center integrals corresponds to a
        # large number of grids. Initializing the opt for the third index is
        # not necessary.
        if cintopt is None:
            # int3c2e opt without the 3rd index.
            cintopt = make_cintopt(atm, bas[:max(i1, j1)], env, intor)

        drv(getattr(libcgto, intor), fill,
            mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
            (ctypes.c_int*6)(*(shls_slice[:6])),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p))
    return mat

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

    def atom_charges(self):
        full_ac = numpy.empty([self.natm], dtype=numpy.int32)
        full_ac[self.QM_atm_lst] = self.molQM.atom_charges()
        full_ac[self.MM_atm_lst] = self.ommp_obj.zmm[self.unfrozenMM]
        return full_ac

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
            """Return the number of float64 which can be safely allocated in  memory."""
            return (self.max_memory - lib.current_memory()[0]) * 1e6 / 8 # In number of float64

        @property
        def ommp_qm_helper(self, inmol=None):
            """Return the qm_helper object exposed by OpenMMPol, if the coordinates are changed
            in the PySCF mol, the object is updated accordingly. If the object still doesen't
            exists, it is created from PySCF mol"""
            if inmol is None:
                mol = self.mol
            else:
                mol = inmol
            if not hasattr(self, '_qmhelper'):
                qmat_q = mol.atom_charges()
                qmat_c = mol.atom_coords()
                qmat_z = [Symbol2Z[a[0]] for a in mol._atom]
                self._qmhelper = ommp.OMMPQmHelper(qmat_c, qmat_q, qmat_z)
            # TODO probably here is more efficient to use crdhash, instead.
            elif not numpy.allclose(mol.atom_coords(), self._qmhelper.cqm):
                self._qmhelper.update_coord(mol.atom_coords())
            return self._qmhelper

        @property
        def crdhash(self):
            """Compute an hash of the coordinaates of the QM part of
            the system; used to check if some quantities (eg. stored
            integrals) should be updated."""
            # TODO probably conversion to string is a bit brutal here
            return hash(self.mol.atom_coords().tostring()+\
                        self.ommp_obj.cmm.tostring())

        def check_crdhash(self):
            """Use crdhash to check for a change in QM coordinates. If such
            a change happened, the quantities which are no longer valid are
            deleted"""
            perishable_attr = ['e_mmpol',
                               '_int1e_screening',
                               '_ef_integrals']

            if hasattr(self, '_crdhash'):
                if self.crdhash == self._crdhash:
                    return True
                else:
                    for pa in perishable_attr:
                        if hasattr(self, pa):
                            delattr(self, pa)
                    self._crdhash = self.crdhash
                    return False
            else:
                for pa in perishable_attr:
                    if hasattr(self, pa):
                        delattr(self, pa)
                self._crdhash = self.crdhash
                return False

        def _compute_screening_list(self, thr=1e-9):
            """Compute a mask to remove from the calculation components of
            electrostatic atomic integrals  which are always close to zero.
            This is used to compress the electric field integrals and save
            them during the SCF. Positive and negative mask are generated
            and accessed through interface function"""
            S = self.mol.intor('int1e_ipnuc_sph')
            S = numpy.linalg.norm(S, axis=0)
            nbas = S.shape[0]
            S = S.flatten()
            keeplist = []
            removelist = []

            for i, s in enumerate(S):
                if abs(s) > thr:
                    keeplist += [i]
                else:
                    removelist += [i]
            # print("Screening integrals {:d}/{:d} {:.2f} %".format(len(sclist), len(S), len(sclist)/len(S) * 100))
            self._int1e_screening_keep = keeplist
            self._int1e_screening_rmv = removelist

            return

        @property
        def int1e_screening(self):
            """Interface function to get positive screening list of electrostatic integrals."""
            self.check_crdhash()
            if not hasattr(self, '_int1e_screening_keep'):
                self._compute_screening_list()
            return self._int1e_screening_keep

        @property
        def int1e_screening_neg(self):
            """Interface function to get negative screening list of electrostatic integrals."""
            self.check_crdhash()
            if not hasattr(self, '_int1e_screening_rmv'):
                self._compute_screening_list()
            return self._int1e_screening_rmv

        @property
        def ef_integrals(self):
            """ Return the compressed electric field atomic integrals, to be contracted with densities
            and/or induced dipoles during the SCF procedure. If they haven't been computed before,
            they are computed and stored"""
            # TODO if the available memory is not enough for saving them they should be computed
            # ecery time on the fly in chunk.
            self.check_crdhash()
            if not hasattr(self, '_ef_integrals'):
                fm = self.fakemol_static
                nao = self.mol.nao
                nmmp = fm.natm

                polatm = []
                mmatm = []
                for i in range(nmmp):
                    if i in self.ommp_obj.polar_mm:
                        polatm += [i]
                    else:
                        mmatm += [i]
                nmm  = len(mmatm)
                npol = len(polatm)

                memblksize = int((0.8*self.fakeget_mem-3*(nmm+npol)*len(self.int1e_screening)) / (3*nao*nao))
                blksize = min(memblksize, 2500//3)

                ommp.time_push()
                efi_pol = numpy.empty([3,npol,len(self.int1e_screening)])
                for j0, j1 in lib.prange(0, npol, blksize):
                    _fm = gto.fakemol_for_charges(fm.atom_coords()[polatm[j0:j1]])
                    _efi = ommp_get_3c2eint(self.mol, _fm, intor='int3c2e_ip1')
                    _efi = _efi.transpose()
                    _efi = _efi.reshape(3, (j1-j0), nao*nao, order='C')
                    _efi = _efi[:,:,self.int1e_screening]
                    efi_pol[:,j0:j1,:] = _efi
                efi_pol = efi_pol.reshape(3*npol, len(self.int1e_screening))

                self._ef_integrals = {'pol': efi_pol}
                ommp.time_pull("Computing and storing EF integrals")
            return self._ef_integrals

        def ef_pol_contract(self, dm=None, dipoles=None, force_direct_alg=False):
            """Contract electric field integrals with density matrix or dipole using
            the stored integrals if possible or direct strategy when it is not possible
            to use stored integrals.
            Electric field integrals
            <mu|\hat(E)|nu> = <\mu|\\nabla r^{-1}|\\nu> =
                            = (\\nabla\mu\\nu|\delta) + (\mu\\nabla\\nu|\delta)
                            = (\\nabla\mu\\nu|\delta) + (\\nabla\mu\\nu|\delta)^\dagger
            at coordinates of MM atoms."""

            mol = self.mol
            nmmp = self.ommp_obj.pol_atoms
            fm = self.fakemol_pol
            nao = mol.nao

            memblksize = int((0.8*self.fakeget_mem) / (3*nao*nao))
            blksize = min(memblksize, 2500//3)

            if force_direct_alg or not self.store_ef_integrals:
                direct_alg = True
            else:
                direct_alg = False

            if dipoles is None and dm is None:
                raise NotImplementedError("Raw integrals cannot be returned, just contracted ones.")

            elif dipoles is not None and dm is None:
                if direct_alg:
                    Ef_dc = numpy.zeros([nao,nao])
                    for j0, j1, in lib.prange(0, fm.natm, blksize):
                        _fm = gto.fakemol_for_charges(fm.atom_coords()[j0:j1])
                        Ef_dc -= numpy.einsum('mnji,ji->mn',
                                              df.incore.aux_e2(mol, _fm, intor='int3c2e_ip1').transpose((1,2,3,0)),
                                              dipoles[j0:j1],
                                              optimize='optimal')
                else:
                    Ef_dc = numpy.zeros(nao*nao)
                    _dipoles = numpy.ascontiguousarray(dipoles.transpose().reshape([1,nmmp*3]))
                    Ef_dc[self.int1e_screening] = -numpy.ravel(lib.numpy_helper.ddot(_dipoles, self.ef_integrals['pol']))
                    Ef_dc = Ef_dc.reshape(nao,nao)

                return Ef_dc + Ef_dc.T

            elif dm is not None:
                if direct_alg:
                    Ef_dc = numpy.empty([3,fm.natm])

                    _dm = dm.reshape(nao*nao, 1)
                    for j0, j1 in lib.prange(0, fm.natm, blksize):
                        _fm = gto.fakemol_for_charges(fm.atom_coords()[j0:j1])
                        _ints = ommp_get_3c2eint(mol, _fm, intor='int3c2e_ip1')
                        _ints = _ints.transpose()
                        _ints = _ints.reshape(3, (j1-j0), nao*nao)

                        for j in range(3):
                            Ef_dc[j,j0:j1] = numpy.ravel(lib.numpy_helper.ddot(_ints[j], _dm))

                    Ef_dc = Ef_dc.transpose()
                else:
                    scrlist = self.int1e_screening
                    ommp.time_push()
                    _dm = dm.reshape([nao*nao,1])[scrlist]
                    Ef_dc = lib.numpy_helper.ddot(self.ef_integrals['pol'], _dm).reshape(3,nmmp)
                    ommp.time_pull("Computing electric field")
                    Ef_dc = Ef_dc.transpose()

                Ef_dc *= 2.

                if dipoles is None:
                    return Ef_dc
                else:
                    return -numpy.einsum('ji,ji->', Ef_dc, dipoles)
            return None

            # """Electric field gradients integrals
            # <mu|\hat(G)|nu> = <\mu|\\nabla\\nabla r^{-1}|\\nu> =
            #                 = ... =
            #                 = (\\nabla\\nabla\mu\\nu|\delta) +
            #                   (\\nabla\\nabla\mu\\nu|\delta)^\dagger +
            #                   2 (\\nabla\mu\\nabla\\nu|\delta)
            # at coordinates of MM atoms. Those integrals have formally
            # 9 different components, but we use it symmetrized and
            # compressed."""
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

            # """Electric field Hessian integrals
            # <mu|\hat(G)|nu> = <\mu|\\nabla \\nabla\\nabla r^{-1}|\\nu> =
            #                 = ... =
            #                 = (\\nabla\\nabla\\nabla\mu\\nu|\delta) +
            #                   (\\nabla\\nabla\\nabla\mu\\nu|\delta)^\dagger +
            #                   3 (\\nabla\\nabla\mu\\nabla\\nu|\delta) +
            #                   3 (\\nabla\\nabla\mu\\nabla\\nu|\delta)^\dagger
            # at coordinates of MM atoms. Those integrals have formally
            # 27 different components, but we use it symmetrized and
            # compressed."""
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

        @property
        def store_ef_integrals(self):
            return True


        def ef_at_pol(self, dm, mol=None, exclude_nuclei=False):
            """Compute the electric field generated by the QM system with density dm
            at polarizable sites"""

            ef = self.ef_pol_contract(dm=dm)
            if not exclude_nuclei:
                ef += self.ef_nucl_at_pol
            return ef

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

            ommp.time_push()
            vhf = method_class.get_veff(self, mol, dm, *args, **kwargs)
            ommp.time_pull("Veff QM")
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
                ommp.time_push()
                current_ef = self.ef_at_pol(dm_tot, mol=mol)
                ommp.time_pull("Computing EF (scf)")

                # 2. update the induced dipoles
                ommp.time_push()
                self.ommp_obj.set_external_field(current_ef)
                ommp.time_pull("Solving LS")

                # 3.1. get the induced dipoles
                if not self.ommp_obj.is_amoeba:
                    current_ipds = self.get_mmpol_induced_dipoles()
                else:
                    current_ipds_d, current_ipds_p = self.get_mmpol_induced_dipoles()
                    current_ipds = (current_ipds_d+current_ipds_p) / 2

                # 3.2. compute the induced dipoles term in the Hamiltonian
                ommp.time_push()
                v_mmpol = self.ef_pol_contract(dipoles=current_ipds)
                ommp.time_pull("Veff Pol")

                # 4. Compute the MMPol contribution to energy
                ommp.time_push()
                if not self.ommp_obj.is_amoeba:
                    e_mmpol = -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds)
                else:
                    e_mmpol = -0.5 * numpy.einsum('nm,nm', current_ef, current_ipds_d)
                e_mmpol += self.ommp_obj.get_polelec_energy()
                ommp.time_pull('MMPol Energy')

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
            ommp.time_push()
            if getattr(method_class, 'get_hcore', None):
                h1e = method_class.get_hcore(self, mol)
            else:
                # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise RuntimeError('openMMPol cannot be applied here in post-HF methods')
            ommp.time_pull('HCore QM')

            ommp.time_push()
            nao = mol.nao
            q = self.ommp_obj.static_charges

            memblksize = int((0.8*self.fakeget_mem) / (nao*nao))
            blksize = min(memblksize, 2500)

            self.h1e_mmpol = numpy.zeros([nao, nao])

            ommp.time_push()

            for i0, i1 in lib.prange(0, self.ommp_obj.mm_atoms, blksize):
                _fm = gto.fakemol_for_charges(self.ommp_obj.cmm[i0:i1])
                _ints = ommp_get_3c2eint(mol, _fm, intor='int3c2e')
                _ints = _ints.transpose()
                _ints = _ints.reshape(i1-i0, nao*nao, order='C')
                self.h1e_mmpol -= lib.numpy_helper.ddot(q[i0:i1].reshape(1, i1-i0), _ints).reshape(nao, nao)
            ommp.time_pull("HCore: q/v")

            if self.ommp_obj.is_amoeba or self.do_pol:
                ommp.time_push()
                memblksize = int((0.8*self.fakeget_mem) / (3*nao*nao))
                blksize = min(memblksize, 2500//3)

                ommp.time_push()
                map_oo2no = numpy.empty(self.ommp_obj.mm_atoms, dtype=numpy.int64)
                map_oo2no[0:self.ommp_obj.pol_atoms] = self.ommp_obj.polar_mm
                map_oo2no[self.ommp_obj.pol_atoms:] = [i for i in range(self.ommp_obj.mm_atoms) if i not in self.ommp_obj.polar_mm]
                ommp.time_pull("Map oo2no creation")

                no_cmm = self.ommp_obj.cmm[map_oo2no]
                no_mus = self.ommp_obj.static_dipoles[map_oo2no]

                if self.store_ef_integrals:
                    self._ef_integrals = {'pol': numpy.empty([3,self.ommp_obj.pol_atoms,len(self.int1e_screening)])}

                for i0, i1 in lib.prange(0, self.ommp_obj.mm_atoms, blksize):
                    _fm = gto.fakemol_for_charges(no_cmm[i0:i1])

                    _ints = ommp_get_3c2eint(mol, _fm, intor='int3c2e_ip1')
                    _ints = _ints.transpose()
                    _ints = _ints.reshape(3*(i1-i0), nao*nao, order='C')

                    _mu = no_mus[i0:i1].transpose().reshape((1,3*(i1-i0)))
                    Ef = numpy.ravel(lib.numpy_helper.ddot(_mu, _ints))
                    Ef = Ef.reshape(nao,nao) + Ef.reshape(nao,nao).T
                    # Old behavior
                    # Ef = Ef.reshape(nao*nao)
                    # Ef[self.int1e_screening_neg] = 0.0
                    # Ef = Ef.reshape(nao*nao)
                    self.h1e_mmpol -= Ef

                    # Save integrals for later use during SCF, if possible
                    if self.store_ef_integrals:
                       i1_pol = min(i1, self.ommp_obj.pol_atoms)
                       _ints = _ints.reshape(3, (i1-i0), nao*nao, order='C')
                       if i1_pol != i1:
                           _ints = _ints[:,:i1_pol-i0,:]
                       _ints = _ints[:,:,self.int1e_screening]
                       self._ef_integrals['pol'][:,i0:i1_pol,:] = _ints
                self._ef_integrals['pol'] = self._ef_integrals['pol'].reshape(3*self.ommp_obj.pol_atoms, len(self.int1e_screening))

                ommp.time_pull("HCore: mu/E")

            if self.ommp_obj.is_amoeba:
                ommp.time_push()
                memblksize = int((0.8*self.fakeget_mem) / (9*nao*nao))
                blksize = min(memblksize, 2500//9)
                for i0, i1 in lib.prange(0, self.ommp_obj.mm_atoms, blksize):
                    _fm = gto.fakemol_for_charges(self.ommp_obj.cmm[i0:i1])

                    nni_j = ommp_get_3c2eint(mol, _fm, intor='int3c2e_ipip1')
                    nni_j = nni_j.transpose()
                    nni_j = nni_j.reshape(9*(i1-i0), nao*nao, order='C')

                    ni_nj = ommp_get_3c2eint(mol, _fm, intor='int3c2e_ipvip1')
                    ni_nj = ni_nj.transpose()
                    ni_nj = ni_nj.reshape(9*(i1-i0), nao*nao, order='C')

                    tmp_quad = numpy.zeros([9,i1-i0])
                    tmp_quad[[1,2,5],:] = self.ommp_obj.static_quadrupoles[i0:i1,[1,3,4]].T
                    tmp_quad[[0,4,8],:] = self.ommp_obj.static_quadrupoles[i0:i1,[0,2,5]].T
                    tmp_quad[[3,6,7],:] = tmp_quad[[1,2,5],:]
                    tmp_quad = tmp_quad.reshape([1,(i1-i0)*9])

                    Gef  = numpy.ravel(lib.numpy_helper.ddot(tmp_quad, nni_j))
                    Gef += numpy.ravel(lib.numpy_helper.ddot(tmp_quad, ni_nj))
                    self.h1e_mmpol -= Gef.reshape(nao,nao) + Gef.reshape(nao,nao).T

                ommp.time_pull("HCore: Q/G")
            ommp.time_pull('HCore MMPol')
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
            """Energy of the electronic part of the system"""
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

            self.check_crdhash()

            if not hasattr(self, 'e_mmpol'):
                self.e_mmpol = self.ommp_obj.get_full_bnd_energy()
                self.e_mmpol += self.ommp_obj.get_full_ele_energy()
                self.e_mmpol += self.ommp_obj.get_vdw_energy()
                # QM-MM VdW interaction
                if(self.ommp_qm_helper.use_nonbonded):
                    self.e_mmpol += self.ommp_qm_helper.vdw_energy(self.ommp_obj)

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

                        vd = self.ef_pol_contract(dipoles=current_ipds)
                        v_mmpol += [vd]

                    v_mmpol = numpy.array(v_mmpol)

                    v += v_mmpol

                return v

            return vind_mmpol

        def energy_analysis(self):
            """Simple analysis of the energy terms composing the total
            energy, mainly here for debug pourposes"""
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
            changes the coordinates of ila, and is only useful in rare cases
            as the functionality is provided in a more clean fashon by json input."""
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
            """Return the number of float64 which can be safely allocated in  memory."""
            return (self.max_memory - lib.current_memory()[0]) * 1e6 / 8 # In number of float64

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

            if not hasattr(self, 'g_mm'):
                self.grad_qmmm()
            g_qm += self.g_mm
            if hasattr(self, 'g_pol'):
                g_qm += self.g_pol
            return g_qm

        def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
            if isinstance(self.mol, QMMMPolMole):
                return grad_class.grad_elec(self, mo_energy, mo_coeff, mo_occ, self.mol.QM_atm_lst[atmlst])
            else:
                return grad_class.grad_elec(self, mo_energy, mo_coeff, mo_occ, atmlst)

        def grad_qmmm(self, mol=None, domm=False):
            """Compute the term of QM/MMPol derivatives coupling involving the
            electrostatic integrals and stores them in memory."""

            if mol is None:
                mol = self.mol

            nao = mol.nao
            memblksize = int((0.8*self.fakeget_mem) / (3*nao*nao))
            blksize = min(memblksize, 2500//3)

            self.g_mm = numpy.zeros([3, nao*nao])
            if self.base.do_pol:
                self.g_pol = numpy.zeros([3, nao*nao])
            if domm:
                self.qmmm_force_mm = numpy.zeros(self.base.ommp_obj.cmm.shape)
                _dm = self.base.make_rdm1().reshape((nao*nao,1))

            # Charges
            ommp.time_push()
            q = self.base.ommp_obj.static_charges
            if domm:
                # This is needed for multipoles torque
                ef_at_MM = numpy.zeros([q.size,3])

            for i0, i1 in lib.prange(0, q.size, blksize):
                ommp.time_push()
                fm = gto.fakemol_for_charges(self.base.ommp_obj.cmm[i0:i1])

                _ints = ommp_get_3c2eint(mol, fm, intor='int3c2e_ip1')
                _ints = _ints.transpose()
                _ints = _ints.reshape(3, (i1-i0), nao*nao, order='C')
                _q = q[i0:i1].reshape(1,i1-i0)
                ommp.time_pull('Integrals qs')
                ommp.time_push()
                ommp.time_push()
                for j in range(3):
                    self.g_mm[j] += numpy.ravel(lib.numpy_helper.ddot(_q, _ints[j]))
                ommp.time_pull("Integral qs - coreh")

                if domm:
                    ommp.time_push()
                    ef = numpy.zeros([3,i1-i0])
                    for j in range(3):
                        ef[j] = 2*numpy.ravel(lib.numpy_helper.ddot(_ints[j], _dm))
                    ommp.time_pull("Integral qs - mm 1")
                    ommp.time_push()
                    # Add nuclear component
                    ef += self.base.ef_nucl_at_static[i0:i1].T
                    ommp.time_pull("Integral qs - mm 2")
                    ommp.time_push()
                    # Save the electric field for later use (multipoles torque)
                    ef_at_MM[i0:i1,:] = ef.T
                    ommp.time_pull("Integral qs - mm 3")
                    ommp.time_push()
                    # Contract qith charges
                    self.qmmm_force_mm[i0:i1] -= numpy.einsum('ji,i->ij', ef, q[i0:i1])
                    ommp.time_pull("Integral qs - mm 4")
                ommp.time_pull('Contraction qs')
            ommp.time_pull('Charges')

            # AMOEBA also have fixed dipoles
            if self.base.ommp_obj.is_amoeba or self.base.do_pol:
                # fixed dipoles
                ommp.time_push()
                memblksize = int((0.8*self.fakeget_mem) / (9*nao*nao))
                blksize = min(memblksize, 2500//9)
                # Create a map to put all polarizable atoms at the beginning
                # that is contiguously to compute simultaneously the components
                # of g_mm due to static dipoles and of g_pol due to induce dipoles.
                ommp.time_push()
                map_oo2no = numpy.empty(self.base.ommp_obj.mm_atoms, dtype=numpy.int64)
                map_oo2no[0:self.base.ommp_obj.pol_atoms] = self.base.ommp_obj.polar_mm
                map_oo2no[self.base.ommp_obj.pol_atoms:] = [i for i in range(self.base.ommp_obj.mm_atoms) if i not in self.base.ommp_obj.polar_mm]
                ommp.time_pull("Map oo2no creation")

                no_cmm = self.base.ommp_obj.cmm[map_oo2no]
                no_mus = self.base.ommp_obj.static_dipoles[map_oo2no]

                if self.base.do_pol:
                    # Contribution of the converged induced dipoles
                    if not self.base.ommp_obj.is_amoeba:
                        mui = self.base.get_mmpol_induced_dipoles()
                    else:
                        mu_d, mu_p = self.base.get_mmpol_induced_dipoles()
                        mui = 0.5 * (mu_p + mu_d)

                if domm:
                    # To have consistency with screened integrals used before
                    _dm_pol = _dm.copy()
                    _dm_pol[self.base.int1e_screening_neg] = 0.0
                    if self.base.ommp_obj.is_amoeba:
                        # This is needed for multipoles torque
                        Gef_at_MM = numpy.zeros([q.size,6])

                if self.base.ommp_obj.is_amoeba:
                    maxidx = self.base.ommp_obj.mm_atoms
                else:
                    maxidx = self.base.ommp_obj.pol_atoms

                for i0, i1 in lib.prange(0, maxidx, blksize):
                    ommp.time_push()
                    fm = gto.fakemol_for_charges(no_cmm[i0:i1])

                    _ints1 = ommp_get_3c2eint(mol, fm, intor='int3c2e_ipip1')
                    _ints1 = _ints1.transpose()
                    _ints1 = _ints1.reshape(3, 3*(i1-i0), nao*nao)
                    _ints2 = ommp_get_3c2eint(mol, fm, intor='int3c2e_ipvip1')
                    _ints2 = _ints2.transpose()
                    _ints2 = _ints2.reshape(3, 3*(i1-i0), nao*nao)
                    ommp.time_pull('Integrals mus and mui')

                    ommp.time_push()
                    # Flag to check if there are polarizabilities to contract
                    contract_pol = self.base.do_pol and i0 < self.base.ommp_obj.pol_atoms

                    if self.base.ommp_obj.is_amoeba:
                        _mu = numpy.ascontiguousarray(no_mus[i0:i1].transpose().reshape(1,3*(i1-i0)))
                        for j in range(3):
                            self.g_mm[j] += numpy.ravel(lib.numpy_helper.ddot(_mu, _ints1[j]))
                            self.g_mm[j] += numpy.ravel(lib.numpy_helper.ddot(_mu, _ints2[j]))

                    if contract_pol:
                        i1_pol = min(i1, self.base.ommp_obj.pol_atoms)
                        if i1_pol == i1:
                            _mui = numpy.ascontiguousarray(mui[i0:i1_pol].transpose().reshape(1,3*(i1_pol-i0)))
                        else:
                            _mui = numpy.zeros([1,3*(i1-i0)])
                            for i in range(3):
                                _mui[0,(i1-i0)*i:(i1-i0)*i+(i1_pol-i0)] = mui[i0:i1_pol,i]
                        for j in range(3):
                            self.g_pol[j] += numpy.ravel(lib.numpy_helper.ddot(_mui, _ints1[j]))
                            self.g_pol[j] += numpy.ravel(lib.numpy_helper.ddot(_mui, _ints2[j]))

                    if domm:
                        if self.base.ommp_obj.is_amoeba:
                            Gef = numpy.zeros([3, 3*(i1-i0)])
                            for j in range(3):
                                Gef[j]  = numpy.ravel(lib.numpy_helper.ddot(_ints1[j], _dm))
                                Gef[j] += numpy.ravel(lib.numpy_helper.ddot(_ints2[j], _dm))
                            Gef = Gef.reshape(9,i1-i0)
                            Gef[[1,2,5]] += Gef[[3,6,7]]
                            Gef[[0,4,8]] *= 2
                            Gef = Gef[[0,1,4,2,5,8]]
                            Gef -= self.base.gef_nucl_at_static[map_oo2no[i0:i1]].T
                            Gef_at_MM[map_oo2no[i0:i1],:] = Gef.T
                            self.qmmm_force_mm[map_oo2no[i0:i1]] += -numpy.einsum('ji,i->ij', Gef[[0,1,3]], no_mus[i0:i1,0])
                            self.qmmm_force_mm[map_oo2no[i0:i1]] += -numpy.einsum('ji,i->ij', Gef[[1,2,4]], no_mus[i0:i1,1])
                            self.qmmm_force_mm[map_oo2no[i0:i1]] += -numpy.einsum('ji,i->ij', Gef[[3,4,5]], no_mus[i0:i1,2])

                        if contract_pol:
                            if i1_pol == i1:
                                Gef = numpy.zeros([3, 3*(i1-i0)])
                                for j in range(3):
                                    Gef[j]  = numpy.ravel(lib.numpy_helper.ddot(_ints1[j], _dm_pol))
                                    Gef[j] += numpy.ravel(lib.numpy_helper.ddot(_ints2[j], _dm_pol))
                                Gef = Gef.reshape(9,i1-i0)
                                Gef[[1,2,5]] += Gef[[3,6,7]]
                                Gef[[0,4,8]] *= 2
                                Gef = Gef[[0,1,4,2,5,8]]
                                Gef -= self.base.gef_nucl_at_pol[i0:i1].T
                                self.qmmm_force_mm[self.base.ommp_obj.polar_mm[i0:i1]] += -numpy.einsum('ji,i->ij', Gef[[0,1,3]], mui[i0:i1,0])
                                self.qmmm_force_mm[self.base.ommp_obj.polar_mm[i0:i1]] += -numpy.einsum('ji,i->ij', Gef[[1,2,4]], mui[i0:i1,1])
                                self.qmmm_force_mm[self.base.ommp_obj.polar_mm[i0:i1]] += -numpy.einsum('ji,i->ij', Gef[[3,4,5]], mui[i0:i1,2])
                            else:
                                Gef = numpy.zeros([3, 3*(i1_pol-i0)])
                                for j in range(3):
                                    Gef[j]  = numpy.ravel(lib.numpy_helper.ddot(_ints1[j,:3*(i1_pol-i0)], _dm_pol))
                                    Gef[j] += numpy.ravel(lib.numpy_helper.ddot(_ints2[j,:3*(i1_pol-i0)], _dm_pol))
                                Gef = Gef.reshape(9,i1_pol-i0)
                                Gef[[1,2,5]] += Gef[[3,6,7]]
                                Gef[[0,4,8]] *= 2
                                Gef = Gef[[0,1,4,2,5,8]]
                                Gef -= self.base.gef_nucl_at_pol[i0:i1_pol].T
                                self.qmmm_force_mm[self.base.ommp_obj.polar_mm[i0:i1_pol]] += -numpy.einsum('ji,i->ij', Gef[[0,1,3]], mui[i0:i1_pol,0])
                                self.qmmm_force_mm[self.base.ommp_obj.polar_mm[i0:i1_pol]] += -numpy.einsum('ji,i->ij', Gef[[1,2,4]], mui[i0:i1_pol,1])
                                self.qmmm_force_mm[self.base.ommp_obj.polar_mm[i0:i1_pol]] += -numpy.einsum('ji,i->ij', Gef[[3,4,5]], mui[i0:i1_pol,2])

                    ommp.time_pull('Contraction mus and mui')
                ommp.time_pull('Static and induced dipoles')

                # Compute torque of multipoles, this is computed by openmmpol using electric field and
                # electric field gradients generated by the QM density.
                ommp.time_push()
                self.qmmm_force_mm += self.base.ommp_obj.rotation_geomgrad(ef_at_MM, -Gef_at_MM)
                ommp.time_pull('Multipoles torque')

                # fixed quadrupoles
                ommp.time_push()
                memblksize = int((0.8*self.fakeget_mem) / (27*nao*nao))
                blksize = min(memblksize, 2500//27)
                quad = self.base.ommp_obj.static_quadrupoles
                for i0, i1 in lib.prange(0, q.size, blksize):
                    ommp.time_push()
                    fm = gto.fakemol_for_charges(self.base.ommp_obj.cmm[i0:i1])

                    _ints1 = ommp_get_3c2eint(mol, fm, intor='int3c2e_ipipip1')
                    _ints1 = _ints1.transpose()
                    _ints1 = _ints1.reshape(27, (i1-i0), nao*nao)

                    _ints2 = ommp_get_3c2eint(mol, fm, intor='int3c2e_ipipvip1')
                    _ints2 = _ints2.transpose()
                    _ints2 = _ints2.reshape(27, (i1-i0), nao*nao)

                    ommp.time_pull('Integrals Qs')
                    ommp.time_push()
                    _quad = quad[i0:i1,[0,1,3,1,2,4,3,4,5]].transpose().flatten().reshape(1,9*(i1-i0))

                    for idx in range(3):
                        self.g_mm[idx] += numpy.ravel(lib.numpy_helper.ddot(_quad, _ints1[idx*9:(idx+1)*9].reshape(9*(i1-i0), nao*nao)))
                        self.g_mm[idx] += 2*numpy.ravel(lib.numpy_helper.ddot(_quad, _ints2[idx*9:(idx+1)*9].reshape(9*(i1-i0), nao*nao)))
                        self.g_mm[idx] += numpy.ravel(lib.numpy_helper.ddot(_quad, _ints2[idx::3].reshape(9*(i1-i0), nao*nao)).reshape(nao,nao).transpose().reshape(nao*nao))

                    if domm:
                        Hef = numpy.zeros([27, i1-i0])
                        for j in range(27):
                            Hef[j] = numpy.ravel(lib.numpy_helper.ddot(_ints1[j], _dm))
                            Hef[j] += numpy.ravel(lib.numpy_helper.ddot(_ints2[j], _dm))*3

                        # Compress and make symmetric
                        Hef[1] =  (Hef[1] +  Hef[3] +  Hef[9]) / 3
                        Hef[2] =  (Hef[2] +  Hef[6] +  Hef[18]) / 3
                        Hef[4] =  (Hef[4] +  Hef[10] + Hef[12]) / 3
                        Hef[5] =  (Hef[5] +  Hef[7] +  Hef[11] + Hef[15] + Hef[19] + Hef[21]) / 6
                        Hef[8] =  (Hef[8] +  Hef[20] + Hef[24]) / 3
                        Hef[14] = (Hef[14] + Hef[16] + Hef[22]) / 3
                        Hef[17] = (Hef[17] + Hef[23] + Hef[25]) / 3
                        Hef = Hef[[0,1,2,4,5,8,13,14,17,26]] * 2.
                        Hef += self.base.Hef_nucl_at_static[i0:i1].T

                        self.qmmm_force_mm[i0:i1] +=     -numpy.einsum('ji,i->ij', Hef[[0,1,2]], quad[i0:i1,0]) #xx
                        self.qmmm_force_mm[i0:i1] += -2.0*numpy.einsum('ji,i->ij', Hef[[1,3,4]], quad[i0:i1,1]) #xy
                        self.qmmm_force_mm[i0:i1] +=     -numpy.einsum('ji,i->ij', Hef[[3,6,7]], quad[i0:i1,2]) #yy
                        self.qmmm_force_mm[i0:i1] += -2.0*numpy.einsum('ji,i->ij', Hef[[2,4,5]], quad[i0:i1,3]) #xz
                        self.qmmm_force_mm[i0:i1] += -2.0*numpy.einsum('ji,i->ij', Hef[[4,7,8]], quad[i0:i1,4]) #yz
                        self.qmmm_force_mm[i0:i1] +=     -numpy.einsum('ji,i->ij', Hef[[5,8,9]], quad[i0:i1,5]) #zz

                    ommp.time_pull('Contraction Qs')
                ommp.time_pull('Static quadrupoles')

            self.g_mm = self.g_mm.reshape(3, nao, nao)
            self.g_mm = numpy.einsum('imn->inm', self.g_mm)

            if self.base.do_pol:
                mask = numpy.ones(nao*nao, dtype=bool)
                mask[self.base.int1e_screening] = False
                self.g_pol[:,mask] = 0.0
                self.g_pol = self.g_pol.reshape(3, nao, nao)
                self.g_pol = numpy.einsum('imn->inm', self.g_pol)

            return


        def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, domm=False):
            ommp.time_push()
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

            # Compute the derivatives of QM/MM copling energy
            ommp.time_push()
            self.grad_qmmm(domm=domm)
            ommp.time_pull("Derivatives of QM/MM coupling")
            ommp.time_push()
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
            ommp.time_pull("Derivatives of QM part")

            if domm:
                ommp.time_push()
                force = self.qmmm_force_mm
                ommp.time_push()
                force += self.base.ommp_obj.polelec_geomgrad()
                force += self.base.ommp_obj.fixedelec_geomgrad()
                ommp.time_pull("MM Electrostatic Forces")
                ommp.time_push()
                force += self.base.ommp_obj.vdw_geomgrad()
                ommp.time_pull("MM VdW Forces")
                ommp.time_push()
                force += self.base.ommp_obj.full_bnd_geomgrad()
                ommp.time_pull("MM Bonded Forces")

                # QM-MM VdW interaction
                if self.base.ommp_qm_helper.use_nonbonded:
                    force += self.base.ommp_qm_helper.vdw_geomgrad(self.base.ommp_obj)['MM']

                if self.base.ommp_obj.use_linkatoms:
                    force += la_contrib['MM']

                self.de_mm = force
                ommp.time_pull("Force on MM atoms")

            logger.timer(self, 'SCF/MMPol gradients', *cput0)
            self._finalize()
            ommp.time_pull("Forces QM/MMPol")
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
