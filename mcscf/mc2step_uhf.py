#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import numpy
import scipy.linalg
import pyscf.lib.logger as logger
import pyscf.scf
from pyscf.mcscf import mc1step_uhf

def kernel(casscf, mo_coeff, tol=1e-7, macro=30, micro=8, \
           ci0=None, verbose=None, **cikwargs):
    if verbose is None:
        verbose = casscf.verbose
    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (time.clock(), time.time())
    log.debug('Start 2-step CASSCF')

    mo = mo_coeff
    nmo = mo[0].shape[1]
    eris = casscf.update_ao2mo(mo)
    e_tot, e_ci, fcivec = casscf.casci(mo, ci0, eris, **cikwargs)
    log.info('CASCI E = %.15g', e_tot)
    if casscf.ncas == nmo:
        return e_tot, e_ci, fcivec, mo
    elast = e_tot
    conv = False
    toloose = casscf.conv_tol_grad
    totmicro = totinner = 0

    t2m = t1m = log.timer('Initializing 2-step CASSCF', *cput0)
    for imacro in range(macro):
        ninner = 0
        t3m = t2m
        for imicro in range(micro):

            casdm1, casdm2 = casscf.fcisolver.make_rdm12s(fcivec, casscf.ncas,
                                                          casscf.nelecas)
            u, dx, g_orb, nin = casscf.rotate_orb(mo, casdm1, casdm2, eris, 0)
            t3m = log.timer('orbital rotation', *t3m)

            mo = list(map(numpy.dot, mo, u))
            #pyscf.scf.chkfile.dump(casscf.chkfile, 'mcscf/mo_coeff', mo)
            casscf.save_mo_coeff(mo, imacro, imicro)

            eris = None # to avoid using too much memory
            eris = casscf.update_ao2mo(mo)
            t3m = log.timer('update eri', *t3m)

            ninner += nin
            norm_dt = numpy.linalg.norm(u-numpy.eye(nmo))
            norm_gorb = numpy.linalg.norm(g_orb)
            log.debug('micro %d, |u-1|=%4.3g, |g[o]|=%4.3g', \
                      imicro, norm_dt, norm_gorb)
            t2m = log.timer('micro iter %d'%imicro, *t2m)
            if norm_dt < toloose or norm_gorb < toloose:
                break

        totinner += ninner
        totmicro += imicro+1

        e_tot, e_ci, fcivec = casscf.casci(mo, fcivec, eris, **cikwargs)
        log.info('macro iter %d (%d ah, %d micro), CASSCF E = %.15g, dE = %.8g,',
                 imacro, ninner, imicro+1, e_tot, e_tot-elast)
        norm_gorb = numpy.linalg.norm(g_orb)
        log.info('                        |grad[o]| = %6.5g',
                 numpy.linalg.norm(g_orb))
        log.timer('CASCI solver', *t3m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        if abs(elast - e_tot) < tol and norm_gorb < toloose:
            conv = True
            break
        else:
            elast = e_tot

    if conv:
        log.info('2-step CASSCF converged in %d macro (%d ah %d micro) steps',
                 imacro+1, totinner, totmicro)
    else:
        log.info('2-step CASSCF not converged, %d macro (%d ah %d micro) steps',
                 imacro+1, totinner, totmicro)
    log.log('2-step CASSCF, energy = %.15g', e_tot)
    log.timer('2-step CASSCF', *cput0)
    return e_tot, e_ci, fcivec, mo


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.mcscf import addons

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()
    emc = kernel(mc1step_uhf.CASSCF(mol, m, 4, (2,1)), m.mo_coeff, verbose=4)[0]
    print(ehf, emc, emc-ehf)
    print(emc - -2.9782774463926618)


    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()
    mc = mc1step_uhf.CASSCF(mol, m, 4, (2,1))
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7), 1)
    emc = mc.mc2step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-75.631870606190233, -75.573930418500652, 0.057940187689581535
    print(emc - -75.573930418500652, emc - -75.648547447838951)


