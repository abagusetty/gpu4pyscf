# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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
"""
Nonadiabatic derivetive coupling matrix element calculation is now in experiment.
This module is under development.
"""

from functools import reduce
import cupy as cp
import numpy as np
from pyscf import lib
import pyscf
from gpu4pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad_cpu
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf
from pyscf import __config__
from gpu4pyscf.lib import utils
from gpu4pyscf import tdscf
from pyscf.scf import _vhf


def get_nacv(td_nac, x_yI, EI, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Only supports for singlet states.
    Ref:
    [1] 10.1063/1.4903986 main reference
    [2] 10.1021/acs.accounts.1c00312
    [3] 10.1063/1.4885817

    Args:
        td_nac (gpu4pyscf.tdscf.rhf.TDA): Non-adiabatic coupling object for TDDFT or TDHF.
        x_yI (tuple): (xI, yI), xI and YI are the eigenvectors corresponding to the excitation and de-excitation.
        EI (float): excitation energy for state I

    Kwargs:
        singlet (bool): Whether calculate singlet states.
        atmlst (list): List of atoms to calculate the NAC.
        verbose (int): Verbosity level.

    Returns:
        nacv (np.ndarray): NAC matrix element.
    """
    if singlet is False:
        raise NotImplementedError('Only supports for singlet states')
    mol = td_nac.mol
    mf = td_nac.base._scf
    mf_grad = mf.nuc_grad_method()
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nao, nmo = mo_coeff.shape
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]
    if getattr(mf, 'with_solvent', None) is not None:
        raise NotImplementedError('With solvent is not supported yet')
    if getattr(mf, 'with_df', None) is not None:
        raise NotImplementedError('With density fitting is not supported yet')

    xI, yI = x_yI
    xI = cp.asarray(xI).reshape(nocc, nvir).T
    if not isinstance(yI, np.ndarray) and not isinstance(yI, cp.ndarray):
        yI = cp.zeros_like(xI)
    yI = cp.asarray(yI).reshape(nocc, nvir).T
    LI = xI-yI    # eq.(83) in Ref. [1]

    vresp = mf.gen_response(singlet=None, hermi=1)

    def fvind(x):
        dm = reduce(cp.dot, (orbv, x.reshape(nvir, nocc) * 2, orbo.T)) # double occupency
        v1ao = vresp(dm + dm.T)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(
        fvind,
        mo_energy,
        mo_occ,
        -LI*1.0*EI, # only one spin, negative in cphf
        max_cycle=td_nac.cphf_max_cycle,
        tol=td_nac.cphf_conv_tol)[0] # eq.(83) in Ref. [1]

    z1 = z1.reshape(nvir, nocc)
    z1ao = reduce(cp.dot, (orbv, z1, orbo.T)) * 2 # double occupency
    # eq.(50) in Ref. [1]
    z1aoS = (z1ao + z1ao.T)*0.5 # 0.5 is in the definition of z1aoS
    # eq.(73) in Ref. [1]
    GZS = vresp(z1aoS) # generate the double occupency 
    GZS_mo = reduce(cp.dot, (mo_coeff.T, GZS, mo_coeff))
    W = cp.zeros((nmo, nmo))  # eq.(75) in Ref. [1]
    W[:nocc, :nocc] = GZS_mo[:nocc, :nocc]
    zeta0 = mo_energy[nocc:, cp.newaxis]
    zeta0 = z1 * zeta0
    W[:nocc, nocc:] = GZS_mo[:nocc, nocc:] + 0.5*yI.T*EI + 0.5*zeta0.T #* eq.(43), (56), (28) in Ref. [1]
    zeta1 = mo_energy[cp.newaxis, :nocc]
    zeta1 = z1 * zeta1
    W[nocc:, :nocc] = 0.5*xI*EI + 0.5*zeta1
    W = reduce(cp.dot, (mo_coeff, W , mo_coeff.T)) * 2.0

    mf_grad = mf.nuc_grad_method()
    dmz1doo = z1aoS
    oo0 = reduce(cp.dot, (orbo, orbo.T)) * 2.0

    if atmlst is None:
        atmlst = range(mol.natm)
    
    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_td = contract("xij,ij->xi", h1, dmz1doo)
    ds = contract("xij,ij->xi", s1, (W + W.T))

    dh1e_td = int3c2e.get_dh1e(mol, dmz1doo)  # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(mol, dmz1doo)  # 1/r like terms
    extra_force = cp.zeros((len(atmlst), 3))

    dvhf_all = 0
    dvhf = td_nac.get_veff(mol, dmz1doo + oo0) 
    for k, ia in enumerate(atmlst):
        extra_force[k] += mf_grad.extra_force(ia, locals())
    dvhf_all += dvhf
    dvhf = td_nac.get_veff(mol, dmz1doo)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= mf_grad.extra_force(ia, locals())
    dvhf_all -= dvhf
    dvhf = td_nac.get_veff(mol, oo0)
    for k, ia in enumerate(atmlst):
        extra_force[k] -= mf_grad.extra_force(ia, locals())
    dvhf_all -= dvhf

    delec = dh_td*2 - ds
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])

    xIao = reduce(cp.dot, (orbo, xI.T, orbv.T)) * 2
    yIao = reduce(cp.dot, (orbv, yI, orbo.T)) * 2
    ds_x = contract("xij,ji->xi", s1, xIao*EI)
    ds_y = contract("xij,ji->xi", s1, yIao*EI)
    ds_x_etf = contract("xij,ij->xi", s1, (xIao*EI + xIao.T*EI) * 0.5)
    ds_y_etf = contract("xij,ij->xi", s1, (yIao*EI + yIao.T*EI) * 0.5)
    dsxy = cp.asarray([cp.sum(ds_x[:, p0:p1] + ds_y[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    dsxy_etf = cp.asarray([cp.sum(ds_x_etf[:, p0:p1] + ds_y_etf[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])
    de = 2.0 * dvhf_all + extra_force + dh1e_td + delec 
    de_etf = de + dsxy_etf
    de += dsxy 
    
    de = de.get()
    de_etf = de_etf.get()
    return de, de/EI, de_etf, de_etf/EI


class NAC(lib.StreamObject):

    cphf_max_cycle = getattr(__config__, "grad_tdrhf_Gradients_cphf_max_cycle", 20)
    cphf_conv_tol = getattr(__config__, "grad_tdrhf_Gradients_cphf_conv_tol", 1e-8)

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {
        "cphf_max_cycle",
        "cphf_conv_tol",
        "mol",
        "base",
        "chkfile",
        "states",
        "atmlst",
        "de",
        "de_scaled",
        "de_etf",
        "de_etf_scaled"
    }

    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.states = (0, 1)  # of which the gradients to be computed.
        self.atmlst = None
        self.de = None
        self.de_scaled = None
        self.de_etf = None
        self.de_etf_scaled = None

    _write      = rhf_grad_cpu.GradientsBase._write

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("\n")
        log.info(
            "******** LR %s gradients for %s ********",
            self.base.__class__,
            self.base._scf.__class__,
        )
        log.info("cphf_conv_tol = %g", self.cphf_conv_tol)
        log.info("cphf_max_cycle = %d", self.cphf_max_cycle)
        log.info("chkfile = %s", self.chkfile)
        log.info(f"States ID = {self.states}")
        log.info("\n")
        return self

    @lib.with_doc(get_nacv.__doc__)
    def get_nacv(self, x_yI, EI, singlet, atmlst=None, verbose=logger.INFO):
        return get_nacv(self, x_yI, EI, singlet, atmlst, verbose)

    def kernel(self, xy_I=None, xy_J=None, E_I=None, E_J=None, singlet=None, atmlst=None):

        logger.warn(self, "This module is under development!!")

        if singlet is None:
            singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        if xy_I is None or xy_J is None:
            states = sorted(self.states)
            I, J = states
            if I < 0 or J < 0:
                raise ValueError("Excited states ID should be non-negetive integers.")
            elif I > 0:
                raise NotImplementedError("Only for ground-excited states nonadiabatic coupling.")
            elif I == 0:
                xy_I = self.base.xy[J-1]
                E_I = self.base.e[J-1]
                self.de, self.de_scaled, self.de_etf, self.de_etf_scaled \
                    = self.get_nacv(xy_I, E_I, singlet, atmlst, verbose=self.verbose)
                self._finalize()
            else:
                raise NotImplementedError("Only for ground-excited states nonadiabatic coupling.")
        return self.de
    
    def get_veff(self, mol=None, dm=None, j_factor=1.0, k_factor=1.0, omega=0.0, hermi=0, verbose=None):
        """
        Computes the first-order derivatives of the energy contributions from
        Veff per atom.

        NOTE: This function is incompatible to the one implemented in PySCF CPU version.
        In the CPU version, get_veff returns the first order derivatives of Veff matrix.
        """
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.base.make_rdm1()
        if omega == 0.0:
            vhfopt = self.base._scf._opt_gpu.get(None, None)
            return rhf_grad._jk_energy_per_atom(mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor, verbose=verbose)
        else:
            vhfopt = self.base._scf._opt_gpu.get(omega, None)
            with mol.with_range_coulomb(omega):
                return rhf_grad._jk_energy_per_atom(
                    mol, dm, vhfopt, j_factor=j_factor, k_factor=k_factor, verbose=verbose)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(
                self,
                "--------- %s nonadiabatic derivative coupling for states %d and %d----------",
                self.base.__class__.__name__,
                self.states[0],
                self.states[1],
            )
            self._write(self.mol, self.de, self.atmlst)
            logger.note(
                self,
                "--------- %s nonadiabatic derivative coupling for states %d and %d after E scaled (divided by E)----------",
                self.base.__class__.__name__,
                self.states[0],
                self.states[1],
            )
            self._write(self.mol, self.de_scaled, self.atmlst)
            logger.note(
                self,
                "--------- %s nonadiabatic derivative coupling for states %d and %d with ETF----------",
                self.base.__class__.__name__,
                self.states[0],
                self.states[1],
            )
            self._write(self.mol, self.de_etf, self.atmlst)
            logger.note(
                self,
                "--------- %s nonadiabatic derivative coupling for states %d and %d with ETF after E scaled (divided by E)----------",
                self.base.__class__.__name__,
                self.states[0],
                self.states[1],
            )
            self._write(self.mol, self.de_etf_scaled, self.atmlst)
            logger.note(self, "----------------------------------------------")

    def solvent_response(self, dm):
        return 0.0

    as_scanner = NotImplemented

    to_gpu = lib.to_gpu


tdscf.rhf.TDA.NAC = tdscf.rhf.TDHF.NAC = lib.class_as_method(NAC)
