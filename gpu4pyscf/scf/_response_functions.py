# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

import cupy
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import hf, uhf, rohf
from gpu4pyscf.hessian.rks import nr_rks_fnlc_mo

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, grids=None, max_memory=None, with_nlc=True):
    '''Generate a function to compute the product of RHF response function and
    RHF density matrices.

    Kwargs:
        singlet (None or boolean) : If singlet is None, response function for
            orbital hessian or CPHF will be generated. If singlet is boolean,
            it is used in TDDFT response kernel.
    '''
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    
    if isinstance(mf, hf.KohnShamDFT):
        if grids is None:
            grids = mf.grids
        if grids and grids.coords is None:
            grids.build(mol=mol, with_non0tab=False, sort_grids=True)
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if singlet is None:
            # for ground state orbital hessian
            spin = 0
        else:
            spin = 1
        rho0, vxc, fxc = ni.cache_xc_kernel(
            mol, grids, mf.xc, mo_coeff, mo_occ, spin, max_memory=max_memory)
        dm0 = None

        if singlet is None:
            # Without specify singlet, used in ground state orbital hessian
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = cupy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(mol, grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, max_memory=max_memory)
                    if mf.do_nlc():
                        if with_nlc:
                            v1 += nr_rks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dm1, return_in_mo = False)
                        else:
                            logger.warn(mf, "NLC contribution in gen_response is NOT included")
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if omega > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1

        elif singlet:
            fxc *= .5
            def vind(dm1):
                if hermi == 2:
                    v1 = cupy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = ni.nr_rks_fxc_st(mol, grids, mf.xc, dm0, dm1, 0, True,
                                          rho0, vxc, fxc, max_memory=max_memory)
                    if mf.do_nlc():
                        if with_nlc:
                            v1 += nr_rks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dm1, return_in_mo = False)
                        else:
                            logger.warn(mf, "NLC contribution in gen_response is NOT included")
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if abs(omega) > 1e-10:  # For range separated Coulomb
                            vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1

        else:  # triplet
            fxc *= .5
            def vind(dm1):
                if hermi == 2:
                    v1 = cupy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = ni.nr_rks_fxc_st(mol, grids, mf.xc, dm0, dm1, 0, False,
                                          rho0, vxc, fxc, max_memory=max_memory)
                    if mf.do_nlc():
                        pass # fxc = 0, do nothing
                if hybrid:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if abs(omega) > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += -.5 * vk
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _gen_uhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, grids=None, max_memory=None, with_nlc=True):
    '''Generate a function to compute the product of UHF response function and
    UHF density matrices.
    '''
    assert isinstance(mf, (uhf.UHF, rohf.ROHF))
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        if grids is None:
            grids = mf.grids
        if grids and grids.coords is None:
            grids.build(mol=mol, with_non0tab=False, sort_grids=True)
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc,
                                            mo_coeff, mo_occ, 1)
        dm0 = None

        def vind(dm1):
            if hermi == 2:
                v1 = cupy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(mol, grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, max_memory=max_memory)
                if mf.do_nlc():
                    if with_nlc:
                        v1 += nr_rks_fnlc_mo(mf, mol, mo_coeff, mo_occ, dm1[0] + dm1[1], return_in_mo = False)
                    else:
                        logger.warn(mf, "NLC contribution in gen_response is NOT included")
            if not hybrid:
                if with_j:
                    vj = mf.get_j(mol, dm1, hermi=hermi)
                    v1 += vj[0] + vj[1]
            else:
                if with_j:
                    vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 += vj[0] + vj[1] - vk
                else:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if omega > 1e-10:  # For range separated Coulomb
                        vk += mf.get_k(mol, dm1, hermi, omega) * (alpha-hyb)
                    v1 -= vk
            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind

hf.RHF.gen_response = _gen_rhf_response
uhf.UHF.gen_response = _gen_uhf_response
rohf.ROHF.gen_response = _gen_uhf_response
