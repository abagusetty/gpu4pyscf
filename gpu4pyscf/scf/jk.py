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

'''
Compute J/K matrices
'''

import ctypes
import math
import numpy as np
import cupy as cp
import scipy.linalg
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pyscf.gto import ANG_OF, ATOM_OF, NPRIM_OF, NCTR_OF, PTR_COORD, PTR_COEFF
from pyscf import lib
from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import (load_library, condense, sandwich_dot, transpose_sum,
                                       reduce_to_device)
from gpu4pyscf.__config__ import _streams, _num_devices, shm_size
from gpu4pyscf.__config__ import props as gpu_specs
from gpu4pyscf.lib import logger
from gpu4pyscf.gto.mole import group_basis

__all__ = [
    'get_jk', 'get_j',
]

libvhf_rys = load_library('libgvhf_rys')
libvhf_rys.RYS_build_jk.restype = ctypes.c_int
libvhf_rys.RYS_init_constant.restype = ctypes.c_int
libvhf_rys.RYS_init_rysj_constant.restype = ctypes.c_int
libvhf_rys.cuda_version.restype = ctypes.c_int
CUDA_VERSION = libvhf_rys.cuda_version()

PTR_BAS_COORD = 7
LMAX = 4
TILE = 2
QUEUE_DEPTH = 262144
UNROLL_ORDER = ctypes.c_int.in_dll(libvhf_rys, 'rys_jk_unrolled_max_order').value
UNROLL_LMAX = ctypes.c_int.in_dll(libvhf_rys, 'rys_jk_unrolled_lmax').value
UNROLL_NFMAX = ctypes.c_int.in_dll(libvhf_rys, 'rys_jk_unrolled_max_nf').value
UNROLL_J_LMAX = ctypes.c_int.in_dll(libvhf_rys, 'rys_j_unrolled_lmax').value
UNROLL_J_MAX_ORDER = ctypes.c_int.in_dll(libvhf_rys, 'rys_j_unrolled_max_order').value
SHM_SIZE = shm_size - 1024
del shm_size
GOUT_WIDTH = 42
THREADS = 256
GROUP_SIZE = 256

def _jk_task(mol, dms, vhfopt, task_list, hermi=0,
             device_id=0, with_j=True, with_k=True, verbose=None):
    n_dm = dms.shape[0]
    nao, _ = vhfopt.coeff.shape
    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    kern = libvhf_rys.RYS_build_jk

    timing_counter = Counter()
    kern_counts = 0
    with cp.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()
        dms = cp.asarray(dms)

        if hermi == 0:
            # Contract the tril and triu parts separately
            dms = cp.vstack([dms, dms.transpose(0,2,1)])
        n_dm = dms.shape[0]
        tile_q_ptr = ctypes.cast(vhfopt.tile_q_cond.data.ptr, ctypes.c_void_p)
        q_ptr = ctypes.cast(vhfopt.q_cond.data.ptr, ctypes.c_void_p)
        s_ptr = lib.c_null_ptr()
        if mol.omega < 0:
            s_ptr = ctypes.cast(vhfopt.s_estimator.data.ptr, ctypes.c_void_p)

        vj = vk = None
        vj_ptr = vk_ptr = lib.c_null_ptr()
        assert with_j or with_k
        if with_k:
            vk = cp.zeros(dms.shape)
            vk_ptr = ctypes.cast(vk.data.ptr, ctypes.c_void_p)
        if with_j:
            vj = cp.zeros(dms.shape)
            vj_ptr = ctypes.cast(vj.data.ptr, ctypes.c_void_p)

        ao_loc = mol.ao_loc
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        log_max_dm = dm_cond.max()
        log_cutoff = math.log(vhfopt.direct_scf_tol)
        tile_mappings = _make_tril_tile_mappings(l_ctr_bas_loc, vhfopt.tile_q_cond,
                                                 log_cutoff-log_max_dm)
        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty((workers, QUEUE_DEPTH*4), dtype=np.uint16)
        info = cp.empty(2, dtype=np.uint32)
        t1 = log.timer_debug1(f'q_cond and dm_cond on Device {device_id}', *cput0)

        for i, j, k, l in task_list:
            ij_shls = (l_ctr_bas_loc[i], l_ctr_bas_loc[i+1],
                       l_ctr_bas_loc[j], l_ctr_bas_loc[j+1])
            tile_ij_mapping = tile_mappings[i,j]
            llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
            kl_shls = (l_ctr_bas_loc[k], l_ctr_bas_loc[k+1],
                        l_ctr_bas_loc[l], l_ctr_bas_loc[l+1])
            tile_kl_mapping = tile_mappings[k,l]
            scheme = quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
            err = kern(
                vj_ptr, vk_ptr, ctypes.cast(dms.data.ptr, ctypes.c_void_p),
                ctypes.c_int(n_dm), ctypes.c_int(nao),
                vhfopt.rys_envs, (ctypes.c_int*2)(*scheme),
                (ctypes.c_int*8)(*ij_shls, *kl_shls),
                ctypes.c_int(tile_ij_mapping.size),
                ctypes.c_int(tile_kl_mapping.size),
                ctypes.cast(tile_ij_mapping.data.ptr, ctypes.c_void_p),
                ctypes.cast(tile_kl_mapping.data.ptr, ctypes.c_void_p),
                tile_q_ptr, q_ptr, s_ptr,
                ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                ctypes.c_float(log_cutoff),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                ctypes.cast(info.data.ptr, ctypes.c_void_p),
                ctypes.c_int(workers),
                mol._atm.ctypes, ctypes.c_int(mol.natm),
                mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
            if err != 0:
                raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
            if log.verbose >= logger.DEBUG1:
                msg = f'processing {llll}, tasks = {info[1].get()} on Device {device_id}'
                t1, t1p = log.timer_debug1(msg, *t1), t1
                timing_counter[llll] += t1[1] - t1p[1]
                kern_counts += 1
        if with_j:
            if hermi == 1:
                vj *= 2.
            else:
                vj, vjT = vj[:n_dm//2], vj[n_dm//2:]
                vj += vjT.transpose(0,2,1)
        if with_k:
            if hermi == 1:
                vk = transpose_sum(vk)
            else:
                vk, vkT = vk[:n_dm//2], vk[n_dm//2:]
                vk += vkT.transpose(0,2,1)
    return vj, vk, kern_counts, timing_counter

def get_jk(mol, dm, hermi=0, vhfopt=None, with_j=True, with_k=True, verbose=None):
    '''Compute J, K matrices
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()

    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    mol = vhfopt.sorted_mol
    nao, nao_orig = vhfopt.coeff.shape

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = sandwich_dot(dms, vhfopt.coeff.T)
    dms = cp.asarray(dms, order='C')

    n_dm = dms.shape[0]

    assert with_j or with_k

    init_constant(mol)

    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = np.count_nonzero(uniq_l <= LMAX)

    tasks = []
    for i in range(n_groups):
        for j in range(i+1):
            for k in range(i+1):
                for l in range(k+1): 
                    tasks.append((i,j,k,l))
    tasks = np.array(tasks)
    task_list = []
    for device_id in range(_num_devices):
        task_list.append(tasks[device_id::_num_devices])

    cp.cuda.get_current_stream().synchronize()
    futures = []
    with ThreadPoolExecutor(max_workers=_num_devices) as executor:
        for device_id in range(_num_devices):
            future = executor.submit(
                _jk_task,
                mol, dms, vhfopt, task_list[device_id], hermi=hermi,
                with_j=with_j, with_k=with_k, verbose=verbose, 
                device_id=device_id)
            futures.append(future)

    kern_counts = 0
    timing_collection = Counter()
    vj_dist = []
    vk_dist = []
    for future in futures:
        vj, vk, counts, counter = future.result()
        kern_counts += counts
        timing_collection += counter
        vj_dist.append(vj)
        vk_dist.append(vk)

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)

    for s in _streams:
        s.synchronize()
    cp.cuda.get_current_stream().synchronize()
    vj = vk = None
    if with_k:
        vk = reduce_to_device(vk_dist, inplace=True)
        #:vk = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, vk, vhfopt.coeff)
        vk = sandwich_dot(vk, vhfopt.coeff)
        
    if with_j:
        vj = reduce_to_device(vj_dist, inplace=True)
        vj = transpose_sum(vj)
        #:vj = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, vj, vhfopt.coeff)
        vj = sandwich_dot(vj, vhfopt.coeff)

    h_shls = vhfopt.h_shls

    if h_shls:
        cput1 = log.timer_debug1('get_jk pass 1 on gpu', *cput0)
        log.debug3('Integrals for %s functions on CPU', l_symb[LMAX+1])
        scripts = []
        if with_j:
            scripts.append('ji->s2kl')
        if with_k:
            if hermi == 1:
                scripts.append('jk->s2il')
            else:
                scripts.append('jk->s1il')
        shls_excludes = [0, h_shls[0]] * 4
        if hermi == 1:
            dms = dms.get()
        else:
            dms = dms[:n_dm//2].get()
        vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                 dms, 1, mol._atm, mol._bas, mol._env,
                                 shls_excludes=shls_excludes)
        if with_j and with_k:
            vj1 = vs_h[0]
            vk1 = vs_h[1]
        elif with_j:
            vj1 = vs_h[0]
        else:
            vk1 = vs_h[0]
        coeff = vhfopt.coeff
        idx, idy = np.tril_indices(nao, -1)
        if with_j:
            vj1[:,idy,idx] = vj1[:,idx,idy]
            for i, v in enumerate(vj1):
                vj[i] += coeff.T.dot(cp.asarray(v)).dot(coeff)
        if with_k:
            if hermi:
                vk1[:,idy,idx] = vk1[:,idx,idy]
            for i, v in enumerate(vk1):
                vk[i] += coeff.T.dot(cp.asarray(v)).dot(coeff)
        log.timer_debug1('get_jk pass 2 for h functions on cpu', *cput1)
    
    if with_j:
        vj = vj.reshape(dm.shape)
    if with_k:
        vk = vk.reshape(dm.shape)

    log.timer('vj and vk', *cput0)
    return vj, vk

def get_j(mol, dm, hermi=0, vhfopt=None, verbose=None):
    '''Compute J matrix
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    mol = vhfopt.sorted_mol
    nao, nao_orig = vhfopt.coeff.shape

    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    n_dm = dms.shape[0]
    assert n_dm == 1
    #:dms = cp.einsum('pi,nij,qj->npq', vhfopt.coeff, dms, vhfopt.coeff)
    dms = sandwich_dot(dms, vhfopt.coeff.T)
    dms = cp.asarray(dms, order='C')
    if hermi != 1:
        dms = transpose_sum(dms)
        dms *= .5

    ao_loc = mol.ao_loc
    dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
    log_max_dm = dm_cond.max()
    log_cutoff = math.log(vhfopt.direct_scf_tol)

    dms = dms.get()
    pair_loc = _make_j_engine_pair_locs(mol)
    dm_xyz = np.empty(pair_loc[-1])
    libvhf_rys.transform_cart_to_xyz(
        dm_xyz.ctypes, dms.ctypes, ao_loc.ctypes, pair_loc.ctypes,
        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
    dm_xyz = cp.asarray(dm_xyz)
    vj_xyz = cp.zeros_like(dm_xyz)

    pair_loc_on_gpu = cp.asarray(pair_loc)
    rys_envs = RysIntEnvVars(
        mol.natm, mol.nbas,
        vhfopt.rys_envs.atm, vhfopt.rys_envs.bas, vhfopt.rys_envs.env,
        pair_loc_on_gpu.data.ptr,
    )

    err = libvhf_rys.RYS_init_rysj_constant(ctypes.c_int(SHM_SIZE))
    if err != 0:
        raise RuntimeError('CUDA kernel initialization')

    uniq_l_ctr = vhfopt.uniq_l_ctr
    uniq_l = uniq_l_ctr[:,0]
    l_ctr_bas_loc = vhfopt.l_ctr_offsets
    l_symb = [lib.param.ANGULAR[i] for i in uniq_l]
    n_groups = np.count_nonzero(uniq_l <= LMAX)
    ntiles = mol.nbas // TILE
    tile_mappings = {}
    workers = gpu_specs['multiProcessorCount']
    pool = cp.empty((workers, QUEUE_DEPTH*4), dtype=np.uint16)
    info = cp.empty(2, dtype=np.uint32)

    for i in range(n_groups):
        for j in range(i+1):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            ij_shls = (ish0, ish1, jsh0, jsh1)
            i0 = ish0 // TILE
            i1 = ish1 // TILE
            j0 = jsh0 // TILE
            j1 = jsh1 // TILE
            sub_tile_q = vhfopt.tile_q_cond[i0:i1,j0:j1]
            mask = sub_tile_q > log_cutoff - log_max_dm
            if i == j:
                mask = cp.tril(mask)
            t_ij = (cp.arange(i0, i1, dtype=np.int32)[:,None] * ntiles +
                    cp.arange(j0, j1, dtype=np.int32))
            idx = cp.argsort(sub_tile_q[mask])[::-1]
            tile_mappings[i,j] = t_ij[mask][idx]
    t1 = t2 = log.timer_debug1('q_cond and dm_cond', *cput0)

    timing_collection = {}
    kern_counts = 0
    kern = libvhf_rys.RYS_build_j

    for i in range(n_groups):
        for j in range(i+1):
            ij_shls = (l_ctr_bas_loc[i], l_ctr_bas_loc[i+1],
                       l_ctr_bas_loc[j], l_ctr_bas_loc[j+1])
            tile_ij_mapping = tile_mappings[i,j]
            for k in range(i+1):
                for l in range(k+1):
                    llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                    kl_shls = (l_ctr_bas_loc[k], l_ctr_bas_loc[k+1],
                               l_ctr_bas_loc[l], l_ctr_bas_loc[l+1])
                    tile_kl_mapping = tile_mappings[k,l]
                    scheme = _j_engine_quartets_scheme(mol, uniq_l_ctr[[i, j, k, l]])
                    err = kern(
                        ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                        ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(n_dm), ctypes.c_int(nao),
                        rys_envs, (ctypes.c_int*3)(*scheme),
                        (ctypes.c_int*8)(*ij_shls, *kl_shls),
                        ctypes.c_int(tile_ij_mapping.size),
                        ctypes.c_int(tile_kl_mapping.size),
                        ctypes.cast(tile_ij_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(tile_kl_mapping.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.tile_q_cond.data.ptr, ctypes.c_void_p),
                        ctypes.cast(vhfopt.q_cond.data.ptr, ctypes.c_void_p),
                        lib.c_null_ptr(),
                        ctypes.cast(dm_cond.data.ptr, ctypes.c_void_p),
                        ctypes.c_float(log_cutoff),
                        ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                        ctypes.cast(info.data.ptr, ctypes.c_void_p),
                        ctypes.c_int(workers),
                        mol._atm.ctypes, ctypes.c_int(mol.natm),
                        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
                    if err != 0:
                        raise RuntimeError(f'RYS_build_jk kernel for {llll} failed')
                    if log.verbose >= logger.DEBUG1:
                        t1, t1p = log.timer_debug1(f'processing {llll}, tasks = {info[1]}', *t1), t1
                        if llll not in timing_collection:
                            timing_collection[llll] = 0
                        timing_collection[llll] += t1[1] - t1p[1]
                        kern_counts += 1

    if log.verbose >= logger.DEBUG1:
        log.debug1('kernel launches %d', kern_counts)
        for llll, t in timing_collection.items():
            log.debug1('%s wall time %.2f', llll, t)
        cp.cuda.Stream.null.synchronize()
        log.timer_debug1('cuda kernel', *t2)

    vj_xyz = vj_xyz.get()
    vj = np.empty_like(dms)
    libvhf_rys.transform_xyz_to_cart(
        vj.ctypes, vj_xyz.ctypes, ao_loc.ctypes, pair_loc.ctypes,
        mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
    #:vj = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, cp.asarray(vj), vhfopt.coeff)
    vj = sandwich_dot(vj, vhfopt.coeff)
    vj = transpose_sum(vj)
    vj *= 2.
    vj = vj.reshape(dm.shape)

    h_shls = vhfopt.h_shls
    if h_shls:
        cput1 = log.timer_debug1('get_j pass 1 on gpu', *cput0)
        log.debug3('Integrals for %s functions on CPU', l_symb[LMAX+1])
        scripts = ['ji->s2kl']
        shls_excludes = [0, h_shls[0]] * 4
        vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                 dms.get(), 1, mol._atm, mol._bas, mol._env,
                                 shls_excludes=shls_excludes)
        vj1 = vs_h[0].reshape(n_dm,nao,nao)
        coeff = vhfopt.coeff
        idx, idy = np.tril_indices(nao, -1)
        vj1[:,idy,idx] = vj1[:,idx,idy]
        for i, v in enumerate(vj1):
            vj[i] += coeff.T.dot(cp.asarray(v)).dot(coeff)
        log.timer_debug1('get_j pass 2 for h functions on cpu', *cput1)

    log.timer('vj', *cput0)
    return vj

class _VHFOpt:
    def __init__(self, mol, cutoff=1e-13):
        self.mol = mol
        self.direct_scf_tol = cutoff
        self.uniq_l_ctr = None
        self.l_ctr_offsets = None
        self.h_shls = None
        self.tile = TILE

        # Hold cache on GPU devices
        self._rys_envs = {}
        self._mol_gpu = {}
        self._q_cond = {}
        self._tile_q_cond = {}
        self._s_estimator = {}

    def build(self, group_size=None, verbose=None):
        mol = self.mol
        log = logger.new_logger(mol, verbose)
        cput0 = log.init_timer()
        mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(mol, self.tile, group_size)
        self.sorted_mol = mol
        self.coeff = cp.asarray(coeff)
        self.uniq_l_ctr = uniq_l_ctr
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))

        # very high angular momentum basis are processed on CPU
        lmax = uniq_l_ctr[:,0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax > LMAX:
            self.h_shls = l_slices[LMAX+1:].tolist()
        else:
            self.h_shls = []

        nbas = mol.nbas
        ao_loc = mol.ao_loc
        q_cond = np.empty((nbas,nbas))
        intor = mol._add_suffix('int2e')
        _vhf.libcvhf.CVHFnr_int2e_q_cond(
            getattr(_vhf.libcvhf, intor), lib.c_null_ptr(),
            q_cond.ctypes, ao_loc.ctypes,
            mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
        q_cond = np.log(q_cond + 1e-300).astype(np.float32)
        self.q_cond_cpu = q_cond

        tile = self.tile
        if tile > 1:
            ntiles = nbas // tile
            self._tile_q_cond_cpu = q_cond.reshape(ntiles,tile,ntiles,tile).max(axis=(1,3))
        else:
            self._tile_q_cond_cpu = q_cond

        if mol.omega < 0:
            # CVHFnr_sr_int2e_q_cond in pyscf has bugs in upper bound estimator.
            # Use the local version of s_estimator instead
            s_estimator = np.empty((nbas,nbas), dtype=np.float32)
            libvhf_rys.sr_eri_s_estimator(
                s_estimator.ctypes, ctypes.c_float(mol.omega),
                mol._atm.ctypes, ctypes.c_int(mol.natm),
                mol._bas.ctypes, ctypes.c_int(mol.nbas), mol._env.ctypes)
            self.s_estimator_cpu = s_estimator
        log.timer('Initialize q_cond', *cput0)
        return self

    @property
    def q_cond(self):
        device_id = cp.cuda.Device().id
        if device_id not in self._q_cond:
            with cp.cuda.Device(device_id), _streams[device_id]:
                self._q_cond[device_id] = cp.asarray(self.q_cond_cpu)
        return self._q_cond[device_id]

    @property
    def tile_q_cond(self):
        device_id = cp.cuda.Device().id
        if device_id not in self._tile_q_cond:
            with cp.cuda.Device(device_id), _streams[device_id]:
                q_cpu = self._tile_q_cond_cpu
                self._tile_q_cond[device_id] = cp.asarray(q_cpu)
        return self._tile_q_cond[device_id]

    @property
    def s_estimator(self):
        if not self.mol.omega < 0:
            return None
        device_id = cp.cuda.Device().id
        if device_id not in self._rys_envs:
            with cp.cuda.Device(device_id), _streams[device_id]:
                s_cpu = self.s_estimator_cpu
                self._s_estimator[device_id] = cp.asarray(s_cpu)
        return self._s_estimator[device_id]

    @property
    def rys_envs(self):
        device_id = cp.cuda.Device().id
        if device_id not in self._rys_envs:
            with cp.cuda.Device(device_id), _streams[device_id]:
                mol = self.sorted_mol
                _atm = cp.array(mol._atm)
                _bas = cp.array(mol._bas)
                _env = cp.array(_scale_sp_ctr_coeff(mol))
                ao_loc = cp.array(mol.ao_loc)
                self._mol_gpu[device_id] = (_atm, _bas, _env, ao_loc)
                self._rys_envs[device_id] = RysIntEnvVars(
                    mol.natm, mol.nbas,
                    _atm.data.ptr, _bas.data.ptr, _env.data.ptr,
                    ao_loc.data.ptr)
        return self._rys_envs[device_id]

class RysIntEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_uint16),
        ('nbas', ctypes.c_uint16),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
    ]

def _scale_sp_ctr_coeff(mol):
    # Match normalization factors of s, p functions in libcint
    _env = mol._env.copy()
    ls = mol._bas[:,ANG_OF]
    ptr, idx = np.unique(mol._bas[:,PTR_COEFF], return_index=True)
    ptr = ptr[ls[idx] < 2]
    idx = idx[ls[idx] < 2]
    fac = ((ls[idx]*2+1) / (4*np.pi)) ** .5
    nprim = mol._bas[idx,NPRIM_OF]
    nctr = mol._bas[idx,NCTR_OF]
    for p, n, f in zip(ptr, nprim*nctr, fac):
        _env[p:p+n] *= f
    return _env

def iter_cart_xyz(n):
    return [(x, y, n-x-y)
            for x in reversed(range(n+1))
            for y in reversed(range(n+1-x))]

def g_pair_idx(ij_inc=None):
    dat = []
    xyz = [np.array(iter_cart_xyz(li)) for li in range(LMAX+1)]
    for li in range(LMAX+1):
        for lj in range(LMAX+1):
            li1 = li + 1
            idx = (xyz[lj][:,None] * li1 + xyz[li]).transpose(2,0,1)
            dat.append(idx.ravel())
    g_idx = np.hstack(dat).astype(np.int32)
    offsets = np.cumsum([0] + [x.size for x in dat]).astype(np.int32)
    return g_idx, offsets

def init_constant(mol):
    g_idx, offsets = g_pair_idx()
    for device_id in range(_num_devices):
        with cp.cuda.Device(device_id), _streams[device_id]:
            err = libvhf_rys.RYS_init_constant(
                g_idx.ctypes, offsets.ctypes, mol._env.ctypes,
                ctypes.c_int(mol._env.size), ctypes.c_int(SHM_SIZE))
            if err != 0:
                raise RuntimeError(f'CUDA kernel initialization on device {device_id}')

def _make_tril_tile_mappings(l_ctr_bas_loc, tile_q_cond, cutoff, tile=TILE):
    n_groups = len(l_ctr_bas_loc) - 1
    ntiles = tile_q_cond.shape[0]
    tile_mask = cp.tril(tile_q_cond > cutoff)
    tile_mappings = {}
    for i in range(n_groups):
        for j in range(i+1):
            ish0, ish1 = l_ctr_bas_loc[i], l_ctr_bas_loc[i+1]
            jsh0, jsh1 = l_ctr_bas_loc[j], l_ctr_bas_loc[j+1]
            i0 = ish0 // tile
            i1 = ish1 // tile
            j0 = jsh0 // tile
            j1 = jsh1 // tile
            sub_tile_q = tile_q_cond[i0:i1,j0:j1]
            mask = tile_mask[i0:i1,j0:j1]
            t_ij = (cp.arange(i0, i1, dtype=np.int32)[:,None] * ntiles +
                    cp.arange(j0, j1, dtype=np.int32))
            idx = cp.argsort(sub_tile_q[mask])[::-1]
            tile_mappings[i,j] = t_ij[mask][idx]
    return tile_mappings

def _make_j_engine_pair_locs(mol):
    ls = mol._bas[:,ANG_OF]
    ll = (ls[:,None]+ls).ravel()
    pair_loc = np.append(0, np.cumsum((ll+1)*(ll+2)*(ll+3)//6))
    return np.asarray(pair_loc, dtype=np.int32)

def quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    nfl = (ll + 1) * (ll + 2) // 2
    gout_size = nfi * nfj * nfk * nfl
    if (gout_size <= UNROLL_NFMAX or order <= UNROLL_ORDER) and all(ls <= UNROLL_LMAX):
        if (CUDA_VERSION >= 12040 and
            order <= 3 and (li,lj,lk,ll) != (1,1,1,0) and (li,lj,lk,ll) != (1,0,1,1)):
            return 512, 1
        return 256, 1

    g_size = (li+1)*(lj+1)*(lk+1)*(ll+1)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    nroots = order // 2 + 1
    unit = nroots*2 + g_size*3 + ij_prims*4
    if mol.omega < 0: # SR
        unit += nroots * 2
    counts = shm_size // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    while gout_stride < 16 and gout_size / (gout_stride*GOUT_WIDTH) > 1:
        n //= 2
        gout_stride *= 2
    return n, gout_stride

def _j_engine_quartets_scheme(mol, l_ctr_pattern, shm_size=SHM_SIZE):
    ls = l_ctr_pattern[:,0]
    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    lij = li + lj
    lkl = lk + ll
    nmax = max(lij, lkl)
    nps = l_ctr_pattern[:,1]
    ij_prims = nps[0] * nps[1]
    g_size = (lij+1)*(lkl+1)
    nf3_ij = (lij+1)*(lij+2)*(lij+3)//6
    nf3_kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    nroots = order // 2 + 1
    # UNROLL_J_LMAX is different to UNROLL_LMAX of orbital basis. see rys_contract_j kernel
    if order <= UNROLL_J_MAX_ORDER and lij <= UNROLL_J_LMAX and lkl <= UNROLL_J_LMAX:
        if CUDA_VERSION >= 12040 and order <= 2:
            return 512, 1, False
        return 256, 1, False

    unit = nroots*2 + g_size*3 + ij_prims*4
    dm_cache_size = nf3_ij + nf3_kl*2 + (lij+1)*(lkl+1)*(nmax+2)
    gout_size = nf3_ij * nf3_kl
    if dm_cache_size < gout_size:
        unit += dm_cache_size
        shm_size -= nf3_ij * TILE*TILE * 8
        with_gout = False
    else:
        unit += gout_size
        with_gout = True

    if mol.omega < 0:
        unit += nroots*2
    counts = shm_size // (unit*8)
    n = min(THREADS, _nearest_power2(counts))
    gout_stride = THREADS // n
    if CUDA_VERSION >= 12040:
        gout_stride *= 2
    return n, gout_stride, with_gout

def _nearest_power2(n, return_leq=True):
    '''nearest 2**x that is leq or geq than n.

    Kwargs:
        return_leq specifies that the return is less or equal than n.
        Otherwise, the return is greater or equal than n.
    '''
    n = int(n)
    assert n > 0
    if return_leq:
        return 1 << (n.bit_length() - 1)
    else:
        return 1 << ((n-1).bit_length())
