/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#if 0
template <int NROOTS, int GOUTSIZE> __global__
static void GINTfill_int2e_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    double norm = envs.fac;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int task_id = task_ij + ntasks_ij * task_kl;
    double *uw = envs.uw + task_id * nprim_ij * nprim_kl * NROOTS * 2;
    double gout[GOUTSIZE];
    double *g = gout + envs.nf;
    int i;

    for (i = 0; i < envs.nf; ++i) {
        gout[i] = 0;
    }

    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }

    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_2e_2d4d<NROOTS>(envs, g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTgout2e<NROOTS>(envs, gout, g);
        uw += NROOTS * 2;
    } }
    GINTwrite_ints_s2(eri, gout, ish, jsh, ksh, lsh);
}
#endif


template <int NROOTS, int GOUTSIZE> __attribute__((always_inline))
void GINTfill_int2e_kernel(GINTEnvVars envs, ERITensor eri, BasisProdOffsets offsets, sycl::nd_item<2>& item)
{
    int ntasks_ij = offsets.ntasks_ij;
    int ntasks_kl = offsets.ntasks_kl;
    int task_ij = static_cast<int>( item.get_global_id(1) );
    int task_kl = static_cast<int>( item.get_global_id(0) );
    if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
        return;
    }

    double norm = envs.fac;
    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *bas_pair2bra = c_bpcache.bas_pair2bra;
    int *bas_pair2ket = c_bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    double gout[GOUTSIZE];
    double *g = gout + envs.nf;
    int i;
    for (i = 0; i < envs.nf; ++i) {
        gout[i] = 0;
    }

    int ij, kl;
    int as_ish, as_jsh, as_ksh, as_lsh;
    if (envs.ibase) {
        as_ish = ish;
        as_jsh = jsh;
    } else {
        as_ish = jsh;
        as_jsh = ish;
    }
    if (envs.kbase) {
        as_ksh = ksh;
        as_lsh = lsh;
    } else {
        as_ksh = lsh;
        as_lsh = ksh;
    }
    for (ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
    for (kl = prim_kl; kl < prim_kl+nprim_kl; ++kl) {
        GINTg0_2e_2d4d<NROOTS>(envs, g, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        GINTgout2e<NROOTS>(envs, gout, g);
    } }

    GINTwrite_ints_s2(eri, gout, ish, jsh, ksh, lsh);
}
