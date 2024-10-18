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

#pragma once

#include <sycl//sycl.hpp>

typedef struct {
    int natm;
    int nbas;
    int *ao_loc;
    double *env;
    double *atom_coordx;
} GTOValEnvVars;

typedef struct {
    int ngrids;
    int nbas;
    int nao;
    int bas_off;
    int nprim;
    int *ao_loc;
    int *bas_indices;
    double fac;
    double *gridx;
    double *data;
} BasOffsets;

#define C_ATOM          0
#define C_EXP           1
#define C_COEFF         2
#define C_BAS_SLOTS     3
#define NBAS_MAX        6000

template <class T>
using sycl_device_global = sycl::ext::oneapi::experimental::device_global<T>;
SYCL_EXTERNAL extern sycl_device_global<uint16_t[NBAS_MAX]> c_bas_atom;
SYCL_EXTERNAL extern sycl_device_global<uint16_t[NBAS_MAX]> c_bas_exp;
SYCL_EXTERNAL extern sycl_device_global<uint16_t[NBAS_MAX]> c_bas_coeff;
SYCL_EXTERNAL extern sycl_device_global<GTOValEnvVars> c_envs;
