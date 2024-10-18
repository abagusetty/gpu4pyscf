#ifndef GPU4PYSCF_CONSTANT_HPP
#define GPU4PYSCF_CONSTANT_HPP

#include "gint/sycl_device.hpp"
#include "gint/gint.h"

SYCL_EXTERNAL sycl_device_global<BasisProdCache> c_bpcache;
SYCL_EXTERNAL sycl_device_global<int16_t[NFffff*3]> c_idx4c;
SYCL_EXTERNAL sycl_device_global<int[TOT_NF*3]> c_idx;
SYCL_EXTERNAL sycl_device_global<int[GPU_LMAX+2]> c_l_locs;

SYCL_EXTERNAL sycl_device_global<BasisProdOffsets[MAX_STREAMS]> c_offsets;
SYCL_EXTERNAL sycl_device_global<GINTEnvVars[MAX_STREAMS]> c_envs;
SYCL_EXTERNAL sycl_device_global<JKMatrix[MAX_STREAMS]> c_jk;

#endif //GPU4PYSCF_CONSTANT_HPP
