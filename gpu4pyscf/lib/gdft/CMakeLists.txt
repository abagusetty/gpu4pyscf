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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -maxrregcount=128")

add_library(gdft SHARED
  nr_eval_gto.cu
  contract_rho.cu
  gen_grids.cu
  nr_numint_sparse.cu
  vv10.cu
  libxc.cu
)

set_target_properties(gdft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Optional ExchCXX backend: libxc_exchcxx.so
#
# A drop-in replacement for deps/lib/libxc.so that implements the same libxc
# C ABI (xc_func_alloc/init/end/free, xc_version_string, ...) plus the three
# GDFT_xc_lda/gga/mgga device entry points, backed by ExchCXX instead of
# libxc's own CUDA kernels. Selected at runtime with GDFT_XC_BACKEND=exchcxx;
# the default libxc build above is always present and untouched.
#
# Functionals ExchCXX has no builtin kernel for fall back to PySCF's CPU libxc
# via xc_func_init returning 3 -- see the note in ../CMakeLists.txt.
#
# Built as plain C++ (not CUDA): exchcxx.cpp only calls host-side CUDA runtime
# functions and ExchCXX's device entry points, which live in libexchcxx.so.
# ---------------------------------------------------------------------------
if(BUILD_EXCHCXX)
  find_package(CUDAToolkit REQUIRED)

  add_library(xc_exchcxx SHARED exchcxx.cpp)
  add_dependencies(xc_exchcxx ExchCXX)

  target_include_directories(xc_exchcxx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}   # exchcxx.h
    ${PROJECT_SOURCE_DIR}
  )
  target_link_libraries(xc_exchcxx PRIVATE ExchCXX::ExchCXX CUDA::cudart)

  set_target_properties(xc_exchcxx PROPERTIES
    OUTPUT_NAME              xc_exchcxx
    POSITION_INDEPENDENT_CODE ON
    # Python looks for it next to libxc.so, in gpu4pyscf/lib/deps/lib
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/deps/lib"
    # libexchcxx.so lives under deps/exchcxx/lib
    BUILD_RPATH              "$ORIGIN/../exchcxx/lib:$ORIGIN/../exchcxx/lib64"
    INSTALL_RPATH            "$ORIGIN/../exchcxx/lib:$ORIGIN/../exchcxx/lib64"
  )
endif()
