set(GPU_SRCS
  md_contract_j.cu unrolled_md_j.cu unrolled_md_j_4dm.cu
  md_j_driver.cu md_pairdata.c
  contract_int3c2e.cu
  pbc_md_contract_j.cu
  contract_Et.cu
)

if(NOT USE_SYCL)
  list(APPEND GPU_SRCS md_indices.cu)
endif()

add_library(gvhf_md SHARED ${GPU_SRCS})

set_target_properties(gvhf_md PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

if(NOT USE_SYCL)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
  set_target_properties(gvhf_md PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries(gvhf_md PRIVATE OpenMP::OpenMP_C)
gpu4pyscf_configure_gpu_target(gvhf_md)

if(USE_HIP)
  # Compile md_contract_j.cu at -O1 on HIP. With amdclang++ from ROCm 7.2.0
  # (LLVM 22), -O2 miscompiles the generic md_j_1dm_kernel / md_j_4dm_kernel
  # for high angular momentum quartets (l >= 4, as exercised by
  # scf/tests/test_rhf.py::test_get_j1): the kernel hits a GPU memory-aperture
  # violation (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION) and the process
  # aborts. The same source is correct at -O0 and -O1, so only this file is
  # lowered. The unrolled md_j kernels and the rest of the library stay at -O2.
  set_source_files_properties(md_contract_j.cu PROPERTIES COMPILE_OPTIONS "-O1")
endif()
