# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(bench_ivf_selector EXCLUDE_FROM_ALL bench_ivf_selector.cpp)
target_link_libraries(bench_ivf_selector PRIVATE faiss_avx512 ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(bench_ivf_selector PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)

add_executable(bench_result_handler_overhead EXCLUDE_FROM_ALL
  bench_result_handler_overhead.cpp)
target_link_libraries(bench_result_handler_overhead PRIVATE faiss_avx512 ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX)
target_compile_options(bench_result_handler_overhead PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)

if(FAISS_ENABLE_COMPRESSED_STORAGE)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LZ4 REQUIRED liblz4)
  pkg_check_modules(ZSTD REQUIRED libzstd)

  add_executable(bench_hnsw_compressed EXCLUDE_FROM_ALL bench_hnsw_compressed.cpp)
  target_link_libraries(bench_hnsw_compressed PRIVATE faiss ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX ${LZ4_LIBRARIES} ${ZSTD_LIBRARIES})
  target_include_directories(bench_hnsw_compressed PRIVATE ${LZ4_INCLUDE_DIRS} ${ZSTD_INCLUDE_DIRS})
  target_compile_definitions(bench_hnsw_compressed PRIVATE FAISS_ENABLE_COMPRESSED_STORAGE)
endif()
