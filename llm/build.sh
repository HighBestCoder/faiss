#!/bin/bash
# =============================================================
# FAISS Build Script
# Builds /ceph/faiss-dev with Intel MKL + Intel OpenMP (libiomp5)
# Build strategy adapted from llm/faiss-build/Dockerfile*
#
# Dependencies (must be pre-installed):
#   - intel-mkl package (apt install intel-mkl)
#   - libiomp5 (comes with intel-mkl or libomp-dev)
#   - gcc/g++
#   - cmake >= 3.24
# =============================================================
set -euo pipefail

# ============================================
# Fixed paths — do not auto-detect
# ============================================
FAISS_SOURCE_DIR="/ceph/faiss-dev"
BUILD_DIR="${FAISS_SOURCE_DIR}/build_mkl"
INSTALL_PREFIX="${FAISS_SOURCE_DIR}/install"

CC="${CC:-gcc}"
CXX="${CXX:-g++}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
NPROC="${NPROC:-$(nproc)}"

# Intel MKL (system intel-mkl package)
MKL_INCLUDE_DIR="/usr/include/mkl"
MKL_LIB_DIR="/usr/lib/x86_64-linux-gnu"

# Intel OpenMP (libiomp5)
IOMP5_LIB="/usr/lib/x86_64-linux-gnu/libiomp5.so"

# ============================================
# Pre-flight checks
# ============================================
echo "========================================"
echo "  FAISS Build Script"
echo "  Source:  ${FAISS_SOURCE_DIR}"
echo "  Build:   ${BUILD_DIR}"
echo "  Install: ${INSTALL_PREFIX}"
echo "========================================"

MISSING=0
for f in \
    "${MKL_INCLUDE_DIR}/mkl.h" \
    "${MKL_LIB_DIR}/libmkl_core.so" \
    "${MKL_LIB_DIR}/libmkl_intel_lp64.so" \
    "${MKL_LIB_DIR}/libmkl_intel_thread.so" \
    "${IOMP5_LIB}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing ${f}"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "Please install: apt install intel-mkl"
    exit 1
fi

echo ">>> MKL:     ${MKL_LIB_DIR}/libmkl_core.so"
echo ">>> libiomp5: ${IOMP5_LIB}"
echo ">>> CC:      ${CC} ($(${CC} -dumpversion 2>/dev/null || echo '?'))"
echo ">>> CXX:     ${CXX} ($(${CXX} -dumpversion 2>/dev/null || echo '?'))"
echo ">>> Build:   ${BUILD_TYPE}"
echo ">>> Jobs:    ${NPROC}"
echo ""

# ============================================
# Compiler/linker flags (from Dockerfile)
# ============================================
CFLAGS="-O3 -march=native -mtune=native -ffast-math -funroll-loops -fno-semantic-interposition"
CXXFLAGS="${CFLAGS}"
LDFLAGS="-L${MKL_LIB_DIR}"

# ============================================
# Prepare build directory
# ============================================
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# ============================================
# CMake Configure
# ============================================
echo "========================================"
echo "  CMake Configure"
echo "========================================"
cmake "${FAISS_SOURCE_DIR}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_MKL=ON \
    -DFAISS_OPT_LEVEL=avx512 \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_USE_LTO=ON \
    \
    -DMKL_ROOT="/usr" \
    -DBLA_VENDOR=Intel10_64lp \
    -DBLA_VENDOR_THREADING=intel \
    \
    -DCMAKE_C_FLAGS="${CFLAGS}" \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS}" \
    \
    -DOpenMP_C_FLAGS="-fopenmp" \
    -DOpenMP_CXX_FLAGS="-fopenmp" \
    -DOpenMP_C_LIB_NAMES="iomp5" \
    -DOpenMP_CXX_LIB_NAMES="iomp5" \
    -DOpenMP_iomp5_LIBRARY="${IOMP5_LIB}"

# ============================================
# Build
# ============================================
echo ""
echo "========================================"
echo "  Building FAISS with ${NPROC} cores ..."
echo "========================================"
make -j"${NPROC}"

# ============================================
# Install
# ============================================
echo ""
echo "========================================"
echo "  Installing to ${INSTALL_PREFIX}"
echo "========================================"
make install

# ============================================
# Verify
# ============================================
echo ""
echo "========================================"
echo "  Build Complete"
echo "========================================"
echo "Libraries:"
ls -lh "${INSTALL_PREFIX}/lib/"libfaiss* 2>/dev/null || echo "  (none)"
echo ""
echo "Linked MKL/OpenMP:"
ldd "${INSTALL_PREFIX}/lib/libfaiss.so" 2>/dev/null | grep -E "mkl|iomp" || true
echo ""
echo "FAISS installed to: ${INSTALL_PREFIX}"
echo "========================================"
