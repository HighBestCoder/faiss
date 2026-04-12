#!/bin/bash
# =============================================================
# V0 FAISS Build Script
# Builds original FAISS v1.14.1 from tag with -march=native
# Uses a temporary git worktree to avoid disturbing current branch
#
# Output: llm/faiss-1.14.1-origin/{lib,include}
# =============================================================
set -euo pipefail

FAISS_ROOT="/ceph/faiss-dev"
V0_INSTALL="${FAISS_ROOT}/llm/faiss-1.14.1-origin"
V0_WORKTREE="/tmp/faiss-v0-build"
V0_BUILD="${V0_WORKTREE}/build_v0"
TAG="v1.14.1"

CC="${CC:-gcc}"
CXX="${CXX:-g++}"
NPROC="${NPROC:-$(nproc)}"

# Intel MKL (system intel-mkl package)
MKL_LIB_DIR="/usr/lib/x86_64-linux-gnu"
IOMP5_LIB="${MKL_LIB_DIR}/libiomp5.so"

CFLAGS="-O3 -march=native -mtune=native -ffast-math -funroll-loops -fno-semantic-interposition"

echo "========================================"
echo "  V0 FAISS Build Script (march=native)"
echo "  Tag:     ${TAG}"
echo "  Install: ${V0_INSTALL}"
echo "  CC:      ${CC} ($(${CC} -dumpversion 2>/dev/null || echo '?'))"
echo "  Jobs:    ${NPROC}"
echo "  CFLAGS:  ${CFLAGS}"
echo "========================================"

# ============================================
# Pre-flight checks
# ============================================
for f in \
    "/usr/include/mkl/mkl.h" \
    "${MKL_LIB_DIR}/libmkl_core.so" \
    "${MKL_LIB_DIR}/libmkl_intel_lp64.so" \
    "${MKL_LIB_DIR}/libmkl_intel_thread.so" \
    "${IOMP5_LIB}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing ${f}"
        echo "Please install: apt install intel-mkl"
        exit 1
    fi
done

# ============================================
# Create worktree from v1.14.1 tag
# ============================================
echo ""
echo ">>> Creating worktree from ${TAG}..."
cd "${FAISS_ROOT}"

# Clean up any leftover worktree
if [ -d "${V0_WORKTREE}" ]; then
    echo "  Removing existing worktree at ${V0_WORKTREE}..."
    git worktree remove "${V0_WORKTREE}" --force 2>/dev/null || rm -rf "${V0_WORKTREE}"
fi

git worktree add "${V0_WORKTREE}" "${TAG}" --detach
echo "  Worktree created at ${V0_WORKTREE}"
echo "  HEAD: $(cd ${V0_WORKTREE} && git log --oneline -1)"

# ============================================
# CMake Configure
# ============================================
echo ""
echo "========================================"
echo "  CMake Configure"
echo "========================================"
rm -rf "${V0_BUILD}"
mkdir -p "${V0_BUILD}"
cd "${V0_BUILD}"

cmake "${V0_WORKTREE}" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${V0_INSTALL}" \
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
    -DCMAKE_CXX_FLAGS="${CFLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${MKL_LIB_DIR}" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${MKL_LIB_DIR}" \
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
echo "  Building FAISS ${TAG} with ${NPROC} cores ..."
echo "========================================"
make -j"${NPROC}"

# ============================================
# Install (backup old lib first)
# ============================================
echo ""
echo "========================================"
echo "  Installing to ${V0_INSTALL}"
echo "========================================"
if [ -d "${V0_INSTALL}/lib" ]; then
    echo "  Backing up old lib to ${V0_INSTALL}/lib.bak.haswell"
    rm -rf "${V0_INSTALL}/lib.bak.haswell"
    mv "${V0_INSTALL}/lib" "${V0_INSTALL}/lib.bak.haswell"
fi
make install

# ============================================
# Verify
# ============================================
echo ""
echo "========================================"
echo "  Verification"
echo "========================================"
echo "Libraries:"
ls -lh "${V0_INSTALL}/lib/"libfaiss* 2>/dev/null || echo "  (none)"

echo ""
echo "Linked MKL/OpenMP:"
ldd "${V0_INSTALL}/lib/libfaiss_avx512.so" 2>/dev/null | grep -E "mkl|iomp" || true

echo ""
echo "AVX-512 check (zmm count in search_from_candidates):"
ADDR=$(nm -D "${V0_INSTALL}/lib/libfaiss_avx512.so" | grep "search_from_candidates" | grep -v panorama | grep " T " | awk '{print "0x"$1}')
if [ -n "$ADDR" ]; then
    END=$(printf "0x%x" $(($ADDR + 0x1000)))
    ZMM_COUNT=$(objdump -d --start-address=${ADDR} --stop-address=${END} "${V0_INSTALL}/lib/libfaiss_avx512.so" 2>/dev/null | grep -c "zmm" || true)
    echo "  zmm instructions: ${ZMM_COUNT}"
    if [ "${ZMM_COUNT}" -gt 0 ]; then
        echo "  ✓ AVX-512 auto-vectorization confirmed!"
    else
        echo "  ✗ WARNING: No AVX-512 auto-vectorization detected"
    fi
else
    echo "  (symbol not found)"
fi

# ============================================
# Cleanup worktree
# ============================================
echo ""
echo ">>> Cleaning up worktree..."
cd "${FAISS_ROOT}"
git worktree remove "${V0_WORKTREE}" --force 2>/dev/null || rm -rf "${V0_WORKTREE}"

echo ""
echo "========================================"
echo "  V0 Build Complete (march=native)"
echo "  Library: ${V0_INSTALL}/lib/libfaiss_avx512.so"
echo "========================================"
echo ""
echo "Next: recompile bench binary:"
echo "  make -C llm clean && make -C llm bench-v0"
