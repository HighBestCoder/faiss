#!/bin/bash
# Build a FAISS source tree with the EXACT flags from
# /db/third-party-build/presets/faiss_presets.json (preset faiss-x64-intel),
# but using system gcc instead of /opt/xtools.
#
# Usage:
#   build_faiss_oneapi.sh <SRC_DIR> <BUILD_DIR> <INSTALL_PREFIX>
#
# Required: oneAPI installed at /opt/intel/oneapi.
set -eo pipefail

SRC="${1:?src dir}"
BUILD="${2:?build dir}"
INSTALL="${3:?install prefix}"

# setvars.sh references unbound vars; do not run with `-u`.
source /opt/intel/oneapi/setvars.sh > /dev/null
set -u
echo ">>> MKLROOT=$MKLROOT"

IOMP5_LIB=$(find /opt/intel/oneapi/compiler -type f -name libiomp5.so | sort -V | tail -1)
test -n "$IOMP5_LIB" || { echo "libiomp5.so not found"; exit 1; }
echo ">>> IOMP5_LIB=$IOMP5_LIB"

OMP_INCLUDE_DIR=$(dirname "$(find /opt/intel/oneapi/compiler -type f -name omp.h | sort -V | tail -1)")
echo ">>> OMP_INCLUDE_DIR=$OMP_INCLUDE_DIR"

# Flags copied verbatim from preset faiss-x64-intel.
CFLAGS="-O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -fno-semantic-interposition -I${OMP_INCLUDE_DIR}"
CXXFLAGS="$CFLAGS"
LDFLAGS="-L${MKLROOT}/lib -L/usr/lib/x86_64-linux-gnu"

NPROC="${NPROC:-$(nproc)}"
CMAKE="${CMAKE:-/db/claude_user/.local/bin/cmake}"

rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"

"$CMAKE" "$SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL" \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_MKL=ON \
    -DFAISS_OPT_LEVEL=avx512 \
    -DFAISS_USE_LTO=ON \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DMKL_ROOT="$MKLROOT" \
    -DBLA_VENDOR=Intel10_64lp \
    -DBLA_VENDOR_THREADING=intel \
    -DMKL_THREADING=INTEL \
    -DCMAKE_C_FLAGS="$CFLAGS" \
    -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
    -DOpenMP_C_FLAGS="-fopenmp" \
    -DOpenMP_CXX_FLAGS="-fopenmp" \
    -DOpenMP_C_LIB_NAMES=iomp5 \
    -DOpenMP_CXX_LIB_NAMES=iomp5 \
    -DOpenMP_iomp5_LIBRARY="$IOMP5_LIB"

"$CMAKE" --build . -j "$NPROC"
"$CMAKE" --install .

echo ""
echo ">>> Verifying $INSTALL/lib/libfaiss_avx512.so links MKL + iomp5 ..."
LIB="$INSTALL/lib/libfaiss_avx512.so"
ldd "$LIB" | grep -E 'mkl|iomp|gomp|openblas' || true
if ldd "$LIB" | grep -qE 'gomp|openblas'; then
    echo "ERROR: $LIB links gomp or openblas; aborting."
    exit 1
fi
ldd "$LIB" | grep -q libmkl_core || { echo "ERROR: not linked against libmkl_core"; exit 1; }
ldd "$LIB" | grep -q libiomp5 || { echo "ERROR: not linked against libiomp5"; exit 1; }
echo ">>> OK"
