# Third-Party Libraries Build Guide

A simple 3-step process to build all third-party libraries (FAISS, gRPC, nlohmann/json, spdlog, Intel MKL, OpenMP).

## Prerequisites

- Docker installed and running
- Base toolchain image: `iyadactian/lnxbld:v1`
- At least 20GB free disk space

## Build Steps

### Step 1: Build Base Image

Build the base image with all dependencies (CMake, Intel MKL, build tools, FAISS source code):

```bash
cd /src/faiss-buld-docs
docker build -f Dockerfile.base -t faiss:v1.13.1-base .
```

**Build time:** ~10-15 minutes  
**Note:** Only need to rebuild when dependencies change.

### Step 2: Build All Libraries

Build all third-party libraries in one go:

```bash
docker build -f Dockerfile -t third-party:v2.0 .
```

**Build time:** ~30-60 minutes (depending on CPU cores)  
**Libraries built:**
- nlohmann/json v3.12.0
- spdlog v1.16.0
- gRPC v1.71.0
- FAISS v1.9.0
- Intel MKL libraries
- OpenMP libraries

### Step 3: Extract Libraries

Copy the compiled libraries from the container to your host:

```bash
# Start container with mounted volume
docker run -it --rm \
    -v /builds:/builds \
    third-party:v2.0 bash

# Inside container, verify installation
ls -lh /builds/cortex.core/third-party/

# Exit container (libraries are already in /builds/cortex.core/third-party/)
exit
```

**Alternatively, copy from stopped container:**

```bash
# Create temporary container
docker create --name temp-extract third-party:v2.0

# Copy entire third-party directory
docker cp temp-extract:/builds/cortex.core/third-party /builds/cortex.core/

# Remove temporary container
docker rm temp-extract
```

## Output Directory Structure

After extraction, your directory structure will be:

```
/builds/cortex.core/third-party/
├── faiss/
│   └── linux-x64/
│       ├── include/
│       └── lib/
├── grpc/
│   └── linux-x64/
│       ├── bin/
│       ├── include/
│       └── lib/
├── intel-mkl/
│   └── linux-x64/
│       └── lib/
├── openmp/
│   └── linux-x64/
│       ├── include/
│       └── lib/
├── nlohmann/
│   └── *.hpp
└── spdlog/
    ├── include/
    └── src/
```

## Environment Variables

The libraries are configured with the following environment variables (automatically set in the container):

```bash
export LD_LIBRARY_PATH=/builds/cortex.core/third-party/faiss/linux-x64/lib:\
/builds/cortex.core/third-party/grpc/linux-x64/lib:\
/builds/cortex.core/third-party/intel-mkl/linux-x64/lib:\
/builds/cortex.core/third-party/openmp/linux-x64/lib:${LD_LIBRARY_PATH}

export CPATH=/builds/cortex.core/third-party/nlohmann:\
/builds/cortex.core/third-party/spdlog/include:\
/builds/cortex.core/third-party/openmp/linux-x64/include:\
/builds/cortex.core/third-party/faiss/linux-x64/include:\
/builds/cortex.core/third-party/grpc/linux-x64/include:${CPATH}

export PATH=/builds/cortex.core/third-party/grpc/linux-x64/bin:${PATH}
```

## Compiler Optimization

Libraries are compiled with `-march=haswell -mtune=haswell` for AVX2 support as it is widely available.

## Troubleshooting

### Disk Space Issues

If you encounter "no space left on device" errors:

```bash
# Clean up Docker cache
docker system prune -a -f --volumes

# Check disk space
df -h
```

### Rebuild Individual Steps

If a build fails, you can rebuild from that step:

```bash
# Rebuild only the final image (if base is already built)
docker build -f Dockerfile -t third-party:v2.0 .

# Force rebuild from scratch
docker build --no-cache -f Dockerfile -t third-party:v2.0 .
```

## Library Versions

**FAISS:** v1.13.1 (with Intel MKL backend)
- **gRPC:** v1.71.0
- **nlohmann/json:** v3.12.0
- **spdlog:** v1.16.0
- **Intel MKL:** 2025.3
- **CMake:** 4.0.0

## Quick Commands Summary

```bash
# Full build from scratch
docker build -f Dockerfile.base -t faiss:v1.13.1-base .
docker build -f Dockerfile -t third-party:v2.0 .

# Extract libraries
docker run -it --rm -v /builds:/builds third-party:v2.0 bash
# (libraries already in /builds/cortex.core/third-party/)

# Verify installation
docker run --rm third-party:v2.0 ls -lh /builds/cortex.core/third-party/
```
