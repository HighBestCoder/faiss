FAISS_ROOT   := /ceph/faiss-dev
V0_LIB       := $(FAISS_ROOT)/llm/faiss-1.14.1-origin/lib
OPT_LIB      := $(FAISS_ROOT)/install/lib
INCLUDE      := $(FAISS_ROOT)/install/include

CXX          := g++
CXXFLAGS     := -O3 -march=native -mtune=native -std=c++17 -I$(INCLUDE)
MKL_FLAGS    := -L/usr/lib/x86_64-linux-gnu \
                -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
                -liomp5 -lpthread -lm -ldl

.PHONY: all clean bench-v0 bench-opt lib

# Build everything
all: bench-v0 bench-opt

# V0 baseline bench (linked against original v1.14.1 lib)
bench-v0: faiss-1.14.1-origin/bench

faiss-1.14.1-origin/bench: faiss-1.14.1-origin/bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< \
		-L$(V0_LIB) -lfaiss_avx512 \
		$(MKL_FLAGS) \
		-Wl,-rpath,$(V0_LIB)

# Optimized bench (linked against our modified lib)
bench-opt: bench_task

bench_task: bench_task.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< \
		-L$(OPT_LIB) -lfaiss_avx512 \
		$(MKL_FLAGS) \
		-Wl,-rpath,$(OPT_LIB)

# Rebuild FAISS library (runs cmake + make)
lib:
	cd $(FAISS_ROOT) && bash llm/build.sh

clean:
	rm -f bench_task faiss-1.14.1-origin/bench
