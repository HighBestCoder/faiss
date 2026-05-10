FAISS_ROOT   := /db/faiss
V0_113_LIB   := $(FAISS_ROOT)/llm/faiss-1.13.1-origin/lib
V0_LIB       := $(FAISS_ROOT)/llm/faiss-1.14.1-origin/lib
OPT_LIB      := $(FAISS_ROOT)/install/lib
OPT_INCLUDE  := $(FAISS_ROOT)/install/include
V0_INCLUDE   := $(FAISS_ROOT)/llm/faiss-1.14.1-origin/include
V0_113_INCLUDE := $(FAISS_ROOT)/llm/faiss-1.13.1-origin/include

CXX          := g++
CXXFLAGS     := -O3 -march=native -mtune=native -std=c++17

# oneAPI MKL + iomp5 (matches Dockerfile / build_faiss_oneapi.sh)
MKLROOT      ?= /opt/intel/oneapi/mkl/2026.0
IOMP5_DIR    := $(shell dirname $$(find /opt/intel/oneapi/compiler -name libiomp5.so | sort -V | tail -1))
MKL_FLAGS    := -L$(MKLROOT)/lib \
                -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
                -L$(IOMP5_DIR) -liomp5 \
                -lpthread -lm -ldl \
                -Wl,-rpath,$(MKLROOT)/lib -Wl,-rpath,$(IOMP5_DIR)

.PHONY: all clean bench-v0 bench-v0-113 bench-opt bench-rabitq lib

all: bench-v0-113 bench-v0 bench-opt

bench-v0-113: faiss-1.13.1-origin/bench

faiss-1.13.1-origin/bench: faiss-1.14.1-origin/bench.cpp
	mkdir -p faiss-1.13.1-origin
	$(CXX) $(CXXFLAGS) -I$(V0_113_INCLUDE) -o $@ $< \
		-L$(V0_113_LIB) -lfaiss_avx512 \
		$(MKL_FLAGS) \
		-Wl,-rpath,$(V0_113_LIB)

bench-v0: faiss-1.14.1-origin/bench

faiss-1.14.1-origin/bench: faiss-1.14.1-origin/bench.cpp
	$(CXX) $(CXXFLAGS) -I$(V0_INCLUDE) -o $@ $< \
		-L$(V0_LIB) -lfaiss_avx512 \
		$(MKL_FLAGS) \
		-Wl,-rpath,$(V0_LIB)

bench-opt: bench_task

bench_task: bench_task.cpp
	$(CXX) $(CXXFLAGS) -I$(OPT_INCLUDE) -o $@ $< \
		-L$(OPT_LIB) -lfaiss_avx512 \
		$(MKL_FLAGS) \
		-Wl,-rpath,$(OPT_LIB)

bench-rabitq: bench_rabitq

bench_rabitq: bench_rabitq.cpp
	$(CXX) $(CXXFLAGS) -I$(OPT_INCLUDE) -o $@ $< \
		-L$(OPT_LIB) -lfaiss_avx512 \
		$(MKL_FLAGS) \
		-Wl,-rpath,$(OPT_LIB)

lib:
	cd $(FAISS_ROOT) && bash llm/build_faiss_oneapi.sh $(FAISS_ROOT) $(FAISS_ROOT)/build $(FAISS_ROOT)/install

clean:
	rm -f bench_task bench_rabitq faiss-1.14.1-origin/bench faiss-1.13.1-origin/bench
