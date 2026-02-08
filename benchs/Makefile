BENCH_BIN = ../build/benchs/bench_hnsw_compare
DATASET_DIR = /src/faiss-dev/dataset
RESULT_DIR = ../perf-test

OMP_THREADS ?= 8
M ?= 16
EF_CONSTRUCTION ?= 100
EF_SEARCH ?= 50,100,200

COMMON_ARGS = -M $(M) -efConstruction $(EF_CONSTRUCTION) -efSearch $(EF_SEARCH)

# --- Rebuild pipeline benchmark ---
REBUILD_BIN = ../build/benchs/bench_rebuild_perf
REBUILD_M ?= 16
REBUILD_EFC ?= 40
REBUILD_EFS ?= 64
REBUILD_DEL ?= 10
REBUILD_ARGS = -M $(REBUILD_M) -efConstruction $(REBUILD_EFC) -efSearch $(REBUILD_EFS) -delete_pct $(REBUILD_DEL)

.PHONY: all test test-l2 test-angular test-sift test-gist test-glove test-nytimes test-deep clean help
.PHONY: share-test rebuild-perf rebuild-perf-sift rebuild-perf-gist rebuild-perf-glove rebuild-perf-nytimes

all: test

help:
	@echo "HNSW Cache Optimization Benchmark"
	@echo ""
	@echo "Usage:"
	@echo "  make test              - Run all benchmarks"
	@echo "  make test-l2           - Run L2/Euclidean datasets (SIFT, GIST)"
	@echo "  make test-angular      - Run angular/IP datasets (GloVe, NYTimes, Deep)"
	@echo "  make test-sift         - Run SIFT-1M benchmark"
	@echo "  make test-gist         - Run GIST-960 benchmark"
	@echo "  make test-glove        - Run GloVe-100 benchmark"
	@echo "  make test-nytimes      - Run NYTimes-256 benchmark"
	@echo "  make test-deep         - Run Deep-96 benchmark (10M vectors, slow)"
	@echo ""
	@echo "  make share-test        - Run SharedVectorStore rebuild benchmark (all datasets)"
	@echo "  make rebuild-perf      - Same as share-test"
	@echo "  make rebuild-perf-sift - Rebuild benchmark on SIFT-1M"
	@echo "  make rebuild-perf-gist - Rebuild benchmark on GIST-960"
	@echo "  make rebuild-perf-glove- Rebuild benchmark on GloVe-100"
	@echo "  make rebuild-perf-nytimes - Rebuild benchmark on NYTimes-256"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  OMP_THREADS=$(OMP_THREADS)"
	@echo "  M=$(M)"
	@echo "  EF_CONSTRUCTION=$(EF_CONSTRUCTION)"
	@echo "  EF_SEARCH=$(EF_SEARCH)"
	@echo ""
	@echo "Rebuild options:"
	@echo "  REBUILD_M=$(REBUILD_M)"
	@echo "  REBUILD_EFC=$(REBUILD_EFC)"
	@echo "  REBUILD_EFS=$(REBUILD_EFS)"
	@echo "  REBUILD_DEL=$(REBUILD_DEL)"

test: test-l2 test-angular

test-l2: test-sift test-gist

test-angular: test-glove test-nytimes

test-sift:
	@echo "=== Running SIFT-1M Benchmark (L2) ==="
	@mkdir -p $(RESULT_DIR)
	OMP_NUM_THREADS=$(OMP_THREADS) $(BENCH_BIN) $(DATASET_DIR)/sift-128-euclidean.hdf5 $(COMMON_ARGS) 2>&1 | tee $(RESULT_DIR)/sift-latest.log
	@echo "Results saved to $(RESULT_DIR)/sift-latest.log"

test-gist:
	@echo "=== Running GIST-960 Benchmark (L2) ==="
	@mkdir -p $(RESULT_DIR)
	OMP_NUM_THREADS=$(OMP_THREADS) $(BENCH_BIN) $(DATASET_DIR)/gist-960-euclidean.hdf5 $(COMMON_ARGS) 2>&1 | tee $(RESULT_DIR)/gist-latest.log
	@echo "Results saved to $(RESULT_DIR)/gist-latest.log"

test-glove:
	@echo "=== Running GloVe-100 Benchmark (Angular/IP) ==="
	@mkdir -p $(RESULT_DIR)
	OMP_NUM_THREADS=$(OMP_THREADS) $(BENCH_BIN) $(DATASET_DIR)/glove-100-angular.hdf5 $(COMMON_ARGS) 2>&1 | tee $(RESULT_DIR)/glove-latest.log
	@echo "Results saved to $(RESULT_DIR)/glove-latest.log"

test-nytimes:
	@echo "=== Running NYTimes-256 Benchmark (Angular/IP) ==="
	@mkdir -p $(RESULT_DIR)
	OMP_NUM_THREADS=$(OMP_THREADS) $(BENCH_BIN) $(DATASET_DIR)/nytimes-256-angular.hdf5 $(COMMON_ARGS) 2>&1 | tee $(RESULT_DIR)/nytimes-latest.log
	@echo "Results saved to $(RESULT_DIR)/nytimes-latest.log"

test-deep:
	@echo "=== Running Deep-96 Benchmark (Angular/IP, 10M vectors - this will take a while) ==="
	@mkdir -p $(RESULT_DIR)
	OMP_NUM_THREADS=$(OMP_THREADS) $(BENCH_BIN) $(DATASET_DIR)/deep-image-96-angular.hdf5 $(COMMON_ARGS) 2>&1 | tee $(RESULT_DIR)/deep-latest.log
	@echo "Results saved to $(RESULT_DIR)/deep-latest.log"

# --- SharedVectorStore rebuild pipeline ---
share-test: rebuild-perf

rebuild-perf: rebuild-perf-sift rebuild-perf-gist rebuild-perf-glove rebuild-perf-nytimes

rebuild-perf-sift:
	@echo "=== Rebuild Pipeline: SIFT-128 (L2) ==="
	@mkdir -p $(RESULT_DIR)
	OMP_NUM_THREADS=$(OMP_THREADS) $(REBUILD_BIN) $(DATASET_DIR)/sift-128-euclidean.hdf5 $(REBUILD_ARGS) 2>&1 | tee $(RESULT_DIR)/rebuild-sift-latest.log
	@echo "Results saved to $(RESULT_DIR)/rebuild-sift-latest.log"

rebuild-perf-gist:
	@echo "=== Rebuild Pipeline: GIST-960 (L2) ==="
	@mkdir -p $(RESULT_DIR)
	OMP_NUM_THREADS=$(OMP_THREADS) $(REBUILD_BIN) $(DATASET_DIR)/gist-960-euclidean.hdf5 $(REBUILD_ARGS) 2>&1 | tee $(RESULT_DIR)/rebuild-gist-latest.log
	@echo "Results saved to $(RESULT_DIR)/rebuild-gist-latest.log"

rebuild-perf-glove:
	@echo "=== Rebuild Pipeline: GloVe-100 (Angular/IP) ==="
	@mkdir -p $(RESULT_DIR)
	OMP_NUM_THREADS=$(OMP_THREADS) $(REBUILD_BIN) $(DATASET_DIR)/glove-100-angular.hdf5 $(REBUILD_ARGS) 2>&1 | tee $(RESULT_DIR)/rebuild-glove-latest.log
	@echo "Results saved to $(RESULT_DIR)/rebuild-glove-latest.log"

rebuild-perf-nytimes:
	@echo "=== Rebuild Pipeline: NYTimes-256 (Angular/IP) ==="
	@mkdir -p $(RESULT_DIR)
	OMP_NUM_THREADS=$(OMP_THREADS) $(REBUILD_BIN) $(DATASET_DIR)/nytimes-256-angular.hdf5 $(REBUILD_ARGS) 2>&1 | tee $(RESULT_DIR)/rebuild-nytimes-latest.log
	@echo "Results saved to $(RESULT_DIR)/rebuild-nytimes-latest.log"

clean:
	rm -f $(RESULT_DIR)/*-latest.log
