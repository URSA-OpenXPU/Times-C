CC = nvcc
I_DIR = -I./include 
BIN = ./bin/test

SRC = ./test/test_kshape.cu

CXXFLAGS := -std=c++11
CXX_SUPPORTS_CXX14 := $(shell $(CXX) -std=c++14 -E -x c++ - </dev/null >/dev/null 2>&1 && echo "1")
CXX_SUPPORTS_CXX17 := $(shell $(CXX) -std=c++17 -E -x c++ - </dev/null >/dev/null 2>&1 && echo "1")
ifeq ($(CXX_SUPPORTS_CXX14), 1)
  CXXFLAGS := -std=c++14
endif
ifeq ($(CXX_SUPPORTS_CXX17), 1)
  CXXFLAGS := -std=c++17
endif

CLEAN = clean
COMPUTE = sm_70
RDC = true
MATRIX_LIB = cublas
FFT_LIB = cufft
EIGEN_LIB = cusolver
RAND_LIB = curand
$(BIN):$(SRC)
	$(CC) $(CXXFLAGS) $^ -o $@ -l$(MATRIX_LIB) -l$(FFT_LIB) -l$(EIGEN_LIB) -l$(RAND_LIB) -arch=$(COMPUTE) -rdc=$(RDC) -lcudadevrt $(I_DIR)

.PHONY:$(CLEAN)
$(CLEAN):
	rm -rf $(BIN)

