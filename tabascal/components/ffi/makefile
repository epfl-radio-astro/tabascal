JAXLIB_HOME ?= $(shell python3 -c "import jaxlib; import os; path = os.path.dirname(jaxlib.__file__); print(path)")

ifeq (, $(shell which nvcc 2> /dev/null))
CUDACXX ?= /usr/local/cuda/bin/nvcc
else
CUDACXX ?= nvcc
endif

CXX ?= c++

CUDAFLAGS ?= -O2 -DNDEBUG  \
			  -gencode arch=compute_70,code=compute_70 \
			  -gencode arch=compute_80,code=compute_80 \
			  -gencode arch=compute_90,code=compute_90 \
			  -gencode arch=compute_90a,code=sm_90a

CXXFLAGS ?= -O2 -DNDEBUG -march=native -Wno-return-type -Wno-attributes


CXXFLAGS += -std=c++17 -fPIC -I $(JAXLIB_HOME)/include

CUDAFLAGS += -std=c++17 -Xcompiler -fPIC -I $(JAXLIB_HOME)/include \
			 -Xcompiler -Wno-attributes \
			 -Xcompiler -Wno-return-type \
			 -Xcudafe "--diag_suppress=940 --diag_suppress=2473"


CXX_SOURCES = $(wildcard *.cpp)
CUDA_SOURCES = $(wildcard *.cu)
CXX_OBJECTS = $(patsubst %.cpp,%.o,$(CXX_SOURCES))
CUDA_OBJECTS = $(patsubst %.cu,%.o,$(CUDA_SOURCES))

all: cpu gpu

cpu: tabascal.so

tabascal.so: $(CXX_OBJECTS)
	$(CXX) $^ -shared -o $@

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

gpu: tabascal_gpu.so

tabascal_gpu.so: $(CUDA_OBJECTS)
	$(CUDACXX) $^ -shared -o $@

%.o: %.cu
	$(CUDACXX) -c $< -o $@ $(CUDAFLAGS)

clean:
	-rm -f $(CXX_OBJECTS) $(CUDA_OBJECTS) tabascal.so tabascal_gpu.so

.PHONY: clean all cpu gpu
