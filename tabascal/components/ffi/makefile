JAXLIB_HOME ?= $(shell python3 -c "import jaxlib; import os; path = os.path.dirname(jaxlib.__file__); print(path)")

ifeq (, $(shell which nvcc 2> /dev/null))
CUDACXX ?= /usr/local/cuda/bin/nvcc
else
CUDACXX ?= nvcc
endif

ifeq (, $(shell which hipcc 2> /dev/null))
HIPCXX ?= /opt/rocm/bin/hipcc
else
HIPCXX ?= hipcc
endif

CXX ?= c++

CUDAFLAGS ?= -O2 -DNDEBUG  \
			  -gencode arch=compute_70,code=compute_70 \
			  -gencode arch=compute_80,code=compute_80 \
			  -gencode arch=compute_90,code=compute_90 \
			  -gencode arch=compute_90a,code=sm_90a

HIPFLAGS ?= -O2 -DNDEBUG \
			--offload-arch=gfx906 \
			--offload-arch=gfx908 \
			--offload-arch=gfx90a \
			--offload-arch=gfx1030 \
			--offload-arch=gfx1100

CXXFLAGS ?= -O2 -DNDEBUG -march=native -Wno-return-type -Wno-attributes


CXXFLAGS += -std=c++17 -fPIC -I $(JAXLIB_HOME)/include

CUDAFLAGS += -std=c++17 -Xcompiler -fPIC -I $(JAXLIB_HOME)/include \
			 -Xcompiler -Wno-attributes \
			 -Xcompiler -Wno-return-type \
			 -Xcudafe "--diag_suppress=940 --diag_suppress=2473"

HIPFLAGS += -std=c++17 -fPIC -I $(JAXLIB_HOME)/include \
			-Wno-return-type -Wno-attributes


CXX_SOURCES = $(wildcard *.cpp)
CUDA_SOURCES = $(wildcard *.cu)
CXX_OBJECTS = $(patsubst %.cpp,%.o,$(CXX_SOURCES))
CUDA_OBJECTS = $(patsubst %.cu,%.o,$(CUDA_SOURCES))
HIP_OBJECTS = $(patsubst %.cu,%.hip.o,$(CUDA_SOURCES))

all: cpu cuda

cpu: tabascal.so

tabascal.so: $(CXX_OBJECTS)
	$(CXX) $^ -shared -o $@

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

cuda: tabascal_cuda.so

tabascal_cuda.so: $(CUDA_OBJECTS)
	$(CUDACXX) $^ -shared -o $@

%.o: %.cu
	$(CUDACXX) -c $< -o $@ $(CUDAFLAGS)

hip: tabascal_hip.so

tabascal_hip.so: $(HIP_OBJECTS)
	$(HIPCXX) $^ -shared -o $@

%.hip.o: %.cu
	$(HIPCXX) -c $< -o $@ $(HIPFLAGS)

clean:
	-rm -f $(CXX_OBJECTS) $(CUDA_OBJECTS) $(HIP_OBJECTS) tabascal.so tabascal_cuda.so tabascal_hip.so

.PHONY: clean all cpu cuda hip
