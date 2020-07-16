APPS = main

all: ${APPS}

%: %.cu
	nvcc -std=c++11 -arch=sm_60 -o $@ $< ../common/common.cu
	./$@
