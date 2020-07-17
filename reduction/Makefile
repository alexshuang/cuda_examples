APPS = main

all: ${APPS}

%: %.cu
	nvcc -std=c++11 -o $@ $< ../common/common.cu
	./$@
