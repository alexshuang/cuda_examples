APPS = main

all: ${APPS}

%: %.cu
	nvcc -std=c++11 -G -o $@ $< ../common/common.cu
	./$@
