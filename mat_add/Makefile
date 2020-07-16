APPS = mat_add_2d_2d  mat_add_1d_1d mat_add_2d_1d

all: ${APPS}

%: %.cu
	nvcc -std=c++11 -o $@ $< ../common.cu
	./$@
