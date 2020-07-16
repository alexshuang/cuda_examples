#include <cuda_runtime.h>
#include "common.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>

void rand_array(float **vec, ssize_t size)
{
	int seed = 42;
	srand(seed);

	float *p = *vec = (float *)malloc(size * sizeof(float));

	for (int i = 0; i < size; i++) {
		p[i] = (float)((rand() & 0xff) / 10.0f);
		//printf("%.2f\n", p[i]);
	}
}

void zero_array(float **vec, ssize_t size)
{
	*vec = (float *)calloc(size, sizeof(float));
}

void empty_array(float **vec, ssize_t size)
{
	*vec = (float *)malloc(size * sizeof(float));
}

void to_gpu(float **dev_p, const float *host_p, ssize_t size)
{
	cudaCHECK(cudaMalloc(dev_p, size * sizeof(float)));
	cudaCHECK(cudaMemcpy(*dev_p, host_p, size * sizeof(float), cudaMemcpyHostToDevice));
}

void from_gpu(float **dest, const float *dev_p, ssize_t size)
{
	*dest = (float *)malloc(size * sizeof(float));
	if (!*dest)
		perror("malloc");
	cudaCHECK(cudaMemcpy(*dest, dev_p, size * sizeof(float), cudaMemcpyDeviceToHost));
}

void empty_array_gpu(float **dev_p, ssize_t size)
{
	cudaCHECK(cudaMalloc(dev_p, size * sizeof(float)));
}

bool validate(float *host_p, float *dev_p, ssize_t size)
{
	bool res = true;
	double eps = 1e-8;

	for (ssize_t i = 0; i < size; i++) {
		if (std::abs(dev_p[i] - host_p[i]) > eps) {
			printf("Array doesn't match at index %d, host[%d] == %f, "
					"device[%d] == %f\n", (int)i, (int)i, host_p[i], (int)i, dev_p[i]);
			res = false;
			break;
		}
	}

	return res;
}
