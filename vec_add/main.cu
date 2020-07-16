#include <cuda_runtime.h>
#include <cstdio>
#include "common.h"
#include <cstdlib>

__global__ void gpu_add(float *c, const float *a, const float *b, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size)
		c[idx] = a[idx] + b[idx];
}

void cpu_add(float *c, const float *a, const float *b, int size)
{
	for (int i = 0; i < size; i++)
		c[i] = a[i] + b[i];
}

int main(int argc, char **args)
{
	float *a, *b, *c, *d;
	float *dev_a, *dev_b, *dev_c;
	int n_elem = 10;
	float duration_ms;
	cudaEvent_t start, stop;
	int warmup_iters = 2;

	if (argc > 1)
		n_elem = atoi(args[1]);

	rand_array(&a, n_elem);
	rand_array(&b, n_elem);
	empty_array(&c, n_elem);
	cpu_add(c, a, b, n_elem);

	to_gpu(&dev_a, a, n_elem);
	to_gpu(&dev_b, b, n_elem);
	empty_array_gpu(&dev_c, n_elem);

	// warmup
	for (int i = 0; i < warmup_iters; i++) {
		gpu_add<<<1, n_elem>>>(dev_c, dev_a, dev_b, n_elem);
		cudaDeviceSynchronize();
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	gpu_add<<<1, n_elem>>>(dev_c, dev_a, dev_b, n_elem);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&duration_ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	from_gpu(&d, dev_c, n_elem);
	cudaDeviceReset();

	validate(c, d, n_elem);

	printf("kernel elapsed time: %.4f ms\n", duration_ms);

	return 0;
}
