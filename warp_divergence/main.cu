#include <cuda_runtime.h>
#include <cstdio>
#include "../common/common.h"
#include <cstdlib>

__global__ void sample1(float *c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float a = 0.0, b = 0.0;

	if (!(idx % 2))
		a = 10.0;
	else
		b = 20.0;

	c[idx] = a + b;
}

__global__ void sample2(float *c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float a, b;
	a = b = 0.0f;

	if ((idx / 32) % 2 == 0)
		a = 10.0;
	else
		b = 20.0;

	c[idx] = a + b;
}

void bench()
{
	float *dev_c;
	float duration_ms;
	int warmup_iters = 2;
	cudaEvent_t start, stop;

	int blk_x = 64;
	int data_size = 64;
	empty_array_gpu(&dev_c, data_size);

	dim3 blk(blk_x);
	dim3 grid((data_size + blk.x - 1) / blk.x);

	for (int i = 0; i < warmup_iters; i++) {
		sample1<<<grid, blk>>>(dev_c);
		cudaDeviceSynchronize();
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	sample1<<<grid, blk>>>(dev_c);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&duration_ms, start, stop);
	printf("sample1 <<<%d, %d>>> elapsed time: %.4f ms\n", grid.x, blk.x, duration_ms);

	cudaEventRecord(start);
	sample2<<<grid, blk>>>(dev_c);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&duration_ms, start, stop);
	printf("sample2 <<<%d, %d>>> elapsed time: %.4f ms\n", grid.x, blk.x, duration_ms);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_c);
}

int main(int argc, char **args)
{
	bench();

	cudaDeviceReset();

	return 0;
}
