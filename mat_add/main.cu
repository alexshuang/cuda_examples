#include <cuda_runtime.h>
#include <cstdio>
#include "../common.h"
#include <cstdlib>

__global__ void gpu_add(float *c, const float *a, const float *b, ssize_t nx, ssize_t ny)
{
	ssize_t ix = blockIdx.x * blockDim.x + threadIdx.x;
	ssize_t iy = blockIdx.y * blockDim.y + threadIdx.y;
	ssize_t i = iy * nx + ix;

	if (ix < nx && iy < ny)
		c[i] = a[i] + b[i];
}

void cpu_add(float *c, const float *a, const float *b, ssize_t nx, ssize_t ny)
{
	const float *_a = a;
	const float *_b = b;
	float *_c = c;
	
	for (ssize_t i = 0; i < ny; i++) {
		for (ssize_t j = 0; j < nx; j++)
			_c[j] = _a[j] + _b[j];
		_c += nx; 
		_a += nx; 
		_b += nx; 
	}
}

void kernel_lanuch(dim3 grid, dim3 blk, float *dev_c, const float *dev_a,
		const float *dev_b, ssize_t nx, ssize_t ny, float *duration_ms)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	gpu_add<<<grid, blk>>>(dev_c, dev_a, dev_b, nx, ny);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(duration_ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void kernel_warmup(dim3 grid, dim3 blk, float *dev_c, const float *dev_a,
		const float *dev_b, ssize_t nx, ssize_t ny, int iters)
{
	for (int i = 0; i < iters; i++) {
		gpu_add<<<grid, blk>>>(dev_c, dev_a, dev_b, nx, ny);
		cudaDeviceSynchronize();
	}
}

void bench(ssize_t nx, ssize_t ny, int blk_x, int blk_y, bool valid)
{
	float *a, *b;
	float *dev_a, *dev_b, *dev_c;
	float *c = NULL,
		  *d = NULL;
	float duration_ms;
	int warmup_iters = 2;
	ssize_t n_elem = nx * ny;

	printf("benchmark: nx=%ld, ny=%ld, block.x=%d, block.y=%d ...\n", nx, ny, blk_x, blk_y);

	rand_array(&a, n_elem);
	rand_array(&b, n_elem);
	to_gpu(&dev_a, a, n_elem);
	to_gpu(&dev_b, b, n_elem);
	empty_array_gpu(&dev_c, n_elem);

	dim3 block(blk_x, blk_y);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	kernel_warmup(grid, block, dev_c, dev_a, dev_b, nx, ny, warmup_iters);
	kernel_lanuch(grid, block, dev_c, dev_a, dev_b, nx, ny, &duration_ms);
	printf("kernel elapsed time: %.4f ms\n", duration_ms);

	if (valid) {
		empty_array(&c, n_elem);
		cpu_add(c, a, b, nx, ny);
		from_gpu(&d, dev_c, n_elem);
		validate(c, d, n_elem);
	}

	free(a);
	free(b);
	if (c)
		free(c);
	if (d)
		free(d);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

int main(int argc, char **args)
{
	int nx = 1 << 14, ny = 1 << 14;

	/*
	bench(nx, ny, 32, 32, true);
	bench(nx, ny, 32, 16, true);
	bench(nx, ny, 16, 32, true);
	*/
	bench(nx, ny, 16, 16, true);

	cudaDeviceReset();

	return 0;
}
