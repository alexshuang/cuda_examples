#include <cuda_runtime.h>
#include <cstdio>
#include "../common/common.h"
#include <cstdlib>

float cpuRecursiveReduce(float *data, int const size)
{
    // stop condition
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return cpuRecursiveReduce(data, stride);
}

__global__ void gpu_reduction1(float *dest, float *src, int size)
{
    int g_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    float *_src = src + blockIdx.x * blockDim.x;
    
    if (g_tid >= size) return;
    
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0)
            _src[tid] += _src[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0) dest[blockIdx.x] = _src[0];
}

float cpu_reduction(const float *src, int size)
{
    float res = 0.0f;
    for (int i = 0; i < size; i++)
        res += src[i];
    return res;
}

int main(int argc, char **args)
{
    float res, d_res;
	float *dest, *src, *d_dest, *d_src;
	float duration_ms;
	int warmup_iters = 2;
	cudaEvent_t start, stop;
    int n_elem = 1 << 14,
        blk_x = 512;

    rand_array(&src, n_elem);
    to_gpu(&d_src, src, n_elem);
    res = cpuRecursiveReduce(src, n_elem);
    
	dim3 blk(blk_x);
	dim3 grid((n_elem + blk.x - 1) / blk.x);
    int n = grid.x;

    empty_array_gpu(&d_dest, n);

	for (int i = 0; i < warmup_iters; i++) {
		gpu_reduction1<<<grid, blk>>>(d_dest, d_src, n_elem);
		cudaDeviceSynchronize();
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	gpu_reduction1<<<grid, blk>>>(d_dest, d_src, n_elem);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
    from_gpu(&dest, d_dest, n);
    d_res = cpu_reduction(dest, n);
    if (abs(res - d_res) >= 1e-8)
        printf("calc incorrect: host=%f, device=%f\n", res, d_res);
	cudaEventElapsedTime(&duration_ms, start, stop);
	printf("sample1 <<<%d, %d>>> elapsed time: %.4f ms\n", grid.x, blk.x, duration_ms);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_src);
	cudaFree(d_dest);
    free(src);
    free(dest);
	cudaDeviceReset();

	return 0;
}
