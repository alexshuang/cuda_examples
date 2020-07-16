#pragma once

#include <cstdio>

#define cudaCHECK(call) do { \
	const cudaError_t err = call; \
	if (err != cudaSuccess) { \
		printf("Error: %s:%d\n", __func__, __LINE__); \
		printf("Error code: %d, reason: %s\n", err, cudaGetErrorString(err)); \
	} } while (0)

#define cudaTIME(call, res_ms, start, stop) do { \
	cudaEventCreate(&start); \
	cudaEventCreate(&stop); \
	cudaRecord(start); \
	call; \
	cudaRecord(end); \
	cudaDeviceSynchronize(); \
	cudaEventElapsedTime(&res_ms, start, stop); \
	cudaEventDestroy(start); \
	cudaEventDestroy(stop); \
} while (0)

extern void rand_array(float **vec, int size);
extern void zero_array(float **vec, int size);
extern void empty_array(float **vec, int size);
extern void to_gpu(float **dev_p, const float *host_p, int size);
extern void from_gpu(float **dest, const float *dev_p, int size);
extern void empty_array_gpu(float **dev_p, int size);
extern bool validate(float *host_p, float *dev_p, int size);

