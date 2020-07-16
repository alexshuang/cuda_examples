#include <cuda_runtime.h>
#include <stdio.h>

__global__ void func(void)
{
	printf("hello world from GPU\n");
}

int main(void)
{
	printf("hello world from CPU\n");
	func <<<1, 10>>>();
	cudaDeviceReset();

	return 0;
}
