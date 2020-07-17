#include <cuda_runtime.h>
#include <cstdio>
#include "../common/common.h"
#include <cstdlib>

int main(int argc, char **args)
{
    int dev_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);

    printf("max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);

	return 0;
}
