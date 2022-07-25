
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void square(int *d_out, int *d_in)
{
    int idx = threadIdx.x;
    int i = d_in[idx];
    d_out[idx] = i * i;
}

int main()
{
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    // generate the input array on the host
    int h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = i;
    }
    int h_out[ARRAY_SIZE];

    // declare GPU memory pointers
    int* d_in;
    int* d_out;

    // allocate GPU memory
    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);

    // transfer the array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // launch the kernel
    square <<<1, ARRAY_SIZE >>> (d_out, d_in);

    // copy back the result array to the CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    // print out the resulting array
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        printf("%d", h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
