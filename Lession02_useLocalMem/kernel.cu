
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void use_local_memory_GPU(float in)
{
    float f;    // variable "f" is in local memory and private to each thread
    f = in;     // parameter "in" is in local memory and private to each thread
}

__global__ void use_shared_memory_GPU(float* array)
{
    // local variables, private to each thread
    int i, index = threadIdx.x;
    float average, sum = 0.0f;

    // __shared__ variables are visible to all threads in the thread block
    // and have the same lifetime as the thread block
    __shared__ float sh_arr[128];

    // copy data from "array" in global memory to sh_arr in shared memory.
    // here, each thread is responsible for copying a single element.
    sh_arr[index] = array[index];

    __syncthreads();    // ensure all the writes to shared memory have completed

    for (i = 0; i < index; i++) { sum += sh_arr[i]; }
    average = sum / (index + 1.0f);

    // if array[index] is greater than the average of array[0..index-1], replace with average.
    // since array[] is in global memory, this change will be seen by the host (and potentially
    // other thread blocks, if any)
    if (array[index] > average) { array[index] = average; }

    // the following code has NO EFFECT: it modifies shared memory, but
    // the resulting modified data never copied back to global memory
    // and vanishes when the thread block completes
    sh_arr[index] = 3.14;
}

__global__ void use_global_memory_GPU(float* array)
{
    array[threadIdx.x] = 2.0f * (float)threadIdx.x;
}

int main()
{
#pragma region local memory
     /*
     * First, call a kernel that shows using local memory 
     */
    use_local_memory_GPU <<<1, 128>>> (2.0f);
#pragma endregion

#pragma region shared memory
    /*
     * Next, call a kernel that shows using shared memory
     */
    float h_arr[128];   // convention: h_ variables live on host
    float* d_arr;       // convention: d_ variable live on device (GPU global mem)

    // as before, pass in a pointer to data in global memory
    use_shared_memory_GPU << <1, 128 >> > (d_arr);
#pragma endregion

#pragma region global memory
    /*
     * Next, call a kernel that shows using global memory
     */

    // allocate global memory on the device, place result in "d_arr"
    cudaMalloc((void**)&d_arr, sizeof(float) * 128);
    // now copy data from host memory "h_arr" to device memory "d_arr"
    cudaMemcpy((void*)d_arr, (void*)h_arr, sizeof(float) * 128, cudaMemcpyKind::cudaMemcpyHostToDevice);
    // lauch the kernel (1 block of 128 threads)
    use_global_memory_GPU << <1, 128 >> > (d_arr);
    // copy the modified array back to the host, overwriting contents of h_arr
    cudaMemcpy((void*)h_arr, (void*)d_arr, sizeof(float) * 128, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    // ... do other stuff ...
#pragma endregion





    return 0;
}
