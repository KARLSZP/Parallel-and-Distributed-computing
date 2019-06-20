#include <cstdio>
#include <cmath>
#include "error_checks.h" // Macros CUDA_CHECK and CHECK_ERROR_MSG


__global__ void vector_add(double *C, const double *A, const double *B, int N)
{
    // Add the kernel code
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not try to access past the allocated memory
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


int main(void)
{
    const int N = 20;
    const int ThreadsInBlock = 128;
    double *dA, *dB, *dC;
    double hA[N], hB[N], hC[N];
  
    for(int i = 0; i < N; ++i) {
        hA[i] = (double) i;
        hB[i] = (double) i * i;
    }

    /* 
       Add memory allocations and copies. Wrap your runtime function
       calls with CUDA_CHECK( ) macro
    */
    CUDA_CHECK( cudaMalloc((void**)&dA, sizeof(double)*N) );
    // #error Add the remaining memory allocations and copies
    CUDA_CHECK( cudaMalloc((void**)&dB, sizeof(double)*N) );
    CUDA_CHECK( cudaMalloc((void**)&dC, sizeof(double)*N) );

    CUDA_CHECK( cudaMemcpy((void*)dA, (void*)hA, sizeof(double)*N, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy((void*)dB, (void*)hB, sizeof(double)*N, cudaMemcpyHostToDevice) );
    
    // Note the maximum size of threads in a block
    dim3 threads(ThreadsInBlock), grid((N + threads.x - 1) / threads.x);

    //// Add the kernel call here
    // #error Add the CUDA kernel call
    // vector_add(double *C, const double *A, const double *B, int N);
    // // dereference host pointer hA, hB.
    // vector_add <<<grid, threads>>> (dC, hA, hB, N);
    
    vector_add <<<grid, threads>>> (dC, dA, dB, N);
    
    // Here we add an explicit synchronization so that we catch errors
    // as early as possible. Don't do this in production code!
    cudaDeviceSynchronize();
    CHECK_ERROR_MSG("vector_add kernel");

    //// Copy back the results and free the device memory
    // #error Copy back the results and free the allocated memory
    CUDA_CHECK( cudaMemcpy((void*)hC, (void*)dC, sizeof(double)*N, cudaMemcpyDeviceToHost) );
    
    // // dereference device pointer dC[i]
    // for (int i = 0; i < N; i++)
    //     printf("%5.1f\n", dC[i]);

    for (int i = 0; i < N; i++)
        printf("%5.1f\n", hC[i]);

    return 0;
}