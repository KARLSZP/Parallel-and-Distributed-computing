/*
*   Monte Carlo Algorithm
*   for Pi calculating.
*   
*   随机取点次数：1000, 10000, 100000
*   本例中，N=10000
*/
#include <sys/time.h> 
#include <cstdio> 
#include <cstdlib>
#include "error_checks.h"
#define PI 3.141592653589793

// GPU kernel
__global__ 
void plot(int *C_count, double* ranx, double* rany, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double pos_x = ranx[idx];
    double pos_y = rany[idx];
    if (idx < N) {
        if(pos_x*pos_x + pos_y*pos_y <= 1.0) {
            C_count[idx]++;
        }
    }
}

double generateRand() {
    return (rand() % 10000) * 0.0001;
}

int main() {
    timeval t1, t2; // Structs for timing
    // const int N = 1000;
    const int N = 10000;
    // const int N = 100000;
    // Result-variables.
    int C_count = 0;
    int T_count = N;
    double pi = 0.0;
    
    // Host-variables.
    int *C_count_h = new int[N];
    double rand_x_h[N];
    double rand_y_h[N];
    
    // Device-variables.
    int *C_count_d;
    double *rand_x_d, *rand_y_d;

    // memory allocation for Device-variables.
    CUDA_CHECK( cudaMalloc( (void**)&C_count_d, N * sizeof(int)) );
    CUDA_CHECK( cudaMalloc( (void**)&rand_x_d, N * sizeof(double)) );
    CUDA_CHECK( cudaMalloc( (void**)&rand_y_d, N * sizeof(double)) ); 

    // generate random position.
    for(int i = 0;i<N;i++){
        rand_x_h[i] = generateRand();
        rand_y_h[i] = generateRand();
    }

    // memory copy from Host to Device.
    CUDA_CHECK( cudaMemcpy( (void*)rand_x_d, (void*)rand_x_h, N * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy( (void*)rand_y_d, (void*)rand_y_h, N * sizeof(double), cudaMemcpyHostToDevice) );

    // GPU version
    dim3 threads(128), grid((N+threads.x-1)/threads.x);
    
    // GPU Kernel called.
    gettimeofday( & t1, NULL);
    plot <<< grid, threads >>> (C_count_d, rand_x_d, rand_y_d, N);
    gettimeofday( & t2, NULL); 

    // memory copy from Device to Host.
    CUDA_CHECK( cudaMemcpy( (void*)C_count_h, (void*)C_count_d, N * sizeof(int), cudaMemcpyDeviceToHost) );
    
    // pi calculation.
    for(int i = 0;i<N;i++){
        C_count += C_count_h[i];
    }

    pi = 4.0 * (double(C_count)/double(T_count));

    // Resulting...
    printf(">> For N = %d...\n\
            >> GPU Running for: %g seconds.\n\
            >> Pi is generated as: %lf\n\
            >> Diff is %lf\n\
            >> Accuracy is %lf\f\
            >> End for N = %d\n", 
           N, t2.tv_sec - t1.tv_sec + 
           (t2.tv_usec - t1.tv_usec)/1.0e6, pi, abs(pi-PI), 1-(abs(pi-PI)/PI), N); 

    return EXIT_SUCCESS; 
}