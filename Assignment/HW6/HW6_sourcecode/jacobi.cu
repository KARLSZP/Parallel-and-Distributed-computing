/*
*   Monte Carlo Algorithm
*   for Pi calculating.
*   
*   随机取点次数：1000, 10000, 100000
*/
#include <sys/time.h> 
#include <cstdio> 
#include <cstdlib>
#include "jacobi.h"
#include "error_checks.h"

using namespace std;

double generateRand() {
    return (rand() % 10000) * 0.0001;
}

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

int main() {
    timeval t1, t2; // Structs for timing
    const int N = 1000;
    // const int N = 10000;
    // const int N = 100000;
    int C_count = 0;
    int T_count = N;
    double pi = 0.0;
    int *C_count_h = new int[N];
    int *C_count_d;
    double rand_x_h[N];
    double rand_y_h[N];
    double *rand_x_d, *rand_y_d;

    CUDA_CHECK( cudaMalloc( (void**)&C_count_d, N * sizeof(int)) );
    CUDA_CHECK( cudaMalloc( (void**)&rand_x_d, N * sizeof(double)) );
    CUDA_CHECK( cudaMalloc( (void**)&rand_y_d, N * sizeof(double)) ); 

    // generate random position.
    for(int i = 0;i<N;i++){
        rand_x_h[i] = generateRand();
        rand_y_h[i] = generateRand();
        printf("%lf, %lf\n", rand_x_h[i], rand_y_h[i]);
    }

    CUDA_CHECK( cudaMemcpy( (void*)rand_x_d, (void*)rand_x_h, N * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy( (void*)rand_y_d, (void*)rand_y_h, N * sizeof(double), cudaMemcpyHostToDevice) );

    // GPU version
    dim3 threads(128), grid((N+threads.x-1)/threads.x);
    
    gettimeofday( & t1, NULL);
    
    plot <<< grid, threads >>> (C_count_d, rand_x_d, rand_y_d, N);

    gettimeofday( & t2, NULL); 

    CUDA_CHECK( cudaMemcpy( (void*)C_count_h, (void*)C_count_d, N * sizeof(int), cudaMemcpyDeviceToHost) );
    
    for(int i = 0;i<N;i++){
        C_count += C_count_h[i];
        // cout<<C_count_h[i]<<endl;
    }
    cout<<C_count<<endl;
    pi = 4.0 * (double(C_count)/double(T_count));

    printf("GPU: %g seconds.\nPi is generated as: %lf\n", 
           t2.tv_sec - t1.tv_sec + 
           (t2.tv_usec - t1.tv_usec)/1.0e6, pi); 

    return EXIT_SUCCESS; 
}