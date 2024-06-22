#include <assert.h>
#include <cuda.h>
#include <stdio.h>

// define necessary variables
// -:YOUR CODE HERE:-
#define ARRAY_SIZE 1000
#define THREADS 5
#define BLOCKS 10

const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

// kernel v1
__global__ void addKernel(int* d_a, int* d_b, int* d_result) {
    // index of the threads in x        -> threadIdx.x
    // index of the blocks in x         -> blockIdx.x
    // number of threads per block in x -> blockDim.x
    // number of blocks per grid in x   -> gridDim.x

    // -:YOUR CODE HERE:-

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx < ARRAY_SIZE) {
        d_result[idx] = d_a[idx] + d_b[idx];
        idx += gridDim.x; 
    }


}

void onDevice(int* h_a, int* h_b, int* h_result) {
    int *d_a, *d_b, *d_result;

    // allocate memory on the device
    // -:YOUR CODE HERE:-
    cudaMalloc((void**)&d_a, ARRAY_BYTES);
    cudaMalloc((void**)&d_b, ARRAY_BYTES);
    cudaMalloc((void**)&d_result, ARRAY_BYTES);

    // copythe arrays 'a' and 'b' to the device
    // -:YOUR CODE HERE:-

    cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // run the kernel
    addKernel<<<BLOCKS, THREADS>>>(d_a, d_b, d_result);

    // copy the array 'result' back from the device to the CPU
    // -:YOUR CODE HERE:-
    cudaMemcpy(h_result, d_result, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // check the results
    for (int i = 0; i < ARRAY_SIZE; i++) {
        assert(h_a[i] + h_b[i] == h_result[i]);
        // printf("%i\n", h_result[i] );
    }

    // free device memory
    // -:YOUR CODE HERE:-
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

void onHost() {
    int *h_a, *h_b, *h_result;

    // allocate memory on the host
    // -:YOUR CODE HERE:-
    h_a = (int *) malloc(ARRAY_BYTES);
    h_b = (int *) malloc(ARRAY_BYTES);
    h_result = (int *) malloc(ARRAY_BYTES);
    
    // filling the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = -i;
        h_b[i] = i * i;
        h_result[i] = 0;
    }

    onDevice(h_a, h_b, h_result);

    // check the results
    for (int i = 0; i < ARRAY_SIZE; i++) {
        assert(h_a[i] + h_b[i] == h_result[i]);
    }

    printf("-: successful execution :-\n");

    // free host memory
    // -:YOUR CODE HERE:-
    free(h_a);
    free(h_b);
    free(h_result);
}

int main() {
    onHost();
    return 0;
}
