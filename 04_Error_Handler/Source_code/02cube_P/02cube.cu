#include <assert.h>
#include <stdio.h>
#include "Error.h"

__global__ void cubeKernel(float* d_out, float* d_in) {
    // -:YOUR CODE HERE:-
    int tid = threadIdx.x;
    d_out[tid] = d_in[tid] * d_in[tid] * d_in[tid];
}

void onDevice(float* h_in, float* h_out, int ARRAY_SIZE, int ARRAY_BYTES) {
    // declare GPU memory pointers
    // -:YOUR CODE HERE:-
    float* d_in, * d_out;

    // allocate GPU memory
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMalloc(&d_in, ARRAY_BYTES));
    
    // allocate GPU memory
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMalloc(&d_out, ARRAY_BYTES));

    // transfer the array to the GPU
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));
    // launch the kernel
    cubeKernel<<<1, ARRAY_SIZE>>>(d_out, d_in);
    HANDLER_ERROR_MSG("CUDA Kernel panic!!");
    // copy back the result array to the CPU
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));
    // free GPU memory pointers
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaFree(d_in));
    HANDLER_ERROR_ERR(cudaFree(d_out));
}

void test(float* h_in, float* h_out, int ARRAY_SIZE, int ARRAY_BYTES) {
    // print out the resulting array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        assert(h_out[i] == (h_in[i] * h_in[i] * h_in[i]));
        printf("%f", h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    printf("-: successful execution :-\n");
}

void onHost() {
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // declare CPU memory pointers
    // -:YOUR CODE HERE:-
    float* h_in = (float*) malloc(ARRAY_BYTES);

    // Allocate CPU memory pointers
    // -:YOUR CODE HERE:-
    float* h_out = (float*) malloc(ARRAY_BYTES);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }

    // call the kernel
    onDevice(h_in, h_out, ARRAY_SIZE, ARRAY_BYTES);
    test(h_in, h_out, ARRAY_SIZE, ARRAY_BYTES);

    // free CPU memory pointers
    // -:YOUR CODE HERE:-
    free(h_in);
    free(h_out);
}

int main(int argc, char** argv) {
    onHost();

    return 0;
}