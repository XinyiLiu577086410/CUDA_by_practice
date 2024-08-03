#include <assert.h>
#include <cstddef>
#include <stdio.h>
#include "Error.h"

#define N 500

__device__ int calcIndex(int x, int y) {
    return y * N + x;
}

__global__ void additionMatricesKernel(int* d_a, int* d_b, int* d_c) {
    // -:YOUR CODE HERE:-
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t Xstride = gridDim.x * blockDim.x;
    size_t Ystride = gridDim.y * blockDim.y;
    while(y < N) {
        size_t x = threadIdx.x + blockIdx.x * blockDim.x;
        while(x < N) {
            size_t idx = calcIndex(x, y);
            d_c[idx] = d_a[idx] + d_b[idx];
            x += Xstride;
        }
        y += Ystride;
    }
}

void onDevice(int h_a[][N], int h_b[][N], int h_c[][N]) {
    // declare GPU memory pointers
    int *d_a, *d_b, *d_c;

    const int ARRAY_BYTES = N * N * sizeof(int);

    // allocate  memory on the GPU
    // -:YOUR CODE HERE:-
    cudaMalloc((void**)&d_a, ARRAY_BYTES);
    cudaMalloc((void**)&d_b, ARRAY_BYTES);
    cudaMalloc((void**)&d_c, ARRAY_BYTES);
    // copy data from CPU the GPU
    // -:YOUR CODE HERE:-
    cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // execution configuration
    dim3 GridBlocks(4, 4);
    dim3 ThreadsBlocks(8, 8);

    // run the kernel
    additionMatricesKernel<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b, d_c);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    // -:YOUR CODE HERE:-
    cudaMemcpy(h_c, d_c, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    // free GPU memory
    // -:YOUR CODE HERE:-
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void test(int h_a[][N], int h_b[][N], int h_c[][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(h_a[i][j] + h_b[i][j] == h_c[i][j]);
        }
    }

    printf("-: successful execution :-\n");
}

void onHost() {
    int i, j;
    int h_a[N][N], h_b[N][N], h_c[N][N];

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            h_a[i][j] = h_b[i][j] = i + j;
            h_c[i][j] = 0;
        }
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);
    test(h_a, h_b, h_c);
}

int main() {
    onHost();
}
