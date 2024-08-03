#include <assert.h>
#include <cstddef>
#include <stdio.h>
#include "Error.h"

#define N 1000

__device__ size_t calcIndex(size_t x, size_t y) {return x + y * N;}

__device__ size_t calcIndexTrans(size_t x, size_t y) {return y + x * N;}

__global__ void transposedMatrixKernel(int* d_a, int* d_b) {
    // -:YOUR CODE HERE:-
    size_t Xstride = gridDim.x * blockDim.x;
    size_t Ystride = gridDim.y * blockDim.y;
    for(size_t y = threadIdx.y + blockIdx.y * blockDim.y; y < N; y += Ystride) {
        for(size_t x = threadIdx.x + blockIdx.x * blockDim.x; x < N; x += Xstride) {
            d_b[calcIndexTrans(x, y)] = d_a[calcIndex(x, y)];
        }
    }
}

void onDevice(int h_a[][N], int h_b[][N]) {
    // declare GPU memory pointers
    int *d_a, *d_b;

    const int ARRAY_BYTES = N * N * sizeof(int);

    // allocate  memory on the GPU
    // -:YOUR CODE HERE:-
    cudaMalloc((void**)&d_a, ARRAY_BYTES);
    cudaMalloc((void**)&d_b, ARRAY_BYTES);

    // copy data from CPU the GPU
    // -:YOUR CODE HERE:-
    cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
    // execution configuration
    dim3 GridBlocks(4, 4);
    dim3 ThreadsBlocks(16, 16);

    // run the kernel
    transposedMatrixKernel<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    // -:YOUR CODE HERE:-
    cudaMemcpy(h_b, d_b, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // free GPU memory
    // -:YOUR CODE HERE:-
    cudaFree(d_a);
    cudaFree(d_b);
}

void test(int h_a[][N], int h_b[][N]) {
    // test  result
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            assert(h_a[j][i] == h_b[i][j]);
        }
    }

    printf("-: successful execution :-\n");
}

void onHost() {
    int i, j, k = 0;
    int h_a[N][N], h_b[N][N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_a[i][j] = k;
            h_b[i][j] = 0;
            k++;
        }
    }

    // call device configuration
    onDevice(h_a, h_b);
    test(h_a, h_b);
}

int main() {
    onHost();
}
