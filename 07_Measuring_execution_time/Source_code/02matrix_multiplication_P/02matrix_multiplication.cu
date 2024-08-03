#include <cstddef>
#include <stdio.h>
#include "Error.h"
#include "GpuTimer.h"
#include "Matrix.h"

#define N 4096

__global__ void matrixMultiplicationKernel(Matrix<float> d_a,
                                           Matrix<float> d_b,
                                           Matrix<float> d_c) {
    // -:YOUR CODE HERE:-
    size_t Xstride = gridDim.x * blockDim.x;
    size_t Ystride = gridDim.y * blockDim.y;
    for(size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < d_a.height; y += Ystride) {
        for(size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < d_b.width; x += Xstride) {
            double sum = 0.0;
            for(size_t z = 0; z < d_a.width; ++z) {
                sum += d_a.getElement(y, z) * d_b.getElement(z, x);
            }
            d_c.setElement(y, x, static_cast<float>(sum));
        }
    }
}

void onDevice(Matrix<float> h_a, Matrix<float> h_b, Matrix<float> h_c) {
    // declare GPU matrices
    // -:YOUR CODE HERE:-
    Matrix<float> d_a = h_a;
    Matrix<float> d_b = h_b;
    Matrix<float> d_c = h_c;
    // start timer
    GpuTimer timer;
    timer.Start();

    const int ARRAY_BYTES = N * N * sizeof(float);

    // allocate  memory on the GPU
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMalloc((void**)&(d_a.elements), ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&(d_b.elements), ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&(d_c.elements), ARRAY_BYTES));

    // copy data from CPU the GPU
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));

    // execution configuration
    // -:YOUR CODE HERE:-
    dim3 GridBlocks(2,71,1);
    dim3 ThreadsBlocks(32, 32, 1);

    // run the kernel
    matrixMultiplicationKernel<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b, d_c);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    // stop timer
    timer.Stop();

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // free GPU memory
    // -:YOUR CODE HERE:-
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
    HANDLER_ERROR_ERR(cudaFree(d_c.elements));
}

void test() {
    Matrix<float> h_a, h_b, h_c;

    h_a.width = N;
    h_a.height = N;

    h_b.width = N;
    h_b.height = N;

    h_c.width = N;
    h_c.height = N;

    h_a.elements = (float*)malloc(h_a.width * h_b.height * sizeof(float));
    h_b.elements = (float*)malloc(h_b.width * h_b.height * sizeof(float));
    h_c.elements = (float*)malloc(h_c.width * h_c.height * sizeof(float));

    int i, j, k = 1;

    #pragma omp parallel for collapse(2)
    for (i = 0; i < h_a.height; i++) {
        for (j = 0; j < h_a.height; j++) {
            h_a.setElement(i, j, 1);
            h_b.setElement(i, j, 1);
            h_c.setElement(i, j, 12.0);
            k++;
        }
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);

    // print  result
    // for (i = 0; i < h_c.width; i++) {
    //     for (j = 0; j < h_c.height; j++) {
    //         printf("%.2f ", h_c.elements[i * h_c.width + j]);
    //     }
    //     printf("\n");
    // }

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements);
}

int main() {
    test();
}
