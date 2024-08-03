#include "Error.h"
#include "GpuTimer.h"
#include "Vector.h"

#define N 16
#define BLOCK_SIZE 2
#define STRIDE 4
#define POW(x) (x) * (x)

__global__ void mseKernel(Vector<float> d_a,
                          Vector<float> d_b,
                          Vector<float> d_c) {
    // -:YOUR CODE HERE:-
    __shared__ float cache[4];
    int idx = (threadIdx.x + blockIdx.x * BLOCK_SIZE) + (threadIdx.y + blockIdx.y * BLOCK_SIZE) * STRIDE;
    int ith = threadIdx.x + threadIdx.y * BLOCK_SIZE;

    cache[ith] = POW( d_a.getElement(idx) - d_b.getElement(idx) );
    __syncthreads();

    int i = blockDim.x * blockDim.y / 2;
    while(i) {
        if(ith < i) {
            cache[ith] += cache[ith + i];
        }
        __syncthreads();
        i /= 2;
    }

    if(ith == 0)
        d_c.setElement(blockIdx.x + blockDim.x * blockIdx.y, cache[0]);
}

void onDevice(Vector<float> h_a, Vector<float> h_b, Vector<float> h_c) {
    // declare GPU vectors
    // -:YOUR CODE HERE:-
    Vector<float> d_a, d_b, d_c;
    // start timer
    GpuTimer timer;
    timer.Start();

    const int ARRAY_BYTES = N * sizeof(float);

    // allocate  memory on the GPU
    // -:YOUR CODE HERE:-
    cudaMalloc((void**)&d_a.elements, ARRAY_BYTES);
    cudaMalloc((void**)&d_b.elements, ARRAY_BYTES);
    cudaMalloc((void**)&d_c.elements, 4 * sizeof(float));

   // copy data from CPU the GPU
    // -:YOUR CODE HERE:-
    cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES, cudaMemcpyHostToDevice);
    // execution configuration
    dim3 GridBlocks(2, 2);
    dim3 ThreadsBlocks(2, 2);

    mseKernel<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b, d_c);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    // -:YOUR CODE HERE:-
    cudaMemcpy(h_c.elements, d_c.elements, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // stop timer
    timer.Stop();

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // free GPU memory
    // -:YOUR CODE HERE:-
    cudaFree(d_a.elements);
    cudaFree(d_b.elements);
    cudaFree(d_c.elements);
}

void test() {
    Vector<float> h_a, h_b, h_c;
    h_a.length = N;
    h_b.length = N;
    h_c.length = 4;

    h_a.elements = (float*)malloc(h_a.length * sizeof(float));
    h_b.elements = (float*)malloc(h_a.length * sizeof(float));
    h_c.elements = (float*)malloc(4 * sizeof(float));

    int i, j = 16, k = 1;

    for (i = 0; i < h_a.length; i++) {
        h_a.setElement(i, k);
        h_b.setElement(i, j);
        // h_c.setElement( i, 0.0 );
        k++;
        j--;
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);

    // verify that the GPU did the work we requested
    for (int i = 0; i < 4; i++) {
        printf(" [%i] = %f \n", i, h_c.getElement(i));
    }

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements);
}

int main(void) {
    test();
    return 0;
}
