#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"

// #define MARK 1
// #define UNMARK 0
#define ARRAY_SIZE 10000000000 // 10 Bilion
#define int long
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(signed);

__global__ void kernelSieve(Vector<signed> d_a) {
    // -:YOUR CODE HERE:-
    int k = 2;
    while(k * k < ARRAY_SIZE) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x + 2;
        if(d_a.elements[k]) {
            while (1) {
                int pos = idx * k;
                if(pos < ARRAY_SIZE) {
                    d_a.elements[pos] = 0;
                }
                else { 
                    break;
                }
                idx += gridDim.x * blockDim.x;
            }
        }
        __syncthreads();
        ++k;
    }
}

void onDevice(Vector<signed> h_a) {
    Vector<signed> d_a;
    // int k;

    // create the streams
    // -:YOUR CODE HERE:-
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    HANDLER_ERROR_ERR(cudaMalloc(&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES,
                                 cudaMemcpyHostToDevice));

    // kernel call
    // -:YOUR CODE HERE:-
    // k = 2;
    // while(k < ARRAY_SIZE) {
        // int inside = ARRAY_SIZE / k;
        kernelSieve<<<256, 1024, 0, stream>>>(d_a);
        // k++;
    // }

    HANDLER_ERROR_ERR(cudaMemcpy(h_a.elements, d_a.elements, ARRAY_BYTES,
                                 cudaMemcpyDeviceToHost));
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));

    // destroy stream
    // -:YOUR CODE HERE:-
    cudaStreamDestroy(stream);
}

void onHost() {
    Vector<signed> h_a;
    h_a.length = ARRAY_SIZE;

    int j;
    // h_a.elements = (int*)malloc(ARRAY_BYTES);
    cudaHostAlloc((void**)&h_a.elements, ARRAY_BYTES, cudaHostAllocDefault);

    for (j = 0; j < ARRAY_SIZE; j++) {
        h_a.setElement(j, j);
    }

    onDevice(h_a);

    // for (j = 0; j < ARRAY_SIZE; j++) {
    //     if (h_a.getElement(j) > 1)
    //         printf("%li \n", h_a.getElement(j));
    // }

    cudaFreeHost(h_a.elements);
}

void checkDeviceProps() {
    // properties validation
    cudaDeviceProp prop;
    signed whichDevice;
    HANDLER_ERROR_ERR(cudaGetDevice(&whichDevice));
    HANDLER_ERROR_ERR(cudaGetDeviceProperties(&prop, whichDevice));
    if (!prop.deviceOverlap) {
        printf(
            "Device will not handle overlaps, so no speed up from streams\n");
    }
}

signed main() {
    checkDeviceProps();
    onHost();
}
