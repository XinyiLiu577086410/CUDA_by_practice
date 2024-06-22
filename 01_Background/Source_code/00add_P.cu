#include <assert.h>
#include <stdio.h>

__global__ void add_kernel(int* device_a, int* device_b, int* device_result, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N && idx >= 0)
        device_result[idx] = device_a[idx] + device_b[idx];
}

// Do the add vector operation
int* add(int* a, int* b, int* result, int N) {
    int ARRAY_BYTES = N * sizeof(int);
    int  *device_a, *device_b, *device_result;
    cudaMalloc(&device_a, ARRAY_BYTES);
    cudaMalloc(&device_b, ARRAY_BYTES);
    cudaMalloc(&device_result, ARRAY_BYTES);
    cudaMemcpy(device_a, a, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, ARRAY_BYTES, cudaMemcpyHostToDevice);

    //kernel invocation
    add_kernel<<<ceil(N/1024.000), 1024>>>(device_a,device_b,device_result, N);

    cudaMemcpy(result, device_result, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_result);
    cudaDeviceReset();
    return result;
}

// Configure the dynamic memory to initialize
// the variables a, b and result as pointers.

void onDevice() {
    int N = 10;

    // -:YOUR CODE HERE:-
    const int ARRAY_SIZE = 10240000;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
    int *a, *b, *result;

    a = (int*)malloc(ARRAY_BYTES);
    b = (int*)malloc(ARRAY_BYTES);
    result = (int*)malloc(ARRAY_BYTES);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = -i;
        b[i] = i * i;
        result[i] = 0;
    }

    add(a, b, result, ARRAY_SIZE);


    for (int i = 0; i < N; i++) {
        assert(a[i] + b[i] == result[i]);
    }

    printf("-: successful execution :-\n");
   
    // -:YOUR CODE HERE:-
}

int main() {
    onDevice();
    return 0;
}