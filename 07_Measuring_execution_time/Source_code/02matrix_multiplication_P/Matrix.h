#ifndef GPU_MATRIX_H__
#define GPU_MATRIX_H__
#include <cassert>
template <typename T>
struct Matrix {
    int width;
    int height;
    T* elements;

    __device__ __host__ T getElement(int row, int col) {
        if(row < 0 || row >= height || col < 0 || col >= width) {
            assert(0);
        }
        return elements[row * width + col];
    }

    __device__ __host__ void setElement(int row, int col, T value) {
        if(row < 0 || row >= height || col < 0 || col >= width) {
            assert(0);
        }
        elements[row * width + col] = value;
    }
};

#endif /* GPU_MATRIX_H__ */
