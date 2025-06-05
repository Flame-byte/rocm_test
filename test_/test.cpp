#include <hip/hip_runtime.h>

#include <iostream>

// Define vector size
#define N 1024

// HIP kernel for vector addition
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Allocate host memory
    float *h_A, *h_B, *h_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, N * sizeof(float));
    hipMalloc(&d_B, N * sizeof(float));
    hipMalloc(&d_C, N * sizeof(float));

    // Copy data from host to device
    hipMemcpy(d_A, h_A, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, N * sizeof(float), hipMemcpyHostToDevice);

    // Define grid and block size
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    hipLaunchKernelGGL(vector_add, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);
    hipDeviceSynchronize();

    // Copy result back to host
    hipMemcpy(h_C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            std::cout << "Error at index " << i << ": " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            break;
        }
    }
    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
    }

    // Free device memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
