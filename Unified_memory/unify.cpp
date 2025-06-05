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
    float *A, *B, *C;

    // hipMallocManaged(&A, N * sizeof(float));
    // hipMallocManaged(&B, N * sizeof(float));
    // hipMallocManaged(&C, N * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }
    // hipMallocManaged(&a, sizeof(*a));
    // hipMallocManaged(&b, sizeof(*b));
    // hipMallocManaged(&c, sizeof(*c));

    // // Copy data from host to device
    // hipMemcpy(d_A, h_A, N * sizeof(float), hipMemcpyHostToDevice);
    // hipMemcpy(d_B, h_B, N * sizeof(float), hipMemcpyHostToDevice);

    // Define grid and block size
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    hipLaunchKernelGGL(vector_add, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, A, B, C, N);

    hipDeviceSynchronize();

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (C[i] != A[i] + B[i]) {
            success = false;
            std::cout << "Error at index " << i << ": " << C[i] << " != " << A[i] + B[i] << std::endl;
            break;
        }
    }
    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
    }

    // Free device memory
    hipFree(A);
    hipFree(B);
    hipFree(C);

    return 0;
}
