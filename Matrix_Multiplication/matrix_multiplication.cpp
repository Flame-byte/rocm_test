#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <hip/hip_fp16.h>


__global__ void matrixMul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // A的行号 / C的行号
    int col = blockIdx.x * blockDim.x + threadIdx.x; // B的列号 / C的列号

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];  // A[i,j] * B[j,k]
            printf("A[%d]=%f, B[%d]=%f, sum=%f\n", row * N + i, A[row * N + i], i * K + col, B[i * K + col], sum);
        }
        C[row * K + col] = sum;
        printf("C[%d]=%f\n", row * K + col, C[row * K + col]);
    }
}

// __global__ __half matrixMul(int n, __half* A, __half* B, __half* C)
// {
//     column = blockIdx.x * blockDim.x + threadIdx.x;
//     row = blockIdx.y * blockDim.y + threadIdx.y;

//     if (column < n && row < n)
//     {
//         __half sum = 0.0f;
//         for (int k = 0; k < n; k++)
//         {
//             sum += A[row * n + k] * B[k * n + column];
//             printf("A[%d]=%f, B[%d]=%f, sum=%f\n", row * n + k, A[row * n + k], k * n + column, B[k * n + column], sum);
//         }
//         C[row * n + column] = sum;
//         printf("C[%d]=%f\n", row * n + column, C[row * n + column]);
//     }
// }

int main() {
    int M = 2, N = 4, K = 3;

    
    std::vector<float> h_A , h_B;

    for (int i = 0; i < M * N; ++i) {
        h_A.push_back(i + 1.0f);
    }

    for (int i = 0; i < N * K; ++i) {
        h_B.push_back(i + 1.0f);
    }

    std::vector<float> h_C(M * K, 0); // 4x2

    float *d_A, *d_B, *d_C;

    hipMalloc(&d_A, sizeof(float) * M * N);
    hipMalloc(&d_B, sizeof(float) * N * K);
    hipMalloc(&d_C, sizeof(float) * M * K);

    hipMemcpy(d_A, h_A.data(), sizeof(float) * M * N, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), sizeof(float) * N * K, hipMemcpyHostToDevice);

    dim3 block(4,4);
    dim3 grid(1,1);

    hipLaunchKernelGGL(matrixMul, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);

    hipMemcpy(h_C.data(), d_C, sizeof(float) * M * K, hipMemcpyDeviceToHost);

    // 输出结果
    // std::cout << "Result C = A x B:" << std::endl;
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < K; ++j) {
    //         std::cout << h_C[i * K + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}


