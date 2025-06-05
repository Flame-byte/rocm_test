#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#include <iostream>

#define TILE_SIZE 2

__global__ void matmul_fp8_to_bf16(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, __hip_bfloat16* c, int m, int n, int k) {
    __shared__ __hip_fp8_e4m3_fnuz tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __hip_fp8_e4m3_fnuz tile_B[TILE_SIZE][TILE_SIZE];

    int row_a = threadIdx.x; // a的tile内行坐标
    int row_b = threadIdx.y; // b的tile内行坐标
    int col = threadIdx.z; // tile内列坐标

    int global_row_a = blockIdx.y * TILE_SIZE + row_a; // a的全局行索引
    int global_row_b = blockIdx.x * TILE_SIZE + row_b; // b的全局行索引（注意b是n×k）

    float result = 0;
    for (int i = 0; i < (k - 1)/TILE_SIZE; i += 1) {
        // load tile_A
        if (global_row_a < m && (i * TILE_SIZE + col) < k) {
            // a是m x k，列优先
            tile_A[row_a][col] = a[global_row_a * k + i * TILE_SIZE + col];
            //printf("tile_A[%d][%d] = %f\n", row_a, col, (float)tile_A[row_a][col]);
        } else {
            tile_A[row_a][col] = static_cast<__hip_fp8_e4m3_fnuz>(0);
        }

        // load tile_B
        if (global_row_b < n && (i * TILE_SIZE + col) < k) {
            // b是n x k，列优先
            tile_B[row_b][col] = b[global_row_b * k + i * TILE_SIZE + col];
            //printf("tile_B[%d][%d] = %f\n", row_b, col, (float)tile_B[row_b][col]);
        } else {
            tile_B[row_b][col] = static_cast<__hip_fp8_e4m3_fnuz>(0);
        }

        __syncthreads(); // 确保tile加载完

        float block_result = 0;
        for(int j = 0; j < TILE_SIZE; ++j) {
            float av = (float)tile_A[row_a][j];
            float bv = (float)tile_B[row_b][j];
            block_result += av * bv;
        }
        result += block_result;
        __syncthreads(); // 确保计算完再加载下一轮tile
    }

    if (global_row_a < m && global_row_b < n) {
        c[global_row_a * n + global_row_b] = (__hip_bfloat16)result;
    }
}


__global__ void matmul_fp8_to_bf16_2(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, __hip_bfloat16* c, int m, int n, int k) {
    __shared__ __hip_fp8_e4m3_fnuz tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __hip_fp8_e4m3_fnuz tile_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_a = bx * TILE_SIZE + ty;  // 使用ty作为行索引，因为block中的线程按TILE_SIZE划分行
    int row_b = by * TILE_SIZE + tx;  // 使用tx作为行索引

    float result = 0.0f;
    for (int i = 0; i < ((k + TILE_SIZE - 1) / TILE_SIZE; ++i) {
        // 计算当前块在k维度上的起始位置
        int k_start = i * TILE_SIZE;

        // 加载A的块到tile_A（列主序）
        int a_col = k_start + tx;
        int a_idx = a_col * m + row_a;
        if (a_col < k && row_a < m)
            tile_A[ty][tx] = a[a_idx];
        else
            tile_A[ty][tx] = static_cast<__hip_fp8_e4m3_fnuz>(0);

        // 加载B的块到tile_B（列主序）
        int b_col = k_start + ty;
        int b_idx = b_col * n + row_b;
        if (b_col < k && row_b < n)
            tile_B[tx][ty] = b[b_idx];  // 注意转置加载
        else
            tile_B[tx][ty] = static_cast<__hip_fp8_e4m3_fnuz>(0);

        __syncthreads();

        // 计算块内乘积并累加
        for (int j = 0; j < TILE_SIZE; ++j) {
            float av = static_cast<float>(tile_A[ty][j]);
            float bv = static_cast<float>(tile_B[tx][j]);
            result += av * bv;
        }

        __syncthreads();
    }

    // 写入结果
    if (row_a < m && row_b < n) {
        c[row_a * n + row_b] = static_cast<__hip_bfloat16>(result);
    }
}

__global__ void matmul_fp8_to_bf16_3(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_a = bx * TILE_SIZE + ty;
    int row_b = by * TILE_SIZE + tx;

    float result = 0;
    for (int i = 0; i < ((k + TILE_SIZE - 1) / TILE_SIZE); ++i) {
        // 计算当前块在k维度上的起始位置
        int k_start = i * TILE_SIZE;

        // 加载A的块到tile_A（列主序）
        int a_col = k_start + tx;
        int a_idx = a_col * m + row_a;
        if (a_col < k && row_a < m)
            tile_A[ty][tx] = a[a_idx];
        else
            tile_A[ty][tx] = 0;

        // 加载B的块到tile_B（列主序）
        int b_col = k_start + ty;
        int b_idx = b_col * n + row_b;
        if (b_col < k && row_b < n)
            tile_B[tx][ty] = b[b_idx];  // 注意转置加载
        else
            tile_B[tx][ty] = 0;
    
        __syncthreads(); // 确保tile加载完

        float block_result = 0;
        for(int j = 0; j < TILE_SIZE; ++j) {
            float av = tile_A[ty][j];
            float bv = tile_B[tx][j];
            block_result += av * bv;
        }
        result += block_result;
        __syncthreads(); // 确保计算完再加载下一轮tile
    }

    if (row_a < m && row_b < n) {
        c[row_a * n + row_b] = result;
    }
}



// CPU reference implementation for column-major fp8 matmul (A: m x k, B: n x k, C: m x n, all column-major)
void cpu_matmul_fp8_to_bf16_column_major(
    const __hip_fp8_e4m3_fnuz* A,
    const __hip_fp8_e4m3_fnuz* B,
    __hip_bfloat16* C,
    int m, int n, int k)
{
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                // A: m x k, column-major: A[row + kk * m]
                // B: n x k, column-major: B[col + kk * n]
                float a_val = (float)A[row + kk * m];
                float b_val = (float)B[col + kk * n];
                acc += a_val * b_val;
            }
            C[col + row * n] = __float2bfloat16(acc);
        }
    }
}


int main()
{
    const int m = 4, n = 8, k = 16;
    size_t size_A = m * k * sizeof(__hip_fp8_e4m3_fnuz);
    size_t size_B = n * k * sizeof(__hip_fp8_e4m3_fnuz);
    size_t size_C = m * n * sizeof(__hip_bfloat16);

    __hip_fp8_e4m3_fnuz* h_A = new __hip_fp8_e4m3_fnuz[m * k];
    __hip_fp8_e4m3_fnuz* h_B = new __hip_fp8_e4m3_fnuz[n * k];
    __hip_bfloat16* h_C = new __hip_bfloat16[m * n];
    __hip_bfloat16* h_C_CPU = new __hip_bfloat16[m * n];
    // 初始化数据 (这里简单初始化，实际可以根据需要填充)
    for (int i = 0; i < m * k; ++i) h_A[i] = __hip_fp8_e4m3_fnuz(i % 13 - 6);
    for (int i = 0; i < n * k; ++i) h_B[i] = __hip_fp8_e4m3_fnuz(i % 7 - 3);

    __hip_fp8_e4m3_fnuz* d_A;
    __hip_fp8_e4m3_fnuz* d_B;
    __hip_bfloat16* d_C;
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    cpu_matmul_fp8_to_bf16_column_major(h_A, h_B, h_C_CPU, m, n, k);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << __bfloat162float(h_C_CPU[i + j * m]) << " ";
        }
        std::cout << "\n";
    }

    hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim( (m + TILE_SIZE - 1) / TILE_SIZE ,(n + TILE_SIZE - 1) / TILE_SIZE );

    matmul_fp8_to_bf16_2<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
    hipDeviceSynchronize();

    hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);

    // 打印部分结果
    std::cout << "Partial C (after computation):\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << __bfloat162float(h_C[i + j * m]) << " ";
        }
        std::cout << "\n";
    }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}

// // CPU reference implementation for column-major fp8 matmul (A: m x k, B: n x k, C: m x n, all column-major)
// void cpu_matmul_fp8_to_bf16_column_major(
//     const float* A,
//     const float* B,
//     float* C,
//     int m, int n, int k)
// {
//     for (int row = 0; row < m; ++row) {
//         for (int col = 0; col < n; ++col) {
//             float acc = 0.0f;
//             for (int kk = 0; kk < k; ++kk) {
//                 // A: m x k, column-major: A[row + kk * m]
//                 // B: n x k, column-major: B[col + kk * n]
//                 float a_val = A[row + kk * m];
//                 float b_val = B[col + kk * n];
//                 acc += a_val * b_val;
//             }
//             C[col + row * n] = acc;
//         }
//     }
// }


// int main()
// {
//     const int m = 4, n = 8, k = 16;
//     size_t size_A = m * k * sizeof(float);
//     size_t size_B = n * k * sizeof(float);
//     size_t size_C = m * n * sizeof(float);

//     float* h_A = new float[m * k];
//     float* h_B = new float[n * k];
//     float* h_C = new float[m * n];
//     float* h_C_CPU = new float[m * n];
//     // 初始化数据 (这里简单初始化，实际可以根据需要填充)
//     for (int i = 0; i < m * k; ++i) h_A[i] = float(i % 13 - 6);
//     for (int i = 0; i < n * k; ++i) h_B[i] = float(i % 7 - 3);

//     float* d_A;
//     float* d_B;
//     float* d_C;
//     hipMalloc(&d_A, size_A);
//     hipMalloc(&d_B, size_B);
//     hipMalloc(&d_C, size_C);

//     cpu_matmul_fp8_to_bf16_column_major(h_A, h_B, h_C_CPU, m, n, k);

//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             std::cout << h_C_CPU[i + j * m] << " ";
//         }
//         std::cout << "\n";
//     }

//     hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
//     hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);

//     dim3 blockDim(TILE_SIZE, TILE_SIZE);
//     dim3 gridDim( (m + TILE_SIZE - 1) / TILE_SIZE ,(n + TILE_SIZE - 1) / TILE_SIZE );

//     matmul_fp8_to_bf16_3<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
//     hipDeviceSynchronize();

//     hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);

//     // 打印部分结果
//     std::cout << "Partial C (after computation):\n";
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             std::cout << h_C[i + j * m] << " ";
//         }
//         std::cout << "\n";
//     }

//     // Clean up
//     delete[] h_A;
//     delete[] h_B;
//     delete[] h_C;
//     hipFree(d_A);
//     hipFree(d_B);
//     hipFree(d_C);

//     return 0;
// }