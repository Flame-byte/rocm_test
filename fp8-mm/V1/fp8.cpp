#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#include <iostream>
//#include <torch/torch.h>

//------------------------------------------------------------------------------------------------//
//custom kernel

constexpr const int BLOCK = 128;

__global__ void custom_kernel(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, 
                   __hip_bfloat16* c, int m, int n, int k) {
    int cx = threadIdx.x + blockDim.x * blockIdx.x;
    int cy = threadIdx.y + blockDim.y * blockIdx.y;
    if(cx >= m || cy >= n) return;
    
    int sn = (n + BLOCK - 1) / BLOCK;
    
    float result = 0;
    for(int i = 0; i < k; i += BLOCK) {
        float block_result = 0;
        for(int ii = 0; ii < BLOCK; ++ii) {
            float av = (float)a[cx + (i + ii) * m];
            float bv = (float)b[cy + (i + ii) * n];
            block_result += av * bv; 
        }
        result += block_result * as[cx + i/BLOCK * m] * bs[cy/BLOCK + i/BLOCK * sn];
    }
    c[cx * n + cy] = (__hip_bfloat16)result;
}

//------------------------------------------------------------------------------------------------//
//my kernel

#define TILE_SIZE 2

__global__ void matmul_fp8_to_bf16(
    const __hip_fp8_e4m3_fnuz* a, 
    const __hip_fp8_e4m3_fnuz* b, 
    float* a_scale, 
    float* b_scale, 
    __hip_bfloat16* c, 
    int m, 
    int n, 
    int k) {
    __shared__ __hip_fp8_e4m3_fnuz tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __hip_fp8_e4m3_fnuz tile_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_a = bx * TILE_SIZE + ty;
    int row_b = by * TILE_SIZE + tx;

    float result = 0.0f;
    for (int i = 0; i < k; i += TILE_SIZE){
        int k_start = i;

        int a_col = k_start + tx;
        int a_idx = a_col * m + row_a;
        if (a_col < k && row_a < m)
            tile_A[ty][tx] = a[a_idx];
        else
            tile_A[ty][tx] = static_cast<__hip_fp8_e4m3_fnuz>(0);

        int b_col = k_start + ty;
        int b_idx = b_col * n + row_b;
        if (b_col < k && row_b < n)
            tile_B[ty][tx] = b[b_idx];
        else
            tile_B[ty][tx] = static_cast<__hip_fp8_e4m3_fnuz>(0);

        __syncthreads();
        float block_result = 0;
        for (int j = 0; j < TILE_SIZE; ++j) {
            float av = static_cast<float>(tile_A[ty][j]);
            float bv = static_cast<float>(tile_B[j][tx]);
            block_result += av * bv;
        }
    result += block_result * a_scale[row_a + (i / 128) * m] * b_scale[(row_b / 128) + (i / 128) * ((n + 127) / 128)];
        __syncthreads();
    }

    if (row_a < m && row_b < n) {
        c[row_a * n + row_b] = static_cast<__hip_bfloat16>(result);
    }
}

//------------------------------------------------------------------------------------------------//

// CPU reference implementation for column-major fp8 matmul (A: m x k, B: n x k, C: m x n, all column-major)
void cpu_matmul_fp8_to_bf16_column_major(
    const __hip_fp8_e4m3_fnuz* A,
    const __hip_fp8_e4m3_fnuz* B,
    __hip_bfloat16* C,
    float* a_scale,
    float* b_scale,
    int m, int n, int k)
{
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                float a_val = static_cast<float>(A[row + kk * m]);
                float b_val = static_cast<float>(B[col + kk * n]);
                acc += a_val * b_val * a_scale[row + (kk/128) * m] * b_scale[(col/128) + (kk/128) * ((n+127)/128)];
            }
            C[row * n + col] = __float2bfloat16(acc);
        }
    }
}

//------------------------------------------------------------------------------------------------//

int main()
{
    //const int m = 1024, n = 1536, k = 7168;
    const int m = 64, n = 576, k = 7168;
    size_t size_A = m * k * sizeof(__hip_fp8_e4m3_fnuz);
    size_t size_B = n * k * sizeof(__hip_fp8_e4m3_fnuz);
    size_t size_C = m * n * sizeof(__hip_bfloat16);

    // Host allocations
    __hip_fp8_e4m3_fnuz* h_A = new __hip_fp8_e4m3_fnuz[m * k];
    __hip_fp8_e4m3_fnuz* h_B = new __hip_fp8_e4m3_fnuz[n * k];
    __hip_bfloat16* h_C_custom = new __hip_bfloat16[m * n];
    __hip_bfloat16* h_C_my = new __hip_bfloat16[m * n];
    __hip_bfloat16* h_C_cpu = new __hip_bfloat16[m * n];

    // Initialize data
    for (int i = 0; i < m * k; ++i) h_A[i] = __hip_fp8_e4m3_fnuz(i % 13 - 6);
    for (int i = 0; i < n * k; ++i) h_B[i] = __hip_fp8_e4m3_fnuz(i % 7 - 3);

    float* h_a_scale = new float[m * (k / 128)];
    float* h_b_scale = new float[(n / 128) * (k / 128)];
    for (int i = 0; i < m * (k / 128); ++i) h_a_scale[i] = 1.0f;
    for (int i = 0; i < (n / 128) * (k / 128); ++i) h_b_scale[i] = 2.0f;

    size_t size_a_scale = m * (k / 128) * sizeof(float);
    size_t size_b_scale = (n / 128) * (k / 128) * sizeof(float);

    // Device allocations
    __hip_fp8_e4m3_fnuz* d_A;
    __hip_fp8_e4m3_fnuz* d_B;
    __hip_bfloat16* d_C_custom;
    __hip_bfloat16* d_C_my;
    float* d_a_scale;
    float* d_b_scale;

    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C_custom, size_C);
    hipMalloc(&d_C_my, size_C);
    hipMalloc(&d_a_scale, size_a_scale);
    hipMalloc(&d_b_scale, size_b_scale);

    // Copy data to device
    hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    hipMemcpy(d_a_scale, h_a_scale, size_a_scale, hipMemcpyHostToDevice);
    hipMemcpy(d_b_scale, h_b_scale, size_b_scale, hipMemcpyHostToDevice);

    // Launch custom_kernel
    dim3 blockDim2(16, 16);
    dim3 gridDim2((m + 15) / 16, (n + 15) / 16);
    custom_kernel<<<gridDim2, blockDim2>>>(d_A, d_B, d_a_scale, d_b_scale, d_C_custom, m, n, k);

    // Launch my kernel
    dim3 blockDim_my(TILE_SIZE, TILE_SIZE);
    dim3 gridDim_my((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    matmul_fp8_to_bf16<<<gridDim_my, blockDim_my>>>(d_A, d_B, d_a_scale, d_b_scale, d_C_my, m, n, k);

    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(h_C_custom, d_C_custom, size_C, hipMemcpyDeviceToHost);
    hipMemcpy(h_C_my, d_C_my, size_C, hipMemcpyDeviceToHost);

    // CPU reference
    cpu_matmul_fp8_to_bf16_column_major(
        h_A, h_B, h_C_cpu, h_a_scale, h_b_scale, m, n, k
    );

    // // Print custom_kernel result
    // std::cout << "Result from custom_kernel:\n";
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << __bfloat162float(h_C_custom[i * n + j]) << " ";
    //     }
    //     std::cout << "\n";
    // }

    // // Print my kernel result
    // std::cout << "Result from my kernel:\n";
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << __bfloat162float(h_C_my[i * n + j]) << " ";
    //     }
    //     std::cout << "\n";
    // }

    // // Print CPU result
    // std::cout << "Result from CPU reference:\n";
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << __bfloat162float(h_C_cpu[i * n + j]) << " ";
    //     }
    //     std::cout << "\n";
    // }

    // Compare the results of custom_kernel and my kernel
    int diff_count_custom_vs_my = 0;
    std::cout << "Comparing results of custom_kernel and my kernel:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float val_custom = __bfloat162float(h_C_custom[i * n + j]);
            float val_my = __bfloat162float(h_C_my[i * n + j]);
            if (fabs(val_custom - val_my) > 1e-3f) {
                std::cout << "Mismatch at (" << i << ", " << j << "): custom=" << val_custom << ", my=" << val_my << std::endl;
                ++diff_count_custom_vs_my;
            }
        }
    }
    if (diff_count_custom_vs_my == 0) {
        std::cout << "All results match between custom_kernel and my kernel.\n";
    } else {
        std::cout << "Total mismatches between custom_kernel and my kernel: " << diff_count_custom_vs_my << std::endl;
    }

    // // Compare the results of CPU and custom_kernel
    // int diff_count_cpu_vs_custom = 0;
    // std::cout << "Comparing results of CPU and custom_kernel:\n";
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         float val_cpu = __bfloat162float(h_C_cpu[i * n + j]);
    //         float val_custom = __bfloat162float(h_C_custom[i * n + j]);
    //         if (fabs(val_cpu - val_custom) > 1e-3f) {
    //             std::cout << "Mismatch at (" << i << ", " << j << "): cpu=" << val_cpu << ", custom=" << val_custom << std::endl;
    //             ++diff_count_cpu_vs_custom;
    //         }
    //     }
    // }
    // if (diff_count_cpu_vs_custom == 0) {
    //     std::cout << "All results match between CPU and custom_kernel.\n";
    // } else {
    //     std::cout << "Total mismatches between CPU and custom_kernel: " << diff_count_cpu_vs_custom << std::endl;
    // }

    // // Compare the results of CPU and my kernel
    // int diff_count_cpu_vs_my = 0;
    // std::cout << "Comparing results of CPU and my kernel:\n";
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         float val_cpu = __bfloat162float(h_C_cpu[i * n + j]);
    //         float val_my = __bfloat162float(h_C_my[i * n + j]);
    //         if (fabs(val_cpu - val_my) > 1e-3f) {
    //             std::cout << "Mismatch at (" << i << ", " << j << "): cpu=" << val_cpu << ", my=" << val_my << std::endl;
    //             ++diff_count_cpu_vs_my;
    //         }
    //     }
    // }
    // if (diff_count_cpu_vs_my == 0) {
    //     std::cout << "All results match between CPU and my kernel.\n";
    // } else {
    //     std::cout << "Total mismatches between CPU and my kernel: " << diff_count_cpu_vs_my << std::endl;
    // }

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_custom;
    delete[] h_C_my;
    delete[] h_C_cpu;
    delete[] h_a_scale;
    delete[] h_b_scale;
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C_custom);
    hipFree(d_C_my);
    hipFree(d_a_scale);
    hipFree(d_b_scale);

    return 0;
}