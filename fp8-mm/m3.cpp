#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#include <iostream>

constexpr const int BLOCK = 128;

__global__ void custom_kernel(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, 
                   __hip_bfloat16* c, int m, int n, int k) {
                   
    // Your implementation here
    int cx = threadIdx.x + blockDim.x * blockIdx.x;
    int cy = threadIdx.y + blockDim.y * blockIdx.y;
    if(cx >= m || cy >= n) return;
    
    int sn = (n + BLOCK - 1) / BLOCK;
    
    float result = 0;
    // split loop into an outer loop over different blocks, and an inner loop within one block.
    // we can assume k % BLOCK == 0.
    for(int i = 0; i < k; i += BLOCK) {
        // block results accumulates the inner product across a single block.
        // within each block, scales are constant, so we can lift the scaling 
        // outside of the inner loop.
        float block_result = 0;
        for(int ii = 0; ii < BLOCK; ++ii) {
            // load input matrix elements and convert to float for computations
            float av = (float)a[cx + (i + ii) * m];
            float bv = (float)b[cy + (i + ii) * n];
            block_result += av * bv; 
        }
        
        // before we can go to the next block, scale the result of the current block
        // and accumulate to final result
        // note the different indexing into as and bs
        result += block_result * as[cx + i/BLOCK * m] * bs[cy/BLOCK + i/BLOCK * sn];
    }
    
    // finally, write the result as bf16
    c[cx * n + cy] = (__hip_bfloat16)result;
}


#define TILE_SIZE 2

__global__ void matmul_fp8_to_bf16(
    const __hip_fp8_e4m3_fnuz* a, 
    const __hip_fp8_e4m3_fnuz* b, 
    __hip_bfloat16* c, 
    float* a_scale, 
    float* b_scale, 
    int m, 
    int n, 
    int k) {
    __shared__ __hip_fp8_e4m3_fnuz tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ __hip_fp8_e4m3_fnuz tile_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_a = bx * TILE_SIZE + ty;  // 使用ty作为行索引，因为block中的线程按TILE_SIZE划分行
    int row_b = by * TILE_SIZE + tx;  // 使用tx作为行索引

    float result = 0.0f;
    //for (int i = 0; i < ((k + TILE_SIZE - 1) / TILE_SIZE); ++i) {
    for (int i = 0; i < k; i += TILE_SIZE){
        // 计算当前块在k维度上的起始位置
        //int k_start = i * TILE_SIZE;
        int k_start = i;

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
        float block_result = 0;
        // 计算块内乘积并累加
        for (int j = 0; j < TILE_SIZE; ++j) {
            float av = static_cast<float>(tile_A[ty][j]);
            float bv = static_cast<float>(tile_B[tx][j]);
            block_result += av * bv;
        }
        result += block_result * a_scale[row_a*k/128+i/128] * b_scale[(row_b/128)*(k/128)+i/128];
        __syncthreads();
    }

    // 写入结果
    if (row_a < m && row_b < n) {
        c[row_a * n + row_b] = static_cast<__hip_bfloat16>(result);
    }
}

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
                // A: m x k, column-major: A[row + kk * m]
                // B: n x k, column-major: B[col + kk * n]
                float a_val = (float)A[row + kk * m];
                float b_val = (float)B[col + kk * n];
                acc += a_val * b_val * a_scale[(row*k/128+kk/128)] * b_scale[((col/128)*(k/128)+kk/128)];
            }
            C[col + row * n] = __float2bfloat16(acc);
        }
    }
}


int main()
{
    const int m = 4, n = 128, k =256;
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

    //a_scale: [float32] of shape [m, k // 128]
    //b_scale: [float32] of shape [n // 128, k // 128]
    float* h_a_scale = new float[m * (k / 128)];
    float* h_b_scale = new float[(n / 128) * (k / 128)];
    for (int i = 0; i < m * (k / 128); ++i) h_a_scale[i] = 1.0f;
    for (int i = 0; i < (n / 128) * (k / 128); ++i) h_b_scale[i] = 2.0f;

    size_t size_a_scale = m * (k / 128) * sizeof(float);
    size_t size_b_scale = (n / 128) * (k / 128) * sizeof(float);

    __hip_fp8_e4m3_fnuz* d_A;
    __hip_fp8_e4m3_fnuz* d_B;
    __hip_bfloat16* d_C;
    float* d_a_scale;
    float* d_b_scale;
    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);
    hipMalloc(&d_a_scale, size_a_scale);
    hipMalloc(&d_b_scale, size_b_scale);

    cpu_matmul_fp8_to_bf16_column_major(h_A, h_B, h_C_CPU, h_a_scale, h_b_scale, m, n, k);

    std::cout << "CPU result:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << __bfloat162float(h_C_CPU[i + j * m]) << " ";
        }
        std::cout << "\n";
    }

    hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);
    hipMemcpy(d_a_scale, h_a_scale, size_a_scale, hipMemcpyHostToDevice);
    hipMemcpy(d_b_scale, h_b_scale, size_b_scale, hipMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim( (m + TILE_SIZE - 1) / TILE_SIZE ,(n + TILE_SIZE - 1) / TILE_SIZE );

    matmul_fp8_to_bf16<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_a_scale, d_b_scale, m, n, k);

    //custom_kernel<<<dim3((m+15)/16, (n+15)/16), dim3(16, 16), 0, 0>>> (d_A, d_B, d_a_scale, d_b_scale, d_C, m, n, k);
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

    // 检查CPU和GPU的计算结果是否相同，输出不同的索引和值
    int diff_count = 0;
    std::cout << "Comparing CPU and GPU results:\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float cpu_val = __bfloat162float(h_C_CPU[i + j * m]);
            float gpu_val = __bfloat162float(h_C[i + j * m]);
            if (fabs(cpu_val - gpu_val) > 1e-3f) {
                std::cout << "Mismatch at (" << i << ", " << j << "): CPU=" << cpu_val << ", GPU=" << gpu_val << std::endl;
                ++diff_count;
            }
        }
    }
    if (diff_count == 0) {
        std::cout << "All results match between CPU and GPU.\n";
    } else {
        std::cout << "Total mismatches: " << diff_count << std::endl;
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