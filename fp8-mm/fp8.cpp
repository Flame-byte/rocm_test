#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_runtime.h>
//#include <torch/torch.h>
#include <iostream>
#include <cstdlib>


#define TILE_SIZE 32

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

    int row_a = bx * TILE_SIZE + ty;  // idx of row a
    int row_b = by * TILE_SIZE + tx;  // idx of row b

    float result = 0.0f;
    //for (int i = 0; i < ((k + TILE_SIZE - 1) / TILE_SIZE); ++i) {
    for (int i = 0; i < k; i += TILE_SIZE){
        // start idx of k
        //int k_start = i * TILE_SIZE;
        int k_start = i;

        // load A's block to tile_A (column-major)
        int a_col = k_start + tx;
        int a_idx = a_col * m + row_a;
        if (a_col < k && row_a < m)
            tile_A[ty][tx] = a[a_idx];
        else
            tile_A[ty][tx] = static_cast<__hip_fp8_e4m3_fnuz>(0);

        // load B's block to tile_B (column-major)
        int b_col = k_start + ty;
        int b_idx = b_col * n + row_b;
        if (b_col < k && row_b < n)
            tile_B[tx][ty] = b[b_idx];
        else
            tile_B[tx][ty] = static_cast<__hip_fp8_e4m3_fnuz>(0);

        __syncthreads();
        float block_result = 0;
        // compute inner product and accumulate
        for (int j = 0; j < TILE_SIZE; ++j) {
            float av = static_cast<float>(tile_A[ty][j]);
            float bv = static_cast<float>(tile_B[tx][j]);
            block_result += av * bv;
        }
        result += block_result * a_scale[(i/128)*m+row_a] * b_scale[(i/128)*n+row_b];
        __syncthreads();
    }

    // write result
    if (row_a < m && row_b < n) {
        c[row_a * n + row_b] = static_cast<__hip_bfloat16>(result);
    }
}

__global__ void forward_kernel(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, __hip_bfloat16* c, int m, int n, int k) {
    __shared__ __hip_fp8_e4m3_fnuz tile_A[TILE_SIZE][BLOCK];
    __shared__ __hip_fp8_e4m3_fnuz tile_B[TILE_SIZE][BLOCK];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    int block_col = col / BLOCK;

    float acc = 0.0f;

    // 每轮处理 k 维度的 tile
    for (int i = 0; i < (k + BLOCK - 1) / BLOCK; ++i) {
        int k_idx = i * BLOCK;

        // 加载 A[row][k]
        if (row < m && k_idx + tx < k)
        {
            tile_A[ty][tx] = a[row * k + k_idx + tx];
            //printf("tile_A[%d][%d] = %d\n", ty, tx, (float)tile_A[ty][tx]);
        }
        else
        {
            tile_A[ty][tx] = __hip_fp8_e4m3_fnuz(0);
        }

        // 加载 B[col][k]，注意现在 B 是 n × k，每一行是一个“向量”
        if (col < n && k_idx + ty < k)
        {
            tile_B[ty][tx] = b[col * k + k_idx + ty];  // 行访问 B 的第 col 行，第 k 维度
            //printf("tile_B[%d][%d] = %d\n", ty, tx, (float)tile_B[ty][tx]);
        }
        else
        {
            tile_B[ty][tx] = __hip_fp8_e4m3_fnuz(0);
        }

        __syncthreads();

        // 累加这一轮 tile 的贡献
        for (int j = 0; j < BLOCK; ++j) {
            acc += (float)tile_A[ty][j] * (float)tile_B[j][tx] * as[row * (k / BLOCK) + i] * bs[block_col * (k / BLOCK) + i];
        }

        __syncthreads();
    }

    if (row < m && col < n)
        c[row * n + col] = (__hip_bfloat16)acc;
}

// CPU reference implementation for fp8 matmul with scaling
void fp8_mm_cpu(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, __hip_bfloat16* c, int m, int n, int k) {
    int scale_k = k / BLOCK;
    int scale_n = (n + BLOCK - 1) / BLOCK;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            int block_col = col / BLOCK;
            for (int i = 0; i < scale_k; ++i) {
                float block_result = 0.0f;
                int k_idx = i * BLOCK;
                for (int j = 0; j < BLOCK && (k_idx + j) < k; ++j) {
                    float av = (float)a[row * k + k_idx + j];
                    float bv = (float)b[col * k + k_idx + j];
                    block_result += av * bv;
                }
                float scale_a = as[row * scale_k + i];
                float scale_b = bs[block_col * scale_k + i];
                acc += block_result * scale_a * scale_b;
            }
            c[row * n + col] = (__hip_bfloat16)acc;
        }
    }
}

void fp8_mm(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, __hip_bfloat16* c, int m, int n, int k) {
    dim3 blocksize(BLOCK, TILE_SIZE);
    dim3 gridsize((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);

    forward_kernel<<<gridsize, blocksize, 0, 0>>> (a, b, as, bs, c, m, n, k);
}

int main() {
    int m = 4, n = 6, k = 128;
    __hip_fp8_e4m3_fnuz* h_a = (__hip_fp8_e4m3_fnuz*)malloc(m * k * sizeof(__hip_fp8_e4m3_fnuz));
    __hip_fp8_e4m3_fnuz* h_b = (__hip_fp8_e4m3_fnuz*)malloc(n * k * sizeof(__hip_fp8_e4m3_fnuz));
    float* h_as = (float*)malloc(m * (k / BLOCK) * sizeof(float));
    float* h_bs = (float*)malloc(((n + BLOCK - 1) / BLOCK) * (k / BLOCK) * sizeof(float));
    __hip_bfloat16* h_c = (__hip_bfloat16*)malloc(m * n * sizeof(__hip_bfloat16));
    __hip_bfloat16* h_c_cpu = (__hip_bfloat16*)malloc(m * n * sizeof(__hip_bfloat16));

    // Fill A and B with some values
    for (int i = 0; i < m * k; ++i) h_a[i] = __hip_fp8_e4m3_fnuz(rand() % 256);
    for (int i = 0; i < n * k; ++i) h_b[i] = __hip_fp8_e4m3_fnuz(rand() % 256);
    // Fill as and bs with some values (avoid zero scaling)
    int scale_k = k / BLOCK;
    int scale_n = (n + BLOCK - 1) / BLOCK;
    for (int i = 0; i < m * scale_k; ++i) h_as[i] = 2.0f;
    for (int i = 0; i < scale_n * scale_k; ++i) h_bs[i] = 3.0f;

    // Device memory
    __hip_fp8_e4m3_fnuz *d_a, *d_b;
    float *d_as, *d_bs;
    __hip_bfloat16 *d_c;
    hipMalloc(&d_a, m * k * sizeof(__hip_fp8_e4m3_fnuz));
    hipMalloc(&d_b, n * k * sizeof(__hip_fp8_e4m3_fnuz));
    hipMalloc(&d_as, m * scale_k * sizeof(float));
    hipMalloc(&d_bs, scale_n * scale_k * sizeof(float));
    hipMalloc(&d_c, m * n * sizeof(__hip_bfloat16));

    hipMemcpy(d_a, h_a, m * k * sizeof(__hip_fp8_e4m3_fnuz), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, n * k * sizeof(__hip_fp8_e4m3_fnuz), hipMemcpyHostToDevice);
    hipMemcpy(d_as, h_as, m * scale_k * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_bs, h_bs, scale_n * scale_k * sizeof(float), hipMemcpyHostToDevice);

    fp8_mm(d_a, d_b, d_as, d_bs, d_c, m, n, k);

    hipMemcpy(h_c, d_c, m * n * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    // CPU reference
    fp8_mm_cpu(h_a, h_b, h_as, h_bs, h_c_cpu, m, n, k);

    // Print result
    std::cout << "(GPU) = \n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << float(h_c[i * n + j]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "(CPU) = \n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << float(h_c_cpu[i * n + j]) << " ";
        }
        std::cout << "\n";
    }

    // 计算GPU和CPU结果的差异
    float max_diff = 0.0f;
    float total_diff = 0.0f;
    int count = 0;

    for (int i = 0; i < m; ++i) {   
        for (int j = 0; j < n; ++j) {
            float diff = std::abs(float(h_c[i * n + j]) - float(h_c_cpu[i * n + j]));
            if (diff > max_diff) max_diff = diff;
            total_diff += diff;
            ++count;
        }   
    }

    float avg_diff = total_diff / count;
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Average difference: " << avg_diff << std::endl;   

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_as);
    hipFree(d_bs);
    hipFree(d_c);

    free(h_a);
    free(h_b);
    free(h_as);
    free(h_bs);
    free(h_c);
    free(h_c_cpu);

    return 0;
}

// void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
//     int m = a.size(0);
//     int n = b.size(0);
//     int k = a.size(1);
//     custom_kernel<<<dim3((m+15)/16, (n+15)/16), dim3(16, 16), 0, 0>>> ((__hip_fp8_e4m3_fnuz*)a.data_ptr(), (__hip_fp8_e4m3_fnuz*)b.data_ptr(), 
//     as.data_ptr<float>(), bs.data_ptr<float>(), (__hip_bfloat16*)c.data_ptr(), m, n, k);
//     //C10_CUDA_CHECK(cudaGetLastError());
// }

// int main() {
//     // 假设 m, n, k 都是 BLOCK 的倍数
//     int m = 4, n = 6, k = 128;
//     int scale_n = (n + BLOCK - 1) / BLOCK;
//     int scale_k = (k + BLOCK - 1) / BLOCK;

//     // 创建输入张量，类型为 torch::kFloat32 以便后续转换
//     auto a_fp32 = torch::full({m, k}, 1.0, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
//     auto b_fp32 = torch::full({n, k}, 2.0, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
//     auto as = torch::rand({m, scale_k}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
//     auto bs = torch::rand({n, scale_k}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
//     auto c = torch::zeros({m, n}, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));

//     // 假设有将 float32 转换为 __hip_fp8_e4m3_fnuz 的函数
//     // 这里只是示意，实际需要自定义转换
//     torch::Tensor a_fp8 = a_fp32.to(torch::kFloat32); // 这里应为 fp8 类型
//     torch::Tensor b_fp8 = b_fp32.to(torch::kFloat32); // 这里应为 fp8 类型

//     // 调用 fp8_mm
//     fp8_mm(a_fp8, b_fp8, as, bs, c);

//     // 同步等待 kernel 完成
//     hipDeviceSynchronize();

//     // 打印部分结果
//     auto c_cpu = c.to(torch::kCPU, torch::kFloat32);
//     std::cout << "Result c[0][0]: " << c_cpu[0][0].item<float>() << std::endl;

//     return 0;
// }
