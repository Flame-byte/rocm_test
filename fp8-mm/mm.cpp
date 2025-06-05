#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_runtime.h>
//#include <torch/torch.h>
#include <iostream>
#include <cstdlib>

#define BLOCK 4
#define TILE_SIZE 4





// __global__ void matrix_multiply_tile(float* a, float* b, float* c, int m, int n, int k)
// {
//     __shared__ float tile_A[TILE_SIZE][BLOCK];
//     __shared__ float tile_B[TILE_SIZE][BLOCK];

//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     int row = by * TILE_SIZE + ty;
//     int col = bx * TILE_SIZE + tx;

//     float acc = 0.0f;

//     // 每轮处理 k 维度的 tile
//     for (int i = 0; i < (k + BLOCK - 1) / BLOCK; ++i) {
//         int k_idx = i * BLOCK;

//         // 加载 A[row][k]
//         if (row < m && k_idx + tx < k)
//             tile_A[ty][tx] = a[row * k + k_idx + tx];
//         else
//             tile_A[ty][tx] = 0.0f;

//         // 加载 B[col][k]，注意现在 B 是 n × k，每一行是一个“向量”
//         if (col < n && k_idx + ty < k)
//             tile_B[ty][tx] = b[col * k + k_idx + ty];  // 行访问 B 的第 col 行，第 k 维度
//         else
//             tile_B[ty][tx] = 0.0f;

//         __syncthreads();

//         // 累加这一轮 tile 的贡献
//         for (int j = 0; j < BLOCK; ++j) {
//             acc += tile_A[ty][j] * tile_B[j][tx];
//         }

//         __syncthreads();
//     }

//     if (row < m && col < n)
//         c[row * n + col] = acc;
// }

// __global__ void mm(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, __hip_bfloat16* c, int m, int n, int k)
// {
//     __shared__ __hip_fp8_e4m3_fnuz tile_A[TILE_SIZE][BLOCK];
//     __shared__ __hip_fp8_e4m3_fnuz tile_B[TILE_SIZE][BLOCK];

//     for(int i = 0; i < (k + BLOCK - 1) / BLOCK; ++i) {

//     }
    
// }

// __global__ void forward_kernel(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, __hip_bfloat16* c, int m, int n, int k) {
//     __shared__ __hip_fp8_e4m3_fnuz tile_A[TILE_SIZE][BLOCK];
//     __shared__ __hip_fp8_e4m3_fnuz tile_B[TILE_SIZE][BLOCK];

//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     int row = by * TILE_SIZE + ty;
//     int col = bx * TILE_SIZE + tx;

//     float acc = 0.0f;

//     for (int i = 0; i < (k + BLOCK - 1) / BLOCK; ++i) {
//         int k_idx = i * BLOCK;

//         if (row < m && k_idx + tx < k)
//         {
//             tile_A[ty][tx] = a[row * k + k_idx + tx];
//             tile_B[ty][tx] = b[row * k + k_idx + tx]; 
//             //printf("tile_A[%d][%d] = %f, tile_B[%d][%d] = %f\n", ty, tx, (float)tile_A[ty][tx], ty, tx, (float)tile_B[ty][tx]);
//         }
//         else
//         {
//             tile_A[ty][tx] = __hip_fp8_e4m3_fnuz(0);
//             tile_B[ty][tx] = __hip_fp8_e4m3_fnuz(0);
//         }

//         // if (col < n && k_idx + ty < k)
//         //     tile_B[ty][tx] = b[col * k + k_idx + ty];
//         // else
//         //     tile_B[ty][tx] = __hip_fp8_e4m3_fnuz(0);

//         __syncthreads();

//         for (int j = 0; j < BLOCK; ++j) {
//             acc += (float)tile_A[ty][j] * (float)tile_B[tx][j];
//         }

//         __syncthreads();
//     }

//     if (row < m && col < n)
//         c[row * n + col] = (__hip_bfloat16)acc;
// }

__global__ void test_fp8_to_float(__hip_fp8_e4m3_fnuz* d_in, __hip_bfloat16* d_out) {
    __shared__ __hip_fp8_e4m3_fnuz tile_in[BLOCK];
    __shared__ float tile_out[BLOCK];

    int tid = threadIdx.x;
    tile_in[tid] = d_in[tid];
    tile_out[tid] = (float)tile_in[tid];
    __syncthreads();
    d_out[tid] = (__hip_bfloat16)tile_out[tid];
}

int main() {
    const int N = 16;
    __hip_fp8_e4m3_fnuz h_in[N];
    __hip_bfloat16 h_out[N];

    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        h_in[i] = __hip_fp8_e4m3_fnuz(i - 8); // -8 ~ 7
    }

    // 分配设备内存
    __hip_fp8_e4m3_fnuz* d_in;
    __hip_bfloat16* d_out;
    hipMalloc(&d_in, N * sizeof(__hip_fp8_e4m3_fnuz));
    hipMalloc(&d_out, N * sizeof(__hip_bfloat16));

    // 拷贝输入到设备
    hipMemcpy(d_in, h_in, N * sizeof(__hip_fp8_e4m3_fnuz), hipMemcpyHostToDevice);

    // 启动 kernel
    int block = 16;
    int grid = (N + block - 1) / block;
    test_fp8_to_float<<<grid, block>>>(d_in, d_out);
    hipDeviceSynchronize();

    // 拷贝结果回主机
    hipMemcpy(h_out, d_out, N * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost);

    // 打印结果
    std::cout << "FP8 input -> float output:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "in: " << int(i - 8) << "  out: " << __bfloat162float(h_out[i]) << std::endl;
    }

    hipFree(d_in);
    hipFree(d_out);

    int load_a_m = 0;
    int load_a_k = 0;
    int a_address = 0;
    __hip_fp8x4_e4m3_fnuz tile_A[4][4];
    __hip_fp8x4_e4m3_fnuz a[4];

    for(int i = 0; i < 4; i++)
    {
        tile_A[load_a_m][load_a_k + i] = a[a_address + i];
    }

    __hip_fp8x4_e4m3_fnuz tile_A[load_a_m][load_a_k] = __hip_fp8x4_e4m3_fnuz(a[a_address]);

    return 0;
}

// // 主函数，调用 forward_kernel 并验证计算结果
// int main() {
//     int m = 4, n = 6, k = 8;
//     size_t size_a = m * k * sizeof(__hip_fp8_e4m3_fnuz);
//     size_t size_b = n * k * sizeof(__hip_fp8_e4m3_fnuz);
//     size_t size_c = m * n * sizeof(__hip_bfloat16);

//     // 分配主机内存
//     __hip_fp8_e4m3_fnuz* h_a = (__hip_fp8_e4m3_fnuz*)malloc(size_a);
//     __hip_fp8_e4m3_fnuz* h_b = (__hip_fp8_e4m3_fnuz*)malloc(size_b);
//     __hip_bfloat16* h_c = (__hip_bfloat16*)malloc(size_c);
//     float* h_c_ref = (float*)malloc(m * n * sizeof(float));

//     // 初始化输入矩阵
//     for (int i = 0; i < m * k; ++i) h_a[i] = __hip_fp8_e4m3_fnuz((i % 13) - 6);
//     for (int i = 0; i < n * k; ++i) h_b[i] = __hip_fp8_e4m3_fnuz((i % 7) - 3);

//     // 分配设备内存
//     __hip_fp8_e4m3_fnuz *d_a, *d_b;
//     __hip_bfloat16 *d_c;
//     hipMalloc(&d_a, size_a);
//     hipMalloc(&d_b, size_b);
//     hipMalloc(&d_c, size_c);

//     // 拷贝数据到设备
//     hipMemcpy(d_a, h_a, size_a, hipMemcpyHostToDevice);
//     hipMemcpy(d_b, h_b, size_b, hipMemcpyHostToDevice);

//     // 启动 kernel
//     dim3 block(TILE_SIZE, BLOCK);
//     dim3 grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);
//     forward_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
//     hipDeviceSynchronize();

//     // 拷贝结果回主机
//     hipMemcpy(h_c, d_c, size_c, hipMemcpyDeviceToHost);

//     // CPU 参考实现
//     for (int row = 0; row < m; ++row) {
//         for (int col = 0; col < n; ++col) {
//             float acc = 0.0f;
//             for (int kk = 0; kk < k; ++kk) {
//                 float av = (float)h_a[row * k + kk];
//                 float bv = (float)h_b[col * k + kk];
//                 acc += av * bv;
//             }
//             h_c_ref[row * n + col] = acc;
//         }
//     }

//     // 打印并比较结果
//     bool correct = true;
//     std::cout << "GPU result:\n";
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             float gpu = float(h_c[i * n + j]);
//             float cpu = h_c_ref[i * n + j];
//             std::cout << gpu << "\t";
//             if (fabs(gpu - cpu) > 1e-2f) correct = false;
//         }
//         std::cout << "\n";
//     }
//     std::cout << "CPU result:\n";
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             std::cout << h_c_ref[i * n + j] << "\t";
//         }
//         std::cout << "\n";
//     }
//     if (correct)
//         std::cout << "Results match!\n";
//     else
//         std::cout << "Results do not match!\n";

//     // 释放内存
//     free(h_a); free(h_b); free(h_c); free(h_c_ref);
//     hipFree(d_a); hipFree(d_b); hipFree(d_c);

//     return 0;
// }

// int main() {
//     int m = 4, k = 8, n = 6;
//     std::vector<float> h_a(m * k), h_b(n * k), h_c(m * n, 0), h_c_cpu(m * n, 0);

//     // Fill A and B with some values
//     for (int i = 0; i < m * k; ++i) h_a[i] = static_cast<float>(i + 1);
//     for (int i = 0; i < n * k; ++i) h_b[i] = static_cast<float>(i + 1);

//     // CPU matrix multiplication: h_c = h_a * h_b^T (since h_b is n x k, treat as n rows of k)
//     for (int row = 0; row < m; ++row) {
//         for (int col = 0; col < n; ++col) {
//             float acc = 0.0f;
//             for (int kk = 0; kk < k; ++kk) {
//                 acc += h_a[row * k + kk] * h_b[col * k + kk];
//             }
//             h_c_cpu[row * n + col] = acc;
//         }
//     }

//     for(int i = 0; i < m; ++i) {
//         for(int j = 0; j < n; ++j) {
//             std::cout << h_c_cpu[i * n + j] << " ";
//         }
//         std::cout << "\n";
//     }

//     float *d_a, *d_b, *d_c;
//     hipMalloc(&d_a, m * k * sizeof(float));
//     hipMalloc(&d_b, n * k * sizeof(float));
//     hipMalloc(&d_c, m * n * sizeof(float));

//     hipMemcpy(d_a, h_a.data(), m * k * sizeof(float), hipMemcpyHostToDevice);
//     hipMemcpy(d_b, h_b.data(), n * k * sizeof(float), hipMemcpyHostToDevice);

//     dim3 block(BLOCK, BLOCK);
//     dim3 grid((n + BLOCK - 1) / BLOCK, (m + BLOCK - 1) / BLOCK);

//     matrix_multiply_tile<<<grid, block>>>(d_a, d_b, d_c, m, n, k);

//     hipMemcpy(h_c.data(), d_c, m * n * sizeof(float), hipMemcpyDeviceToHost);

//     // Print result
//     std::cout << "C = \n";
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             std::cout << h_c[i * n + j] << " ";
//         }
//         std::cout << "\n";
//     }

//     hipFree(d_a);
//     hipFree(d_b);
//     hipFree(d_c);

//     return 0;
// }




