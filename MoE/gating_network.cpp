#include "gating_network.h"
#include "dot_product.h"
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include "linear.h"
#include <cmath>
#include "softmax.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>


// Top-k selection kernel: for each row of size K, select topk values and indices
__global__ void topk_kernel(__half* scores, __half* topk_scores, __half* topk_indices, int M, int K, int topk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        __half* row = scores + idx * K;
        __half* out_s = topk_scores + idx * topk;
        __half* out_i = topk_indices + idx * topk;
        for (int t = 0; t < topk; ++t) {
            float max_val = -1e20f;
            int max_idx = 0;
            #pragma unroll 4
            for (int j = 0; j < K; ++j) {
                float v = __half2float(row[j]);
                if (v > max_val) {
                    max_val = v;
                    max_idx = j;
                }
            }
            out_s[t] = __float2half(max_val);
            out_i[t] = __float2half((float)max_idx);
            // mark selected so next iteration won't pick it
            row[max_idx] = __float2half(-1e20f);
        }
    }
}

extern "C" void gating_network(
    __half* W_gate,
    __half* a,
    __half* logits,
    __half* scores,
    __half* topk_scores,
    __half* topk_indices,
    int batch_size,
    int seq_len,
    int d_hidden,
    int n_experts,
    int topk,
    hipStream_t stream,
    rocblas_handle handle
)
{
    int M = batch_size * seq_len;
    int K = n_experts;

    // // Allocate device memory
    // __half *d_W_gate, *d_a, *d_logits, *d_scores, *d_topk_scores, *d_topk_indices;
    // hipMalloc(&d_W_gate, d_hidden * K * sizeof(__half));
    // hipMalloc(&d_a, M * d_hidden * sizeof(__half));
    // hipMalloc(&d_logits, M * K * sizeof(__half));
    // hipMalloc(&d_scores, M * K * sizeof(__half));
    // hipMalloc(&d_topk_scores, M * topk * sizeof(__half));
    // hipMalloc(&d_topk_indices, M * topk * sizeof(__half));

    // // Copy host to device
    // hipMemcpyAsync(d_W_gate, h_W_gate, d_hidden * K * sizeof(__half), hipMemcpyHostToDevice, stream);
    // hipMemcpyAsync(d_a, a, M * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream);

    // // Create rocBLAS handle and set stream
    // rocblas_handle handle;
    // rocblas_create_handle(&handle);
    // rocblas_set_stream(handle, stream);

    // Compute logits: (batch*seq, d_hidden) x (d_hidden, K) -> (batch*seq, K)
    linear(a, W_gate, logits, batch_size, seq_len, d_hidden, K, handle);

    // Softmax
    softmax(logits, scores, M, K, stream);

    // Top-k selection
    int threads = 128;
    int blocks = (M + threads - 1) / threads;
    hipLaunchKernelGGL(topk_kernel, dim3(blocks), dim3(threads), 0, stream, scores, topk_scores, topk_indices, M, K, topk);

    // // Copy results back
    // hipMemcpyAsync(topk_scores, d_topk_scores, M * topk * sizeof(__half), hipMemcpyDeviceToHost, stream);
    // hipMemcpyAsync(topk_indices, d_topk_indices, M * topk * sizeof(__half), hipMemcpyDeviceToHost, stream);

    // // Synchronize to ensure completion
    // hipStreamSynchronize(stream);

    // // Cleanup
    // hipFree(d_W_gate);
    // hipFree(d_a);
    // hipFree(d_logits);
    // hipFree(d_scores);
    // hipFree(d_topk_scores);
    // hipFree(d_topk_indices);
    // hipFree(logits);
    // hipFree(scores);
    // rocblas_destroy_handle(handle);
}


// // 辅助函数
// float half2float(__half h) { return __half2float(h); }
// __half float2half(float f) { return __float2half(f); }

// int main() {
//     int m = 5;      // 行数
//     int k = 10;     // 每行元素数
//     int topk = 3;   // 取前3大

//     int size = m * k;
//     std::vector<__half> h_scores(size);

//     // 随机填充输入
//     std::mt19937 rng(42);
//     std::uniform_real_distribution<float> dist(-10, 10);
//     for (int i = 0; i < size; ++i) {
//         h_scores[i] = float2half(dist(rng));
//     }

//     // 设备分配
//     __half *d_scores, *d_topk_scores, *d_topk_indices;
//     hipMalloc(&d_scores, size * sizeof(__half));
//     hipMalloc(&d_topk_scores, m * topk * sizeof(__half));
//     hipMalloc(&d_topk_indices, m * topk * sizeof(__half));

//     // 拷贝输入到设备
//     hipMemcpy(d_scores, h_scores.data(), size * sizeof(__half), hipMemcpyHostToDevice);

//     // 启动 kernel
//     int threads = 128;
//     int blocks = (m + threads - 1) / threads;
//     hipLaunchKernelGGL(topk_kernel, dim3(blocks), dim3(threads), 0, 0, d_scores, d_topk_scores, d_topk_indices, m, k, topk);

//     // 拷回结果
//     std::vector<__half> h_topk_scores(m * topk);
//     std::vector<__half> h_topk_indices(m * topk);
//     hipMemcpy(h_topk_scores.data(), d_topk_scores, m * topk * sizeof(__half), hipMemcpyDeviceToHost);
//     hipMemcpy(h_topk_indices.data(), d_topk_indices, m * topk * sizeof(__half), hipMemcpyDeviceToHost);

//     // CPU 参考
//     for (int row = 0; row < m; ++row) {
//         std::vector<std::pair<float, int>> row_vals;
//         for (int col = 0; col < k; ++col) {
//             row_vals.emplace_back(half2float(h_scores[row * k + col]), col);
//         }
//         std::partial_sort(row_vals.begin(), row_vals.begin() + topk, row_vals.end(),
//                           [](const auto& a, const auto& b) { return a.first > b.first; });

//         std::cout << "Row " << row << " top-" << topk << " (CPU): ";
//         for (int t = 0; t < topk; ++t) {
//             std::cout << "[" << row_vals[t].first << ", idx=" << row_vals[t].second << "] ";
//         }
//         std::cout << "\n";

//         std::cout << "Row " << row << " top-" << topk << " (GPU): ";
//         for (int t = 0; t < topk; ++t) {
//             float v = half2float(h_topk_scores[row * topk + t]);
//             int idx = static_cast<int>(half2float(h_topk_indices[row * topk + t]));
//             std::cout << "[" << v << ", idx=" << idx << "] ";
//         }
//         std::cout << "\n";
//     }

//     // 分开打印 d_topk_scores
//     std::cout << "d_topk_scores (GPU):\n";
//     for (int row = 0; row < m; ++row) {
//         std::cout << "Row " << row << ": ";
//         for (int t = 0; t < topk; ++t) {
//             float v = half2float(h_topk_scores[row * topk + t]);
//             std::cout << v << " ";
//         }
//         std::cout << "\n";
//     }

//     // 分开打印 d_topk_indices
//     std::cout << "d_topk_indices (GPU):\n";
//     for (int row = 0; row < m; ++row) {
//         std::cout << "Row " << row << ": ";
//         for (int t = 0; t < topk; ++t) {
//             int idx = static_cast<int>(half2float(h_topk_indices[row * topk + t]));
//             std::cout << idx << " ";
//         }
//         std::cout << "\n";
//     }


//     // 释放
//     hipFree(d_scores);
//     hipFree(d_topk_scores);
//     hipFree(d_topk_indices);

//     return 0;
// }

// extern "C" void gating_network(
//     __half* h_W_gate,
//     __half* a,
//     __half* topk_scores,
//     __half* topk_indices,
//     int batch_size,
//     int seq_len,
//     int d_hidden,
//     int n_experts,
//     int topk,
//     hipStream_t stream
// )
// {
//     int M = batch_size * seq_len;
//     int K = n_experts;

//     // Allocate device memory
//     __half *d_W_gate, *d_a, *d_logits, *d_scores, *d_topk_scores, *d_topk_indices;
//     hipMalloc(&d_W_gate, d_hidden * K * sizeof(__half));
//     hipMalloc(&d_a, M * d_hidden * sizeof(__half));
//     hipMalloc(&d_logits, M * K * sizeof(__half));
//     hipMalloc(&d_scores, M * K * sizeof(__half));
//     hipMalloc(&d_topk_scores, M * topk * sizeof(__half));
//     hipMalloc(&d_topk_indices, M * topk * sizeof(__half));

//     // Copy host to device
//     hipMemcpyAsync(d_W_gate, h_W_gate, d_hidden * K * sizeof(__half), hipMemcpyHostToDevice, stream);
//     hipMemcpyAsync(d_a, a, M * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream);

//     // Create rocBLAS handle and set stream
//     rocblas_handle handle;
//     rocblas_create_handle(&handle);
//     rocblas_set_stream(handle, stream);

//     // Compute logits: (batch*seq, d_hidden) x (d_hidden, K) -> (batch*seq, K)
//     linear(d_a, d_W_gate, d_logits, batch_size, seq_len, d_hidden, K, handle);

//     // Softmax
//     softmax(d_logits, d_scores, M, K, stream);

//     // Top-k selection
//     int threads = 128;
//     int blocks = (M + threads - 1) / threads;
//     hipLaunchKernelGGL(topk_kernel, dim3(blocks), dim3(threads), 0, stream, d_scores, d_topk_scores, d_topk_indices, M, K, topk);

//     // Copy results back
//     hipMemcpyAsync(topk_scores, d_topk_scores, M * topk * sizeof(__half), hipMemcpyDeviceToHost, stream);
//     hipMemcpyAsync(topk_indices, d_topk_indices, M * topk * sizeof(__half), hipMemcpyDeviceToHost, stream);

//     // Synchronize to ensure completion
//     hipStreamSynchronize(stream);

//     // Cleanup
//     hipFree(d_W_gate);
//     hipFree(d_a);
//     hipFree(d_logits);
//     hipFree(d_scores);
//     hipFree(d_topk_scores);
//     hipFree(d_topk_indices);
//     rocblas_destroy_handle(handle);
// }