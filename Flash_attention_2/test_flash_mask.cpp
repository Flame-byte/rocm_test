#include "flash.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

void cpu_attention(const __half* h_Q, const __half* h_K, const __half* h_V, __half* h_O_cpu,
                   int seq_len, int head_dim, int num_heads_q, int num_heads_kv) {
    const float sqrt_dim = sqrtf(head_dim);
    
    // 遍历每个Q头
    for (int q_head = 0; q_head < num_heads_q; ++q_head) {
        int kv_head = q_head / (num_heads_q / num_heads_kv); // GQA分组
        
        // 提取Q头数据 [seq_len, head_dim]
        std::vector<std::vector<float>> Q(seq_len, std::vector<float>(head_dim));
        for (int i = 0; i < seq_len; ++i) {
            int base = i * (num_heads_q * head_dim) + q_head * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                Q[i][d] = __half2float(h_Q[base + d]);
            }
        }
        
        // 提取K头数据 [seq_len, head_dim]
        std::vector<std::vector<float>> K(seq_len, std::vector<float>(head_dim));
        for (int j = 0; j < seq_len; ++j) {
            int base = j * (num_heads_kv * head_dim) + kv_head * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                K[j][d] = __half2float(h_K[base + d]);
            }
        }
        
        // 计算S = Q*K^T / sqrt(d)
        std::vector<std::vector<float>> S(seq_len, std::vector<float>(seq_len));
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += Q[i][d] * K[j][d];
                }
                S[i][j] = dot / sqrt_dim;
            }
        }

        // 应用上三角mask (causal mask)
        // 将下三角区域（不包括对角线）设置为负无穷大
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                // 如果j > i，则是上三角区域，保持不变
                // 如果j < i，则是下三角区域，设置为负无穷大
                if (j > i) {
                    S[i][j] = -INFINITY;  // 使用负无穷大表示mask
                }
            }
        }

        
        // 行Softmax
        for (int i = 0; i < seq_len; ++i) {
            float max_val = S[i][0];
            for (float val : S[i]) if (val > max_val) max_val = val;
            
            float sum_exp = 0.0f;
            for (float& val : S[i]) {
                val = expf(val - max_val);
                sum_exp += val;
            }
            for (float& val : S[i]) val /= sum_exp;
        }
        
        // 提取V头数据 [seq_len, head_dim]
        std::vector<std::vector<float>> V(seq_len, std::vector<float>(head_dim));
        for (int j = 0; j < seq_len; ++j) {
            int base = j * (num_heads_kv * head_dim) + kv_head * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                V[j][d] = __half2float(h_V[base + d]);
            }
        }
        
        // 计算O = S*V
        std::vector<std::vector<float>> O(seq_len, std::vector<float>(head_dim, 0));
        for (int i = 0; i < seq_len; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                for (int j = 0; j < seq_len; ++j) {
                    O[i][d] += S[i][j] * V[j][d];
                }
            }
        }
        
        // 写入结果
        for (int i = 0; i < seq_len; ++i) {
            int base = i * (num_heads_q * head_dim) + q_head * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                h_O_cpu[base + d] = __float2half(O[i][d]);
            }
        }
    }
}

int main() {
    // Parameters
    const int seq_len = 2048; // Example sequence length
    const int head_dim = 128; // Example head dimension
    const int num_heads_q = 32; // Number of query heads
    const int num_heads_kv = 8; // Number of key/value heads

    // Initialize Q, K, V with some values
    // std::vector<__half> h_Q(seq_len * num_heads_q * head_dim, __float2half(1.0f));
    // std::vector<__half> h_K(seq_len * num_heads_kv * head_dim, __float2half(1.0f));
    // std::vector<__half> h_V(seq_len * num_heads_kv * head_dim, __float2half(1.0f));

    __half *h_Q = (__half*)malloc(seq_len * num_heads_q * head_dim * sizeof(__half));
    __half *h_K = (__half*)malloc(seq_len * num_heads_kv * head_dim * sizeof(__half));
    __half *h_V = (__half*)malloc(seq_len * num_heads_kv * head_dim * sizeof(__half));

    for (int i = 0; i < seq_len * num_heads_q * head_dim; i++) {
        //奇数为1，偶数为2
        h_Q[i] = __float2half((i % 2 == 0) ? 1.0f : 2.0f);
        //随机数
        //h_Q[i] = __float2half(rand() % 10000 / 10000.0f);
    }

    for (int i = 0; i < seq_len * num_heads_kv * head_dim; i++) {
        //奇数为1，偶数为2
        h_K[i] = __float2half((i % 2 == 0) ? 1.0f : 2.0f);
        h_V[i] = __float2half((i % 2 == 0) ? 1.0f : 2.0f);
        // //随机数
        // h_K[i] = __float2half(rand() % 10000 / 10000.0f);
        // h_V[i] = __float2half(rand() % 10000 / 10000.0f);
    }

    __half *h_O_cpu = (__half*)malloc(seq_len * num_heads_q * head_dim * sizeof(__half));
    cpu_attention(h_Q, h_K, h_V, h_O_cpu, seq_len, head_dim, num_heads_q, num_heads_kv);

    std::cout << "Output O_cpu (for each head, first and last dimension):" << std::endl;
    for (int head = 0; head < num_heads_q; ++head) {
        int base_idx = head * head_dim;
        std::cout << "Head " << head << ": first=" << __half2float(h_O_cpu[base_idx]) 
                  << ", last=" << __half2float(h_O_cpu[base_idx + head_dim - 1]) << std::endl;
    }
    std::cout << std::endl;
    

    // Allocate device memory
    __half *d_Q, *d_K, *d_V, *d_O;
    hipMalloc(&d_Q, seq_len * num_heads_q * head_dim * sizeof(__half));
    hipMalloc(&d_K, seq_len * num_heads_kv * head_dim * sizeof(__half));
    hipMalloc(&d_V, seq_len * num_heads_kv * head_dim * sizeof(__half));
    hipMalloc(&d_O, seq_len * num_heads_q * head_dim * sizeof(__half)); // Output size matches Q

    // Copy data to device
    hipMemcpy(d_Q, h_Q, seq_len * num_heads_q * head_dim * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_K, h_K, seq_len * num_heads_kv * head_dim * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V, seq_len * num_heads_kv * head_dim * sizeof(__half), hipMemcpyHostToDevice);

    // Call the flash_attention function
    flash_attention(d_Q, d_K, d_V, d_O, seq_len, head_dim, num_heads_q, num_heads_kv);

    // Copy result back to host
    std::vector<__half> h_O(seq_len * num_heads_q * head_dim);
    hipMemcpy(h_O.data(), d_O, seq_len * num_heads_q * head_dim * sizeof(__half), hipMemcpyDeviceToHost);

    // Print the GPU output for each head's first and last dimension
    std::cout << "Output O_gpu (for each head, first and last dimension):" << std::endl;
    for (int head = 0; head < num_heads_q; ++head) {
        int base_idx = head * head_dim;
        std::cout << "Head " << head << ": first=" << __half2float(h_O[base_idx]) 
                  << ", last=" << __half2float(h_O[base_idx + head_dim - 1]) << std::endl;
    }
    std::cout << std::endl;
    

    // Compare CPU and GPU results
    std::cout << "\nComparing CPU and GPU results:" << std::endl;
    
    // Calculate maximum error and average error
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < seq_len * num_heads_q * head_dim; i++) {
        float cpu_val = __half2float(h_O_cpu[i]);
        float gpu_val = __half2float(h_O[i]);
        float error = fabs(cpu_val - gpu_val);
        
        avg_error += error;
        max_error = std::max(max_error, error);
        
        // Count elements with significant error
        if (error > 1e-3) {
            error_count++;
        }
    }
    
    avg_error /= (seq_len * num_heads_q * head_dim);
    
    std::cout << "Maximum error: " << max_error << std::endl;
    std::cout << "Average error: " << avg_error << std::endl;
    std::cout << "Elements with significant error (>1e-3): " << error_count 
              << " out of " << (seq_len * num_heads_q * head_dim) << std::endl;
    
        //计算所有元素的平均误差
    float avg_error_all = 0.0f;
    for (int i = 0; i < seq_len * num_heads_q * head_dim; i++) {
        float cpu_val = __half2float(h_O_cpu[i]);
        float gpu_val = __half2float(h_O[i]);
        float error = fabs(cpu_val - gpu_val);
        avg_error_all += error;
    }
    
    // Free host memory
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O_cpu);

    // Free device memory
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_O);

    return 0;
} 