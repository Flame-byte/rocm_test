#include "flash.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>

// 初始化半精度数组
void init_half_array(__half* arr, int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = __float2half( 1.0f);
    }
}

// 比较半精度数组（带容差）
bool compare_half(const __half* result, const __half* expected, int size, float eps = 1e-2f) {
    for (int i = 0; i < size; ++i) {
        float r = __half2float(result[i]);
        float e = __half2float(expected[i]);
        if (fabs(r - e) > eps) {
            printf("Mismatch at %d: GPU=%f, CPU=%f\n", i, r, e);
            return false;
        }
    }
    return true;
}


// CPU参考实现
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
    const int seq_len = 4;
    const int head_dim = 4;
    const int num_heads_q = 4;
    const int num_heads_kv = 1;
    
    // 分配内存
    const int q_size = seq_len * num_heads_q * head_dim;
    const int kv_size = seq_len * num_heads_kv * head_dim;
    const int o_size = q_size;
    
    __half *h_Q = new __half[q_size];
    __half *h_K = new __half[kv_size];
    __half *h_V = new __half[kv_size];
    __half *h_O_gpu = new __half[o_size];
    __half *h_O_cpu = new __half[o_size];
    
    // 初始化输入
    init_half_array(h_Q, q_size);
    init_half_array(h_K, kv_size);
    init_half_array(h_V, kv_size);
    init_half_array(h_O_cpu, o_size); // CPU结果初始化为0

    // for (int i = 0; i < q_size; i++) {
    //     printf("Q:%f ", __half2float(h_Q[i]));
    // }
    // printf("\n");

    // for (int i = 0; i < kv_size; i++) {
    //     printf("K:%f ", __half2float(h_K[i]));
    // }
    // printf("\n");

    // for (int i = 0; i < kv_size; i++) {
    //     printf("V:%f ", __half2float(h_V[i]));
    // }
    
    // 分配设备内存
    __half *d_Q, *d_K, *d_V, *d_O;
    hipMalloc(&d_Q, q_size * sizeof(__half));
    hipMalloc(&d_K, kv_size * sizeof(__half));
    hipMalloc(&d_V, kv_size * sizeof(__half));
    hipMalloc(&d_O, o_size * sizeof(__half));
    
    // 数据拷贝到设备
    hipMemcpy(d_Q, h_Q, q_size * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_K, h_K, kv_size * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V, kv_size * sizeof(__half), hipMemcpyHostToDevice);
    
    // 执行GPU kernel
    flash_attention(d_Q, d_K, d_V, d_O, seq_len, head_dim, num_heads_q, num_heads_kv);
    
    // 回拷结果
    hipMemcpy(h_O_gpu, d_O, o_size * sizeof(__half), hipMemcpyDeviceToHost);
    
    
    // 执行CPU参考计算
    cpu_attention(h_Q, h_K, h_V, h_O_cpu, seq_len, head_dim, num_heads_q, num_heads_kv);
    
    // 验证结果
    bool pass = compare_half(h_O_gpu, h_O_cpu, o_size);
    printf("Test %s!\n", pass ? "passed" : "failed");
    
    
    printf("GPU computation completed!\n");

    // 打印GPU计算结果
    printf("\nGPU Computation Results:\n");
    for (int head = 0; head < num_heads_q; head++) {
        printf("\nHead %d:\n", head);
        for (int seq = 0; seq < seq_len; seq++) {
            printf("Seq %d: ", seq);
            for (int dim = 0; dim < head_dim; dim++) {
                int idx = seq * num_heads_q * head_dim + head * head_dim + dim;
                printf("%f ", __half2float(h_O_gpu[idx]));
            }
            printf("\n");
        }
    }

    for (int i = 0; i < o_size; i++) {
        printf("O_cpu:%f ", __half2float(h_O_cpu[i]));
    }
    // 资源释放
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_gpu;
    // delete[] h_O_cpu;
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_O);
    
    // return !pass;
    return 0;
}