#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "rope.h"

#define CHECK_HIP(cmd) {                                                        \
    hipError_t error = cmd;                                                     \
    if (error != hipSuccess) {                                                  \
        fprintf(stderr, "HIP error: %s:%d '%s'\n",                              \
                __FILE__, __LINE__, hipGetErrorString(error));                  \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

int main() {
    const int seq_len = 4;
    const int num_heads = 1;
    const int head_dim = 4;
    const int total_elements = seq_len * num_heads * head_dim;

    // 初始化主机内存
    __half *h_data = (__half*)malloc(total_elements * sizeof(__half));
    __half *h_data_original = (__half*)malloc(total_elements * sizeof(__half)); // 保存原始数据副本
    __half *h_cis = (__half*)malloc(seq_len * head_dim * sizeof(__half));

    // 初始化数据矩阵：每个元素为递增的1-16
    for (int i = 0; i < total_elements; ++i) {
        h_data[i] = __float2half(i + 1.0f); // 1.0, 2.0,...16.0
    }
    memcpy(h_data_original, h_data, total_elements * sizeof(__half)); // 备份原始数据

    // 初始化频率矩阵：每个复数对为(0.0, 1.0)
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < head_dim; ++d) {
            int idx = s * head_dim + d;
            h_cis[idx] = __float2half((d % 2 == 0) ? 0.0f : 1.0f);
        }
    }

    // 分配设备内存
    __half *d_data, *d_cis;
    CHECK_HIP(hipMalloc(&d_data, total_elements * sizeof(__half)));
    CHECK_HIP(hipMalloc(&d_cis, seq_len * head_dim * sizeof(__half)));

    // 拷贝数据到设备
    CHECK_HIP(hipMemcpy(d_data, h_data, total_elements * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_cis, h_cis, seq_len * head_dim * sizeof(__half), hipMemcpyHostToDevice));

    // 执行RoPE变换
    rope(d_data, d_cis, seq_len, head_dim, num_heads);

    // 拷贝结果回主机
    CHECK_HIP(hipMemcpy(h_data, d_data, total_elements * sizeof(__half), hipMemcpyDeviceToHost));

    // 验证结果
    int errors = 0;
    for (int s = 0; s < seq_len; ++s) {
        for (int pair = 0; pair < head_dim / 2; ++pair) {
            // 原始数据中的复数对位置
            int original_base = s * num_heads * head_dim + 2 * pair;
            float original_real = __half2float(h_data_original[original_base]);
            float original_imag = __half2float(h_data_original[original_base + 1]);

            // 预期结果：实部 = -原虚部，虚部 = 原实部
            float expected_real = -original_imag;
            float expected_imag = original_real;

            // 实际结果位置
            int result_base = s * num_heads * head_dim + 2 * pair;
            __half real = h_data[result_base];
            __half imag = h_data[result_base + 1];

            // 转换当前值为float
            float real_f = __half2float(real);
            float imag_f = __half2float(imag);

            // 允许1e-3的误差
            if (fabs(real_f - expected_real) > 1e-3 || 
                fabs(imag_f - expected_imag) > 1e-3) {
                printf("Error at seq[%d] pair[%d]: ", s, pair);
                printf("Got (%.2f, %.2f), Expected (%.2f, %.2f)\n",
                       real_f, imag_f, expected_real, expected_imag);
                errors++;
            }
        }
    }

    // 输出测试结果
    if (errors == 0) {
        printf("All tests passed successfully!\n");
    } else {
        printf("Found %d errors!\n", errors);
    }

    // 释放资源
    free(h_data);
    free(h_data_original);
    free(h_cis);
    CHECK_HIP(hipFree(d_data));
    CHECK_HIP(hipFree(d_cis));

    return errors ? 1 : 0;
}