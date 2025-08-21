#include <iostream>
#include <hip/hip_runtime.h>
//#include <cmath>
#include <hip/hip_fp16.h>

__global__ void precompute_freqs_cis_kernel(__half* freqs_cis, int dim, int end, float theta) {
    const int half_dim = dim / 2;
    const int total_elements = end * half_dim;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int pos = idx; pos < total_elements; pos += blockDim.x * gridDim.x) {
        const int i = pos / half_dim;   // token position index
        const int k = pos % half_dim;   // head dimension half index
        
        // Calculate frequency scale
        const float scale = 1.0f / powf(theta, static_cast<float>(2 * k) / static_cast<float>(dim));
        const float freq = static_cast<float>(i) * scale;
        
        // Calculate complex rotation components
        const int base_index = i * dim + 2 * k;
        freqs_cis[base_index] = __float2half(cosf(freq));     // real part
        freqs_cis[base_index + 1] = __float2half(sinf(freq)); // imaginary part
    }
}

__half* precompute_freqs_cis_gpu(int dim, int end, float theta = 10000.0f) {
    // Allocate device memory
    const size_t buffer_size = end * dim * sizeof(__half);
    __half* d_freqs_cis;
    hipMalloc(&d_freqs_cis, buffer_size);
    
    // Calculate kernel launch parameters
    const int half_dim = dim / 2;
    const int total_elements = end * half_dim;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    hipLaunchKernelGGL(precompute_freqs_cis_kernel,
                      dim3(blocks),
                      dim3(threads_per_block),
                      0, 0,
                      d_freqs_cis, dim, end, theta);
    
    return d_freqs_cis;
}

// 示例调用
int main() {
    int dim = 128;
    int max_seq_len = 2048;
    float rope_theta = 10000.0f;

    // GPU计算
    __half* d_freqs_cis = precompute_freqs_cis_gpu(dim, max_seq_len, rope_theta);
    
    // 拷贝结果回主机
    const size_t buffer_size = max_seq_len * dim * sizeof(__half);
    __half* h_freqs_cis = new __half[max_seq_len * dim];
    hipMemcpy(h_freqs_cis, d_freqs_cis, buffer_size, hipMemcpyDeviceToHost);

    // 打印形状
    std::cout << "Shape: [" << max_seq_len << ", " << dim << "]" << std::endl;

    // 打印前4行前6列数据
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            std::cout << __half2float(h_freqs_cis[i * dim + j]) << " ";
        }
        std::cout << std::endl;
    }

    // 清理资源
    delete[] h_freqs_cis;
    hipFree(d_freqs_cis);

    return 0;
}

