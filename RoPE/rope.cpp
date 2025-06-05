#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "rope.h"

__global__ void rope_kernel(
    __half* data,
    const __half* freqs_cis,
    int seq_len,
    int head_dim,
    int num_heads,
    int total_elements
) {
    // 计算全局索引（现在每个线程处理一个复数对）
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算复数对总数（每个复数包含2个元素）
    const int total_pairs = total_elements / 2;
    
    // 使用grid-stride循环处理所有复数对
    for (; idx < total_pairs; idx += gridDim.x * blockDim.x) {
        // 将一维索引分解为多维坐标
        int dim_pair_idx = idx % (head_dim/2);     // 复数对在头内的索引 [0, 63]
        int head_idx = (idx / (head_dim/2)) % num_heads; // 头索引 [0, num_heads-1]
        int seq_pos = idx / ((head_dim/2) * num_heads);  // 序列位置 [0, 2047]
        
        // 转换为原始维度索引（实部位置）
        int dim_idx = 2 * dim_pair_idx;  // 0, 2, 4...126
        
        // 计算数据索引（三维到一维的映射）
        int data_idx = seq_pos * num_heads * head_dim  // 序列偏移
                     + head_idx * head_dim             // 头偏移
                     + dim_idx;                        // 维度偏移
                     
        // 获取对应的位置编码复数
        int cis_idx = seq_pos * head_dim + dim_idx;
        __half cis_real = freqs_cis[cis_idx];
        __half cis_imag = freqs_cis[cis_idx + 1];
        
        // 获取当前元素的复数对
        __half data_real = data[data_idx];
        __half data_imag = data[data_idx + 1];
        
        // 执行复数乘法
        __half new_real = data_real * cis_real - data_imag * cis_imag;
        __half new_imag = data_real * cis_imag + data_imag * cis_real;
        
        // 写回结果
        data[data_idx] = new_real;
        data[data_idx + 1] = new_imag;
    }
}

// 修改后的调用函数
void rope(
    __half* d_data,
    const __half* d_cis,
    int seq_len,
    int head_dim,
    int num_heads
) {
    int total_elements = seq_len * num_heads * head_dim;
    int total_pairs = total_elements / 2;  // 需要处理的复数对数
    
    const int block_size = 256;
    int grid_size = (total_pairs + block_size - 1) / block_size;
    
    rope_kernel<<<grid_size, block_size>>>(d_data, d_cis, seq_len, head_dim, num_heads, total_elements);
    
}