#include "RMSnorm.h"
#include <iostream>
#include <hip/hip_fp16.h>

// Kernel函数，计算RMS值并应用归一化
__global__ void RMSNormKernel(__half* input_tensor, __half* output_tensor, int seq_len, int dim) {
    // In 3D tensor with shape [batch_size, seq_len, dim]
    // blockIdx.x represents batch index
    // blockIdx.y represents sequence index
    // threadIdx.x handles elements along the dim dimension
    
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y; 
    int dim_idx = threadIdx.x;
    
    // Calculate indices
    int token_idx = batch_idx * seq_len + seq_idx;  // Index for the current token
    int base_idx = token_idx * dim;                 // Starting index of this token in the tensor
    
    // Shared memory for reduction
    __shared__ float temp[1024]; // 使用float进行中间计算以提高精度
    
    // Step 1: Calculate squares and store in shared memory
    temp[dim_idx] = 0.0f;
    
    if (dim_idx < dim) {
        float val = __half2float(input_tensor[base_idx + dim_idx]);
        temp[dim_idx] = val * val; // 存储元素的平方
    }
    
    __syncthreads(); // 确保所有线程都完成了平方计算
    
    // Step 2: Parallel reduction to compute sum of squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (dim_idx < stride && dim_idx + stride < dim) {
            temp[dim_idx] += temp[dim_idx + stride];
        }
        __syncthreads(); // 确保reduction的每一步都同步
    }
    
    // Step 3: Thread 0 computes the final RMS value
    if (dim_idx == 0) {
        // 计算RMS值: sqrt(sum of squares / dim)
        float rms = sqrt(temp[0] / dim + 1e-5f); // 添加小常数避免除零
        
        // 将RMS值写回到共享内存的第一个位置，供所有线程访问
        temp[0] = rms;
    }
    
    __syncthreads(); // 确保RMS值计算完成，所有线程可见
    
    // Step 4: 每个线程应用归一化
    if (dim_idx < dim) {
        float rms_value = temp[0]; // 从共享内存读取RMS值
        float input_val = __half2float(input_tensor[base_idx + dim_idx]);
        output_tensor[base_idx + dim_idx] = __float2half(input_val / rms_value);
    }
}

void RMSnorm(int batch_size, int seq_len, int dim, __half* input_tensor, __half* output_tensor) {
    // 定义grid和block维度
    dim3 gridDim(batch_size, seq_len);  // 2D grid, 每个block处理一个token
    dim3 blockDim(1024);                // Block size设为2的幂，方便reduction
    
    // 执行RMS归一化kernel
    RMSNormKernel<<<gridDim, blockDim>>>(input_tensor, output_tensor, seq_len, dim);
    
    // 同步等待kernel完成
    hipDeviceSynchronize();
} 