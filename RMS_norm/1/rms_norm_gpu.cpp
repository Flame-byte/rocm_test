#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>  // For sqrt and fabs functions

__global__ void RMSNorm(float* input_tensor, float* output_tensor, float* rms_values, int seq_len, int dim) {
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
    __shared__ float temp[4096]; // 用于平方和计算的共享内存
    
    // 使用grid-stride loop确保所有元素都被处理
    temp[dim_idx] = 0.0f;
    
    // Step 1: Calculate squares and store in shared memory
    // 每个线程处理多个元素（跨步循环）
    for (int i = dim_idx; i < dim; i += blockDim.x) {
        float val = input_tensor[base_idx + i];
        temp[dim_idx] += val * val; // 累加元素的平方到线程对应的共享内存位置
    }
    
    __syncthreads(); // 确保所有线程都完成了平方计算
    
    // Step 2: Parallel reduction to compute sum of squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (dim_idx < stride) {
            temp[dim_idx] += temp[dim_idx + stride];
        }
        __syncthreads(); // 确保reduction的每一步都同步
    }
    
    // Step 3: Thread 0 computes the final RMS value
    if (dim_idx == 0) {
        // 计算RMS值: sqrt(sum of squares / dim)
        float rms = sqrt(temp[0] / dim + 1e-5); // 添加小常数避免除零
        rms_values[token_idx] = rms; // 保存RMS值供CPU验证
        
        // 将RMS值写回到共享内存的第一个位置，供所有线程访问
        temp[0] = rms;
    }
    
    __syncthreads(); // 确保RMS值计算完成，所有线程可见
    
    // Step 4: 每个线程应用归一化（同样使用跨步循环处理多个元素）
    float rms_value = temp[0]; // 从共享内存读取RMS值
    for (int i = dim_idx; i < dim; i += blockDim.x) {
        output_tensor[base_idx + i] = input_tensor[base_idx + i] / rms_value;
    }
}

int main() {
    // Define matrix dimensions
    const int batch_size = 5;
    const int seq_len = 2048;
    const int dim = 4096;

    // 使用一个3D的tensor (batch_size x seq_len x dim)
    // 在内存中表示为一维数组
    float *input_tensor;  // 输入张量
    float *output_tensor; // 归一化后的输出
    float *rms_values;    // RMS值 (用于验证)

    // 分配内存
    hipMallocManaged((void**)&input_tensor, batch_size * seq_len * dim * sizeof(float));
    hipMallocManaged((void**)&output_tensor, batch_size * seq_len * dim * sizeof(float));
    hipMallocManaged((void**)&rms_values, batch_size * seq_len * sizeof(float));

    // 初始化tensor
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < dim; d++) {
                int idx = b * seq_len * dim + s * dim + d;
                input_tensor[idx] = 1; 
            }
        }
    }
    
    // 定义grid和block维度
    dim3 gridDim(batch_size, seq_len);  // 2D grid, 每个block处理一个token
    dim3 blockDim(1024);                 // Block size 设为2的幂，方便reduction
    
    // 执行合并的RMS归一化kernel
    RMSNorm<<<gridDim, blockDim>>>(input_tensor, output_tensor, rms_values, seq_len, dim);
    
    // 同步等待kernel完成
    hipDeviceSynchronize();
    
    // 打印RMS值
    printf("RMS Values:\n");
    for (int b = 0; b < batch_size; b++) {
        printf("Batch %d:\n", b);
        for (int s = 0; s < seq_len; s++) {
            printf("  Seq %d: %f\n", s, rms_values[b * seq_len + s]);
        }
    }
    
    // 打印归一化后的结果样本
    // printf("\nNormalized Tensor Samples:\n");
    // for (int b = 0; b < batch_size; b++) {
    //     printf("Batch %d:\n", b);
    //     for (int s = 0; s < seq_len; s++) {
    //         printf("  Seq %d: [", s);
    //         //打印所有元素
    //         for (int d = 0; d < dim; d++) {
    //             int idx = b * seq_len * dim + s * dim + d;
    //             printf("%f ", output_tensor[idx]);
    //         }
    //         printf("...]\n");
    //     }
    // }
    
    // 验证结果
    printf("\nValidating results...\n");
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // 计算预期的RMS值
            float expected_rms = 0.0f;
            for (int d = 0; d < dim; d++) {
                int idx = b * seq_len * dim + s * dim + d;
                float val = input_tensor[idx];
                expected_rms += val * val;
            }
            expected_rms = sqrt(expected_rms / dim + 1e-5);
            
            // 比较RMS值
            float actual_rms = rms_values[b * seq_len + s];
            float rms_diff = fabs(actual_rms - expected_rms);
            printf("  Batch %d, Seq %d: Expected RMS=%f, Actual RMS=%f, Diff=%f\n", 
                   b, s, expected_rms, actual_rms, rms_diff);
            
            // 验证归一化结果
            for (int d = 0; d < dim; d++) {
                int idx = b * seq_len * dim + s * dim + d;
                float expected_norm = input_tensor[idx] / expected_rms;
                float actual_norm = output_tensor[idx];
                float norm_diff = fabs(actual_norm - expected_norm);
                
                // 只检查第一个元素，如果差异过大，则报告
                if (d == 0 && norm_diff > 1e-5) {
                    printf("    WARNING: Large normalization difference at element 0: Expected=%f, Actual=%f, Diff=%f\n",
                          expected_norm, actual_norm, norm_diff);
                }
            }
        }
    }
    
    // 释放内存
    hipFree(input_tensor);
    hipFree(output_tensor);
    hipFree(rms_values);
    
    return 0;
}






