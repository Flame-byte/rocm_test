#ifndef RMS_NORM_H
#define RMS_NORM_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// RMSnorm函数：执行RMS归一化
// 参数:
//   batch_size: 批次大小
//   seq_len: 序列长度
//   dim: 特征维度大小
//   input_tensor: 输入张量 [batch_size, seq_len, dim]
//   output_tensor: 输出张量 [batch_size, seq_len, dim]
void RMSnorm(int batch_size, int seq_len, int dim, __half* input_tensor, __half* output_tensor);

#endif // RMS_NORM_H 