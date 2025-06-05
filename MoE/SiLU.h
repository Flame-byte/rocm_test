#ifndef SiLU_H
#define SiLU_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// SiLU activation function kernel declaration
__global__ void silu_activation_kernel(
    const __half* input,
    __half* output,
    const int seq_len,
    const int hidden_dim);

// Helper function to launch the SiLU kernel
extern "C" void silu_activation(
    const __half* input,
    __half* output,
    const int seq_len,
    const int hidden_dim,
    hipStream_t stream);

#endif  // SiLU_H
