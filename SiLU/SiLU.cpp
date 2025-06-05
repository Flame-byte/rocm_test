#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include "SiLU.h"

// SiLU activation function: SiLU(x) = x * sigmoid(x)
// where sigmoid(x) = 1/(1 + exp(-x))
__global__ void silu_activation_kernel(
    const __half* input,
    __half* output,
    const int seq_len,
    const int hidden_dim)
{
    // Calculate global thread ID using HIP indexing
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Check if thread is within the valid range
    if (tid < seq_len * hidden_dim) {
        // Load input value
        __half x = input[tid];

        //__half t = __ocml_exp_f16(x);
        
        // Convert to float for computation (for numerical stability)
        float x_float = __half2float(x);
        
        // Compute sigmoid: 1/(1 + exp(-x))
        float sigmoid_val = 1.0f / (1.0f + expf(-x_float));
        
        // Compute SiLU: x * sigmoid(x)
        float silu_val = x_float * sigmoid_val;
        
        // Convert back to half and store in output
        output[tid] = __float2half(silu_val);
    }
}

// Helper function to launch the SiLU kernel
void silu_activation(
    const __half* input,
    __half* output,
    const int seq_len,
    const int hidden_dim,
    hipStream_t stream
)
{
    // Calculate total number of elements
    int num_elements = seq_len * hidden_dim;
    
    // Define block size and grid size
    const unsigned int block_size = 256;
    const unsigned int grid_size = (num_elements + block_size - 1) / block_size;
    
    // Launch the kernel using HIP syntax
    hipLaunchKernelGGL(
        silu_activation_kernel,
        grid_size,          // grid size
        block_size,         // block size
        0,                  // shared memory size
        stream,             // stream
        input, output, seq_len, hidden_dim);
}

// Example usage function with C linkage
extern "C" void silu_activation_wrapper(
    const __half* input,
    __half* output,
    const int seq_len,
    const int hidden_dim)
{
    silu_activation(input, output, seq_len, hidden_dim, nullptr);
    // Wait for kernel completion
    hipDeviceSynchronize();
}

