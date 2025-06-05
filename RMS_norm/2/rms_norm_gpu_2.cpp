#include <hip/hip_runtime.h>
#include <iostream>

#define block_size 256

__global__ void RMSNorm(const float* input, float* output, int dim) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    float sum = 0;
    for(int i = global_id; i < dim; i += blockDim.x * gridDim.x) {
        sum += input[i] * input[i];
    }
    
    __shared__ float shared_mem[block_size];
    shared_mem[local_id] = sum;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if(local_id < stride) {
            shared_mem[local_id] += shared_mem[local_id + stride];
        }
        __syncthreads();
    }
    
    if(local_id == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

int main() {
    int batch_size = 2, seq_len = 32, dim = 4096;
    int total_size = batch_size * seq_len * dim;

    // Create input and output arrays
    float *input, *output;

    // Allocate memory for input and output
    hipMallocManaged(&input, sizeof(float) * total_size);
    hipMallocManaged(&output, sizeof(float) * batch_size * seq_len);

    // Initialize input data
    for (int i = 0; i < total_size; i++) {
        input[i] = i * 1.0f;
    }

    int grid_size = (total_size + block_size - 1) / block_size;

    // Launch kernel for each sequence
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // Copy the current sequence to device memory
            float* current_input = &input[(b * seq_len + s) * dim];
            hipLaunchKernelGGL(RMSNorm, grid_size, block_size, 0, 0, current_input, output, dim);

            // Wait for GPU to finish before accessing on host
            hipDeviceSynchronize();

            // Print the result for the current sequence
            std::cout << "Batch " << b << ", Sequence " << s << ": " << output[0] << std::endl;
        }
    }

    // Free memory
    hipFree(input);
    hipFree(output);

    return 0;
}