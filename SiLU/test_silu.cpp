#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <iomanip>
#include "SiLU.h"

// CPU implementation of SiLU for reference
void silu_cpu(const std::vector<__half>& input, std::vector<__half>& output) {
    for (size_t i = 0; i < input.size(); ++i) {
        float x = __half2float(input[i]);
        float sigmoid_val = 1.0f / (1.0f + std::exp(-x));
        float silu_val = x * sigmoid_val;
        output[i] = __float2half(silu_val);
    }
}

int main() {
    // Example dimensions
    const int seq_len = 2;
    const int hidden_dim = 4;
    const int total_elements = seq_len * hidden_dim;
    
    // Allocate host memory
    std::vector<__half> h_input(total_elements);
    std::vector<__half> h_output_gpu(total_elements);
    std::vector<__half> h_output_cpu(total_elements);
    
    // Initialize input data (for example: [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    float init_values[8] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    for (int i = 0; i < total_elements; ++i) {
        h_input[i] = __float2half(init_values[i]);
    }
    
    // Calculate SiLU on CPU
    silu_cpu(h_input, h_output_cpu);
    
    // Allocate device memory
    __half* d_input;
    __half* d_output;
    hipMalloc(&d_input, total_elements * sizeof(__half));
    hipMalloc(&d_output, total_elements * sizeof(__half));
    
    // Copy input data from host to device
    hipMemcpy(d_input, h_input.data(), total_elements * sizeof(__half), hipMemcpyHostToDevice);
    
    // Execute SiLU activation on GPU
    silu_activation_wrapper(d_input, d_output, seq_len, hidden_dim);
    
    // Copy results back to host
    hipMemcpy(h_output_gpu.data(), d_output, total_elements * sizeof(__half), hipMemcpyDeviceToHost);
    
    // Print results and compare CPU vs GPU
    std::cout << "=== SiLU Activation Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Input | CPU Output | GPU Output | Difference | Relative Error (%)" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    
    double max_abs_diff = 0.0;
    double max_rel_error = 0.0;
    
    for (int i = 0; i < total_elements; ++i) {
        float input_val = __half2float(h_input[i]);
        float cpu_val = __half2float(h_output_cpu[i]);
        float gpu_val = __half2float(h_output_gpu[i]);
        
        // Calculate difference and relative error
        float abs_diff = std::abs(cpu_val - gpu_val);
        float rel_error = (cpu_val != 0.0f) ? (abs_diff / std::abs(cpu_val)) * 100.0f : 0.0f;
        
        // Track maximum differences
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(abs_diff));
        max_rel_error = std::max(max_rel_error, static_cast<double>(rel_error));
        
        std::cout << std::setw(6) << input_val << " | "
                  << std::setw(10) << cpu_val << " | "
                  << std::setw(10) << gpu_val << " | "
                  << std::setw(10) << abs_diff << " | "
                  << std::setw(10) << rel_error << "%" << std::endl;
    }
    
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "Maximum absolute difference: " << max_abs_diff << std::endl;
    std::cout << "Maximum relative error: " << max_rel_error << "%" << std::endl;
    
    // Determine if results are acceptable (for example, max error < 0.1%)
    bool results_match = (max_rel_error < 0.1);
    std::cout << "Test result: " << (results_match ? "PASSED" : "FAILED") << std::endl;
    
    // Free device memory
    hipFree(d_input);
    hipFree(d_output);
    
    return results_match ? 0 : 1; // Return success or failure code
} 