#include <iostream>
#include <cmath>

void rms_norm_cpu(const float* input, float* output, int batch_size, int seq_len, int dim) {
    // For each batch and sequence
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float sum_squares = 0.0f;
            
            // Calculate sum of squares for current sequence
            for (int d = 0; d < dim; d++) {
                float val = input[b * seq_len * dim + s * dim + d];
                sum_squares += val * val;
            }
            
            // Store the result
            output[b * seq_len + s] = sum_squares;
        }
    }
}

// Test function
void test_rms_norm_cpu() {
    // Test data dimensions
    const int batch_size = 2;
    const int seq_len = 3;
    const int dim = 5;
    
    // Create test input data
    float input[batch_size * seq_len * dim] = {
        // Layer 0
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        // Layer 1
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
        26, 27, 28, 29, 30
    };
    
    // Output array
    float output[batch_size * seq_len];
    
    // Call the function
    rms_norm_cpu(input, output, batch_size, seq_len, dim);
    
    // Print results
    std::cout << "Results:\n";
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            std::cout << "Batch " << b << ", Sequence " << s << ": " 
                      << output[b * seq_len + s] << std::endl;
        }
    }
}

int main() {
    test_rms_norm_cpu();
    return 0;
} 