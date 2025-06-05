#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>

// Configuration parameters
constexpr int SEQ_LEN = 2048;        // Sequence length
constexpr int HEAD_DIM = 128;        // Head dimension
constexpr int NUM_Q_HEADS = 32;      // Number of Q heads
constexpr int NUM_KV_HEADS = 8;      // Number of K/V heads
constexpr int Q_FEATURE_DIM = 4096;  // Q feature dimension (NUM_Q_HEADS * HEAD_DIM)
constexpr int KV_FEATURE_DIM = 1024; // K/V feature dimension (NUM_KV_HEADS * HEAD_DIM)

// Block sizes for tiling
constexpr int BLOCK_SIZE_R = 64;     // Block size for rows (queries)
constexpr int BLOCK_SIZE_C = 64;     // Block size for columns (keys)

// Amount of shared memory available (64KB)
constexpr int SHARED_MEM_SIZE = 64 * 1024;

// Calculate the number of blocks
constexpr int NUM_BLOCKS_R = (SEQ_LEN + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R;
constexpr int NUM_BLOCKS_C = (SEQ_LEN + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C;

// Helper function to check HIP errors
#define CHECK_HIP_ERROR(error) checkHipError(error, __FILE__, __LINE__)
inline void checkHipError(hipError_t error, const char* file, int line) {
    if (error != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(error)
                  << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

// CUDA-like helpers for half precision operations
__device__ __half max(__half a, __half b) {
    return __hgt(a, b) ? a : b;
}

__device__ __half exp(__half x) {
    return __float2half(expf(__half2float(x)));
}

// Kernel for Flash Attention 2 forward pass
__global__ void flashAttention2Forward(
    const __half* __restrict__ q,    // [SEQ_LEN, NUM_Q_HEADS, HEAD_DIM]
    const __half* __restrict__ k,    // [SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]
    const __half* __restrict__ v,    // [SEQ_LEN, NUM_KV_HEADS, HEAD_DIM]
    __half* __restrict__ o,          // [SEQ_LEN, NUM_Q_HEADS, HEAD_DIM]
) {
    // Each block is responsible for one head
    const int q_head_idx = blockIdx.x % NUM_Q_HEADS;
    const int kv_head_idx = q_head_idx / (NUM_Q_HEADS / NUM_KV_HEADS); // Map Q head to KV head
    
    // Each thread block processes a BLOCK_SIZE_R chunk of queries for a specific head
    const int block_row_idx = blockIdx.y;
    const int q_start_idx = block_row_idx * BLOCK_SIZE_R;
    
    // Thread indices
    const int tid = threadIdx.x;
    const int row_in_block = tid / BLOCK_SIZE_C;
    const int col_in_block = tid % BLOCK_SIZE_C;
    
    // Allocate shared memory
    extern __shared__ __half shared_mem[];
    
    // Partition shared memory
    __half* q_block = shared_mem;
    __half* k_block = q_block + BLOCK_SIZE_R * HEAD_DIM;
    __half* v_block = k_block + BLOCK_SIZE_C * HEAD_DIM;
    __half* s_block = v_block + BLOCK_SIZE_C * HEAD_DIM;
    __half* p_block = s_block + BLOCK_SIZE_R * BLOCK_SIZE_C;
    __half* o_block = p_block + BLOCK_SIZE_R * BLOCK_SIZE_C;
    __half* m_block = o_block + BLOCK_SIZE_R * HEAD_DIM;
    __half* l_block = m_block + BLOCK_SIZE_R;
    
    // Initialize O, l, m values for this block
    for (int i = tid; i < BLOCK_SIZE_R * HEAD_DIM; i += blockDim.x) {
        o_block[i] = __float2half(0.0f);
    }
    
    if (tid < BLOCK_SIZE_R) {
        l_block[tid] = __float2half(0.0f);
        m_block[tid] = __float2half(-INFINITY);
    }
    
    __syncthreads();
    
    // Load Q block for this head to shared memory
    if (q_start_idx + row_in_block < SEQ_LEN && tid < BLOCK_SIZE_R * HEAD_DIM) {
        int row = q_start_idx + row_in_block;
        int feature_idx = tid % HEAD_DIM;
        int q_offset = row * Q_FEATURE_DIM + q_head_idx * HEAD_DIM + feature_idx;
        q_block[row_in_block * HEAD_DIM + feature_idx] = q[q_offset];
    }
    
    __syncthreads();
    
    // Process each block of K, V
    for (int block_col_idx = 0; block_col_idx < NUM_BLOCKS_C; block_col_idx++) {
        const int kv_start_idx = block_col_idx * BLOCK_SIZE_C;
        
        // Load K, V blocks to shared memory
        if (kv_start_idx + row_in_block < SEQ_LEN && tid < BLOCK_SIZE_C * HEAD_DIM) {
            int row = kv_start_idx + row_in_block;
            int feature_idx = tid % HEAD_DIM;
            
            int k_offset = row * KV_FEATURE_DIM + kv_head_idx * HEAD_DIM + feature_idx;
            int v_offset = row * KV_FEATURE_DIM + kv_head_idx * HEAD_DIM + feature_idx;
            
            k_block[row_in_block * HEAD_DIM + feature_idx] = k[k_offset];
            v_block[row_in_block * HEAD_DIM + feature_idx] = v[v_offset];
        }
        
        __syncthreads();
        
        // Compute S = Q * K^T
        if (row_in_block < BLOCK_SIZE_R && col_in_block < BLOCK_SIZE_C) {
            if (q_start_idx + row_in_block < SEQ_LEN && kv_start_idx + col_in_block < SEQ_LEN) {
                __half dot_product = __float2half(0.0f);
                
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot_product = __hadd(dot_product, 
                                        __hmul(q_block[row_in_block * HEAD_DIM + d], 
                                              k_block[col_in_block * HEAD_DIM + d]));
                }
                
                s_block[row_in_block * BLOCK_SIZE_C + col_in_block] = dot_product;
            } else {
                s_block[row_in_block * BLOCK_SIZE_C + col_in_block] = __float2half(-INFINITY);
            }
        }
        
        __syncthreads();
        
        // Compute row max for stable softmax
        if (tid < BLOCK_SIZE_R) {
            __half row_max = __float2half(-INFINITY);
            
            for (int j = 0; j < BLOCK_SIZE_C; j++) {
                row_max = max(row_max, s_block[tid * BLOCK_SIZE_C + j]);
            }
            
            // Store previous m value
            __half prev_m = m_block[tid];
            
            // Update m value
            m_block[tid] = max(prev_m, row_max);
            
            // Compute scaling factors
            __half m_diff = __hsub(prev_m, m_block[tid]);
            __half scale = exp(m_diff);
            
            // Scale previous cumulative sum
            l_block[tid] = __hmul(l_block[tid], scale);
            
            // Compute softmax denominators and P matrix
            for (int j = 0; j < BLOCK_SIZE_C; j++) {
                __half s_scaled = __hsub(s_block[tid * BLOCK_SIZE_C + j], m_block[tid]);
                p_block[tid * BLOCK_SIZE_C + j] = exp(s_scaled);
                l_block[tid] = __hadd(l_block[tid], p_block[tid * BLOCK_SIZE_C + j]);
            }
        }
        
        __syncthreads();
        
        // Compute O = P * V
        if (row_in_block < BLOCK_SIZE_R && kv_start_idx + col_in_block < SEQ_LEN) {
            for (int d = 0; d < HEAD_DIM; d++) {
                __half weighted_value = __hmul(p_block[row_in_block * BLOCK_SIZE_C + col_in_block], 
                                              v_block[col_in_block * HEAD_DIM + d]);
                
                atomicAdd(&o_block[row_in_block * HEAD_DIM + d], weighted_value);
            }
        }
        
        __syncthreads();
    }
    
    // Finalize O by dividing by the cumulative sum
    if (tid < BLOCK_SIZE_R * HEAD_DIM && q_start_idx + row_in_block < SEQ_LEN) {
        int row = row_in_block;
        int feature_idx = tid % HEAD_DIM;
        
        o_block[row * HEAD_DIM + feature_idx] = __hdiv(o_block[row * HEAD_DIM + feature_idx], 
                                                       l_block[row]);
    }
    
    // Compute final logsumexp values
    if (tid < BLOCK_SIZE_R && q_start_idx + tid < SEQ_LEN) {
        logsumexp[(q_start_idx + tid) * NUM_Q_HEADS + q_head_idx] = 
            __hadd(m_block[tid], log(l_block[tid]));
    }
    
    // Write final O values back to global memory
    if (tid < BLOCK_SIZE_R * HEAD_DIM && q_start_idx + row_in_block < SEQ_LEN) {
        int row = q_start_idx + row_in_block;
        int feature_idx = tid % HEAD_DIM;
        
        o[row * Q_FEATURE_DIM + q_head_idx * HEAD_DIM + feature_idx] = 
            o_block[row_in_block * HEAD_DIM + feature_idx];
    }
}

int main() {
    // Allocate host memory
    std::vector<__half> h_q(SEQ_LEN * Q_FEATURE_DIM);
    std::vector<__half> h_k(SEQ_LEN * KV_FEATURE_DIM);
    std::vector<__half> h_v(SEQ_LEN * KV_FEATURE_DIM);
    std::vector<__half> h_o(SEQ_LEN * Q_FEATURE_DIM);
    std::vector<__half> h_logsumexp(SEQ_LEN * NUM_Q_HEADS);
    
    // Initialize input data (in a real application, these would be your input tensors)
    for (int i = 0; i < h_q.size(); i++) {
        h_q[i] = __float2half((float)rand() / RAND_MAX);
    }
    
    for (int i = 0; i < h_k.size(); i++) {
        h_k[i] = __float2half((float)rand() / RAND_MAX);
    }
    
    for (int i = 0; i < h_v.size(); i++) {
        h_v[i] = __float2half((float)rand() / RAND_MAX);
    }
    
    // Allocate device memory
    __half *d_q, *d_k, *d_v, *d_o, *d_logsumexp;
    CHECK_HIP_ERROR(hipMalloc(&d_q, SEQ_LEN * Q_FEATURE_DIM * sizeof(__half)));
    CHECK_HIP_ERROR(hipMalloc(&d_k, SEQ_LEN * KV_FEATURE_DIM * sizeof(__half)));
    CHECK_HIP_ERROR(hipMalloc(&d_v, SEQ_LEN * KV_FEATURE_DIM * sizeof(__half)));
    CHECK_HIP_ERROR(hipMalloc(&d_o, SEQ_LEN * Q_FEATURE_DIM * sizeof(__half)));
    CHECK_HIP_ERROR(hipMalloc(&d_logsumexp, SEQ_LEN * NUM_Q_HEADS * sizeof(__half)));
    
    // Copy input data to device
    CHECK_HIP_ERROR(hipMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__half), hipMemcpyHostToDevice));
    
    // Calculate shared memory size
    size_t sharedMemSize = 
        (BLOCK_SIZE_R * HEAD_DIM +                 // q_block
         BLOCK_SIZE_C * HEAD_DIM +                 // k_block
         BLOCK_SIZE_C * HEAD_DIM +                 // v_block
         BLOCK_SIZE_R * BLOCK_SIZE_C +             // s_block
         BLOCK_SIZE_R * BLOCK_SIZE_C +             // p_block
         BLOCK_SIZE_R * HEAD_DIM +                 // o_block
         BLOCK_SIZE_R +                            // m_block
         BLOCK_SIZE_R) * sizeof(__half);           // l_block
    
    // Make sure shared memory size doesn't exceed limit
    if (sharedMemSize > SHARED_MEM_SIZE) {
        std::cerr << "Shared memory requirement exceeds available shared memory!" << std::endl;
        std::cerr << "Required: " << sharedMemSize << " bytes, Available: " << SHARED_MEM_SIZE << " bytes" << std::endl;
        return 1;
    }
    
    // Set up grid dimensions
    dim3 gridDim(NUM_Q_HEADS, NUM_BLOCKS_R);
    dim3 blockDim(256); // Using 256 threads per block
    
    // Launch kernel
    std::cout << "Launching Flash Attention 2 kernel with shared memory size: " 
              << sharedMemSize << " bytes" << std::endl;
    flashAttention2Forward<<<gridDim, blockDim, sharedMemSize>>>(d_q, d_k, d_v, d_o, d_logsumexp);
    
    // Check for kernel launch errors
    CHECK_HIP_ERROR(hipGetLastError());
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    
    // Copy results back to host
    CHECK_HIP_ERROR(hipMemcpy(h_o.data(), d_o, h_o.size() * sizeof(__half), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(h_logsumexp.data(), d_logsumexp, h_logsumexp.size() * sizeof(__half), hipMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_HIP_ERROR(hipFree(d_q));
    CHECK_HIP_ERROR(hipFree(d_k));
    CHECK_HIP_ERROR(hipFree(d_v));
    CHECK_HIP_ERROR(hipFree(d_o));
    CHECK_HIP_ERROR(hipFree(d_logsumexp));
    
    std::cout << "Flash Attention 2 completed successfully!" << std::endl;
    
    return 0;
}
