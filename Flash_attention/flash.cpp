#include "flash.h"

// __device__ __half hmax(__half a, __half b) {
//     return a > b ? a : b;
// }

// // __device__ __half hlog(__half x) {
// //     return __float2half(logf(__half2float(x)));
// // }

// __device__ __half custom_hexp(__half x) {
//     return __float2half(expf(__half2float(x)));
// }


// #define MAX_LDS_SIZE 65536  // 64 KB
// // Example blocking sizes for LDS
// #define BLOCK_ROWS 32       // Br
// #define BLOCK_COLS 32       // Bc

// __global__ void forward_kernel(const __half* __restrict__ Q,
//                                const __half* __restrict__ K,
//                                const __half* __restrict__ V,
//                                __half* __restrict__ O,
//                                const int seq_len,
//                                const int head_dim,
//                                const int num_heads_q,
//                                const int num_heads_kv
//                                ) {
//     extern __shared__ __half shared_mem[];  // dynamic LDS
//     __half* Q_block = shared_mem;           // Q_i BLOCK_ROWS * head_dim
//     __half* K_block = Q_block + BLOCK_ROWS * head_dim;  // K_j BLOCK_COLS * head_dim
//     __half* V_block = K_block + BLOCK_COLS * head_dim;  // V_j BLOCK_COLS * head_dim

//     const __half scale = __float2half(1.0f / sqrtf((float)head_dim));

//     // ---------------------------
//     // Head/block indexing
//     // ---------------------------
//     int q_head_id = blockIdx.x;          // [0, 31]
//     int kv_head_id = q_head_id / 4;      // [0, 7], one K head for 4 Q heads

//     // thread index in block
//     int tid = threadIdx.x;

//     // Q offset: Q is [seq_len, Q_HEADS * head_dim]
//     int q_offset = q_head_id * head_dim;

//     // K/V offset: K is [seq_len, KV_HEADS * head_dim]
//     int kv_offset = kv_head_id * head_dim;

//     // ---------------------------
//     // Process in tiles (Br × d), (Bc × d)
//     // ---------------------------

//     for (int qi = 0; qi < seq_len; qi += BLOCK_ROWS) {
//         // load Q_i block (Br × d)
//         if (tid < BLOCK_ROWS * head_dim) {
//             int local_row = tid / head_dim;
//             int local_col = tid % head_dim;
//             int global_row = qi + local_row;

//             if (global_row < seq_len) {
//                 Q_block[local_row * head_dim + local_col] =
//                     Q[global_row * (num_heads_q * head_dim) + q_offset + local_col];
//             }
//         }
//         __syncthreads();

//         // 初始化 m, ℓ, O
//         __half* m_i = V_block + BLOCK_COLS * head_dim;   // m_i BLOCK_ROWS
//         __half* l_i = m_i + BLOCK_ROWS;   // l_i BLOCK_ROWS  
//         __half* O_i = l_i + BLOCK_ROWS;  // O_i BLOCK_ROWS * head_dim

//         if (tid < BLOCK_ROWS) {
//             m_i[tid] = __half(-65504.0f); // -∞
//             l_i[tid] = __half(0.0f);
//             O_i[tid] = __half(0.0f);
//         }

//         __syncthreads();

//         for (int kj = 0; kj < seq_len; kj += BLOCK_COLS) {

//             int local_row = tid / head_dim;
//             int local_col = tid % head_dim;

//             // load K_j block (Bc × d)
//             if (tid < BLOCK_COLS * head_dim) {
//                 // int local_row = tid / head_dim;
//                 // int local_col = tid % head_dim;
//                 int global_row = kj + local_row;

//                 if (global_row < seq_len) {
//                     K_block[local_row * head_dim + local_col] =
//                         K[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];

//                     V_block[local_row * head_dim + local_col] =
//                         V[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];
//                 }
//             }
//             __syncthreads();

//             // Compute S = Q_block × K_block^T
//             for (int i = 0; i < BLOCK_ROWS; ++i) {
//                 if (qi + i >= seq_len) continue;
//                 __half row_max = __half(-65504.0f);
//                 __half s_row[BLOCK_COLS];

//                 for (int j = 0; j < BLOCK_COLS; ++j) {
//                     if (kj + j >= seq_len) continue;
//                     __half dot = __half(0.0f);
//                     for (int k = 0; k < head_dim; ++k) {
//                         __half q_val = Q_block[i * head_dim + k];
//                         __half k_val = K_block[j * head_dim + k];
//                         dot = __hadd(dot, __hmul(q_val, k_val));
//                     }
//                     __half scaled_dot = __hmul(dot, scale);  
//                     s_row[j] = scaled_dot;
//                     row_max = __hmax(row_max, scaled_dot); 
//                 }

//                 m_i[i] = __hmax(m_i[i], row_max);  
//             }

//             __syncthreads();

//             // Now compute exp(S - m), update l_i, O_i
//             for (int i = 0; i < BLOCK_ROWS; ++i) {
//                 if (qi + i >= seq_len) continue;
//                 for (int j = 0; j < BLOCK_COLS; ++j) {
//                     if (kj + j >= seq_len) continue;
//                     __half dot = __half(0.0f);
//                     for (int k = 0; k < head_dim; ++k) {
//                         __half q_val = Q_block[i * head_dim + k];
//                         __half k_val = K_block[j * head_dim + k];
//                         dot = __hadd(dot, __hmul(q_val, k_val));
//                     }
//                     __half p = custom_hexp(dot - m_i[i]);
//                     l_i[i] = __hadd(l_i[i], p);

//                     // O_i += p * V[j]
//                     for (int k = 0; k < head_dim; ++k) {
//                         __half v_val = V_block[j * head_dim + k];
//                         __half delta = __hmul(p, v_val);
//                         O_i[i * head_dim + k] = __hadd(O_i[i * head_dim + k], delta);
//                     }
//                 }
//             }
//             __syncthreads();
//         }

//         // Normalize O_i by l_i
//         if (tid < BLOCK_ROWS * head_dim) {
//             int row = tid / head_dim;
//             int col = tid % head_dim;
//             if (qi + row < seq_len) {
//                 __half norm = __hdiv(O_i[row * head_dim + col], l_i[row]);
//                 O[(qi + row) * (num_heads_q * head_dim) + q_offset + col] = norm;

//                 // if (col == 0)
//                 //     L[(qi + row) * num_heads_q + q_head_id] = m_i[row] + hlog(l_i[row]);
//             }
//         }

//         __syncthreads();
//     }
// }

// #define MAX_LDS_SIZE 65536  // 64 KB
// #define BLOCK_ROWS 32
// #define BLOCK_COLS 32

// __device__ __half hmax(__half a, __half b) {
//     return __hgt(a, b) ? a : b;
// }
// __device__ __half custom_hexp(__half x) {
//     return __float2half(expf(__half2float(x)));
// }
// __device__ __half hlog(__half x) {
//     return __float2half(logf(__half2float(x)));
// }

// __global__ void forward_kernel(const __half* __restrict__ Q,
//                                const __half* __restrict__ K,
//                                const __half* __restrict__ V,
//                                __half* __restrict__ O,
//                                const int seq_len,
//                                const int head_dim,
//                                const int num_heads_q,
//                                const int num_heads_kv) {
//     extern __shared__ __half shared_mem[];
//     __half* Q_block = shared_mem;
//     __half* K_block = Q_block + BLOCK_ROWS * head_dim;
//     __half* V_block = K_block + BLOCK_COLS * head_dim;

//     const __half scale = __float2half(1.0f / sqrtf((float)head_dim));
//     int q_head_id = blockIdx.x;
//     int kv_head_id = q_head_id / 4;

//     int tid = threadIdx.x;
//     int q_offset = q_head_id * head_dim;
//     int kv_offset = kv_head_id * head_dim;

//     for (int qi = 0; qi < seq_len; qi += BLOCK_ROWS) {
//         int local_row = tid / head_dim;
//         int local_col = tid % head_dim;
//         if (tid < BLOCK_ROWS * head_dim) {
//             int global_row = qi + local_row;
//             if (global_row < seq_len) {
//                 Q_block[local_row * head_dim + local_col] =
//                     Q[global_row * (num_heads_q * head_dim) + q_offset + local_col];
//             }
//         }

//         __syncthreads();

//         // Allocate LDS space for m_i, l_i, o_i
//         __half* m_i = V_block + BLOCK_COLS * head_dim;
//         __half* l_i = m_i + BLOCK_ROWS;
//         __half* O_i = l_i + BLOCK_ROWS; // BLOCK_ROWS * head_dim

//         if (tid < BLOCK_ROWS) {
//             m_i[tid] = __half(-65504.0f); // -∞
//             l_i[tid] = __half(0.0f);
//         }
//         if (tid < BLOCK_ROWS * head_dim) {
//             O_i[tid] = __half(0.0f);
//         }

//         __syncthreads();

//         for (int kj = 0; kj < seq_len; kj += BLOCK_COLS) {
//             if (tid < BLOCK_COLS * head_dim) {
//                 // int local_row = tid / head_dim;
//                 // int local_col = tid % head_dim;
//                 int global_row = kj + local_row;

//                 if (global_row < seq_len) {
//                     K_block[local_row * head_dim + local_col] =
//                         K[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];
//                     // V_block[j * head_dim + local_col] =
//                     //     V[(kj + j) * (num_heads_kv * head_dim) + kv_offset + local_col];

//                     V_block[local_row * head_dim + local_col] =
//                         V[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];
//                 }
//             }

//             __syncthreads();

//             // // Iterate over rows of Q
//             // for (int i = 0; i < BLOCK_ROWS; ++i) {
//             //     if (qi + i >= seq_len) continue;

//             //     __half m_new = m_i[i];
//             //     __half m_prev;
//             //     __half l_new = l_i[i];
//             //     __half l_prev;
//             //     __half o_tmp[BLOCK_COLS];  // store p
//             //     #pragma unroll
//             //     for (int j = 0; j < BLOCK_COLS; ++j) {
//             //         if (kj + j >= seq_len) continue;
//             //         __half dot = __half(0.0f);
//             //         for (int k = 0; k < head_dim; ++k) {
//             //             __half q_val = Q_block[i * head_dim + k];
//             //             __half k_val = K_block[j * head_dim + k];
//             //             __half v_val = V_block[j * head_dim + k];
//             //             dot = __hadd(dot, __hmul(q_val, k_val));
//             //         }
//             //         dot = __hmul(dot, scale);
//             //         o_tmp[j] = dot;

//             //         m_prev = m_new;
//             //         m_new = hmax(m_new, dot);
//             //         m_i[i] = m_new;

//             //         __half p = custom_hexp(o_tmp[j] - m_new);

//             //         l_prev = l_new;
//             //         l_new = __hadd(__hmul(l_prev, custom_hexp(m_prev - m_new)), p);
//             //         l_i[i] = l_new;

//             //         __half delta = __hmul(p, v_val);
//             //         O_i[i * head_dim + k] = __hadd(__hmul(__hmul(l_prev, custom_hexp(m_prev - m_new)), O_i[i * head_dim + k]), delta);
//             //     }

//             //     O_i[i * head_dim + local_col] = __hdiv(O_i[i * head_dim + local_col], l_i[i]);

//             // }
//             for (int i = 0; i < BLOCK_ROWS; ++i) {
//                 if (qi + i >= seq_len) continue;

//                 __half m_prev = m_i[i];
//                 __half m_new = m_prev;

//                 // 1. Compute S_i^{(j)} = Q_i ⋅ K_j^T
//                 __half S_row[BLOCK_COLS];
//                 for (int j = 0; j < BLOCK_COLS; ++j) {
//                     if (kj + j >= seq_len) continue;
//                     __half dot = __float2half(0.0f);
//                     for (int k = 0; k < head_dim; ++k) {
//                         dot = __hadd(dot, __hmul(
//                             Q_block[i * head_dim + k],
//                             K_block[j * head_dim + k]
//                         ));
//                     }
//                     dot = __hmul(dot, scale);
//                     S_row[j] = dot;
//                     m_new = hmax(m_new, dot);
//                 }

//                 // 2. Compute P̃ = exp(S - m_new), l_new
//                 __half l_prev = l_i[i];
//                 __half l_new = __float2half(0.0f);
//                 __half p_row[BLOCK_COLS];
//                 for (int j = 0; j < BLOCK_COLS; ++j) {
//                     if (kj + j >= seq_len) continue;
//                     p_row[j] = custom_hexp(__hsub(S_row[j], m_new));
//                     l_new = __hadd(l_new, p_row[j]);
//                 }
//                 l_new = __hadd(__hmul(custom_hexp(__hsub(m_prev, m_new)), l_prev), l_new);
//                 l_i[i] = l_new;
//                 m_i[i] = m_new;

//                 // 3. Update O_i
//                 for (int k = 0; k < head_dim; ++k) {
//                     __half acc = __hmul(O_i[i * head_dim + k], custom_hexp(__hsub(m_prev, m_new)));
//                     for (int j = 0; j < BLOCK_COLS; ++j) {
//                         if (kj + j >= seq_len) continue;
//                         acc = __hadd(acc, __hmul(p_row[j], V_block[j * head_dim + k]));
//                     }
//                     O_i[i * head_dim + k] = acc;
//                 }
//             }

//             __syncthreads();
//         }

//         for (int i = 0; i < BLOCK_ROWS; ++i) {
//             for (int k = 0; k < head_dim; ++k) {
//                 O_i[i * head_dim + k] = __hdiv(O_i[i * head_dim + k], l_i[i]);
//             }
//         }


//         __syncthreads();
//     }
// }

// #define MAX_LDS_SIZE 65536  // 64 KB
// #define BLOCK_ROWS 1
// #define BLOCK_COLS 4

// __device__ __half hmax(__half a, __half b) {
//     return __hgt(a, b) ? a : b;
// }
// __device__ __half custom_hexp(__half x) {
//     return __float2half(expf(__half2float(x)));
// }

// __global__ void forward_kernel(const __half* __restrict__ Q,
//                                const __half* __restrict__ K,
//                                const __half* __restrict__ V,
//                                __half* __restrict__ O,
//                                const int seq_len,
//                                const int head_dim,
//                                const int num_heads_q,
//                                const int num_heads_kv) {
//     extern __shared__ __half shared_mem[];
//     __half* Q_block = shared_mem;
//     __half* K_block = Q_block + BLOCK_ROWS * head_dim;
//     __half* V_block = K_block + BLOCK_COLS * head_dim;

//     // Allocate LDS space for m_i, l_i, o_i
//     __half* m_i = V_block + BLOCK_COLS * head_dim;
//     __half* l_i = m_i + BLOCK_ROWS;
//     __half* O_i = l_i + BLOCK_ROWS; // Correct pointer arithmetic - no sizeof here

//     const __half scale = __float2half(1.0f / sqrtf((float)head_dim));
    
//     int q_head_id = blockIdx.x;         //当前的block在处理第q_head_id个Q头
//     int kv_head_id = q_head_id / 4;    //当前的block在处理第kv_head_id个KV头

//     // int q_start = blockIdx.y * BLOCK_ROWS;   //本 block 负责 Q 的第 q_start ~ q_start+BLOCK_ROWS-1 行
//     // int kv_start = blockIdx.z * BLOCK_COLS; //本 block 负责 K、V 的第 kv_start ~ kv_start+BLOCK_COLS-1 行

//     int tid = threadIdx.x;
//     int q_offset = q_head_id * head_dim;
//     int kv_offset = kv_head_id * head_dim;

//     for (int qi = 0; qi < seq_len; qi += BLOCK_ROWS) {
//         // int local_row = tid / head_dim - qi;
//         // int local_col = tid % head_dim - 1;
        
//         // // Load Q_i block from HBM to on-chip SRAM
//         // if (tid < ((qi + BLOCK_ROWS) * head_dim)) {
//         //     if (local_row < seq_len) {
//         //         Q_block[local_row * head_dim + local_col] =
//         //             Q[global_row * (num_heads_q * head_dim) + q_offset + local_col];
//         //     }
//         // }

//         int local_row = tid / head_dim;
//         int local_col = tid % head_dim - 1;
        
//         // Load Q_i block from HBM to on-chip SRAM
//         if (tid < BLOCK_ROWS * head_dim) {
//             int global_row = qi + local_row;
//             if (global_row < seq_len) {
//                 Q_block[local_row * head_dim + local_col] =
//                     Q[global_row * (num_heads_q * head_dim) + q_offset + local_col];
//             }
//         }
        
//         __syncthreads();
        
//         // Initialize m_i, l_i, O_i for this block of Q
//         if (tid < BLOCK_ROWS) {
//             m_i[tid] = __half(-65504.0f); // -∞
//             l_i[tid] = __half(0.0f);
//         }
//         if (tid < BLOCK_ROWS * head_dim) {
//             O_i[tid] = __half(0.0f);
//         }
        
//         __syncthreads();

//         for (int kj = 0; kj < seq_len; kj += BLOCK_COLS) {
//             // Load K_j, V_j blocks from HBM to on-chip SRAM
//             if (tid < BLOCK_COLS * head_dim) {
//                 int global_row = kj + local_row;

//                 if (global_row < seq_len) {
//                     K_block[local_row * head_dim + local_col] =
//                         K[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];
//                     V_block[local_row * head_dim + local_col] =
//                         V[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];
//                 }
//             }

//             __syncthreads();

//             // Process each row of Q_i
//             for (int i = 0; i < BLOCK_ROWS; ++i) {
//                 if (qi + i >= seq_len) continue;

//                 __half m_prev = m_i[i];
//                 __half l_prev = l_i[i];
//                 __half new_m = m_prev;
                
//                 // Compute S_i^(j) = Q_i K_j^T and find new max
//                 __half S_ij[BLOCK_COLS];
//                 for (int j = 0; j < BLOCK_COLS; ++j) {
//                     if (kj + j >= seq_len) continue;
//                     __half dot = __half(0.0f);
//                     for (int k = 0; k < head_dim; ++k) {
//                         __half q_val = Q_block[i * head_dim + k];
//                         __half k_val = K_block[j * head_dim + k];
//                         dot = __hadd(dot, __hmul(q_val, k_val));
//                     }
//                     dot = __hmul(dot, scale);
//                     S_ij[j] = dot;
//                     new_m = hmax(new_m, dot);
//                 }
                
//                 // Calculate P̃_i^(j) = exp(S_i^(j) - m_i^(j))
//                 __half l_new = __half(0.0f);
//                 __half P_ij[BLOCK_COLS];
                
//                 for (int j = 0; j < BLOCK_COLS; ++j) {
//                     if (kj + j >= seq_len) continue;
//                     P_ij[j] = custom_hexp(__hsub(S_ij[j], new_m));
//                     l_new = __hadd(l_new, P_ij[j]);
//                 }
                
//                 // Update l_i according to line 9 in the algorithm
//                 l_i[i] = __hadd(__hmul(l_prev, custom_hexp(__hsub(m_prev, new_m))), l_new);
                
//                 // Save the new maximum
//                 m_i[i] = new_m;
                
//                 // Update O_i according to line 10 in the algorithm
//                 __half scale_factor = custom_hexp(__hsub(m_prev, new_m));
//                 for (int k = 0; k < head_dim; ++k) {
//                     __half acc = __hmul(O_i[i * head_dim + k], scale_factor);
                    
//                     for (int j = 0; j < BLOCK_COLS; ++j) {
//                         if (kj + j >= seq_len) continue;
//                         __half v_val = V_block[j * head_dim + k];
//                         __half p_val = P_ij[j];
//                         acc = __hadd(acc, __hmul(p_val, v_val));
//                     }
                    
//                     O_i[i * head_dim + k] = acc;
//                 }
//             }

//             __syncthreads();
//         }
        
//         // After processing all K_j blocks, compute final O_i
//         for (int i = 0; i < BLOCK_ROWS; ++i) {
//             if (qi + i >= seq_len) continue;
            
//             // Normalize O_i by l_i (line 12 in the algorithm)
//             for (int k = 0; k < head_dim; ++k) {
//                 O_i[i * head_dim + k] = __hdiv(O_i[i * head_dim + k], l_i[i]);
//             }
//         }
        
//         // Write O_i to HBM as the i-th block of O (line 14)
//         if (tid < BLOCK_ROWS * head_dim) {
//             if (qi + local_row < seq_len) {
//                 O[(qi + local_row) * (num_heads_q * head_dim) + q_offset + local_col] = 
//                     O_i[local_row * head_dim + local_col];
//             }
//         }

//         __syncthreads();
//     }
// }

// #define MAX_LDS_SIZE 65536  // 64 KB
// #define BLOCK_ROWS 4
// #define BLOCK_COLS 4

// __device__ __half hmax(__half a, __half b) {
//     return __hgt(a, b) ? a : b;
// }
// __device__ __half custom_hexp(__half x) {
//     return __float2half(expf(__half2float(x)));
// }

// __global__ void forward_kernel(const __half* __restrict__ Q,
//                                const __half* __restrict__ K,
//                                const __half* __restrict__ V,
//                                __half* __restrict__ O,
//                                const int seq_len,
//                                const int head_dim,
//                                const int num_heads_q,
//                                const int num_heads_kv) {
//     extern __shared__ __half shared_mem[];
//     __half* Q_block = shared_mem;
//     __half* K_block = Q_block + BLOCK_ROWS * head_dim;
//     __half* V_block = K_block + BLOCK_COLS * head_dim;

//     // Allocate LDS space for m_i, l_i, o_i
//     __half* m_i = V_block + BLOCK_COLS * head_dim;
//     __half* l_i = m_i + BLOCK_ROWS;
//     __half* O_i = l_i + BLOCK_ROWS; 

//     const __half scale = __float2half(1.0f / sqrtf((float)head_dim));
    
//     int Q_head_idx = blockIdx.x;     //当前的block在处理第q_head_id个Q头
//     int Q_block_idx = blockIdx.y;   //当前的block在处理Q_head_idx个Q头中的第Q_block_idx个Q块
//     int KV_head_idx = Q_head_idx / (num_heads_q/num_heads_kv); //当前的block在处理第kv_head_id个KV头

    

//     // int tid = threadIdx.x;
//     // int q_offset = (q_head_id + 1)* head_dim;
//     // int kv_offset = (kv_head_id + 1)* head_dim;

//     // for (; tid < seq_len; tid += BLOCK_ROWS)
//     // {
//     //     for(int i =0; i < head_dim; i++)
//     //     {
//     //         Q_block[tid * head_dim + i] = Q[tid * q_offset + head_dim * q_head_id + i];
//     //     }
//     //     __syncthreads();

//     //     for (int j = 0; j < seq_len; j += BLOCK_COLS)
//     //     {
//     //         for (int k = 0; k < BLOCK_COLS; k++)
//     //         {

//     //         }
//     //     }
        
//     // }

//     // for (int qi = 0; qi < seq_len; qi += BLOCK_ROWS) {
//     //     int local_row = tid / head_dim;
//     //     int local_col = tid % head_dim;
        
//         // // Load Q_i block from HBM to on-chip SRAM
//         // if (tid < BLOCK_ROWS * head_dim) {
//         //     int global_row = qi + local_row;
//         //     if (global_row < seq_len) {
//         //         Q_block[local_row * head_dim + local_col] =
//         //             Q[global_row * (num_heads_q * head_dim) + q_offset + local_col];
//         //     }
//         // }



        
//     //     __syncthreads();
        
//     //     // Initialize m_i, l_i, O_i for this block of Q
//     //     if (tid < BLOCK_ROWS) {
//     //         m_i[tid] = __half(-65504.0f); // -∞
//     //         l_i[tid] = __half(0.0f);
//     //     }
//     //     if (tid < BLOCK_ROWS * head_dim) {
//     //         O_i[tid] = __half(0.0f);
//     //     }
        
//     //     __syncthreads();

//     //     for (int kj = 0; kj < seq_len; kj += BLOCK_COLS) {
//     //         // Load K_j, V_j blocks from HBM to on-chip SRAM
//     //         if (tid < BLOCK_COLS * head_dim) {
//     //             int global_row = kj + local_row;

//     //             if (global_row < seq_len) {
//     //                 K_block[local_row * head_dim + local_col] =
//     //                     K[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];
//     //                 V_block[local_row * head_dim + local_col] =
//     //                     V[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];
//     //             }
//     //         }

//     //         __syncthreads();

//     //         // Process each row of Q_i
//     //         for (int i = 0; i < BLOCK_ROWS; ++i) {
//     //             if (qi + i >= seq_len) continue;

//     //             __half m_prev = m_i[i];
//     //             __half l_prev = l_i[i];
//     //             __half new_m = m_prev;
                
//     //             // Compute S_i^(j) = Q_i K_j^T and find new max
//     //             __half S_ij[BLOCK_COLS];
//     //             for (int j = 0; j < BLOCK_COLS; ++j) {
//     //                 if (kj + j >= seq_len) continue;
//     //                 __half dot = __half(0.0f);
//     //                 for (int k = 0; k < head_dim; ++k) {
//     //                     __half q_val = Q_block[i * head_dim + k];
//     //                     __half k_val = K_block[j * head_dim + k];
//     //                     dot = __hadd(dot, __hmul(q_val, k_val));
//     //                 }
//     //                 dot = __hmul(dot, scale);
//     //                 S_ij[j] = dot;
//     //                 new_m = hmax(new_m, dot);
//     //             }
                
//     //             // Calculate P̃_i^(j) = exp(S_i^(j) - m_i^(j))
//     //             __half l_new = __half(0.0f);
//     //             __half P_ij[BLOCK_COLS];
                
//     //             for (int j = 0; j < BLOCK_COLS; ++j) {
//     //                 if (kj + j >= seq_len) continue;
//     //                 P_ij[j] = custom_hexp(__hsub(S_ij[j], new_m));
//     //                 l_new = __hadd(l_new, P_ij[j]);
//     //             }
                
//     //             // Update l_i according to line 9 in the algorithm
//     //             l_i[i] = __hadd(__hmul(l_prev, custom_hexp(__hsub(m_prev, new_m))), l_new);
                
//     //             // Save the new maximum
//     //             m_i[i] = new_m;
                
//     //             // Update O_i according to line 10 in the algorithm
//     //             __half scale_factor = custom_hexp(__hsub(m_prev, new_m));
//     //             for (int k = 0; k < head_dim; ++k) {
//     //                 __half acc = __hmul(O_i[i * head_dim + k], scale_factor);
                    
//     //                 for (int j = 0; j < BLOCK_COLS; ++j) {
//     //                     if (kj + j >= seq_len) continue;
//     //                     __half v_val = V_block[j * head_dim + k];
//     //                     __half p_val = P_ij[j];
//     //                     acc = __hadd(acc, __hmul(p_val, v_val));
//     //                 }
                    
//     //                 O_i[i * head_dim + k] = acc;
//     //             }
//     //         }

//     //         __syncthreads();
//     //     }
        
//     //     // After processing all K_j blocks, compute final O_i
//     //     for (int i = 0; i < BLOCK_ROWS; ++i) {
//     //         if (qi + i >= seq_len) continue;
            
//     //         // Normalize O_i by l_i (line 12 in the algorithm)
//     //         for (int k = 0; k < head_dim; ++k) {
//     //             O_i[i * head_dim + k] = __hdiv(O_i[i * head_dim + k], l_i[i]);
//     //         }
//     //     }
        
//     //     // Write O_i to HBM as the i-th block of O (line 14)
//     //     if (tid < BLOCK_ROWS * head_dim) {
//     //         if (qi + local_row < seq_len) {
//     //             O[(qi + local_row) * (num_heads_q * head_dim) + q_offset + local_col] = 
//     //                 O_i[local_row * head_dim + local_col];
//     //         }
//     //     }

//     //     __syncthreads();
//     // }
// }


#define MAX_LDS_SIZE 65536  // 64 KB
#define BLOCK_ROWS 1
#define BLOCK_COLS 4

__device__ __half hmax(__half a, __half b) {
    return __hgt(a, b) ? a : b;
}
__device__ __half custom_hexp(__half x) {
    return __float2half(expf(__half2float(x)));
}

__global__ void forward_kernel(const __half* __restrict__ Q,
                               const __half* __restrict__ K,
                               const __half* __restrict__ V,
                               __half* __restrict__ O,
                               const int seq_len,
                               const int head_dim,
                               const int num_heads_q,
                               const int num_heads_kv) {
    extern __shared__ __half shared_mem[];
    __half* Q_block = shared_mem;
    __half* K_block = Q_block + BLOCK_ROWS * head_dim;
    __half* V_block = K_block + BLOCK_COLS * head_dim;

    // Allocate LDS space for m_i, l_i, o_i
    __half* m_i = V_block + BLOCK_COLS * head_dim;
    __half* l_i = m_i + BLOCK_ROWS;
    __half* O_i = l_i + BLOCK_ROWS; // Correct pointer arithmetic - no sizeof here

    const __half scale = __float2half(1.0f / sqrtf((float)head_dim));
    
    int q_head_id = blockIdx.x;
    int kv_head_id = q_head_id / 4;

    int tid = threadIdx.x;
    int q_offset = q_head_id * head_dim;
    int kv_offset = kv_head_id * head_dim;

    for (int qi = 0; qi < seq_len; qi += BLOCK_ROWS) {
        int local_row = tid / head_dim;
        int local_col = tid % head_dim;
        
        // Load Q_i block from HBM to on-chip SRAM
        if (tid < BLOCK_ROWS * head_dim) {
            int global_row = qi + local_row;
            if (global_row < seq_len) {
                Q_block[local_row * head_dim + local_col] =
                    Q[global_row * (num_heads_q * head_dim) + q_offset + local_col];
            }
        }
        
        __syncthreads();
        
        // Initialize m_i, l_i, O_i for this block of Q
        if (tid < BLOCK_ROWS) {
            m_i[tid] = __half(-65504.0f); // -∞
            l_i[tid] = __half(0.0f);
        }
        if (tid < BLOCK_ROWS * head_dim) {
            O_i[tid] = __half(0.0f);
        }
        
        __syncthreads();

        for (int kj = 0; kj < seq_len; kj += BLOCK_COLS) {
            // Load K_j, V_j blocks from HBM to on-chip SRAM
            if (tid < BLOCK_COLS * head_dim) {
                int global_row = kj + local_row;

                if (global_row < seq_len) {
                    K_block[local_row * head_dim + local_col] =
                        K[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];
                    V_block[local_row * head_dim + local_col] =
                        V[global_row * (num_heads_kv * head_dim) + kv_offset + local_col];
                }
            }

            __syncthreads();

            // Process each row of Q_i
            for (int i = 0; i < BLOCK_ROWS; ++i) {
                if (qi + i >= seq_len) continue;

                __half m_prev = m_i[i];
                __half l_prev = l_i[i];
                __half new_m = m_prev;
                
                // Compute S_i^(j) = Q_i K_j^T and find new max
                __half S_ij[BLOCK_COLS];
                for (int j = 0; j < BLOCK_COLS; ++j) {
                    if (kj + j >= seq_len) continue;
                    __half dot = __half(0.0f);
                    for (int k = 0; k < head_dim; ++k) {
                        __half q_val = Q_block[i * head_dim + k];
                        __half k_val = K_block[j * head_dim + k];
                        dot = __hadd(dot, __hmul(q_val, k_val));
                    }
                    dot = __hmul(dot, scale);
                    S_ij[j] = dot;
                    new_m = hmax(new_m, dot);
                }
                
                // Calculate P̃_i^(j) = exp(S_i^(j) - m_i^(j))
                __half l_new = __half(0.0f);
                __half P_ij[BLOCK_COLS];
                
                for (int j = 0; j < BLOCK_COLS; ++j) {
                    if (kj + j >= seq_len) continue;
                    P_ij[j] = custom_hexp(__hsub(S_ij[j], new_m));
                    l_new = __hadd(l_new, P_ij[j]);
                }
                
                // Update l_i according to line 9 in the algorithm
                l_i[i] = __hadd(__hmul(l_prev, custom_hexp(__hsub(m_prev, new_m))), l_new);
                
                // Save the new maximum
                m_i[i] = new_m;
                
                // Update O_i according to line 10 in the algorithm
                __half scale_factor = custom_hexp(__hsub(m_prev, new_m));
                for (int k = 0; k < head_dim; ++k) {
                    __half acc = __hmul(O_i[i * head_dim + k], scale_factor);
                    
                    for (int j = 0; j < BLOCK_COLS; ++j) {
                        if (kj + j >= seq_len) continue;
                        __half v_val = V_block[j * head_dim + k];
                        __half p_val = P_ij[j];
                        acc = __hadd(acc, __hmul(p_val, v_val));
                    }
                    
                    O_i[i * head_dim + k] = acc;
                }
            }

            __syncthreads();
        }
        
        // After processing all K_j blocks, compute final O_i
        for (int i = 0; i < BLOCK_ROWS; ++i) {
            if (qi + i >= seq_len) continue;
            
            // Normalize O_i by l_i (line 12 in the algorithm)
            for (int k = 0; k < head_dim; ++k) {
                O_i[i * head_dim + k] = __hdiv(O_i[i * head_dim + k], l_i[i]);
            }
        }
        
        // Write O_i to HBM as the i-th block of O (line 14)
        if (tid < BLOCK_ROWS * head_dim) {
            if (qi + local_row < seq_len) {
                O[(qi + local_row) * (num_heads_q * head_dim) + q_offset + local_col] = 
                    O_i[local_row * head_dim + local_col];
            }
        }

        __syncthreads();
    }
}


// // 调整分块大小以满足shared memory限制
// #define BLOCK_Q_ROWS 32   // 每次处理 Q 的 32 行
// #define BLOCK_ROWS 32     // 每次处理 K、V 的 32 行

// __device__ __half hmax(__half a, __half b) {
//     return __hgt(a, b) ? a : b;
// }

// __device__ __half custom_hexp(__half x) {
//     return __float2half(expf(__half2float(x)));
// }

// // 新的 kernel：每个 block 只处理 (BLOCK_Q_ROWS x head_dim) 大小的 Q 子块，与 (BLOCK_ROWS x head_dim) 大小的 K、V 子块。
// __global__ void flash_attn_tile_kernel(
//     const __half* __restrict__ Q,  // [seq_len, num_heads_q * head_dim]
//     const __half* __restrict__ K,  // [seq_len, num_heads_kv * head_dim]
//     const __half* __restrict__ V,  // [seq_len, num_heads_kv * head_dim]
//     __half* __restrict__ O,        // [seq_len, num_heads_q * head_dim]
//     const int seq_len,
//     const int head_dim,
//     const int num_heads_q,
//     const int num_heads_kv
// ) {
//     // 1. 确定当前处理的 head
//     int head_id = blockIdx.z; 
//     // 假设 4个Q头对应1个KV头（示例），每4个Q头共用1个KV head
//     int kv_head_id = head_id / (num_heads_q / num_heads_kv);

//     // 计算 Q, K/V 的列偏移
//     int q_offset = head_id * head_dim;     
//     int kv_offset = kv_head_id * head_dim; 

//     // 2. 计算 Q、K/V 在行维度的分块起始
//     int q_start = blockIdx.x * BLOCK_Q_ROWS;   // 本 block 负责 Q 的第 q_start ~ q_start+BLOCK_Q_ROWS-1 行
//     int kv_start = blockIdx.y * BLOCK_ROWS;    // 本 block 负责 K、V 的第 kv_start ~ kv_start+BLOCK_ROWS-1 行

//     // 线程索引
//     int tid = threadIdx.x;
//     int block_threads = blockDim.x;

//     // 3. 分配共享内存：
//     //   Q_block : [BLOCK_Q_ROWS, head_dim]
//     //   K_block : [BLOCK_ROWS, head_dim]
//     //   V_block : [BLOCK_ROWS, head_dim]
//     //   m_i, l_i : [BLOCK_Q_ROWS]
//     //   O_i      : [BLOCK_Q_ROWS, head_dim]
//     extern __shared__ __half shared_mem[];
//     __half* Q_block = shared_mem;                               
//     __half* K_block = Q_block + (BLOCK_Q_ROWS * head_dim);      
//     __half* V_block = K_block + (BLOCK_ROWS * head_dim);        
//     __half* m_i     = V_block + (BLOCK_ROWS * head_dim);        
//     __half* l_i     = m_i + BLOCK_Q_ROWS;                       
//     __half* O_i     = l_i + BLOCK_Q_ROWS;                      

//     // 缩放因子
//     __half scale = __float2half(1.0f / sqrtf((float)head_dim));

//     // 4. 初始化 m_i, l_i, O_i
//     for (int i = tid; i < BLOCK_Q_ROWS; i += block_threads) {
//         m_i[i] = __float2half(-65504.0f); // -∞
//         l_i[i] = __float2half(0.0f);
//     }
//     for (int x = tid; x < BLOCK_Q_ROWS * head_dim; x += block_threads) {
//         O_i[x] = __float2half(0.0f);
//     }
//     __syncthreads();

//     // 5. 加载 Q 的当前子块到共享内存 Q_block
//     for (int x = tid; x < BLOCK_Q_ROWS * head_dim; x += block_threads) {
//         int row = x / head_dim; // Q_block 的行下标
//         int col = x % head_dim; // Q_block 的列下标
//         int global_row = q_start + row;
//         if (global_row < seq_len) {
//             // Q 存储：Q[global_row, q_offset + col]
//             Q_block[x] = Q[global_row * (num_heads_q * head_dim) + (q_offset + col)];
//         } else {
//             Q_block[x] = __float2half(0.0f);
//         }
//     }
//     __syncthreads();

//     // 6. 如果 kv_start 超出 seq_len，就不需要加载 K、V
//     if (kv_start < seq_len) {
//         // 6a. 加载本块 K, V 到共享内存
//         for (int x = tid; x < BLOCK_ROWS * head_dim; x += block_threads) {
//             int row = x / head_dim;
//             int col = x % head_dim;
//             int global_row = kv_start + row;
//             if (global_row < seq_len) {
//                 K_block[x] = K[global_row * (num_heads_kv * head_dim) + (kv_offset + col)];
//                 V_block[x] = V[global_row * (num_heads_kv * head_dim) + (kv_offset + col)];
//             } 
//             // else {
//             //     K_block[x] = __float2half(0.0f);
//             //     V_block[x] = __float2half(0.0f);
//             // }
//         }
//         __syncthreads();

//         // 6b. 计算 Q_block 与 K_block 的点积，更新 m_i, l_i 并累加到 O_i
//         //     这里只作为示例。实际中可在 tid 维度上并行展开。
//         for (int qi = 0; qi < BLOCK_Q_ROWS; ++qi) {
//             // 若实际 Q 行越界，则跳过
//             int real_q_row = q_start + qi;
//             if (real_q_row >= seq_len) break;

//             // 读取前一次 m_i, l_i
//             __half m_prev = m_i[qi];
//             __half l_prev = l_i[qi];

//             // 首先找 rowmax(S_ij)，更新 m_i
//             // (示例中没有使用原子操作，真实应用需注意并发写 m_i / l_i 同一元素)
//             __half local_max = m_prev;
//             for (int kv_row = tid; kv_row < BLOCK_ROWS; kv_row += block_threads) {
//                 int real_k_row = kv_start + kv_row;
//                 if (real_k_row >= seq_len) continue;

//                 // 点积
//                 __half dot = __float2half(0.0f);
//                 for (int d = 0; d < head_dim; d++) {
//                     dot = __hadd(dot, __hmul(Q_block[qi * head_dim + d],
//                                              K_block[kv_row * head_dim + d]));
//                 }
//                 dot = __hmul(dot, scale);

//                 // 记录最大值
//                 local_max = hmax(local_max, dot);
//             }

//             // 用一个 thread 来写回 m_i[qi]（简化示例）
//             __syncthreads();
//             if (tid == 0) {
//                 m_i[qi] = hmax(m_i[qi], local_max);
//             }
//             __syncthreads();

//             // 重新读取更新后的 m_i
//             __half m_new = m_i[qi];

//             // 累加 l_i, 并更新到 O_i
//             // Pseudocode: p_ij = exp(dot - m_new)
//             // l_i = exp(m_prev - m_new)*l_prev + sum_{j} p_ij
//             // O_i = exp(m_prev - m_new)*O_i + p_ij * V
//             __half l_new = __float2half(0.0f);

//             for (int kv_row = tid; kv_row < BLOCK_ROWS; kv_row += block_threads) {
//                 int real_k_row = kv_start + kv_row;
//                 if (real_k_row >= seq_len) continue;

//                 // 点积
//                 __half dot = __float2half(0.0f);
//                 for (int d = 0; d < head_dim; d++) {
//                     dot = __hadd(dot, __hmul(Q_block[qi * head_dim + d],
//                                              K_block[kv_row * head_dim + d]));
//                 }
//                 dot = __hmul(dot, scale);

//                 // p = exp(dot - m_new)
//                 __half p_val = custom_hexp(__hsub(dot, m_new));

//                 // 原子累加到 l_new (示例)
//                 // 无原子操作版本可写成: l_new += p_val; 但要注意并发访问
//                 // 这里演示则用原子:
//                 //atomicAdd((half*)&l_new, p_val);
//                 l_new = __hadd(l_new, p_val);

//                 // 累加到 O_i
//                 for (int d = 0; d < head_dim; d++) {
//                     __half delta = __hmul(p_val, V_block[kv_row * head_dim + d]);
//                     //atomicAdd((half*)&O_i[qi * head_dim + d], delta);
//                     O_i[qi * head_dim + d] = __hadd(O_i[qi * head_dim + d], delta);
//                 }
//             }

//             __syncthreads();
//             // 单线程更新 l_i[qi] => l_new
//             if (tid == 0) {
//                 __half scale_factor = custom_hexp(__hsub(m_prev, m_new));
//                 l_i[qi] = __hadd(__hmul(l_prev, scale_factor), l_new);

//                 // 同时也要将 O_i[qi] 做 exp(m_prev - m_new) 缩放
//                 for (int d = 0; d < head_dim; d++) {
//                     O_i[qi * head_dim + d] = __hmul(O_i[qi * head_dim + d], scale_factor);
//                 }
//             }
//             __syncthreads();
//         }
//     } // end if (kv_start < seq_len)

//     // 7. 将本子块的 O_i 正式归一化，然后写回全局 O
//     for (int qi = 0; qi < BLOCK_Q_ROWS; ++qi) {
//         int real_q_row = q_start + qi;
//         if (real_q_row >= seq_len) break;

//         // 先做  O_i / l_i
//         __half denom = l_i[qi];
//         for (int x = tid; x < head_dim; x += block_threads) {
//             if (__half2float(denom) != 0.f) {
//                 O_i[qi * head_dim + x] =
//                     __hdiv(O_i[qi * head_dim + x], denom);
//             }
//         }
//         __syncthreads();

//         // 写回全局
//         for (int x = tid; x < head_dim; x += block_threads) {
//             O[real_q_row * (num_heads_q * head_dim) + (q_offset + x)] = 
//                 O_i[qi * head_dim + x];
//         }
//         __syncthreads();
//     }
// }

void flash_attention(const __half* Q,
                     const __half* K,
                     const __half* V,
                     __half* O,
                     const int seq_len,
                     const int head_dim,
                     const int num_heads_q,
                     const int num_heads_kv) 
{
    // 计算网格和块规模
    // grid.z = num_heads_q  (每个 head 分配一个"平面")
    // grid.x = ceil(seq_len / BLOCK_Q_ROWS)
    // grid.y = ceil(seq_len / BLOCK_ROWS)
    int grid_x = (seq_len + BLOCK_Q_ROWS - 1) / BLOCK_Q_ROWS;   //计算将Q分成的块数
    int grid_y = (seq_len + BLOCK_ROWS - 1) / BLOCK_ROWS;       //计算将K和V分成的块数
    dim3 grid_dim(grid_x, grid_y, num_heads_q);

    // 线程块大小
    int threads_per_block = 256; // 可以视情况调整

    // 计算共享内存大小 (确保不超过 64KB)
    // Q_block(BLOCK_Q_ROWS*head_dim) + K_block(BLOCK_ROWS*head_dim) + V_block(BLOCK_ROWS*head_dim)
    // + m_i(BLOCK_Q_ROWS) + l_i(BLOCK_Q_ROWS) + O_i(BLOCK_Q_ROWS*head_dim)
    size_t shared_mem_size = (BLOCK_Q_ROWS * head_dim 
                              + BLOCK_ROWS * head_dim * 2
                              + BLOCK_Q_ROWS * (1+1+head_dim)) * sizeof(__half);
    if (shared_mem_size > 64*1024) {
        printf("Error: Shared memory size exceeds 64KB limit.\n");
        return;
    }

    // 启动 kernel
    flash_attn_tile_kernel<<<grid_dim, threads_per_block, shared_mem_size>>>(
        Q, K, V, O, seq_len, head_dim, num_heads_q, num_heads_kv
    );

    // 检查错误
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Kernel launch error: %s\n", hipGetErrorString(err));
        return;
    }
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("Device sync error: %s\n", hipGetErrorString(err));
    }
}