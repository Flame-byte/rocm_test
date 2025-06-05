#include "flash_encoding.h"

#define BLOCK_ROWS 8
#define BLOCK_COLS 4
#define KV_TILE_SIZE 64

__device__ __half hmax(__half a, __half b) {
    return __hgt(a, b) ? a : b;
}
__device__ __half custom_hexp(__half x) {
    //return __float2half(expf(__half2float(x)));
    return hexp(x);
}

__global__ void forward_kernel_tiled(const __half* __restrict__ Q,
                                     const __half* __restrict__ K,
                                     const __half* __restrict__ V,
                                     __half* __restrict__ partial_O,
                                     __half* __restrict__ partial_m,
                                     __half* __restrict__ partial_l,    
                                     const int seq_len_q,
                                     const int seq_len_kv,
                                     const int head_dim,
                                     const int num_heads_q,
                                     const int num_heads_kv) {
    extern __shared__ __half shared_mem[];
    __half* Q_block = shared_mem;
    __half* K_block = Q_block + BLOCK_ROWS * head_dim;
    __half* V_block = K_block + BLOCK_COLS * head_dim;
    __half* m_i = V_block + BLOCK_COLS * head_dim;
    __half* l_i = m_i + BLOCK_ROWS;
    __half* O_i = l_i + BLOCK_ROWS;

    const __half scale = __float2half(1.0f / sqrtf((float)head_dim));

    const int Q_head_idx = blockIdx.x;
    const int Q_block_idx = blockIdx.y;
    const int KV_tile_idx = blockIdx.z;

    const int num_q_per_kv = num_heads_q / num_heads_kv;
    const int KV_head_idx = Q_head_idx / num_q_per_kv;

    for (int idx = threadIdx.x; idx < BLOCK_ROWS * head_dim; idx += blockDim.x) {
        const int local_row = idx / head_dim;
        const int local_col = idx % head_dim;
        const int global_row = Q_block_idx * BLOCK_ROWS + local_row;
        if (global_row < seq_len_q) {
            const int global_pos = global_row * (num_heads_q * head_dim) + Q_head_idx * head_dim + local_col;
            Q_block[idx] = Q[global_pos];
        }
    }

    for (int r = threadIdx.x; r < BLOCK_ROWS; r += blockDim.x) {
        m_i[r] = __half(-65504.0f);
        l_i[r] = __half(0.0f);
    }
    for (int idx = threadIdx.x; idx < BLOCK_ROWS * head_dim; idx += blockDim.x) {
        O_i[idx] = __half(0.0f);
    }
    __syncthreads();

    const int j_start = KV_tile_idx * KV_TILE_SIZE;
    const int j_end = min(seq_len_kv, j_start + KV_TILE_SIZE);

    for (int j = j_start; j < j_end; j += BLOCK_COLS) {
        for (int idx = threadIdx.x; idx < BLOCK_COLS * head_dim; idx += blockDim.x) {
            const int local_row = idx / head_dim;
            const int local_col = idx % head_dim;
            if ((j + local_row) < seq_len_kv) {
                const int global_row = j + local_row;
                const int global_pos = global_row * (num_heads_kv * head_dim) + KV_head_idx * head_dim + local_col;
                K_block[idx] = K[global_pos];
                V_block[idx] = V[global_pos];
            }
        }
        __syncthreads();

        __shared__ __half S_ij[BLOCK_ROWS][BLOCK_COLS];
        int q_row = threadIdx.x;
        int k_col = threadIdx.y;

        if (q_row < BLOCK_ROWS && k_col < BLOCK_COLS) {
            int global_q_idx = Q_block_idx * BLOCK_ROWS + q_row;
            int global_k_idx = j + k_col;

            __half sum = __half(0.0f);
            if (global_q_idx >= global_k_idx && global_q_idx < seq_len_q && global_k_idx < seq_len_kv) {
                for (int k = 0; k < head_dim; ++k) {
                    sum = __hadd(sum, __hmul(Q_block[q_row * head_dim + k], K_block[k_col * head_dim + k]));
                }
                sum = __hmul(sum, scale);
            } else {
                sum = __half(-65504.0f);
            }
            S_ij[q_row][k_col] = sum;
            __syncthreads();

            __half old_m = m_i[q_row];
            __half localMax = S_ij[q_row][0];
            for (int col = 0; col < BLOCK_COLS; col++) {
                localMax = hmax(localMax, S_ij[q_row][col]);
            }
            m_i[q_row] = hmax(old_m, localMax);
            __syncthreads();

            __half old_l = l_i[q_row];
            __half scale_factor = custom_hexp(__hsub(old_m, m_i[q_row]));
            __half l_new = __hmul(old_l, scale_factor);

            __half rowsumP = __half(0.0f);
            for (int col = 0; col < BLOCK_COLS; col++) {
                __half p_val = custom_hexp(__hsub(S_ij[q_row][col], m_i[q_row]));
                S_ij[q_row][col] = p_val;
                rowsumP = __hadd(rowsumP, p_val);
            }
            l_i[q_row] = __hadd(l_new, rowsumP);
            __syncthreads();

            for (int c = 0; c < head_dim; ++c) {
                __half old_O_val = O_i[q_row * head_dim + c];
                __half acc = __hmul(old_O_val, scale_factor);
                __half sumPV = __half(0.0f);
                for (int col = 0; col < BLOCK_COLS; col++) {
                    __half p_val = S_ij[q_row][col];
                    __half v_val = V_block[col * head_dim + c];
                    sumPV = __hadd(sumPV, __hmul(p_val, v_val));
                }
                acc = __hadd(acc, sumPV);
                O_i[q_row * head_dim + c] = acc;
            }
            __syncthreads();
        }
    }

    __syncthreads();
    for (int idx = threadIdx.x; idx < BLOCK_ROWS * head_dim; idx += blockDim.x) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        if (row < BLOCK_ROWS) {
            __half inv_l = __hdiv(__half(1.0f), l_i[row]);
            O_i[row * head_dim + col] = __hmul(O_i[row * head_dim + col], inv_l);
        }
    }
    __syncthreads();

    for (int row = threadIdx.x; row < BLOCK_ROWS; row += blockDim.x) {
        int global_row = Q_block_idx * BLOCK_ROWS + row;
        if (global_row < seq_len_q) {
            int index = ((Q_head_idx * ((seq_len_q + BLOCK_ROWS - 1) / BLOCK_ROWS) + Q_block_idx) * ((seq_len_kv + KV_TILE_SIZE - 1) / KV_TILE_SIZE) + KV_tile_idx);
            partial_m[index * BLOCK_ROWS + row] = m_i[row];
            partial_l[index * BLOCK_ROWS + row] = l_i[row];
        }
    }

    for (int idx = threadIdx.x; idx < BLOCK_ROWS * head_dim; idx += blockDim.x) {
        const int row = idx / head_dim;
        const int col = idx % head_dim;
        const int global_row = Q_block_idx * BLOCK_ROWS + row;
        if (global_row < seq_len_q) {
            int index = ((Q_head_idx * ((seq_len_q + BLOCK_ROWS - 1) / BLOCK_ROWS) + Q_block_idx) * ((seq_len_kv + KV_TILE_SIZE - 1) / KV_TILE_SIZE) + KV_tile_idx);
            partial_O[(index * BLOCK_ROWS + row) * head_dim + col] = O_i[row * head_dim + col];
        }
    }
}


__global__ void merge_kernel(const __half* partial_m,
                             const __half* partial_l,
                             const __half* partial_O,
                             __half* O,
                             int seq_len_q,
                             int head_dim,
                             int num_heads_q,
                             int num_tiles) {
    int head_idx = blockIdx.x;
    int q_block_idx = blockIdx.y;
    int row = threadIdx.x;
    if (row >= BLOCK_ROWS) return;
    int global_row = q_block_idx * BLOCK_ROWS + row;
    if (global_row >= seq_len_q) return;

    __half m_max = __half(-65504.0f);
    for (int t = 0; t < num_tiles; ++t) {
        int idx = ((head_idx * ((seq_len_q + BLOCK_ROWS - 1) / BLOCK_ROWS) + q_block_idx) * num_tiles + t) * BLOCK_ROWS + row;
        m_max = hmax(m_max, partial_m[idx]);
    }

    __half denom = __half(0.0f);
    for (int t = 0; t < num_tiles; ++t) {
        int idx = ((head_idx * ((seq_len_q + BLOCK_ROWS - 1) / BLOCK_ROWS) + q_block_idx) * num_tiles + t) * BLOCK_ROWS + row;
        __half coeff = __hmul(custom_hexp(__hsub(partial_m[idx], m_max)), partial_l[idx]);
        denom = __hadd(denom, coeff);
        for (int d = 0; d < head_dim; ++d) {
            __hadd(&O[(global_row * num_heads_q + head_idx) * head_dim + d], __hmul(coeff, partial_O[(idx) * head_dim + d]));
        }
    }

    for (int d = 0; d < head_dim; ++d) {
        O[(global_row * num_heads_q + head_idx) * head_dim + d] = __hdiv(O[(global_row * num_heads_q + head_idx) * head_dim + d], denom);
    }
}


void flash_encoding(const __half* Q,
                    const __half* K,
                    const __half* V,
                    __half* O,
                    const int seq_len_q,
                    const int seq_len_kv,
                    const int head_dim,
                    const int num_heads_q,
                    const int num_heads_kv) {
    size_t shared_mem_size = (2 * BLOCK_ROWS * head_dim + 2 * BLOCK_COLS * head_dim + 2 * BLOCK_ROWS + BLOCK_ROWS * BLOCK_COLS) * sizeof(__half);
    dim3 grid(num_heads_q, (seq_len_q + BLOCK_ROWS - 1) / BLOCK_ROWS, (seq_len_kv + KV_TILE_SIZE - 1) / KV_TILE_SIZE);
    dim3 block(BLOCK_ROWS, BLOCK_COLS);

    forward_kernel_tiled<<<grid, block, shared_mem_size>>>(Q, K, V, O, seq_len_q, seq_len_kv, head_dim, num_heads_q, num_heads_kv);
}
