#include "flash.h"

#define BLOCK_ROWS 2
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
    __half* O_i = l_i + BLOCK_ROWS; 
    
    const __half scale = __float2half(1.0f / sqrtf((float)head_dim));
    
    // ---- 核心索引计算 ----
    // 当前处理的Q头索引和块索引
    const int Q_head_idx = blockIdx.x; 
    const int Q_block_idx = blockIdx.y;
    
    // 计算对应的KV头索引 (Grouped-Query Attention逻辑)
    const int num_q_per_kv = num_heads_q / num_heads_kv;
    const int KV_head_idx = Q_head_idx / num_q_per_kv; // 整数除法

    // 加载Q_i到共享内存时，仅加载有效行
    for (int idx = threadIdx.x; idx < BLOCK_ROWS * head_dim; idx += blockDim.x) {
        const int local_row = idx / head_dim;
        const int local_col = idx % head_dim;
        const int global_row = Q_block_idx * BLOCK_ROWS + local_row;
        if (global_row < seq_len) {   
            const int global_pos = global_row * (num_heads_q * head_dim) + Q_head_idx * head_dim + local_col;
            Q_block[idx] = Q[global_pos];
        }
    } 
    
    // ---- 初始化中间变量 ----
    // 并行初始化 m_i、l_i
    for (int r = threadIdx.x; r < BLOCK_ROWS; r += blockDim.x) {
        m_i[r] = __half(-65504.0f); // -∞
        l_i[r] = __half(0.0f);
    }

    // 并行初始化 O_i
    for (int idx = threadIdx.x; idx < BLOCK_ROWS * head_dim; idx += blockDim.x) {
        O_i[idx] = __half(0.0f);
    }

    __syncthreads();
    
    // ---- 内循环处理所有K_j/V_j块 ----
    for (int j = 0; j < seq_len; j += BLOCK_COLS) {
        
        // ---- 加载K_j到共享内存 ----
        for (int idx = threadIdx.x; idx < BLOCK_COLS * head_dim; idx += blockDim.x) {
            const int local_row = idx / head_dim;
            const int local_col = idx % head_dim;
            if ((j + local_row) < seq_len) {
                const int global_row = j + local_row;
                const int global_pos = global_row * (num_heads_kv * head_dim) + KV_head_idx * head_dim + local_col;
                K_block[idx] = K[global_pos];
                V_block[idx] = V[global_pos];
            }
            //printf("K_block[%d] = %f, V_block[%d] = %f\n", idx, __half2float(K_block[idx]), idx, __half2float(V_block[idx]));
        }

        __syncthreads();
        
        // S_ij = Q_i * K_j^T,每个线程负责 S_ij[i][j] 的点积
        __shared__ __half S_ij[BLOCK_ROWS][BLOCK_COLS];

        // 获取线程在 2D block 中的坐标 (i, j)
        int q_row = threadIdx.x;
        int k_col = threadIdx.y;

        // 确保 (i, j) 在合法范围内
        if (q_row < BLOCK_ROWS && k_col < BLOCK_COLS) {
            __half sum = __half(0.0f);
            for (int k = 0; k < head_dim; ++k) {
                sum = __hadd(sum, __hmul(Q_block[q_row * head_dim + k], K_block[k_col * head_dim + k]));
            }
            // 缩放因子
            sum = __hmul(sum, scale);
            //printf("sum = %f\n", __half2float(sum));
            // 将结果写回共享内存
            S_ij[q_row][k_col] = sum;

            __syncthreads();

            // ---- 行级最大值 m_i^{(j)} ----
        
            __half old_m = m_i[q_row];  // 取上一轮的 m_i^(j-1)
            // 先在当前 block 范围内，找到 S_ij[i][*] 的最大值
            __half localMax = S_ij[q_row][0];
            for (int col = 0; col < BLOCK_COLS; col++) {
                localMax = hmax(localMax, S_ij[q_row][col]);
            }

            m_i[q_row] = hmax(old_m, localMax);    // 与之前的 m_i[i] 做比较，获得新的 m_i^(j)

            __syncthreads();

            //   P̃_i^(j) = exp( S_ij[i][col] - m_i[i] )
            //   l_i^(j) = e^(m_i^(j-1)-m_i^(j)) * l_i^(j-1) + rowsum(P̃_i^(j))
            __half old_l = l_i[q_row];

            //   e^(m_i^(j-1) - m_i^(j))
            __half scale_factor = custom_hexp(__hsub(old_m, m_i[q_row]));
            
            // l_new = e^(m_i^(j-1)-m_i^(j)) * l_i^(j-1)
            __half l_new = __hmul(old_l, scale_factor);

            // 遍历列，计算 P̃_ij 并累加到 l_new
            __half rowsumP = __half(0.0f);
            for (int col = 0; col < BLOCK_COLS; col++) {
                // P_ij = exp(S_ij[i][col] - m_i[i])
                __half p_val = custom_hexp(__hsub(S_ij[q_row][col], m_i[q_row])); 
                S_ij[q_row][col] = p_val;
                rowsumP = __hadd(rowsumP, p_val);
            }

            // l_i^(j) = e^(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P̃_i^(j))
            l_i[q_row] = __hadd(l_new, rowsumP);

            __syncthreads();

            // 在行级 softmax 之后，rowsumP 与 l_i[q_row] 已经更新完毕
            // 在同一个 j 内，对 Oᵢ^(j) 进行更新：

            // 针对每个 q_row 的 head_dim 维度，更新 Oᵢ^(j)
            for (int c = 0; c < head_dim; ++c)
            {
                // 取上一轮的 Oᵢ^(j-1)
                __half old_O_val = O_i[q_row * head_dim + c];

                // 先做 diag(e^(mᵢ^(j-1)-mᵢ^(j)))⁻¹ * Oᵢ^(j-1)
                __half acc = __hmul(old_O_val, scale_factor);

                // 再加 ∑(p_ij * v_j)
                // BLOCK_COLS 个列，每列存储 p_ij = S_ij[q_row][col]
                // V_j 存在 V_block[col * head_dim + c]
                __half sumPV = __half(0.0f);
                for (int col = 0; col < BLOCK_COLS; col++) {
                    __half p_val = S_ij[q_row][col];                // 即 pij
                    __half v_val = V_block[col * head_dim + c];     // V_j 对应列 col
                    sumPV = __hadd(sumPV, __hmul(p_val, v_val));
                }
                __syncthreads();

                // 累加到 acc
                acc = __hadd(acc, sumPV);

                // 写回新的 O_i^(j)
                O_i[q_row * head_dim + c] = acc;
            }

            __syncthreads();
        }
    }
    
    //最终步骤: 对 O_i 做归一化
    // 每个线程处理一个或多个 (row, col)
    __syncthreads(); // 确保内循环计算全部完成

    for (int idx = threadIdx.x; idx < BLOCK_ROWS * head_dim; idx += blockDim.x) {
        int row = idx / head_dim;  // 行索引
        int col = idx % head_dim;  // 列索引
        
        // 确保 row 没超出这个 block 负责的行数
        if (row < BLOCK_ROWS) {
            // O_i[row, col] /= l_i[row]
            __half inv_l = __hdiv(__half(1.0f), l_i[row]); // 1.0 / l_i[row]
            O_i[row * head_dim + col] = __hmul(O_i[row * head_dim + col], inv_l);
        }
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < BLOCK_ROWS * head_dim; idx += blockDim.x) {
        const int local_row = idx / head_dim;
        const int local_col = idx % head_dim;
        const int global_row = Q_block_idx * BLOCK_ROWS + local_row;
        if (global_row < seq_len) {   
            const int global_pos = global_row * (num_heads_q * head_dim) + Q_head_idx * head_dim + local_col;
            O[global_pos] = O_i[local_row * head_dim + local_col];
        }
    } 
}




void flash_attention(const __half* Q,
                     const __half* K,
                     const __half* V,
                     __half* O,
                     const int seq_len,
                     const int head_dim,
                     const int num_heads_q,
                     const int num_heads_kv) 
{
    //size_t shared_mem_size = (BLOCK_ROWS*head_dim +BLOCK_COLS*head_dim*3 +BLOCK_ROWS*2)*sizeof(__half);
    size_t shared_mem_size = (2 * BLOCK_ROWS * head_dim + 2 * BLOCK_COLS * head_dim + 2 * BLOCK_ROWS + BLOCK_ROWS * BLOCK_COLS) * sizeof(__half);

    dim3 grid(num_heads_q, (seq_len + BLOCK_ROWS - 1) / BLOCK_ROWS);
    dim3 block(BLOCK_ROWS, BLOCK_COLS);

    forward_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, O, seq_len, head_dim, num_heads_q, num_heads_kv);
}