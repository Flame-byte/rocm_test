// Wave Matrix Multiply Accumulate (WMMA) using HIP compiler intrinsic
// Does a matrix multiplication of two 16x16, fp16 matrices, and stores them into a 16x16 fp16 result matrix

#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using namespace std;

#define WAVE_SIZE 32
#define T_BLOCK_X 4*WAVE_SIZE
#define T_BLOCK_Y 4

#define X_direction_prefill T_BLOCK_X/2 //64
#define X_direction_decoder T_BLOCK_X/8 //16
#define Y_direction T_BLOCK_Y*4

// Use half16 as an alias of the internal clang vector type of 16 fp16 values
typedef _Float16 half16 __attribute__((ext_vector_type(16)));

typedef uint16_t bf16x16 __attribute__((ext_vector_type(16)));


// __global__ void flash_attention_mma_kernel(
//                 hip_bfloat16* Q,
//                 hip_bfloat16* K,
//                 hip_bfloat16* V,
//                 hip_bfloat16* O,
//                 int batch_size,
//                 int seq_len_q,
//                 int seq_len_kv,
//                 int n_heads,
//                 int qk_nope_head_dim,
//                 int qk_rope_head_dim,
//                 int v_head_dim,
//                 int X_direction_dim
// )
// {
//     extern __shared__ hip_bfloat16 Q_block[];

//     int head_idx = num_heads * blockIdx.y + blockIdx.x;
//     int Q_global_offset = head_idx * (qk_nope_head_dim + qk_rope_head_dim) * seq_len_q;
//     int K_global_offset = head_idx * (qk_nope_head_dim + qk_rope_head_dim) * seq_len_kv;
//     int V_global_offset = head_idx * v_head_dim * seq_len_kv;

//     for(int Q_loop_idx = 0; Q_loop_idx < seq_len_q; Q_loop_idx += X_direction_dim)
//     {
//         int Q_local_idx = Q_loop_idx * (qk_nope_head_dim + qk_rope_head_dim);
//         for(int i = 0; i < (qk_nope_head_dim + qk_rope_head_dim); i += 64)
//         {
//             int idx = threadIdx.y * blockDim.x + threadIdx.x;
//             int x = idx % 8;   //0-7
//             int y = idx / 8;   //0-63 or 0-15

//             Q_block[idx] = Q[Q_global_offset + Q_local_idx + i*64 + y*64 + x*8];
//             bf16x8 Q_transpose;
//             int idx = threadIdx.y * blockDim.x + threadIdx.x;
//             int x = idx % 8;   //0-7
//             int y = idx / 8;   //0-63 or 0-15

//             int col = i + y*64 + x*8;
//             int row = y;

//             Q_transpose = reinterpret_cast<bf16x8>(Q[Q_global_offset + Q_local_idx + col]);

//             Q_block[(col+0) * X_direction_dim + row] = Q_transpose[0];
//             Q_block[(col+1) * X_direction_dim + row] = Q_transpose[1];
//             Q_block[(col+2) * X_direction_dim + row] = Q_transpose[2];
//             Q_block[(col+3) * X_direction_dim + row] = Q_transpose[3];
//             Q_block[(col+4) * X_direction_dim + row] = Q_transpose[4];
//             Q_block[(col+5) * X_direction_dim + row] = Q_transpose[5];
//             Q_block[(col+6) * X_direction_dim + row] = Q_transpose[6];
//             Q_block[(col+7) * X_direction_dim + row] = Q_transpose[7];
            
//             for(int t = 0; t < 8; ++t) {
//                 int col = i + y*64 + x*8 + t;  // original column index within this block
//                 int row = y;                   // sub-row index within the X_direction_dim chunk
//                 // store as column-major for WMMA (A[col, row] => Q_block[col*ld + row])
//                 Q_block[col * X_direction_dim + row] = Q_transpose[t];
//             }
//         }
//     }

    
// }
// __global__ void flash_attention_mma_kernel(
//                 hip_bfloat16* Q,
//                 hip_bfloat16* K,
//                 hip_bfloat16* V,
//                 hip_bfloat16* O,
//                 int batch_size,
//                 int seq_len_q,
//                 int seq_len_kv,
//                 int n_heads,
//                 int qk_nope_head_dim,
//                 int qk_rope_head_dim,
//                 int v_head_dim,
//                 int X_direction_dim
// )
// {
//     extern __shared__ hip_bfloat16 Q_block[];

//     int head_idx = num_heads * blockIdx.y + blockIdx.x;
//     int Q_global_offset = head_idx * (qk_nope_head_dim + qk_rope_head_dim) * seq_len_q;
//     int K_global_offset = head_idx * (qk_nope_head_dim + qk_rope_head_dim) * seq_len_kv;
//     int V_global_offset = head_idx * v_head_dim * seq_len_kv;

//     for(int Q_loop_idx = 0; Q_loop_idx < seq_len_q; Q_loop_idx += X_direction_dim)
//     {
//         int Q_local_idx = Q_loop_idx * (qk_nope_head_dim + qk_rope_head_dim);
//         for(int i = 0; i < (qk_nope_head_dim + qk_rope_head_dim); i += 64)
//         {
//             int idx = threadIdx.y * blockDim.x + threadIdx.x;
//             int x = idx % 8;   //0-7
//             int y = idx / 8;   //0-63 or 0-15

//             Q_block[idx] = Q[Q_global_offset + Q_local_idx + i*64 + y*64 + x*8];
//             // bf16x8 Q_transpose;
//             // int idx = threadIdx.y * blockDim.x + threadIdx.x;
//             // int x = idx % 8;   //0-7
//             // int y = idx / 8;   //0-63 or 0-15

//             // int col = i + y*64 + x*8;
//             // int row = y;

//             // Q_transpose = reinterpret_cast<bf16x8>(Q[Q_global_offset + Q_local_idx + col]);

//             // Q_block[(col+0) * X_direction_dim + row] = Q_transpose[0];
//             // Q_block[(col+1) * X_direction_dim + row] = Q_transpose[1];
//             // Q_block[(col+2) * X_direction_dim + row] = Q_transpose[2];
//             // Q_block[(col+3) * X_direction_dim + row] = Q_transpose[3];
//             // Q_block[(col+4) * X_direction_dim + row] = Q_transpose[4];
//             // Q_block[(col+5) * X_direction_dim + row] = Q_transpose[5];
//             // Q_block[(col+6) * X_direction_dim + row] = Q_transpose[6];
//             // Q_block[(col+7) * X_direction_dim + row] = Q_transpose[7];
            
//             // for(int t = 0; t < 8; ++t) {
//             //     int col = i + y*64 + x*8 + t;  // original column index within this block
//             //     int row = y;                   // sub-row index within the X_direction_dim chunk
//             //     // store as column-major for WMMA (A[col, row] => Q_block[col*ld + row])
//             //     Q_block[col * X_direction_dim + row] = Q_transpose[t];
//             // }
//         }
//     }

    
// }

__device__ __hip_bfloat16 bfmax(__hip_bfloat16 a, __hip_bfloat16 b) {
    return __hgt(a, b) ? a : b;
}

__device__ __hip_bfloat16 custom_bfexp(__hip_bfloat16 x) {
    return hexp(x);
}

__global__ void flash_attention_mma_kernel(
                __hip_bfloat16* Q,
                __hip_bfloat16* K,
                __hip_bfloat16* V,
                __hip_bfloat16* O,
                int seq_len_q,
                int seq_len_kv,
                int n_heads,
                int qk_nope_head_dim,
                int qk_rope_head_dim,
                int v_head_dim,
                int X_direction_dim
)
{
    extern __shared__ __hip_bfloat16 Q_block[];   //X_direction(64or16)*head_dim
    __hip_bfloat16* S_block = Q_block + X_direction_dim * head_dim;  //(64or16)*(64or16)
    __hip_bfloat16* m_i = S_block + X_direction_dim * X_direction_dim;  //seq_len_q
    __hip_bfloat16* l_i = m_i + seq_len_q;  //seq_len_q
    __hip_bfloat16* o_i = l_i + seq_len_q;  //X_direction(64or16)*head_dim

    int head_idx = blockIdx.x;
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    int hidden_dim_Q = (qk_nope_head_dim + qk_rope_head_dim) * n_heads;
    int hidden_dim_K = (qk_nope_head_dim + qk_rope_head_dim) * n_heads;
    int hidden_dim_V = v_head_dim * n_heads;
    int Q_global_offset = head_idx * (qk_nope_head_dim + qk_rope_head_dim);
    int K_global_offset = head_idx * (qk_nope_head_dim + qk_rope_head_dim);
    int V_global_offset = head_idx * v_head_dim;

    for(int Q_loop_idx = 0; Q_loop_idx < seq_len_q; Q_loop_idx += X_direction_dim) //16or64 outloop
    {
        // int Q_local_idx = Q_loop_idx * Q_global_offset;
        for(int i = 0; i < (qk_nope_head_dim + qk_rope_head_dim); i += 64)
        {
            int Q_block_idx_x = Q_global_offset + i;
            int Q_block_idx_y = Q_loop_idx;
            int x = idx % 8;   //0-7
            int y = idx / 8;   //0-63 or 0-15

            #pragma unroll 8
            for(int t = 0; t < 8; ++t)
            {
                int col = Q_block_idx_x + x + t*8;
                int row = Q_block_idx_y + y;

                Q_block[y * (qk_nope_head_dim + qk_rope_head_dim) + i + (x + t*8)] = Q[row * hidden_dim_Q + col];
            }
            
        }

        __syncthreads();

        for(int KV_loop_idx = 0; KV_loop_idx < seq_len_kv; KV_loop_idx += X_direction_dim) //inner loop
        {
            bf16x16 frag_Q;
            bf16x16 frag_K;
            bf16x16 frag_S;
            bf16x16 frag_V;

            int warp_idx_x = threadIdx.x/32;
            int warp_idx_y = threadIdx.y;
            int threadIdx_in_warp = idx/16;
            int lane = threadIdx_in_warp%16;

            for(int j = 0; j < (qk_nope_head_dim + qk_rope_head_dim); j += 64)
            {
                for (int ele = 0; ele < 16; ++ele)
                {
                    __hip_bfloat16_raw bq = Q_block[(warp_idx_y * 16 + lane) * hidden_dim_Q + 
                                                    (j + warp_idx_x * 16) + ele];
                    frag_Q[ele] = (uint16_t)bq.x;

                    int K_block_idx_x = K_global_offset + j;
                    int K_block_idx_y = KV_loop_idx;

                    __hip_bfloat16_raw bk = K[(K_block_idx_y + warp_idx_y * 16 + lane) * hidden_dim_Q + 
                                              (K_global_offset + j + warp_idx_x * 16) + ele];
                    frag_K[ele] = (uint16_t)bk.x;
                }

                frag_S = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(frag_Q, frag_K, frag_S, false);

                for(int ele = 0; ele < 8; ++ele)
                {   
                    int S_block_idx_x = j + warp_idx_x * 16;
                    int S_block_idx_y = warp_idx_y * 16 + ele * 2 + idx/(X_direction_dim*4);
                    __hip_bfloat16_raw raw = (uint16_t)frag_S[2 * ele];
                    __hip_bfloat16_raw bs{raw};
                    S_block[S_block_idx_y * seq_len_kv + S_block_idx_x + lane] = __hip_bfloat16(bs);
                }
            }

            __syncthreads();

            __hip_bfloat16 max_old = 0;
            __hip_bfloat16 max_new = 0;
            __hip_bfloat16 sum_old = 0;
            __hip_bfloat16 sum_new = 0;
            if(idx >= 0 && idx < X_direction_dim)
            {   
                max_old = m_i[Q_loop_idx + idx];
                sum_old = l_i[Q_loop_idx + idx];
                for(int cnt = 0; cnt < 64; cnt++)
                {
                    max_new = bfmax(S_block[idx * X_direction_dim + cnt], max_old);

                    __hip_bfloat16 old = __hmul(sum_old, custom_bfexp(__hsub(max_old, max_new)));
                    __hip_bfloat16 new = custom_bfexp(__hsub(S_block[idx * X_direction_dim + cnt], max_new));
                    sum_new = __hadd(old, new);

                }
                m_i[Q_loop_idx + idx] = max_new;
                l_i[Q_loop_idx + idx] = sum_new;
            }

            bf16x16 frag_O_old;
            bf16x16 frag_O_new;
            for(int k = 0; k < v_head_dim; k += 64)
            {
                for (int ele = 0; ele < 16; ++ele)
                {
                    int V_block_idx_x = V_global_offset + k;
                    int V_block_idx_y = KV_loop_idx;

                    __hip_bfloat16_raw bk = K[(V_block_idx_y + warp_idx_y * 16 + ele) * hidden_dim_Q + 
                                              (V_block_idx_x + warp_idx_x * 16) + lane];
                    frag_V[ele] = (uint16_t)bk.x;

                    __hip_bfloat16_raw bs =  S_block[(warp_idx_y * 16 + lane) * X_direction_dim + 
                                                    (warp_idx_x * 16 + ele)];
                    //frag_S[ele] = __hdiv(custom_bfexp(__hsub((uint16_t)bs.x, m_i[idx/(num_threads/X_direction_dim)])), l_i[idx/(num_threads/X_direction_dim)]);
                    frag_S[ele] = (uint16_t)__hdiv(custom_bfexp(__hsub((uint16_t)bs.x, max_new)), sum_new);

                    __hip_bfloat16_raw bo = o_i[(warp_idx_y * 16 + lane) * v_head_dim +
                                                (k + warp_idx_x * 16 + ele)];
                    frag_O_old[ele] = (uint16_t)__hmul(bo.x,__hmul(__hdiv(sum_old, sum_new), custom_bfexp(__hsub(max_old, max_new))));
                }

                frag_O_new = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(frag_S, frag_V, frag_O_new, true);

                for(int ele = 0; ele < 8; ++ele)
                {
                    int o_block_idx_x = k + warp_idx_x * 16;
                    int o_block_idx_y = warp_idx_y * 16 + ele * 2 + idx/(X_direction_dim*4);
                    __hip_bfloat16_raw raw = (uint16_t)frag_O_new[2 * ele];
                    __hip_bfloat16_raw bs{raw};
                }
 
            }
        }

        __syncthreads();

    }
}

// __global__ void flash_attention_mma_kernel(
//                 hip_bfloat16* Q,
//                 hip_bfloat16* K,
//                 hip_bfloat16* V,
//                 hip_bfloat16* O,
//                 int seq_len_q,
//                 int seq_len_kv,
//                 int n_heads,
//                 int qk_nope_head_dim,
//                 int qk_rope_head_dim,
//                 int v_head_dim,
//                 int X_direction_dim
// )
// {
//     extern __shared__ hip_bfloat16 Q_block[];   //X_direction(64or16)*head_dim
//     int head_idx = blockIdx.x;
//     int hidden_dim_Q = (qk_nope_head_dim + qk_rope_head_dim) * n_heads;
//     int hidden_dim_K = (qk_nope_head_dim + qk_rope_head_dim) * n_heads;
//     int hidden_dim_V = v_head_dim * n_heads;
//     int Q_global_offset = head_idx * (qk_nope_head_dim + qk_rope_head_dim);
//     int K_global_offset = head_idx * (qk_nope_head_dim + qk_rope_head_dim);
//     int V_global_offset = head_idx * v_head_dim;

//     for(int Q_loop_idx = 0; Q_loop_idx < seq_len_q; Q_loop_idx += X_direction_dim) //16or64
//     {
//         // int Q_local_idx = Q_loop_idx * Q_global_offset;
//         for(int i = 0; i < (qk_nope_head_dim + qk_rope_head_dim); i += 64)
//         {
//             int Q_block_idx_x = Q_global_offset + i;
//             int Q_block_idx_y = Q_loop_idx;
//             int idx = threadIdx.y * blockDim.x + threadIdx.x;
//             int warp_idx_x = threadIdx.x/32;
//             int warp_idx_y = threadIdx.y;
//             int threadIdx_in_warp = idx/32;
//             int lane = threadIdx_in_warp%16;

//             bf16x16 frag_Q;
//             bf16x16 frag_K;

//             for(int ele = 0; ele < 16; ++ele)
//             {
//                 __hip_bfloat16_raw qr = Q[(Q_block_idx_y + warp_idx_y * 16 + lane) * hidden_dim_Q + 
//                                            Q_block_idx_x + warp_idx_x * 16 + ele];
//                 __hip_bfloat16_raw kr = K[(Q_block_idx_y + warp_idx_y * 16 + lane) * hidden_dim_Q + 
//                                            Q_block_idx_x + warp_idx_x * 16 + ele];
//                 frag_Q[ele] = (uint16_t)br.x;

//             }

//             // #pragma unroll 8
//             // for(int t = 0; t < 8; ++t)
//             // {
//             //     int col = Q_block_idx_x + x + t*8;
//             //     int row = Q_block_idx_y + y;

//             //     Q_block[y * (qk_nope_head_dim + qk_rope_head_dim) + i + (x + t*8)] = Q[row * hidden_dim_Q + col];
//             // }
            
//         }

//     }
// }

void flash_attention_mma(
        hip_bfloat16* Q,
        hip_bfloat16* K,
        hip_bfloat16* V,
        hip_bfloat16* O,
        int batch_size,
        int seq_len_q,
        int seq_len_kv,
        int n_heads,
        int qk_nope_head_dim,
        int qk_rope_head_dim,
        int v_head_dim
)
{

    if(seq_len_q <= 15) //decoder
    {
        dim3 block(T_BLOCK_X, T_BLOCK_Y);
        dim3 grid(n_heads);

        flash_attention_mma_kernel<<<grid, block, X_direction_decoder * (qk_nope_head_dim + qk_rope_head_dim) * sizeof(hip_bfloat16)>>>(Q, K, V, O, batch_size, seq_len_q, seq_len_kv, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, X_direction_decoder);
    }
    else    //prefill
    {
        dim3 block(T_BLOCK_X/4, T_BLOCK_Y);
        dim3 grid(n_heads, batch_size);

        flash_attention_mma_kernel<<<grid, block, X_direction_prefill * (qk_nope_head_dim + qk_rope_head_dim) * sizeof(hip_bfloat16)>>>(Q, K, V, O, batch_size, seq_len_q, seq_len_kv, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, X_direction_prefill);
    }

}



__global__ void wmma_matmul(__half* a, __half* b, __half* c)
{
    const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lIdx = threadIdx.x;

    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b
    // a_frag will store one column of the 16x16 matrix A tile
    // b_frag will store one row of the 16x16 matrix B tile
    half16 a_frag;
    half16 b_frag;
    // initialize c fragment to 0
    half16 c_frag = {};

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    const int lane = lIdx % 16;

    for (int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16*ele + lane];
    }

    for (int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * lane + ele];
    }

    // call the WMMA intrinsic with OPSEL set to "false"
    c_frag = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(a_frag, b_frag, c_frag, false);

    for (int ele = 0; ele < 8; ++ele)
    {
        const int r = ele * 2 + (lIdx / 16);
        // store results from unpacked c_frag output
        c[16 * r + lane] = c_frag[ele*2];
        // if OPSEL was set to "true", the line above would instead be
        // c[16 * r + lane] = c_frag[ele*2 + 1];
    }

}

int main(int argc, char* argv[])

{
    __half a[16 * 16] = {};
    __half b[16 * 16] = {};
    __half c[16 * 16] = {};
    __half *a_gpu, *b_gpu, *c_gpu;
    hipMalloc(&a_gpu, 16*16 * sizeof(__half));
    hipMalloc(&b_gpu, 16*16 * sizeof(__half));
    hipMalloc(&c_gpu, 16*16 * sizeof(__half));

    // fill in some data into matrices A and B
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            a[i * 16 + j] = (__half)1.f;
            b[i * 16 + j] = (__half)1.f;
        }
    }

    hipMemcpy(a_gpu, a, (16*16) * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(b_gpu, b, (16*16) * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(c_gpu, c, (16*16) * sizeof(__half), hipMemcpyHostToDevice);

    wmma_matmul<<<dim3(1), dim3(32, 1, 1), 0, 0>>>(a_gpu, b_gpu, c_gpu);

    hipMemcpy(c, c_gpu, (16 * 16) * sizeof(__half), hipMemcpyDeviceToHost);

    hipFree(a_gpu);
    hipFree(b_gpu);
    hipFree(c_gpu);

    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            printf("%f ", (float)c[i * 16 + j]);
        }
        printf("\\n");
    }

    return 0;
}

