#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <iostream>
#include <cassert>
#include "expert.h"
#include "dot_product.h"
#include "linear.h"
#include "gating_network.h"
#include "router.h"
#include "SiLU.h"
#include "softmax.h"

__global__ void Add_kernel(
    const __half* in1,
    const __half* in2,
    __half* out,
    int d_hidden,
    int seq_len,
    int batch_size
){
    // Each thread handles 16 consecutive elements
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * 16;
    int total = batch_size * seq_len * d_hidden;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int idx = start + i;
        if (idx < total) {
            // element-wise add
            out[idx] = __hadd(in1[idx], in2[idx]);
        }
    }
}

void Add(
    const __half* in1,
    const __half* in2,
    __half* out,
    int d_hidden,
    int seq_len,
    int batch_size,
    hipStream_t stream
){
    dim3 block(d_hidden/16,seq_len);
    dim3 grid(batch_size);
    hipLaunchKernelGGL(Add_kernel, grid, block, 0, stream, in1, in2, out, d_hidden, seq_len, batch_size);
}


__global__ void merge_kernel(
    __half* in,
    __half* out,
    int* token_indices,
    int num_tokens,
    int d_hidden)
{
    // 每个线程负责 x 方向上 8 个隐藏单元的累加
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;
    int base_dim = thread_id * 16;  // 该线程处理的第一个隐藏单元索引

    if (token < num_tokens && base_dim < d_hidden) {
        // 对输入和输出的基地址进行预计算
        int out_base = token_indices[token] * d_hidden + base_dim;
        int in_base  = token * d_hidden + base_dim;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int dim = base_dim + i;
            if (dim < d_hidden) {
                // 累加并写回
                float a = __half2float(out[out_base + i]);
                float b = __half2float(in[in_base + i]);
                out[out_base + i] = __float2half(a + b);
            }
        }
    }
}

void merge(
    __half* in,
    __half* out,
    int* token_indices,
    int num_tokens,
    int d_hidden,
    hipStream_t stream
){
    dim3 block(d_hidden/16,num_tokens);
    dim3 grid(1,1);
    hipLaunchKernelGGL(merge_kernel, grid, block, 0, stream, in, out, token_indices, num_tokens, d_hidden);
}

extern "C" void MOE(
    const __half* x,
    __half* h_output,
    const __half* W_up_shared,  // [d_hidden, d_expert * n_shared_experts]
    const __half* W_gate_shared,// [d_hidden, d_expert * n_shared_experts]
    const __half* W_down_shared,// [d_expert * n_shared_experts, d_hidden]
    const __half* W_up_router,  // [d_hidden, d_expert] * n_router_experts
    const __half* W_gate_router,// [d_hidden, d_expert] * n_router_experts
    const __half* W_down_router,// [d_expert * d_hidden] * n_router_experts
    const __half* W_gating_network,// [d_hidden, n_experts]
    int batch_size,
    int seq_len,
    int d_hidden,
    int d_expert,
    int n_shared_experts,
    int n_router_experts,
    int n_experts,
    int topk
){

    hipInit(0);

    __half* d_x, *d_x_router_cache_all, *d_x_output;
    hipMalloc(&d_x, batch_size * seq_len * d_hidden * sizeof(__half));
    hipMalloc(&d_x_router_cache_all, batch_size * seq_len * d_hidden * sizeof(__half));
    hipMalloc(&d_x_output, batch_size * seq_len * d_hidden * sizeof(__half));
    hipMemcpy(d_x, x, batch_size * seq_len * d_hidden * sizeof(__half), hipMemcpyHostToDevice);

    hipEvent_t ev_shared, ev_gating;
    hipEventCreate(&ev_shared);
    hipEventCreate(&ev_gating);


    //shared experts
    hipStream_t stream_shared;
    hipStreamCreate(&stream_shared);

    rocblas_handle handle_shared;
    rocblas_create_handle(&handle_shared);
    rocblas_set_stream(handle_shared, stream_shared);

    __half* d_W_up_shared, *d_W_gate_shared, *d_W_down_shared;
    hipMallocAsync(&d_W_up_shared, d_hidden * d_expert * n_shared_experts * sizeof(__half), stream_shared);
    hipMallocAsync(&d_W_gate_shared, d_hidden * d_expert * n_shared_experts * sizeof(__half), stream_shared);
    hipMallocAsync(&d_W_down_shared, d_expert * d_hidden * n_shared_experts * sizeof(__half), stream_shared);

    __half* d_UP_cache, *d_GATE_cache, *d_DOWN_cache, *d_SiLU_cache;
    hipMallocAsync(&d_UP_cache, batch_size * seq_len * d_expert * n_shared_experts * sizeof(__half), stream_shared);
    hipMallocAsync(&d_GATE_cache, batch_size * seq_len * d_expert * n_shared_experts * sizeof(__half), stream_shared);
    hipMallocAsync(&d_DOWN_cache, batch_size * seq_len * d_expert * n_shared_experts * sizeof(__half), stream_shared);
    hipMallocAsync(&d_SiLU_cache, batch_size * seq_len * d_expert * n_shared_experts * sizeof(__half), stream_shared);

    __half* x_shared_cache;
    hipMallocAsync(&x_shared_cache, batch_size * seq_len * d_hidden * sizeof(__half), stream_shared);

    hipMemcpyAsync(d_x, x, batch_size * seq_len * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream_shared);
    hipMemcpyAsync(d_W_up_shared, W_up_shared, d_hidden * d_expert * n_shared_experts * sizeof(__half), hipMemcpyHostToDevice, stream_shared);
    hipMemcpyAsync(d_W_gate_shared, W_gate_shared, d_hidden * d_expert * n_shared_experts * sizeof(__half), hipMemcpyHostToDevice, stream_shared);
    hipMemcpyAsync(d_W_down_shared, W_down_shared, d_expert * d_hidden * n_shared_experts * sizeof(__half), hipMemcpyHostToDevice, stream_shared);

    expert_shared(d_W_up_shared, d_W_gate_shared, d_W_down_shared, d_x, x_shared_cache, d_UP_cache, d_GATE_cache, d_DOWN_cache, d_SiLU_cache, batch_size, seq_len, d_hidden, d_expert, n_shared_experts, stream_shared, handle_shared);

    rocblas_destroy_handle(handle_shared);
    hipStreamDestroy(stream_shared);
    hipFree(d_W_up_shared);
    hipFree(d_W_gate_shared);
    hipFree(d_W_down_shared);   

    hipEventRecord(ev_shared, stream_shared);
    
    //gating network
    hipStream_t stream_router;
    hipStreamCreate(&stream_router);

    rocblas_handle handle_router;
    rocblas_create_handle(&handle_router);
    rocblas_set_stream(handle_router, stream_router);

    __half* h_topk_scores, *h_topk_indices;
    h_topk_scores = (__half*)malloc(batch_size * seq_len * topk * sizeof(__half));
    h_topk_indices = (__half*)malloc(batch_size * seq_len * topk * sizeof(__half));

    __half* d_W_gating_network;
    hipMallocAsync(&d_W_gating_network, d_hidden * n_experts * sizeof(__half), stream_router);

    __half* d_logits, *d_scores, *d_topk_scores, *d_topk_indices;
    hipMallocAsync(&d_logits, batch_size * seq_len * n_experts * sizeof(__half), stream_router);
    hipMallocAsync(&d_scores, batch_size * seq_len * n_experts * sizeof(__half), stream_router);
    hipMallocAsync(&d_topk_scores, batch_size * seq_len * topk * sizeof(__half), stream_router);
    hipMallocAsync(&d_topk_indices, batch_size * seq_len * topk * sizeof(__half), stream_router);

    hipMemcpyAsync(d_W_gating_network, W_gating_network, d_hidden * n_experts * sizeof(__half), hipMemcpyHostToDevice, stream_router);

    gating_network(d_W_gating_network, d_x, d_logits, d_scores, d_topk_scores, d_topk_indices, batch_size, seq_len, d_hidden, n_experts, topk, stream_router, handle_router);

    hipMemcpyAsync(h_topk_scores, d_topk_scores, batch_size * seq_len * topk * sizeof(__half), hipMemcpyDeviceToHost, stream_router);
    hipMemcpyAsync(h_topk_indices, d_topk_indices, batch_size * seq_len * topk * sizeof(__half), hipMemcpyDeviceToHost, stream_router);
    
    std::vector<Expert> experts = assign_tokens_to_experts(h_topk_scores, h_topk_indices, batch_size, seq_len, topk, n_experts);

    __half* d_W_up_router, *d_W_gate_router, *d_W_down_router;
    hipMallocAsync(&d_W_up_router, d_hidden * d_expert * n_router_experts * sizeof(__half), stream_router);
    hipMallocAsync(&d_W_gate_router, d_hidden * d_expert * n_router_experts * sizeof(__half), stream_router);
    hipMallocAsync(&d_W_down_router, d_expert * d_hidden * n_router_experts * sizeof(__half), stream_router);

    hipMemcpyAsync(d_W_up_router, W_up_router, d_hidden * d_expert * n_router_experts * sizeof(__half), hipMemcpyHostToDevice, stream_router);
    hipMemcpyAsync(d_W_gate_router, W_gate_router, d_hidden * d_expert * n_router_experts * sizeof(__half), hipMemcpyHostToDevice, stream_router);
    hipMemcpyAsync(d_W_down_router, W_down_router, d_expert * d_hidden * n_router_experts * sizeof(__half), hipMemcpyHostToDevice, stream_router);

    rocblas_destroy_handle(handle_router);
    hipStreamDestroy(stream_router);
    hipFree(d_W_gate_router);
    hipFree(d_logits);
    hipFree(d_scores);
    hipFree(d_topk_scores);
    hipFree(d_topk_indices);
    free(h_topk_scores);
    free(h_topk_indices);

    std::vector<rocblas_handle> handles;
    std::vector<hipStream_t> streams;
    std::vector<hipEvent_t> events;
    std::vector<__half*> dx_routers, dx_router_cache, dUP_cache, dGATE_cache, dDOWN_cache, dSiLU_cache, dW_up_expert, dW_gate_expert, dW_down_expert, dscores;
    std::vector<int*> d_token_indices_list;
    std::vector<int>   num_tokens_list;

    hipEventRecord(ev_gating, stream_router);

    hipStreamWaitEvent(stream_router, ev_gating, 0);
    //router_experts
    for (auto &asn : experts) {
        int expert_id = asn.id;
        int num_tokens = asn.token_indices.size();
        std::vector<int> tokens_indices = asn.token_indices;
        std::vector<__half> scores = asn.scores;

        hipEvent_t ev_ex;
        hipEventCreate(&ev_ex);
        hipStream_t stream_ex;
        hipStreamCreate(&stream_ex);
        rocblas_handle handle_ex;
        rocblas_create_handle(&handle_ex);
        rocblas_set_stream(handle_ex, stream_ex);

        streams.push_back(stream_ex);
        handles.push_back(handle_ex);
        events.push_back(ev_ex);

        // Copy token indices to device for merge
        int* d_token_indices;
        hipMalloc(&d_token_indices, num_tokens * sizeof(int));
        hipMemcpyAsync(d_token_indices, tokens_indices.data(), num_tokens * sizeof(int), hipMemcpyHostToDevice, stream_ex);
        d_token_indices_list.push_back(d_token_indices);
        num_tokens_list.push_back(num_tokens);
        
        __half* d_x_router;
        hipMallocAsync(&d_x_router, num_tokens * d_hidden * sizeof(__half), stream_ex);
        // Gather the selected tokens from the device input d_x into x_router_cache
        for (int i = 0; i < num_tokens; ++i) {
            int idx = tokens_indices[i];
            hipMemcpyAsync(
                d_x_router + static_cast<size_t>(i) * d_hidden,
                d_x + static_cast<size_t>(idx) * d_hidden,
                d_hidden * sizeof(__half),
                hipMemcpyDeviceToDevice,
                stream_ex
            );
        }
        dx_routers.push_back(d_x_router);

        __half* d_x_router_cache, *d_UP_cache, *d_GATE_cache, *d_DOWN_cache, *d_SiLU_cache;
        hipMallocAsync(&d_x_router_cache, num_tokens * d_hidden * sizeof(__half), stream_ex);
        hipMallocAsync(&d_UP_cache, num_tokens * d_expert * sizeof(__half), stream_ex);
        hipMallocAsync(&d_GATE_cache, num_tokens * d_expert * sizeof(__half), stream_ex);
        hipMallocAsync(&d_DOWN_cache, num_tokens * d_expert * sizeof(__half), stream_ex);
        hipMallocAsync(&d_SiLU_cache, num_tokens * d_expert * sizeof(__half), stream_ex);

        dx_router_cache.push_back(d_x_router_cache);
        dUP_cache.push_back(d_UP_cache);
        dGATE_cache.push_back(d_GATE_cache);
        dDOWN_cache.push_back(d_DOWN_cache);
        dSiLU_cache.push_back(d_SiLU_cache);

        __half* d_W_up_expert, *d_W_gate_expert, *d_W_down_expert, * d_scores;
        hipMallocAsync(&d_W_up_expert, d_hidden * d_expert * sizeof(__half), stream_ex);
        hipMallocAsync(&d_W_gate_expert, d_hidden * d_expert * sizeof(__half), stream_ex);
        hipMallocAsync(&d_W_down_expert, d_expert * d_hidden * sizeof(__half), stream_ex);
        hipMallocAsync(&d_scores, num_tokens * sizeof(__half), stream_ex);

        dW_up_expert.push_back(d_W_up_expert);
        dW_gate_expert.push_back(d_W_gate_expert);
        dW_down_expert.push_back(d_W_down_expert);
        dscores.push_back(d_scores);
        hipMemcpyAsync(d_W_up_expert, d_W_up_router + expert_id * d_hidden * d_expert, d_hidden * d_expert * sizeof(__half), hipMemcpyDeviceToDevice, stream_ex);
        hipMemcpyAsync(d_W_gate_expert, d_W_gate_router + expert_id * d_hidden * d_expert, d_hidden * d_expert * sizeof(__half), hipMemcpyDeviceToDevice, stream_ex);
        hipMemcpyAsync(d_W_down_expert, d_W_down_router + expert_id * d_expert * d_hidden, d_expert * d_hidden * sizeof(__half), hipMemcpyDeviceToDevice, stream_ex);
        hipMemcpyAsync(d_scores, scores.data(), num_tokens * sizeof(__half), hipMemcpyHostToDevice, stream_ex);
        // hipMemcpyAsync(d_UP_cache, d_UP_cache, num_tokens * d_expert * sizeof(__half), hipMemcpyDeviceToDevice, stream_ex);
        // hipMemcpyAsync(d_GATE_cache, d_GATE_cache, num_tokens * d_expert * sizeof(__half), hipMemcpyDeviceToDevice, stream_ex);
        // hipMemcpyAsync(d_DOWN_cache, d_DOWN_cache, num_tokens * d_expert * sizeof(__half), hipMemcpyDeviceToDevice, stream_ex);
        // hipMemcpyAsync(d_SiLU_cache, d_SiLU_cache, num_tokens * d_expert * sizeof(__half), hipMemcpyDeviceToDevice, stream_ex);

        expert_router(d_W_up_expert, d_W_gate_expert, d_W_down_expert, d_scores, d_x_router, d_x_router_cache, d_UP_cache, d_GATE_cache, d_DOWN_cache, d_SiLU_cache, num_tokens, d_hidden, d_expert, stream_ex, handle_ex);

        merge(d_x_router_cache, d_x_router_cache_all, d_token_indices, num_tokens, d_hidden, stream_ex);
    }

    for (int i = 0; i < experts.size(); ++i) {
        rocblas_destroy_handle(handles[i]);
        hipStreamDestroy(streams[i]);
        hipFree(dx_routers[i]);
        //hipFree(dx_router_cache[i]); 这块先不释放，因为后面还要用
        hipFree(dUP_cache[i]);
        hipFree(dGATE_cache[i]);
        hipFree(dDOWN_cache[i]);
        hipFree(dSiLU_cache[i]);
        hipFree(dW_up_expert[i]);
        hipFree(dW_gate_expert[i]);
        hipFree(dW_down_expert[i]);
        hipFree(dscores[i]);
        hipEventRecord(events[i], streams[i]);
    }

    // Start Generation Here: wait for all experts to finish
    for (auto evt : events) {
        hipEventSynchronize(evt);
    }
    hipEventSynchronize(ev_shared);
    // End Generation Here

    Add(x_shared_cache, d_x_router_cache_all, d_x_output, d_hidden, seq_len, batch_size, stream_shared);

    hipMemcpyAsync(h_output, d_x_output, batch_size * seq_len * d_hidden * sizeof(__half), hipMemcpyDeviceToHost, stream_shared);

    // Ensure output copy completes before freeing
    hipStreamSynchronize(stream_shared);
    // Free intermediate device buffers
    hipFree(d_x);
    hipFree(x_shared_cache);
    hipFree(d_x_router_cache_all);
    hipFree(d_x_output);
    hipFree(d_W_up_router);
    hipFree(d_W_down_router);
    // Free per-expert merge caches
    for (auto ptr : dx_router_cache) {
        hipFree(ptr);
    }
    // Free token index buffers
    for (auto ptr : d_token_indices_list) {
        hipFree(ptr);
    }

    hipEventDestroy(ev_shared);
    hipEventDestroy(ev_gating);
}


int main() {
    // 测试参数
    int batch_size = 2;
    int seq_len = 4;
    int d_hidden = 16;
    int d_expert = 8;
    int n_shared_experts = 2;
    int n_router_experts = 2;
    int n_experts = n_shared_experts + n_router_experts;
    int topk = 2;

    // 分配并初始化输入和权重
    std::vector<__half> x(batch_size * seq_len * d_hidden, __float2half(1.0f));
    std::vector<__half> h_output(batch_size * seq_len * d_hidden, __float2half(0.0f));
    std::vector<__half> W_up_shared(d_hidden * d_expert * n_shared_experts, __float2half(0.1f));
    std::vector<__half> W_gate_shared(d_hidden * d_expert * n_shared_experts, __float2half(0.2f));
    std::vector<__half> W_down_shared(d_expert * d_hidden * n_shared_experts, __float2half(0.3f));
    std::vector<__half> W_up_router(d_hidden * d_expert * n_router_experts, __float2half(0.4f));
    std::vector<__half> W_gate_router(d_hidden * n_experts, __float2half(0.5f));
    std::vector<__half> W_down_router(d_expert * d_hidden * n_router_experts, __float2half(0.6f));
    std::vector<__half> W_gating_network(d_hidden * n_experts, __float2half(0.7f));

    // 调用MOE函数
    MOE(
        x.data(),
        h_output.data(),
        W_up_shared.data(),
        W_gate_shared.data(),
        W_down_shared.data(),
        W_up_router.data(),
        W_gate_router.data(),
        W_down_router.data(),
        W_gating_network.data(),
        batch_size,
        seq_len,
        d_hidden,
        d_expert,
        n_shared_experts,
        n_router_experts,
        n_experts,
        topk
    );

    // 打印部分输出结果
    std::cout << "h_output 部分结果: ";
    for (int i = 0; i < std::min(10, (int)h_output.size()); ++i) {
        std::cout << __half2float(h_output[i]) << " ";
    }
    std::cout << std::endl;

    return 0;
}







