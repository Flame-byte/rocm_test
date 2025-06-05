// ... existing code up to expert_shared 调用 ...

    expert_shared(
        d_W_up_shared, d_W_gate_shared, d_W_down_shared,
        d_x, x_shared_cache,
        batch_size, seq_len, d_hidden, d_expert, n_shared_experts,
        stream_shared, handle_shared
    );

// 1) 标记 shared 流结束
    hipEvent_t event_shared;
    hipEventCreate(&event_shared);
    hipEventRecord(event_shared, stream_shared);

// 2) 等待 expert_shared 真正跑完
    hipEventSynchronize(event_shared);
// 或者直接用：
//    hipStreamSynchronize(stream_shared);

// 3) 释放所有中间资源，保留 x_shared_cache
    // 3.a) 先销毁 handle/stream/event
    rocblas_destroy_handle(handle_shared);
    hipStreamDestroy(stream_shared);
    hipEventDestroy(event_shared);

    // 3.b) 释放输入、权重等 buffer
    hipFree(d_x);
    hipFree(d_W_up_shared);
    hipFree(d_W_gate_shared);
    hipFree(d_W_down_shared);
//    // 注意：不要释放 x_shared_cache，它是你要保留的数据

// ... 下面继续搭建 gating 和 router 的逻辑 ... 