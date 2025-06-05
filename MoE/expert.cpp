#include "expert.h"
#include "linear.h"
#include <hip/hip_runtime.h> 
#include "SiLU.h"
#include "dot_product.h"

extern "C" void expert_shared(
    __half* W_up,
    __half* W_gate,
    __half* W_down,
    __half* a,  //input(batch_size, seq_len, d_hidden)
    __half* b,  //output(batch_size, seq_len, d_hidden)
    __half* UP_cache,
    __half* GATE_cache,
    __half* DOWN_cache,
    __half* SiLU_cache,
    int batch_size,
    int seq_len,
    int d_hidden,
    int d_expert,
    int n_experts,
    hipStream_t stream,
    rocblas_handle handle
)
{
    // // Allocate memory on device
    // __half* d_W_up, *d_W_gate, *d_W_down;
    // __half* d_a, *d_b;
    // hipMalloc(&d_W_up, d_expert * d_hidden * n_experts * sizeof(__half));       //W_up(d_hidden, d_expert*n_experts)
    // hipMalloc(&d_W_gate, d_expert * d_hidden * n_experts * sizeof(__half));     //W_gate(d_hidden, d_expert*n_experts)
    // hipMalloc(&d_W_down, d_expert * d_hidden * n_experts * sizeof(__half));     //W_down(d_expert*n_experts, d_hidden)
    // hipMalloc(&d_a, batch_size * seq_len * d_hidden * sizeof(__half));           //input(batch_size, seq_len, d_hidden)
    // hipMalloc(&d_b, batch_size * seq_len * d_hidden * sizeof(__half));           //output(batch_size, seq_len, d_hidden)

    // __half* UP, *GATE, *DOWN, *SiLU;
    // hipMalloc(&UP, batch_size * seq_len * d_expert * n_experts * sizeof(__half));
    // hipMalloc(&GATE, batch_size * seq_len * d_expert * n_experts * sizeof(__half));
    // hipMalloc(&DOWN, batch_size * seq_len * d_hidden * n_experts * sizeof(__half));
    // hipMalloc(&SiLU, batch_size * seq_len * d_hidden * n_experts * sizeof(__half));

    // rocblas_handle handle1;
    // rocblas_handle handle2;
    // rocblas_create_handle(&handle1);
    // rocblas_create_handle(&handle2);

    // rocblas_set_stream(handle1, stream1);
    // rocblas_set_stream(handle2, stream2);

    // // Copy data to device
    // hipMemcpyAsync(d_W_up, h_W_up, d_expert * d_hidden * n_experts * sizeof(__half), hipMemcpyHostToDevice, stream1);
    // hipMemcpyAsync(d_W_gate, h_W_gate, d_expert * d_hidden * n_experts * sizeof(__half), hipMemcpyHostToDevice, stream1);
    // hipMemcpyAsync(d_W_down, h_W_down, d_expert * d_hidden * n_experts * sizeof(__half), hipMemcpyHostToDevice, stream1);
    // hipMemcpyAsync(d_a, a, batch_size * seq_len * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1); 

    // hipEvent_t event_up, event_silu;
    // hipEventCreate(&event_up);
    // hipEventCreate(&event_silu);
    
    // Launch kernels
    linear(a, W_up, UP_cache, batch_size, seq_len, d_hidden, d_expert*n_experts, handle);
    // hipEventRecord(event_up, stream1);

    linear(a, W_gate, GATE_cache, batch_size, seq_len, d_hidden, d_expert*n_experts, handle);
    silu_activation(GATE_cache, SiLU_cache, batch_size*seq_len, d_expert*n_experts, stream);
    // hipEventRecord(event_silu, stream2);

    // hipStreamWaitEvent(stream2, event_up, 0);
    // hipStreamWaitEvent(stream2, event_silu, 0);

    dot_product(SiLU_cache, UP_cache, DOWN_cache, batch_size*seq_len*d_expert*n_experts, stream);

    // hipEventDestroy(event_up);
    // hipEventDestroy(event_silu);

    linear(DOWN_cache, W_down, b, batch_size, seq_len, d_hidden, d_expert*n_experts, handle);

    // hipMemcpy(b, d_b, batch_size * seq_len * d_hidden * sizeof(__half), hipMemcpyDeviceToHost);

    // hipFree(d_W_up);
    // hipFree(d_W_gate);
    // hipFree(d_W_down);
    // hipFree(d_a);
    // hipFree(d_b);
    // hipFree(UP);
    // hipFree(GATE);
    // hipFree(DOWN);
    // hipFree(SiLU);

    // rocblas_destroy_handle(handle1);
    // rocblas_destroy_handle(handle2);

}

extern "C" void expert_router(
    __half* W_up,
    __half* W_gate,
    __half* W_down,
    __half* scores,
    __half* a,  //input(seq_len, d_hidden)
    __half* b,  //output(seq_len, d_hidden)
    __half* UP_cache,   //(seq_len, d_expert)
    __half* GATE_cache, //(seq_len, d_expert)
    __half* DOWN_cache, //(seq_len, d_expert)
    __half* SiLU_cache, //(seq_len, d_expert)
    int seq_len,
    int d_hidden,
    int d_expert,
    hipStream_t stream,
    rocblas_handle handle
)
{
    // // Allocate memory on device
    // __half* d_W_up, *d_W_gate, *d_W_down;
    // __half* d_a, *d_b;
    // hipMalloc(&d_W_up, d_expert * d_hidden * sizeof(__half));       //W_up(d_hidden, d_expert)
    // hipMalloc(&d_W_gate, d_expert * d_hidden * sizeof(__half));     //W_gate(d_hidden, d_expert)
    // hipMalloc(&d_W_down, d_expert * d_hidden * sizeof(__half));     //W_down(d_expert, d_hidden)
    // hipMalloc(&d_a, seq_len * d_hidden * sizeof(__half));           //input(seq_len, d_hidden)
    // hipMalloc(&d_b, seq_len * d_hidden * sizeof(__half));           //output(seq_len, d_hidden)

    // __half* UP, *GATE, *DOWN, *SiLU;
    // hipMalloc(&UP, seq_len * d_expert * sizeof(__half));
    // hipMalloc(&GATE, seq_len * d_expert * sizeof(__half));
    // hipMalloc(&DOWN, seq_len * d_hidden * sizeof(__half));
    // hipMalloc(&SiLU, seq_len * d_hidden * sizeof(__half));

    // rocblas_handle handle1;
    // rocblas_handle handle2;
    // rocblas_create_handle(&handle1);
    // rocblas_create_handle(&handle2);

    // rocblas_set_stream(handle1, stream1);
    // rocblas_set_stream(handle2, stream2);

    // // Copy data to device
    // hipMemcpyAsync(d_W_up, h_W_up, d_expert * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1);
    // hipMemcpyAsync(d_W_gate, h_W_gate, d_expert * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1);
    // hipMemcpyAsync(d_W_down, h_W_down, d_expert * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1);
    // hipMemcpyAsync(d_a, a, seq_len * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1); 

    // hipEvent_t event_up, event_silu;
    // hipEventCreate(&event_up);
    // hipEventCreate(&event_silu);
    
    // Launch kernels
    linear(a, W_up, UP_cache, 1, seq_len, d_hidden, d_expert, handle);
    // hipEventRecord(event_up, stream1);

    linear(a, W_gate, GATE_cache, 1, seq_len, d_hidden, d_expert, handle);

    silu_activation(GATE_cache, SiLU_cache, seq_len, d_expert, stream);
    // hipEventRecord(event_silu, stream2);

    // hipStreamWaitEvent(stream2, event_up, 0);
    // hipStreamWaitEvent(stream2, event_silu, 0);

    dot_product_scores(SiLU_cache, UP_cache, scores, DOWN_cache, seq_len, d_expert, stream);

    // hipEventDestroy(event_up);
    // hipEventDestroy(event_silu);

    linear(DOWN_cache, W_down, b, 1, seq_len, d_hidden, d_expert, handle);

    // hipMemcpy(b, d_b, seq_len * d_hidden * sizeof(__half), hipMemcpyDeviceToHost);

    // hipFree(d_W_up);
    // hipFree(d_W_gate);
    // hipFree(d_W_down);
    // hipFree(d_a);
    // hipFree(d_b);
    // hipFree(UP);
    // hipFree(GATE);
    // hipFree(DOWN);
    // hipFree(SiLU);

    // rocblas_destroy_handle(handle1);
    // rocblas_destroy_handle(handle2);

}

// extern "C" void expert_shared(
//     __half* h_W_up,
//     __half* h_W_gate,
//     __half* h_W_down,
//     __half* a,  //input(batch_size, seq_len, d_hidden)
//     __half* b,  //output(batch_size, seq_len, d_hidden)
//     int batch_size,
//     int seq_len,
//     int d_hidden,
//     int d_expert,
//     int n_experts,
//     hipStream_t stream1,
//     hipStream_t stream2
// )
// {
//     // Allocate memory on device
//     __half* d_W_up, *d_W_gate, *d_W_down;
//     __half* d_a, *d_b;
//     hipMalloc(&d_W_up, d_expert * d_hidden * n_experts * sizeof(__half));       //W_up(d_hidden, d_expert*n_experts)
//     hipMalloc(&d_W_gate, d_expert * d_hidden * n_experts * sizeof(__half));     //W_gate(d_hidden, d_expert*n_experts)
//     hipMalloc(&d_W_down, d_expert * d_hidden * n_experts * sizeof(__half));     //W_down(d_expert*n_experts, d_hidden)
//     hipMalloc(&d_a, batch_size * seq_len * d_hidden * sizeof(__half));           //input(batch_size, seq_len, d_hidden)
//     hipMalloc(&d_b, batch_size * seq_len * d_hidden * sizeof(__half));           //output(batch_size, seq_len, d_hidden)

//     __half* UP, *GATE, *DOWN, *SiLU;
//     hipMalloc(&UP, batch_size * seq_len * d_expert * n_experts * sizeof(__half));
//     hipMalloc(&GATE, batch_size * seq_len * d_expert * n_experts * sizeof(__half));
//     hipMalloc(&DOWN, batch_size * seq_len * d_hidden * n_experts * sizeof(__half));
//     hipMalloc(&SiLU, batch_size * seq_len * d_hidden * n_experts * sizeof(__half));

//     rocblas_handle handle1;
//     rocblas_handle handle2;
//     rocblas_create_handle(&handle1);
//     rocblas_create_handle(&handle2);

//     rocblas_set_stream(handle1, stream1);
//     rocblas_set_stream(handle2, stream2);

//     // Copy data to device
//     hipMemcpyAsync(d_W_up, h_W_up, d_expert * d_hidden * n_experts * sizeof(__half), hipMemcpyHostToDevice, stream1);
//     hipMemcpyAsync(d_W_gate, h_W_gate, d_expert * d_hidden * n_experts * sizeof(__half), hipMemcpyHostToDevice, stream1);
//     hipMemcpyAsync(d_W_down, h_W_down, d_expert * d_hidden * n_experts * sizeof(__half), hipMemcpyHostToDevice, stream1);
//     hipMemcpyAsync(d_a, a, batch_size * seq_len * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1); 

//     hipEvent_t event_up, event_silu;
//     hipEventCreate(&event_up);
//     hipEventCreate(&event_silu);
    
//     // Launch kernels
//     linear(d_a, d_W_up, UP, batch_size, seq_len, d_hidden, d_expert*n_experts, handle1);
//     hipEventRecord(event_up, stream1);

//     linear(d_a, d_W_gate, GATE, batch_size, seq_len, d_hidden, d_expert*n_experts, handle2);
//     silu_activation(GATE, SiLU, batch_size*seq_len, d_expert*n_experts, stream2);
//     hipEventRecord(event_silu, stream2);

//     hipStreamWaitEvent(stream2, event_up, 0);
//     hipStreamWaitEvent(stream2, event_silu, 0);

//     dot_product(SiLU, UP, DOWN, batch_size*seq_len*d_expert*n_experts, stream2);

//     hipEventDestroy(event_up);
//     hipEventDestroy(event_silu);

//     linear(DOWN, d_W_down, d_b, batch_size, seq_len, d_hidden, d_expert*n_experts, handle2);

//     hipMemcpy(b, d_b, batch_size * seq_len * d_hidden * sizeof(__half), hipMemcpyDeviceToHost);

//     hipFree(d_W_up);
//     hipFree(d_W_gate);
//     hipFree(d_W_down);
//     hipFree(d_a);
//     hipFree(d_b);
//     hipFree(UP);
//     hipFree(GATE);
//     hipFree(DOWN);
//     hipFree(SiLU);

//     rocblas_destroy_handle(handle1);
//     rocblas_destroy_handle(handle2);

// }

// extern "C" void expert_router(
//     __half* h_W_up,
//     __half* h_W_gate,
//     __half* h_W_down,
//     __half* scores,
//     __half* a,  //input(seq_len, d_hidden)
//     __half* b,  //output(seq_len, d_hidden)
//     int seq_len,
//     int d_hidden,
//     int d_expert,
//     hipStream_t stream1,
//     hipStream_t stream2
// )
// {
//     // Allocate memory on device
//     __half* d_W_up, *d_W_gate, *d_W_down;
//     __half* d_a, *d_b;
//     hipMalloc(&d_W_up, d_expert * d_hidden * sizeof(__half));       //W_up(d_hidden, d_expert)
//     hipMalloc(&d_W_gate, d_expert * d_hidden * sizeof(__half));     //W_gate(d_hidden, d_expert)
//     hipMalloc(&d_W_down, d_expert * d_hidden * sizeof(__half));     //W_down(d_expert, d_hidden)
//     hipMalloc(&d_a, seq_len * d_hidden * sizeof(__half));           //input(seq_len, d_hidden)
//     hipMalloc(&d_b, seq_len * d_hidden * sizeof(__half));           //output(seq_len, d_hidden)

//     __half* UP, *GATE, *DOWN, *SiLU;
//     hipMalloc(&UP, seq_len * d_expert * sizeof(__half));
//     hipMalloc(&GATE, seq_len * d_expert * sizeof(__half));
//     hipMalloc(&DOWN, seq_len * d_hidden * sizeof(__half));
//     hipMalloc(&SiLU, seq_len * d_hidden * sizeof(__half));

//     rocblas_handle handle1;
//     rocblas_handle handle2;
//     rocblas_create_handle(&handle1);
//     rocblas_create_handle(&handle2);

//     rocblas_set_stream(handle1, stream1);
//     rocblas_set_stream(handle2, stream2);

//     // Copy data to device
//     hipMemcpyAsync(d_W_up, h_W_up, d_expert * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1);
//     hipMemcpyAsync(d_W_gate, h_W_gate, d_expert * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1);
//     hipMemcpyAsync(d_W_down, h_W_down, d_expert * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1);
//     hipMemcpyAsync(d_a, a, seq_len * d_hidden * sizeof(__half), hipMemcpyHostToDevice, stream1); 

//     hipEvent_t event_up, event_silu;
//     hipEventCreate(&event_up);
//     hipEventCreate(&event_silu);
    
//     // Launch kernels
//     linear(d_a, d_W_up, UP, seq_len, d_hidden, d_expert, handle1);
//     hipEventRecord(event_up, stream1);

//     linear(d_a, d_W_gate, GATE, seq_len, d_hidden, d_expert, handle2);
//     silu_activation(GATE, SiLU, seq_len, d_expert, stream2);
//     hipEventRecord(event_silu, stream2);

//     hipStreamWaitEvent(stream2, event_up, 0);
//     hipStreamWaitEvent(stream2, event_silu, 0);

//     dot_product_scores(SiLU, UP, scores, DOWN, seq_len, d_expert, stream2);

//     hipEventDestroy(event_up);
//     hipEventDestroy(event_silu);

//     linear(DOWN, d_W_down, d_b, seq_len, d_hidden, d_expert, handle2);

//     hipMemcpy(b, d_b, seq_len * d_hidden * sizeof(__half), hipMemcpyDeviceToHost);

//     hipFree(d_W_up);
//     hipFree(d_W_gate);
//     hipFree(d_W_down);
//     hipFree(d_a);
//     hipFree(d_b);
//     hipFree(UP);
//     hipFree(GATE);
//     hipFree(DOWN);
//     hipFree(SiLU);

//     rocblas_destroy_handle(handle1);
//     rocblas_destroy_handle(handle2);

// }