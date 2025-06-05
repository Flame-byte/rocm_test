#include "dot_product.h"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cmath>

// __global__ void dot_product_kernel(
//     const __half* a,
//     const __half* b,
//     __half* c,
//     const int n
// )
// {
//     int tid = blockDim.x * blockIdx.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (int i = tid; i < n; i += stride) {
//         __half a_val = a[i];
//         __half b_val = b[i];
//         __half c_val = __hmul(a_val, b_val);
//         c[i] = c_val;
//     }
// }

__global__ void dot_product_kernel(
    const __half* a,
    const __half* b,
    __half* c,
    const int n
)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n2 = n / 2;
    const half2* a2 = reinterpret_cast<const half2*>(a);
    const half2* b2 = reinterpret_cast<const half2*>(b);
    half2* c2 = reinterpret_cast<half2*>(c);
    for (int i = tid; i < n2; i += stride) {
        c2[i] = __hmul2(a2[i], b2[i]);
    }
    // 处理最后一个元素（如果 n 是奇数）
    if (tid == 0 && n % 2) {
        c[n-1] = __hmul(a[n-1], b[n-1]);
    }
}

__global__ void dot_product_kernel_scores(
    const __half* a,
    const __half* b,
    const __half* scores, // [seq_len]
    __half* c,
    const int seq_len,
    const int d_expert
) {
    extern __shared__ __half shared_scores[]; // shared memory for scores

    int row = blockIdx.x; // 当前处理的行
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // block 内第一个线程将 score 读入 shared memory
    if (tid == 0) {
        shared_scores[0] = scores[row];
    }
    __syncthreads();

    __half score = shared_scores[0];

    int base = row * d_expert;
    for (int col = tid; col < d_expert; col += stride) {
        int idx = base + col;
        c[idx] = __hmul(__hmul(a[idx], b[idx]), score);
    }
}

extern "C" void dot_product(
    const __half* a,
    const __half* b,
    __half* c,
    const int n,
    hipStream_t stream
)
{
    dot_product_kernel<<<(n + 256 - 1) / 256, 256, 0, stream>>>(a, b, c, n);
    hipStreamSynchronize(stream);
}

extern "C" void dot_product_scores(
    const __half* a,
    const __half* b,
    const __half* scores,
    __half* c,
    const int seq_len,
    const int d_expert,
    hipStream_t stream
)
{
    int threads = (d_expert < 128) ? d_expert : 128;
    int blocks = seq_len;
    size_t shared_mem = sizeof(__half);
    dot_product_kernel_scores<<<blocks, threads, shared_mem, stream>>>(a, b, scores, c, seq_len, d_expert);
    hipStreamSynchronize(stream);
}

// void cpu_dot_product_scores(
//     const std::vector<__half>& a,
//     const std::vector<__half>& b,
//     const std::vector<__half>& scores,
//     std::vector<__half>& c,
//     int seq_len,
//     int d_expert
// ) {
//     for (int row = 0; row < seq_len; ++row) {
//         float score = __half2float(scores[row]);
//         for (int col = 0; col < d_expert; ++col) {
//             int idx = row * d_expert + col;
//             float va = __half2float(a[idx]);
//             float vb = __half2float(b[idx]);
//             c[idx] = __float2half(va * vb * score);
//         }
//     }
// }

// int main() {
//     int seq_len = 4;
//     int d_expert = 8;
//     int n = seq_len * d_expert;

//     std::vector<__half> h_a(n), h_b(n), h_scores(seq_len), h_c(n), h_c_cpu(n);

//     // 随机初始化
//     for (int i = 0; i < n; ++i) {
//         h_a[i] = __float2half((float)(i % 7 + 1));
//         h_b[i] = __float2half((float)(i % 5 + 2));
//     }
//     for (int i = 0; i < seq_len; ++i) {
//         h_scores[i] = __float2half(0.1f * (i + 1));
//     }

//     // CPU baseline
//     cpu_dot_product_scores(h_a, h_b, h_scores, h_c_cpu, seq_len, d_expert);

//     // 分配GPU内存
//     __half *d_a, *d_b, *d_scores, *d_c;
//     hipMalloc(&d_a, n * sizeof(__half));
//     hipMalloc(&d_b, n * sizeof(__half));
//     hipMalloc(&d_scores, seq_len * sizeof(__half));
//     hipMalloc(&d_c, n * sizeof(__half));

//     hipMemcpy(d_a, h_a.data(), n * sizeof(__half), hipMemcpyHostToDevice);
//     hipMemcpy(d_b, h_b.data(), n * sizeof(__half), hipMemcpyHostToDevice);
//     hipMemcpy(d_scores, h_scores.data(), seq_len * sizeof(__half), hipMemcpyHostToDevice);

//     hipStream_t stream;
//     hipStreamCreate(&stream);

//     // 调用 kernel
//     dot_product_scores(d_a, d_b, d_scores, d_c, seq_len, d_expert, stream);
//     hipStreamSynchronize(stream);

//     hipMemcpy(h_c.data(), d_c, n * sizeof(__half), hipMemcpyDeviceToHost);

//     // 对比结果
//     bool ok = true;
//     for (int i = 0; i < n; ++i) {
//         float v_gpu = __half2float(h_c[i]);
//         float v_cpu = __half2float(h_c_cpu[i]);
//         if (std::abs(v_gpu - v_cpu) > 1e-3f) {
//             printf("Mismatch at %d: GPU=%f CPU=%f\n", i, v_gpu, v_cpu);
//             ok = false;
//         }
//     }
//     if (ok) {
//         printf("dot_product_scores kernel matches CPU result!\n");
//     } else {
//         printf("dot_product_scores kernel does NOT match CPU result!\n");
//     }

//     hipFree(d_a);
//     hipFree(d_b);
//     hipFree(d_scores);
//     hipFree(d_c);
//     hipStreamDestroy(stream);
//     return 0;
// }

// int main()
// {
//     const int batch_size = 2;
//     const int seq_len = 4;
//     const int d_expert = 4;
//     const int n_experts = 4;
//     const int n = batch_size * seq_len * d_expert * n_experts;
//     __half* h_a = (__half*)malloc(sizeof(__half) * n);
//     __half* h_b = (__half*)malloc(sizeof(__half) * n);
//     __half* h_c = (__half*)malloc(sizeof(__half) * n);

//     for (int i = 0; i < n; i++) {
//         h_a[i] = __float2half(i);
//         h_b[i] = __float2half(i);
//     }

//     // CPU计算
//     float cpu_dot[n];
//     for (int i = 0; i < n; i++) {
//         cpu_dot[i] = __half2float(h_a[i]) * __half2float(h_b[i]);
//     }
//     printf("CPU dot product: ");
//     for (int i = 0; i < n; i++) {
//         printf("%f ", cpu_dot[i]);
//     }
//     printf("\n");

//     __half* d_a, *d_b, *d_c;
//     hipMalloc(&d_a, sizeof(__half) * n);
//     hipMalloc(&d_b, sizeof(__half) * n);
//     hipMalloc(&d_c, sizeof(__half) * n);

//     hipMemcpy(d_a, h_a, sizeof(__half) * n, hipMemcpyHostToDevice);
//     hipMemcpy(d_b, h_b, sizeof(__half) * n, hipMemcpyHostToDevice);

//     hipStream_t stream;
//     hipStreamCreate(&stream);

//     dot_product(d_a, d_b, d_c, n, stream);
//     hipStreamSynchronize(stream);

//     hipMemcpy(h_c, d_c, sizeof(__half) * n, hipMemcpyDeviceToHost);

//     for (int i = 0; i < n; i++) {
//         printf("GPU dot product: %f ", __half2float(h_c[i]));
//     }

//     hipStreamDestroy(stream);

//     free(h_a);
//     free(h_b);
//     free(h_c);
//     hipFree(d_a);
//     hipFree(d_b);
//     hipFree(d_c);
// }