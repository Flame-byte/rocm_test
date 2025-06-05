#include <hip/hip_runtime.h>
#include <iostream>
#include "weight_loader.h"
#include <vector>

__global__ void fetch_embeddings_kernel(
    const int* tokens,
    const __half* weights,
    __half* output,
    int vocab_size,
    int embed_dim,
    int num_tokens
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tokens) return;

    int token_id = tokens[tid];

    //printf("tid=%d, token_id=%d, vocab_size=%d\n", tid, token_id, vocab_size);
    if (token_id < 0 || token_id >= vocab_size) {
        //printf("Invalid token_id=%d\n", token_id);
        return;
    }

    // 计算源地址和目标地址
    const __half* src = weights + token_id * embed_dim;
    __half* dst = output + tid * embed_dim;

    //printf("src=%p, dst=%p\n", src, dst);

    // 拷贝整个嵌入向量
    for (int i = 0; i < embed_dim; i++) {
        dst[i] = src[i];
    }
}



int main() {
    // 初始化加载器
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    WeightLoader loader(
        "/home/qin/rocm_test/Embedding/model_index.json",
        "/home/qin/rocm_test/Embedding/model_weights.bin"
    );

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << "get_weights: " << milliseconds << " ms" << std::endl;

    std::vector<int> tokens = {128000, 128006, 882, 128007, 271, 74745, 76, 374, 459, 1825, 31874, 5729, 11, 24306};
    const int N = tokens.size();
    const int embed_dim = 4096;

    int* d_tokens;
    __half* d_weights = loader.get_weights("tok_embeddings.weight");      // 来自 WeightLoader 的 get_weights("tok_embeddings.weight")
    __half* d_output;       // 输出矩阵：[N, embed_dim]

    // hipMallocManaged(&d_tokens, N * sizeof(int));
    // hipMallocManaged(&d_output, N * embed_dim * sizeof(__half));

    hipError_t err1,err2;
    err1 = hipMalloc(&d_tokens, N * sizeof(int));
    if (err1 != hipSuccess) {
        std::cerr << "hipMalloc failed (d_tokens): " << hipGetErrorString(err1) << std::endl;
        exit(1);
    }

    err2 = hipMallocManaged(&d_output, N * embed_dim * sizeof(__half));
    if (err2 != hipSuccess) {
        std::cerr << "hipMalloc failed (d_output): " << hipGetErrorString(err2) << std::endl;
        exit(1);
    }

    //d_tokens = tokens.data();
    //将tokens数据拷贝到d_tokens
    //hipMemcpy(d_tokens, tokens.data(), N * sizeof(int), hipMemcpyHostToDevice);
    for (int i = 0; i < N; i++) {
        d_tokens[i] = tokens[i];
    }

    // 配置线程块（每个线程处理一个 token）
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    // 调用内核
    fetch_embeddings_kernel<<<grid_size, block_size>>>(
        d_tokens,
        d_weights,
        d_output,
        128256,  // vocab_size
        embed_dim,
        N
    );

    // 同步等待内核完成
    hipDeviceSynchronize();

    //输出每个d_output的前5个元素
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << __half2float(d_output[i * embed_dim + j]) << " ";
        }
        std::cout << std::endl;
    }

    // 释放内存
    hipFree(d_tokens);
    hipFree(d_output);
    hipFree(d_weights);

    return 0;
}