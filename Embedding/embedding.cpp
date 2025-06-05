// #include <hip/hip_runtime.h>
// #include <hip/hip_fp16.h>
// #include <iostream>

// // GPU 核函数：并行查找 token 对应的 embedding
// __global__ void embedding_kernel(int* tokens, int token_count, __half* weights, int vocab_size, int embed_dim, __half* output) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= token_count * embed_dim) return;  // 确保索引合法

//     int token_idx = idx / embed_dim;  // 当前 Token 的索引
//     int embed_idx = idx % embed_dim;  // 当前 Token 维度中的位置

//     int vocab_index = tokens[token_idx];  // 获取 Token 值（词汇索引）
//     if (vocab_index >= vocab_size) return;  // 超出词汇表

//     // 计算在权重矩阵中的位置
//     int weight_pos = vocab_index * embed_dim + embed_idx;
//     output[idx] = weights[weight_pos];  // 复制权重
// }

// // 入口函数，Python 端调用
// extern "C" void embedding_lookup(int* tokens, int token_count, __half* weights, int vocab_size, int embed_dim, __half* output) {
//     // 1. 申请 GPU 内存
//     int* d_tokens;
//     __half* d_weights;
//     __half* d_output;

//     hipMalloc(&d_tokens, token_count * sizeof(int));
//     hipMalloc(&d_weights, vocab_size * embed_dim * sizeof(__half));
//     hipMalloc(&d_output, token_count * embed_dim * sizeof(__half));

//     // 2. 拷贝数据到 GPU
//     hipMemcpy(d_tokens, tokens, token_count * sizeof(int), hipMemcpyHostToDevice);
//     hipMemcpy(d_weights, weights, vocab_size * embed_dim * sizeof(__half), hipMemcpyHostToDevice);

//     // 3. 启动 Kernel
//     int threads_per_block = 256;
//     int total_threads = token_count * embed_dim;
//     int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;
    
//     hipLaunchKernelGGL(embedding_kernel, dim3(blocks_per_grid), dim3(threads_per_block), 0, 0,
//                        d_tokens, token_count, d_weights, vocab_size, embed_dim, d_output);

//     // 4. 拷贝结果回 Host
//     hipMemcpy(output, d_output, token_count * embed_dim * sizeof(__half), hipMemcpyDeviceToHost);

//     // 5. 释放 GPU 资源
//     hipFree(d_tokens);
//     hipFree(d_weights);
//     hipFree(d_output);
// }

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <hip/hip_runtime.h>
#include <json/json.h>
#include <hip/hip_fp16.h>  // HIP FP16 支持

#define VOCAB_SIZE 32000   // 词汇表大小
#define EMBED_DIM 4096     // 嵌入维度

// 读取 JSON 索引文件
std::map<std::string, size_t> load_index(const std::string& json_file) {
    std::ifstream file(json_file);
    if (!file) {
        std::cerr << "无法打开索引文件：" << json_file << std::endl;
        exit(1);
    }

    Json::Value root;
    file >> root;
    std::map<std::string, size_t> index_map;
    for (const auto& key : root.getMemberNames()) {
        index_map[key] = root[key]["offset"].asUInt();
    }
    return index_map;
}

// 读取 FP16 权重
void read_weights_fp16(const std::string& bin_file, size_t offset, size_t num_elements, std::vector<__half>& buffer) {
    std::ifstream file(bin_file, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开权重文件：" << bin_file << std::endl;
        exit(1);
    }
    file.seekg(offset, std::ios::beg);
    buffer.resize(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(__half));
    file.close();
}

// **HIP 核函数：查找 Token 对应的向量**
__global__ void embedding_lookup(__half* embedding_matrix, int* tokens, __half* output, int num_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 线程 ID
    if (idx < num_tokens) {
        int token_id = tokens[idx];  // 读取 token ID
        if (token_id < VOCAB_SIZE) {
            for (int i = 0; i < EMBED_DIM; i++) {
                output[idx * EMBED_DIM + i] = embedding_matrix[token_id * EMBED_DIM + i];
            }
        }
    }
}

int main() {
    //读取 JSON 索引
    std::map<std::string, size_t> index_map = load_index("model_index.json");

    //查找 "tok_embeddings.weight" 偏移量
    std::string target_param = "tok_embeddings.weight";
    if (index_map.find(target_param) == index_map.end()) {
        std::cerr << "未找到参数：" << target_param << std::endl;
        return -1;
    }

    size_t offset = index_map[target_param];  // 偏移量
    size_t num_elements = VOCAB_SIZE * EMBED_DIM;

    //读取 .bin 的 FP16 Embedding
    std::vector<__half> weights;
    read_weights_fp16("model_weights_fp16.bin", offset, num_elements, weights);

    //传入 tokens（假设 Python 传入了一组 tokens）
    std::vector<int> host_tokens = { 123, 456, 789 };  // 这里换成 Python 传入的 tokens
    int num_tokens = host_tokens.size();

    //分配 GPU 内存
    __half* d_embedding;
    int* d_tokens;
    __half* d_output;
    hipMalloc((void**)&d_embedding, num_elements * sizeof(__half));
    hipMalloc((void**)&d_tokens, num_tokens * sizeof(int));
    hipMalloc((void**)&d_output, num_tokens * EMBED_DIM * sizeof(__half));

    //拷贝数据到 GPU
    hipMemcpy(d_embedding, weights.data(), num_elements * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_tokens, host_tokens.data(), num_tokens * sizeof(int), hipMemcpyHostToDevice);

    //启动核函数
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;
    hipLaunchKernelGGL(embedding_lookup, dim3(grid_size), dim3(block_size), 0, 0, d_embedding, d_tokens, d_output, num_tokens);

    //拷回结果
    std::vector<__half> host_output(num_tokens * EMBED_DIM);
    hipMemcpy(host_output.data(), d_output, num_tokens * EMBED_DIM * sizeof(__half), hipMemcpyDeviceToHost);

    //打印部分输出
    std::cout << "Token 123 的前 5 维 Embedding：" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << (float)host_output[i] << " ";
    }
    std::cout << std::endl;

    //释放内存
    hipFree(d_embedding);
    hipFree(d_tokens);
    hipFree(d_output);

    return 0;
}

