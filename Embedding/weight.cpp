#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <hip/hip_runtime.h>
#include <json/json.h>   // 需要安装 jsoncpp
#include <hip/hip_fp16.h> // FP16 支持

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
        index_map[key] = root[key]["offset"].asUInt64();
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

    file.seekg(offset, std::ios::beg);  // 移动到偏移位置
    buffer.resize(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(__half));
    file.close();
}

// GPU 代码：拷贝到 GPU
void copy_to_gpu(const std::vector<__half>& host_data, __half* device_data, size_t num_elements) {
    hipMemcpy(device_data, host_data.data(), num_elements * sizeof(__half), hipMemcpyHostToDevice);
}

int main() {
    // 1. 读取 JSON 索引
    std::map<std::string, size_t> index_map = load_index("/home/qin/rocm_test/Embedding/model_index.json");

    // 2. 选择目标参数，例如 "tok_embeddings.weight"
    std::string target_param = "tok_embeddings.weight";
    if (index_map.find(target_param) == index_map.end()) {
        std::cerr << "未找到参数：" << target_param << std::endl;
        return -1;
    }

    // 3. 读取 .bin 的 FP16 数据
    size_t offset = index_map[target_param];  // 偏移量
    size_t num_elements = 32000 * 4096;  // 词汇表大小 × 嵌入维度
    std::vector<__half> weights;
    read_weights_fp16("/home/qin/rocm_test/Embedding/model_weights.bin", offset, num_elements, weights);

    // 4. 分配 GPU 内存
    __half* d_weights;
    hipMallocManaged((void**)&d_weights, num_elements * sizeof(__half));

    // 5. 复制数据到 GPU
    copy_to_gpu(weights, d_weights, num_elements);

    // 6. 打印数据
    std::cout << "Data:" << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << "Element " << i << ": " << static_cast<float>(weights[i]) << std::endl;
    }

    std::cout << "FP16 权重 " << target_param << " 成功加载到 GPU" << std::endl;

    // 释放 GPU 内存
    hipFree(d_weights);
    return 0;
}
