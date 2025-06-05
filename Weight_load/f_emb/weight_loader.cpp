#include "weight_loader.h"
#include <iostream>
#include <fstream>
#include <json/json.h>

// 构造函数，初始化json_path_和bin_path_
WeightLoader::WeightLoader(const std::string& json_path, const std::string& bin_path)
    : json_path_(json_path), bin_path_(bin_path) {
    load_index();
}

// 加载index文件
void WeightLoader::load_index() {
    std::ifstream file(json_path_);
    if (!file) {
        std::cerr << "Error opening index file: " << json_path_ << std::endl;
        exit(1);
    }

    Json::Value root;
    file >> root;

    for (const auto& key : root.getMemberNames()) { // 遍历json文件中的所有参数
        ParamInfo info; // 存储参数信息 
        info.offset = root[key]["offset"].asUInt64(); // 获取参数的偏移量
        info.num_elements = root[key]["num_elements"].asUInt64(); // 获取参数的元素个数
        for (const auto& dim : root[key]["shape"]) {
            info.shape.push_back(dim.asUInt64());
        }
        index_map_[key] = info; // 将参数信息存储到index_map中
    }
}

// 获取权重参数
__half* WeightLoader::get_weights(const std::string& param_name) {
    if (index_map_.find(param_name) == index_map_.end()) { 
        std::cerr << "Parameter not found: " << param_name << std::endl;
        return nullptr;
    }

    // 从index_map获取元数据
    const ParamInfo& info = index_map_[param_name];
    
    // 分配统一内存
    __half* d_ptr;
    hipMallocManaged(&d_ptr, info.num_elements * sizeof(__half));
    
    // 直接从文件读取到统一内存
    read_weights_fp16(info.offset, info.num_elements, d_ptr);

    gpu_ptrs_.push_back(d_ptr);
    return d_ptr;
}

// 获取参数的维度
std::vector<size_t> WeightLoader::get_shape(const std::string& param_name) {
    return index_map_[param_name].shape;
}

// 从文件读取权重参数
void WeightLoader::read_weights_fp16(size_t offset, size_t num_elements, __half* dst) {
    std::ifstream file(bin_path_, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening weights file: " << bin_path_ << std::endl;
        exit(1);
    }

    file.seekg(offset, std::ios::beg); // 移动到偏移位置
    file.read(reinterpret_cast<char*>(dst), num_elements * sizeof(__half)); // 读取权重参数
    file.close();
}

// 释放所有权重参数 
void WeightLoader::free_all() {
    for (auto ptr : gpu_ptrs_) {
        hipFree(ptr);
    }
    gpu_ptrs_.clear();
}