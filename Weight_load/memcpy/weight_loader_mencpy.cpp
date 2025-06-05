#include "weight_loader_all.h"
#include <iostream>
#include <fstream>
#include <json/json.h>

WeightLoader::WeightLoader(const std::string& json_path, const std::string& bin_path)
    : json_path_(json_path), bin_path_(bin_path), entire_weights_(nullptr) {
    load_index();
    load_entire_weights();  // 初始化时直接加载全部权重
}

WeightLoader::~WeightLoader() {
    free_all();  // 析构时释放内存
}

void WeightLoader::load_index() {
    std::ifstream file(json_path_);
    if (!file) {
        std::cerr << "Error opening index file: " << json_path_ << std::endl;
        exit(1);
    }

    Json::Value root;
    file >> root;

    for (const auto& key : root.getMemberNames()) {
        ParamInfo info;
        info.offset = root[key]["offset"].asUInt64();       // offset是字节单位
        info.num_elements = root[key]["num_elements"].asUInt64();
        for (const auto& dim : root[key]["shape"]) {
            info.shape.push_back(dim.asUInt64());
        }
        index_map_[key] = info;
    }
}

//一次性加载整个.bin文件到统一内存
void WeightLoader::load_entire_weights() {
    std::ifstream file(bin_path_, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening weights file: " << bin_path_ << std::endl;
        exit(1);
    }

    // 获取文件大小并分配内存
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    hipError_t err = hipMallocManaged(&entire_weights_, file_size);
    if (err != hipSuccess) {
        std::cerr << "Failed to allocate unified memory: " << hipGetErrorString(err) << std::endl;
        exit(1);
    }

    // 读取整个文件到统一内存
    file.read(reinterpret_cast<char*>(entire_weights_), file_size);
    file.close();
}

//直接返回内存指针，无需重复加载
__half* WeightLoader::get_weights(const std::string& param_name) {
    if (index_map_.find(param_name) == index_map_.end()) {
        std::cerr << "Parameter not found: " << param_name << std::endl;
        return nullptr;
    }

    const ParamInfo& info = index_map_[param_name];
    // 计算指针偏移量（offset为字节，需转换为__half*的步长）
    size_t element_offset = info.offset / sizeof(__half);
    return entire_weights_ + element_offset;
}

//获取参数的维度
std::vector<size_t> WeightLoader::get_shape(const std::string& param_name) {
    return index_map_[param_name].shape;
}

//简化释放逻辑
void WeightLoader::free_all() {
    if (entire_weights_ != nullptr) {
        hipFree(entire_weights_);
        entire_weights_ = nullptr;
    }
}