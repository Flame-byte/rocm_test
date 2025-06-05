#ifndef WEIGHT_LOADER_ALL_H
#define WEIGHT_LOADER_ALL_H

#include <string>
#include <hip/hip_fp16.h>
#include <unordered_map>
#include <hip/hip_runtime.h>  // 包含HIP类型和API声明

// 前向声明（避免直接包含JSON头文件）
namespace Json {
    class Value;
}

// 参数元数据结构体
struct ParamInfo {
    size_t offset;        // 字节偏移量（需为sizeof(__half)的整数倍）
    size_t num_elements;  // 元素数量（每个元素为__half）
    std::vector<size_t> shape;
};

class WeightLoader {
public:
    // 构造函数：加载JSON索引并全量读取.bin文件
    WeightLoader(const std::string& json_path, const std::string& bin_path);
    // 析构函数：自动释放内存
    ~WeightLoader();

    // 获取参数指针（内存已预加载）
    __half* get_weights(const std::string& param_name);
    std::vector<size_t> get_shape(const std::string& param_name);

    // 释放统一内存（通常由析构函数自动调用）
    void free_all();

private:
    // 禁止拷贝和赋值
    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    // 私有方法
    void load_index();          // 解析JSON索引文件
    void load_entire_weights(); // 全量加载.bin文件到统一内存

    // 成员变量
    std::string json_path_;     // JSON索引文件路径
    std::string bin_path_;      // 二进制权重文件路径
    __half* entire_weights_;    // 统一内存指针（全量加载的权重）

    // 参数元数据映射表（参数名 -> 元数据）
    std::unordered_map<std::string, ParamInfo> index_map_;
};

#endif // WEIGHT_LOADER_H