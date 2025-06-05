#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include <string>
#include <map>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// 新增结构体存储参数信息
struct ParamInfo {
    size_t offset;
    size_t num_elements;
    std::vector<size_t> shape;
};

class WeightLoader {
public:
    WeightLoader(const std::string& json_path, const std::string& bin_path);
    __half* get_weights(const std::string& param_name); // 移除num_elements参数
    std::vector<size_t> get_shape(const std::string& param_name);
    void free_all();

    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

private:
    std::string json_path_;
    std::string bin_path_;
    std::map<std::string, ParamInfo> index_map_; 
    std::vector<__half*> gpu_ptrs_;

    void load_index();
    void read_weights_fp16(size_t offset, size_t num_elements, __half* dst);
};

#endif // WEIGHT_LOADER_H