#include "weight_loader.h"
#include <iostream>
int main() {
    
    // 初始化加载器
    WeightLoader loader(
        "/home/qin/rocm_test/Embedding/model_index.json",
        "/home/qin/rocm_test/Embedding/model_weights.bin"
    );

    // 获取权重指针 
    __half* emb_weights = loader.get_weights("tok_embeddings.weight");

    //打印前10个权重
    for (int i = 0; i < 10; i++) {
        std::cout << (float)emb_weights[i] << " ";
    }
    std::cout << std::endl;

    // 释放所有权重参数 
    loader.free_all();

    return 0;
}