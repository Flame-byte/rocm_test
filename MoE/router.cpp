#include "router.h"
#include <hip/hip_fp16.h>
#include <vector>

std::vector<Expert> assign_tokens_to_experts(
    const __half* topk_scores,
    const __half* topk_indices,
    int batch_size,
    int seq_len,
    int topk,
    int n_experts
) {
    int M = batch_size * seq_len;
    int N = M * topk;

    // Initialize experts
    std::vector<Expert> experts;
    experts.reserve(n_experts);
    for (int e = 0; e < n_experts; ++e) {
        experts.emplace_back(e);
    }

    // Assign each token to the corresponding expert
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < topk; ++k) {
            int idx = i * topk + k;
            int expert_id = static_cast<int>(__half2float(topk_indices[idx]));
            experts[expert_id].token_indices.push_back(i);
            experts[expert_id].scores.push_back(topk_scores[idx]);
        }
    }

    return experts;
} 

// #include <iostream>
// #include <iomanip>

// int main() {
//     // 模拟数据，参考 file_context_0
//     // batch_size = 5, seq_len = 1, topk = 3, n_experts = 10
//     int batch_size = 5;
//     int seq_len = 1;
//     int topk = 3;
//     int n_experts = 10;
//     int M = batch_size * seq_len;

//     // d_topk_scores (GPU):
//     // Row 0: 9.01562 5.92969 5.59375 
//     // Row 1: 7.32422 4.16016 3.01758 
//     // Row 2: 9.84375 9.39844 8.77344 
//     // Row 3: 2.34961 2.23242 0.495605 
//     // Row 4: 9.47656 2.23633 -0.878418 
//     __half topk_scores[15] = {
//         __float2half(9.01562f), __float2half(5.92969f), __float2half(5.59375f),
//         __float2half(7.32422f), __float2half(4.16016f), __float2half(3.01758f),
//         __float2half(9.84375f), __float2half(9.39844f), __float2half(8.77344f),
//         __float2half(2.34961f), __float2half(2.23242f), __float2half(0.495605f),
//         __float2half(9.47656f), __float2half(2.23633f), __float2half(-0.878418f)
//     };

//     // d_topk_indices (GPU):
//     // Row 0: 2 1 5 
//     // Row 1: 4 8 9 
//     // Row 2: 9 2 5 
//     // Row 3: 1 3 9 
//     // Row 4: 5 0 8 
//     __half topk_indices[15] = {
//         __float2half(2), __float2half(1), __float2half(5),
//         __float2half(4), __float2half(8), __float2half(9),
//         __float2half(9), __float2half(2), __float2half(5),
//         __float2half(1), __float2half(3), __float2half(9),
//         __float2half(5), __float2half(0), __float2half(8)
//     };

//     // 调用分组函数
//     std::vector<Expert> experts = assign_tokens_to_experts(
//         topk_scores, topk_indices, batch_size, seq_len, topk, n_experts
//     );

//     // 打印每个expert分配到的token和score
//     for (const auto& expert : experts) {
//         std::cout << "Expert " << expert.id << ": ";
//         if (expert.token_indices.empty()) {
//             std::cout << "(no tokens)";
//         } else {
//             for (size_t i = 0; i < expert.token_indices.size(); ++i) {
//                 std::cout << "[token " << expert.token_indices[i]
//                           << ", score " << std::fixed << std::setprecision(5)
//                           << __half2float(expert.scores[i]) << "] ";
//             }
//         }
//         std::cout << std::endl;
//     }

//     return 0;
// }
