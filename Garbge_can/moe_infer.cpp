#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

// Step 1: Build lookup table mapping each expert to the tokens it should process,
// and gather corresponding sorted top-k scores (weights).
static void build_expert_token_lookup(
    const __half* topk_scores,
    const __half* topk_indices,
    int M,
    int topk,
    int n_experts,
    std::vector<int>& tokens_per_expert_end,
    std::vector<int>& token_indices,
    std::vector<__half>& sorted_scores
) {
    int N = M * topk;
    // Convert half indices to ints
    std::vector<int> flat_expert_indices(N);
    for (int i = 0; i < N; ++i) {
        flat_expert_indices[i] = static_cast<int>(__half2float(topk_indices[i]));
    }
    // Count tokens per expert
    tokens_per_expert_end.assign(n_experts, 0);
    for (int id : flat_expert_indices) {
        tokens_per_expert_end[id]++;
    }
    // Prefix sum to get end offsets
    for (int e = 1; e < n_experts; ++e) {
        tokens_per_expert_end[e] += tokens_per_expert_end[e - 1];
    }
    // Sort flat positions by expert id
    std::vector<int> sorted_positions(N);
    std::iota(sorted_positions.begin(), sorted_positions.end(), 0);
    std::stable_sort(sorted_positions.begin(), sorted_positions.end(),
        [&](int a, int b) {
            return flat_expert_indices[a] < flat_expert_indices[b];
        }
    );
    // Build token index list and sorted score list
    token_indices.resize(N);
    sorted_scores.resize(N);
    for (int i = 0; i < N; ++i) {
        int pos = sorted_positions[i];
        token_indices[i] = pos / topk;
        sorted_scores[i]  = topk_scores[pos];
    }
}

// TODO: implement moe_infer to call build_expert_token_lookup and then
// dispatch experts on separate streams using the lookup tables.
extern "C" void moe_infer(
    const __half* x,           // input tokens: shape (M, d_hidden)
    const __half* topk_scores, // gating scores: shape (M, topk)
    const __half* topk_indices,// gating indices: shape (M, topk)
    int batch_size,
    int seq_len,
    int d_hidden,
    int n_experts,
    int topk
) {
    int M = batch_size * seq_len;
    // Lookup tables
    std::vector<int> tokens_per_expert_end;
    std::vector<int> token_indices;
    std::vector<__half> sorted_scores;
    build_expert_token_lookup(
        topk_scores, topk_indices, M, topk, n_experts,
        tokens_per_expert_end, token_indices, sorted_scores
    );
    
}

// // Test main to verify build_expert_token_lookup
// int main() {
//     const int batch_size = 2;
//     const int seq_len = 10;
//     const int topk = 3;
//     const int n_experts = 3;
//     int M = batch_size * seq_len;
//     int N = M * topk;

//     std::vector<__half> topk_indices(N);
//     std::vector<__half> topk_scores(N);
//     int ids[] = {2, 1, 0, 0, 2, 1};
//     for (int i = 0; i < N; ++i) {
//         topk_indices[i] = __float2half(float(ids[i]));
//         topk_scores[i]  = __float2half(float(i) / 10.0f);
//     }

//     std::vector<int> tokens_per_expert_end;
//     std::vector<int> token_indices;
//     std::vector<__half> sorted_scores;
//     build_expert_token_lookup(
//         topk_scores.data(), topk_indices.data(),
//         M, topk, n_experts,
//         tokens_per_expert_end, token_indices, sorted_scores
//     );

//     // CPU计算验证
//     // 统计每个expert分配到的token数量
//     std::vector<int> expert_token_count(n_experts, 0);
//     std::vector<int> ref_token_indices;
//     std::vector<__half> ref_sorted_scores;

//     // (token, slot)遍历
//     for (int i = 0; i < M; ++i) {
//         for (int k = 0; k < topk; ++k) {
//             int idx = i * topk + k;
//             int expert = static_cast<int>(__half2float(topk_indices[idx]));
//             expert_token_count[expert]++;
//         }
//     }
//     // 构造tokens_per_expert_end
//     std::vector<int> ref_tokens_per_expert_end(n_experts, 0);
//     int acc = 0;
//     for (int e = 0; e < n_experts; ++e) {
//         acc += expert_token_count[e];
//         ref_tokens_per_expert_end[e] = acc;
//     }

//     // 构造token_indices和sorted_scores
//     std::vector<int> offset(n_experts, 0);
//     for (int e = 1; e < n_experts; ++e) {
//         offset[e] = ref_tokens_per_expert_end[e-1];
//     }
//     ref_token_indices.resize(N);
//     ref_sorted_scores.resize(N);
//     for (int i = 0; i < M; ++i) {
//         for (int k = 0; k < topk; ++k) {
//             int idx = i * topk + k;
//             int expert = static_cast<int>(__half2float(topk_indices[idx]));
//             int pos = offset[expert]++;
//             ref_token_indices[pos] = i;
//             ref_sorted_scores[pos] = topk_scores[idx];
//         }
//     }

//     // 打印CPU参考结果
//     std::cout << "ref_tokens_per_expert_end: ";
//     for (int e = 0; e < n_experts; ++e) std::cout << ref_tokens_per_expert_end[e] << " ";
//     std::cout << std::endl;

//     std::cout << "ref_token_indices: ";
//     for (int i = 0; i < N; ++i) std::cout << ref_token_indices[i] << " ";
//     std::cout << std::endl;

//     std::cout << "ref_sorted_scores: ";
//     for (int i = 0; i < N; ++i) std::cout << __half2float(ref_sorted_scores[i]) << " ";
//     std::cout << std::endl;

//     std::cout << "tokens_per_expert_end: ";
//     for (int e = 0; e < n_experts; ++e) std::cout << tokens_per_expert_end[e] << " ";
//     std::cout << std::endl;

//     std::cout << "token_indices: ";
//     for (int i = 0; i < N; ++i) std::cout << token_indices[i] << " ";
//     std::cout << std::endl;

//     std::cout << "sorted_scores: ";
//     for (int i = 0; i < N; ++i) std::cout << __half2float(sorted_scores[i]) << " ";
//     std::cout << std::endl;

//     return 0;
// }
