#ifndef ROUTER_H
#define ROUTER_H

#include <vector>
#include <hip/hip_fp16.h>

// Class representing an expert and its assigned tokens and scores
class Expert {
public:
    int id;                       // Expert identifier
    std::vector<int> token_indices; // Indices of tokens assigned to this expert
    std::vector<__half> scores;     // Corresponding gating scores for each token

    Expert(int id_) : id(id_) {}
};

// Group tokens and their scores by expert
std::vector<Expert> assign_tokens_to_experts(
    const __half* topk_scores,   // [M * topk]
    const __half* topk_indices,  // [M * topk]
    int batch_size,
    int seq_len,
    int topk,
    int n_experts
);

#endif // ROUTER_H 