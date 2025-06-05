#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "load.h"
#include "tokenizer.h"

class TokenizerManager {
    public:
    
    TokenizerManager(const std::string& model_path){
        init(model_path);
    }

    std::map<std::vector<unsigned char>, int> special_tokens;
    int num_reserved_special_tokens = 256;
    std::string pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    Tokenizer* tokenizer;

    void init(const std::string& model_path) {
        // Get encoder directly in the required format
        auto encoder = load_tiktoken_bpe(model_path);
        int num_base_tokens = encoder.size();

        // Define special tokens
        std::vector<std::string> special_tokens = {
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>"  // end of turn
        };

        // Add additional reserved special tokens
        for (int i = 5; i < num_reserved_special_tokens - 5; ++i) {
            special_tokens.push_back("<|reserved_special_token_" + std::to_string(i) + "|>");
        }

        // Convert special tokens to the expected format
        std::vector<std::pair<std::string, Rank>> special_tokens_pairs;
        for (size_t i = 0; i < special_tokens.size(); ++i) {
            special_tokens_pairs.emplace_back(special_tokens[i], static_cast<Rank>(num_base_tokens + i));
        }

        // Initialize Tokenizer
        tokenizer = new Tokenizer(encoder, special_tokens_pairs, pattern);
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<Rank> rank_tokens = tokenizer->Encode(text);
        return std::vector<int>(rank_tokens.begin(), rank_tokens.end());
    }

    std::string decode(const std::vector<int>& tokens) {
        std::vector<Rank> rank_tokens(tokens.begin(), tokens.end());
        return tokenizer->Decode(rank_tokens);
    }
    
    ~TokenizerManager() {
        if (tokenizer != nullptr) {
            delete tokenizer;
        }
    }
};

std::string encode_dialog_prompt(std::vector<std::string> messages, bool user, bool assistant)
{
    std::string formatted;
    if(user)
    {
        // Insert user message with appropriate tokens
        formatted = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n";
        
        // Add the user message content
        if (!messages.empty()) {
            formatted += messages[0];
        }
        
        // Add end of text token and start assistant header
        formatted += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
        
    }
    else if(assistant)
    {
        // Insert user message with appropriate tokens
        formatted = "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n";
        
        // Add the user message content
        if (!messages.empty()) {
            formatted += messages[0];
        }
        
        // Add end of text token and start assistant header
        formatted += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
        
    }

    return formatted;
}



int main() {
    std::string model_path = "/home/qin/rocm_test/Tokenizer/tokenizer.model";
    TokenizerManager tokenizerManager(model_path);

    std::vector<std::string> messages = {"Hello, world!"};
    std::string messages_str = encode_dialog_prompt(messages, true, false);
    std::cout << "Encoded messages_str: ";
    std::cout << messages_str << std::endl;
    std::cout << std::endl;

    std::vector<int> tokens = tokenizerManager.encode(messages_str);
    std::cout << "Encoded tokens: ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    std::string decoded_messages = tokenizerManager.decode(tokens);
    std::cout << "Decoded messages: " << decoded_messages << std::endl;

    return 0;
}