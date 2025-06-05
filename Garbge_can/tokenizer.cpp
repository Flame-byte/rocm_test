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
    Tokenizer* tokenizer;

    std::map<std::vector<unsigned char>, int> special_tokens;
    int num_reserved_special_tokens = 256;
    //std::string pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    //std::string  pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    std::string pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    TokenizerManager() : tokenizer(nullptr) {}
    
    ~TokenizerManager() {
        if (tokenizer) {
            delete tokenizer;
            tokenizer = nullptr;
        }
    }
    
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

        // Convert special tokens to the expected format with IDs starting from 128000
        // This matches the Python implementation
        const int SPECIAL_TOKEN_START_ID = 128000;
        std::vector<std::pair<std::string, Rank>> special_tokens_pairs;
        for (size_t i = 0; i < special_tokens.size(); ++i) {
            special_tokens_pairs.emplace_back(special_tokens[i], static_cast<Rank>(SPECIAL_TOKEN_START_ID + i));
        }

        // Initialize Tokenizer
        tokenizer = new Tokenizer(encoder, special_tokens_pairs, pattern);
    }

    std::vector<int> encode(const std::vector<std::string>& texts) {
        // Create a single string from all text parts to properly handle merging
        std::string combined_text;
        for (const auto& text : texts) {
            combined_text += text;
        }
        
        // Allow all special tokens during encoding
        std::vector<std::string> allowed_special = {
            "<|begin_of_text|>", 
            "<|end_of_text|>", 
            "<|reserved_special_token_0|>", 
            "<|reserved_special_token_1|>", 
            "<|reserved_special_token_2|>", 
            "<|reserved_special_token_3|>", 
            "<|start_header_id|>", 
            "<|end_header_id|>", 
            "<|reserved_special_token_4|>", 
            "<|eot_id|>"
        };
        
        // Get tokens for the combined text
        std::vector<Rank> rank_tokens = tokenizer->Encode(combined_text, allowed_special);
        
        // Convert to vector<int>
        std::vector<int> result(rank_tokens.begin(), rank_tokens.end());
        return result;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::vector<Rank> rank_tokens(tokens.begin(), tokens.end());
        return tokenizer->Decode(rank_tokens);
    }

};



std::vector<std::string> encode_dialog_prompt(std::vector<std::string> messages, bool user, bool assistant)
{
    std::vector<std::string> formatted;
    if(user)
    {
        // Insert user message with appropriate tokens
        formatted.push_back("<|begin_of_text|>");
        formatted.push_back("<|start_header_id|>");
        formatted.push_back("user");
        formatted.push_back("<|end_header_id|>");
        formatted.push_back("\n\n");  // Add double newline to match Python version
        
        // Add the user message content
        if (!messages.empty()) {
            formatted.push_back(messages[0]);
        }
        
        // Add end of text token and start assistant header
        formatted.push_back("<|eot_id|>");
        formatted.push_back("<|start_header_id|>");
        formatted.push_back("assistant");
        formatted.push_back("<|end_header_id|>");
        formatted.push_back("\n\n");
        
    }
    else if(assistant)
    {
        // Insert user message with appropriate tokens
        formatted.push_back("<|begin_of_text|>");
        formatted.push_back("<|start_header_id|>");
        formatted.push_back("assistant");
        formatted.push_back("<|end_header_id|>");
        formatted.push_back("\n\n");  // Add double newline to match Python version

        // Add the user message content
        if (!messages.empty()) {
            formatted.push_back(messages[0]);
        }
        
        // Add end of text token and start assistant header
        formatted.push_back("<|eot_id|>");
        formatted.push_back("<|start_header_id|>");
        formatted.push_back("assistant");
        formatted.push_back("<|end_header_id|>");
        formatted.push_back("\n\n");
        
    }

    return formatted;
}



int main() {
    std::string model_path = "/home/qin/rocm_test/Tokenizer/tokenizer.model";
    TokenizerManager tokenizerManager;
    tokenizerManager.init(model_path);

    // Check if tokenizer is initialized
    if (!tokenizerManager.tokenizer) {
        std::cerr << "Failed to initialize tokenizer!" << std::endl;
        return 1;
    }

    std::vector<std::string> messages = {"The issue seems to be related to the merging of tokens during the encoding process in tokenizer.cpp."};
    std::vector<std::string> messages_str = encode_dialog_prompt(messages, true, false);
    std::cout << "Encoded messages_str: ";
    for (const auto& str : messages_str) {
        std::cout << str << " ";
    }
    std::cout << std::endl;

    std::vector<int> tokens = tokenizerManager.encode(messages_str);
    
    // Print tokens in a format similar to Python's output (as a nested array)
    std::cout << "Encoded tokens: [[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]]" << std::endl;
    
    std::cout << "Number of elements in the first dimension of prompt_tokens: " << tokens.size() << std::endl;
    
    std::string decoded_messages = tokenizerManager.decode(tokens);
    std::cout << "Decoded prompt: " << decoded_messages << std::endl;

    return 0;
}