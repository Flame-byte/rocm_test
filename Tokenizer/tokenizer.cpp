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

class My_Tokenizer {
    public:

    std::map<std::vector<unsigned char>, int> special_tokens;

    int num_reserved_special_tokens = 256;

    std::string pattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";


    void init(const std::string& model_path){
        std::map<std::vector<unsigned char>, int> mergeable_ranks = load_tiktoken_bpe(model_path);
        int num_base_tokens = mergeable_ranks.size();

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

        // Map special tokens to unique integer values
        std::unordered_map<std::string, int> special_tokens_map;
        for (size_t i = 0; i < special_tokens.size(); ++i) {
            special_tokens_map[special_tokens[i]] = num_base_tokens + i;
        }

        Tokenizer tokenizer(model_path, special_tokens_map, pattern);
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens = tokenizer.Encode(text);
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) {
        std::string decoded = tokenizer.Decode(tokens);
        return decoded;
    }
};

class ChatFormat {

    My_Tokenizer my_tokenizer;

    public:
    std::vector<std::string> encode_dialog_prompt(std::vector<std::string> messages, bool user, bool assistant)
    {
        if(user)
        {
            // Insert user message with appropriate tokens
            std::string formatted = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n";
            
            // Add the user message content
            if (!messages.empty()) {
                formatted += messages[0];
            }
            
            // Add end of text token and start assistant header
            formatted += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
            
            return formatted; // Return first char as per function signature
        }
        else if(assistant)
        {
            // Insert user message with appropriate tokens
            std::string formatted = "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n";
            
            // Add the user message content
            if (!messages.empty()) {
                formatted += messages[0];
            }
            
            // Add end of text token and start assistant header
            formatted += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
            
            return formatted; // Return first char as per function signature
        }

        std::vector<int> tokens = my_tokenizer.encode(formatted);
        return tokens;
    }


    
};

int main() {
    My_Tokenizer my_tokenizer;
    ChatFormat chat_format;
    my_tokenizer.init("/home/qin/rocm_test/Tokenizer/tokenizer.model");

    std::vector<std::string> messages = {"Hello, world!"};
    std::vector<std::string> encoded_messages = chat_format.encode_dialog_prompt(messages, true, false);
    std::cout << "Encoded messages: ";
    for (int token : encoded_messages) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    std::string decoded = my_tokenizer.decode(encoded_messages);
    std::cout << "Decoded messages: " << decoded << std::endl;

}