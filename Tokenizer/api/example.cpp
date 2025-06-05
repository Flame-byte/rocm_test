#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

// Helper functions for string operations in case C++20 is not available
bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Helper function to create a basic character tokenizer
std::vector<std::pair<std::vector<uint8_t>, Rank>> createBasicVocabulary() {
    std::vector<std::pair<std::vector<uint8_t>, Rank>> vocab;
    
    // Add all ASCII characters as individual tokens
    for (uint8_t c = 0; c < 128; c++) {
        std::vector<uint8_t> token = {c};
        vocab.emplace_back(token, c);
    }
    
    // Add some common tokens
    vocab.emplace_back(std::vector<uint8_t>{'h', 'e', 'l', 'l', 'o'}, 128);
    vocab.emplace_back(std::vector<uint8_t>{' ', 'w', 'o', 'r', 'l', 'd'}, 129);
    vocab.emplace_back(std::vector<uint8_t>{'t', 'e', 's', 't'}, 130);
    vocab.emplace_back(std::vector<uint8_t>{'h', 'e'}, 131);
    vocab.emplace_back(std::vector<uint8_t>{'l', 'l'}, 132);
    vocab.emplace_back(std::vector<uint8_t>{'o', ' '}, 133);
    vocab.emplace_back(std::vector<uint8_t>{'w', 'o'}, 134);
    vocab.emplace_back(std::vector<uint8_t>{'r', 'l'}, 135);
    vocab.emplace_back(std::vector<uint8_t>{'d', ' '}, 136);
    vocab.emplace_back(std::vector<uint8_t>{'t', 'e'}, 137);
    vocab.emplace_back(std::vector<uint8_t>{'s', 't'}, 138);
    
    return vocab;
}

// Helper function to load a tokenizer from saved files (not used in this example)
Tokenizer loadTokenizer(const std::string& vocabPath, const std::string& mergesPath) {
    // Load vocabulary (token -> id)
    std::vector<std::pair<std::vector<uint8_t>, Rank>> encoder;
    std::vector<std::pair<std::string, Rank>> special_tokens;
    
    // Read vocab file - format: token id
    std::ifstream vocabFile(vocabPath);
    std::string line;
    while (std::getline(vocabFile, line)) {
        std::istringstream iss(line);
        std::string tokenStr;
        Rank id;
        if (iss >> tokenStr >> id) {
            // Check if it's a special token (starting with <| and ending with |>)
            if (starts_with(tokenStr, "<|") && ends_with(tokenStr, "|>")) {
                special_tokens.emplace_back(tokenStr, id);
            } else {
                // Regular token - convert from text representation to bytes
                std::vector<uint8_t> tokenBytes(tokenStr.begin(), tokenStr.end());
                encoder.emplace_back(tokenBytes, id);
            }
        }
    }
    
    // Use standard GPT-2 pattern for BPE tokenization
    const std::string pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    
    // Create tokenizer
    return Tokenizer(encoder, special_tokens, pattern);
}

int main() {
    try {
        // Create a more complete vocabulary including individual characters
        auto encoder = createBasicVocabulary();
        
        std::vector<std::pair<std::string, Rank>> special_tokens = {
            {"<|endoftext|>", 200}
        };
        
        // Use a simpler pattern for our basic tokenizer
        std::string pattern = " ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+";
        
        std::cout << "Creating tokenizer with " << encoder.size() << " tokens in vocabulary" << std::endl;
        
        // Create tokenizer
        Tokenizer tokenizer(encoder, special_tokens, pattern);
        
        // Test encoding
        std::string text = "hello world test";
        std::cout << "Encoding text: '" << text << "'" << std::endl;
        
        std::vector<Rank> tokens = tokenizer.EncodeOrdinary(text);
        
        std::cout << "Encoded tokens: ";
        for (Rank token : tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        
        // Test decoding
        std::string decoded = tokenizer.Decode(tokens);
        std::cout << "Decoded text: '" << decoded << "'" << std::endl;
        
        // Test encoding with special tokens
        std::string text_with_special = "hello world <|endoftext|>";
        std::vector<std::string> allowed_special = {"<|endoftext|>"};
        std::vector<Rank> tokens_with_special = tokenizer.Encode(text_with_special, allowed_special);
        
        std::cout << "Encoded tokens with special: ";
        for (Rank token : tokens_with_special) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        
        // Get all special tokens
        auto special_tokens_list = tokenizer.GetSpecialTokens();
        std::cout << "Special tokens: " << std::endl;
        for (const auto& pair : special_tokens_list) {
            std::cout << "  " << pair.first << ": " << pair.second << std::endl;
        }
        
        // Test with individual characters
        std::string char_text = "ABC";
        std::cout << "\nEncoding individual characters: '" << char_text << "'" << std::endl;
        std::vector<Rank> char_tokens = tokenizer.EncodeOrdinary(char_text);
        
        std::cout << "Encoded character tokens: ";
        for (Rank token : char_tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        
        std::string char_decoded = tokenizer.Decode(char_tokens);
        std::cout << "Decoded characters: '" << char_decoded << "'" << std::endl;
        
    } catch (const TokenizerException& e) {
        std::cerr << "Tokenizer error: " << e.what() 
                  << " (error code: " << static_cast<int>(e.error_code) << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 