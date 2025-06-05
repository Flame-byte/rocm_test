#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

// Type alias to match Rust's Rank
using Rank = uint32_t;

// Error codes from Rust
enum class TokenizerError {
    Success = 0,
    InvalidToken = 1,
    DecodingError = 2,
    EncodingError = 3,
    InvalidArgument = 4,
    OutOfMemory = 5
};

// Define C struct types to match Rust FFI types exactly
struct CoreBPEHandle {
    void* _0;
};

struct EncodedTokens {
    Rank* tokens;
    size_t length;
};

struct DecodedBytes {
    uint8_t* bytes;
    size_t length;
};

// Declare the extern "C" functions
extern "C" {
    CoreBPEHandle create_core_bpe(
        const uint8_t* const* encoder_bytes,
        const size_t* encoder_sizes,
        const Rank* encoder_ranks,
        size_t encoder_count,
        const char* const* special_tokens_strs,
        const Rank* special_tokens_ranks,
        size_t special_tokens_count,
        const char* pattern,
        TokenizerError* err_code
    );

    void free_core_bpe(CoreBPEHandle handle);

    EncodedTokens encode_ordinary(
        CoreBPEHandle handle,
        const char* text,
        TokenizerError* err_code
    );

    EncodedTokens encode(
        CoreBPEHandle handle,
        const char* text,
        const char* const* allowed_special,
        size_t allowed_special_count,
        TokenizerError* err_code
    );

    DecodedBytes decode_bytes(
        CoreBPEHandle handle,
        const Rank* tokens,
        size_t tokens_length,
        TokenizerError* err_code
    );

    int encode_single_token(
        CoreBPEHandle handle,
        const uint8_t* bytes,
        size_t bytes_length,
        Rank* out_token,
        TokenizerError* err_code
    );

    void free_tokens(EncodedTokens tokens);
    void free_bytes(DecodedBytes bytes);

    size_t get_special_tokens_count(CoreBPEHandle handle);
    
    char* get_special_token_at_index(
        CoreBPEHandle handle,
        size_t index,
        Rank* out_token,
        TokenizerError* err_code
    );

    void free_string(char* ptr);
}

// C++ exception for tokenizer errors
class TokenizerException : public std::runtime_error {
public:
    TokenizerException(TokenizerError code, const char* message)
        : std::runtime_error(message), error_code(code) {}
    
    TokenizerError error_code;
};

// C++ wrapper for the tokenizer
class Tokenizer {
public:
    // Constructor that takes encoder map, special tokens and regex pattern
    Tokenizer(
        const std::vector<std::pair<std::vector<uint8_t>, Rank>>& encoder,
        const std::vector<std::pair<std::string, Rank>>& special_tokens,
        const std::string& pattern
    ) {
        // Prepare encoder data
        std::vector<const uint8_t*> encoder_bytes;
        std::vector<size_t> encoder_sizes;
        std::vector<Rank> encoder_ranks;
        
        for (const auto& pair : encoder) {
            encoder_bytes.push_back(pair.first.data());
            encoder_sizes.push_back(pair.first.size());
            encoder_ranks.push_back(pair.second);
        }
        
        // Prepare special tokens data
        std::vector<const char*> special_tokens_strs;
        std::vector<Rank> special_tokens_ranks;
        
        for (const auto& pair : special_tokens) {
            special_tokens_strs.push_back(pair.first.c_str());
            special_tokens_ranks.push_back(pair.second);
        }
        
        // Create the CoreBPE instance
        TokenizerError err = TokenizerError::Success;
        handle_ = create_core_bpe(
            encoder_bytes.data(),
            encoder_sizes.data(),
            encoder_ranks.data(),
            encoder.size(),
            special_tokens_strs.data(),
            special_tokens_ranks.data(),
            special_tokens.size(),
            pattern.c_str(),
            &err
        );
        
        if (err != TokenizerError::Success) {
            throw TokenizerException(err, "Failed to create tokenizer");
        }
    }
    
    // Destructor
    ~Tokenizer() {
        if (handle_._0 != nullptr) {
            free_core_bpe(handle_);
        }
    }
    
    // Encode ordinary text (ignoring special tokens)
    std::vector<Rank> EncodeOrdinary(const std::string& text) {
        TokenizerError err = TokenizerError::Success;
        EncodedTokens tokens = encode_ordinary(handle_, text.c_str(), &err);
        
        if (err != TokenizerError::Success) {
            free_tokens(tokens);
            throw TokenizerException(err, "Failed to encode text");
        }
        
        std::vector<Rank> result;
        if (tokens.tokens != nullptr && tokens.length > 0) {
            result.assign(tokens.tokens, tokens.tokens + tokens.length);
        }
        free_tokens(tokens);
        return result;
    }
    
    // Encode text with special tokens
    std::vector<Rank> Encode(
        const std::string& text,
        const std::vector<std::string>& allowed_special = {}
    ) {
        std::vector<const char*> allowed_special_cstr;
        for (const auto& token : allowed_special) {
            allowed_special_cstr.push_back(token.c_str());
        }
        
        TokenizerError err = TokenizerError::Success;
        EncodedTokens tokens = encode(
            handle_,
            text.c_str(),
            allowed_special_cstr.data(),
            allowed_special_cstr.size(),
            &err
        );
        
        if (err != TokenizerError::Success) {
            free_tokens(tokens);
            throw TokenizerException(err, "Failed to encode text with special tokens");
        }
        
        std::vector<Rank> result;
        if (tokens.tokens != nullptr && tokens.length > 0) {
            result.assign(tokens.tokens, tokens.tokens + tokens.length);
        }
        free_tokens(tokens);
        return result;
    }
    
    // Decode tokens to string
    std::string Decode(const std::vector<Rank>& tokens) {
        if (tokens.empty()) {
            return "";
        }
        
        TokenizerError err = TokenizerError::Success;
        DecodedBytes bytes = decode_bytes(
            handle_,
            tokens.data(),
            tokens.size(),
            &err
        );
        
        if (err != TokenizerError::Success) {
            free_bytes(bytes);
            throw TokenizerException(err, "Failed to decode tokens");
        }
        
        std::string result;
        if (bytes.bytes != nullptr && bytes.length > 0) {
            result.assign(reinterpret_cast<char*>(bytes.bytes), bytes.length);
        }
        free_bytes(bytes);
        return result;
    }
    
    // Encode a single token
    Rank EncodeSingleToken(const std::vector<uint8_t>& bytes) {
        TokenizerError err = TokenizerError::Success;
        Rank token = 0;
        
        int success = encode_single_token(
            handle_,
            bytes.data(),
            bytes.size(),
            &token,
            &err
        );
        
        if (success == 0 || err != TokenizerError::Success) {
            throw TokenizerException(err, "Failed to encode single token");
        }
        
        return token;
    }
    
    // Encode a single token from string
    Rank EncodeSingleToken(const std::string& text) {
        std::vector<uint8_t> bytes(text.begin(), text.end());
        return EncodeSingleToken(bytes);
    }
    
    // Get all special tokens as a map
    std::vector<std::pair<std::string, Rank>> GetSpecialTokens() {
        std::vector<std::pair<std::string, Rank>> result;
        size_t count = get_special_tokens_count(handle_);
        
        for (size_t i = 0; i < count; i++) {
            TokenizerError err = TokenizerError::Success;
            Rank token = 0;
            
            char* token_str = get_special_token_at_index(handle_, i, &token, &err);
            
            if (err != TokenizerError::Success || token_str == nullptr) {
                if (token_str) free_string(token_str);
                throw TokenizerException(err, "Failed to get special token");
            }
            
            std::string str(token_str);
            free_string(token_str);
            
            result.emplace_back(str, token);
        }
        
        return result;
    }
    
private:
    CoreBPEHandle handle_ = {nullptr};
    
    // Prevent copying
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
};

#endif // TOKENIZER_H 