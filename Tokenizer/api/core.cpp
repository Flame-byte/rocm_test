#include "tokenizer.h"
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

class Encoding {
    public:
    void init(
        std::vector<unsigned char> &bpe_ranks,
        std::vector<unsigned char> &special_tokens,
        bool explicit_n_vocab
    )
    {

    }

    void special_token_set

    void encode(
        const std::string& text,
        std::vector<Rank>& tokens,
        const std::vector<std::string>& allowed_special = {},
        bool add_special_tokens = true
        const std::vector<std::string>& disallowed_special = {},
    )
    {
        if(add_special_tokens)
        {

        }
    }
    
};