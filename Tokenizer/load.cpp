#include "load.h"

// Function to decode a base64 encoded string
std::vector<unsigned char> base64_decode(const std::string &in) {
    std::string base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    std::vector<unsigned char> out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[base64_chars[i]] = i;

    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// Function to encode a string to base64
std::string base64_encode(const std::vector<unsigned char> &in) {
    std::string base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    std::string out;
    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

// Function to load BPE ranks from a file
std::map<std::vector<unsigned char>, int> load_tiktoken_bpe(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::map<std::vector<unsigned char>, int> bpe_ranks;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string token_str;
        int rank;
        if (!(iss >> token_str >> rank)) {
            throw std::runtime_error("Error parsing line: " + line);
        }

        std::vector<unsigned char> token = base64_decode(token_str);
        bpe_ranks[token] = rank;
    }

    return bpe_ranks;
}

int main() {

    std::string file_path = "/home/qin/rocm_test/Tokenizer/tokenizer.model";
    auto bpe_ranks = load_tiktoken_bpe(file_path);

    // Convert the map to a vector of pairs and sort by rank
    std::vector<std::pair<std::vector<unsigned char>, int>> sorted_bpe_ranks(bpe_ranks.begin(), bpe_ranks.end());
    std::sort(sorted_bpe_ranks.begin(), sorted_bpe_ranks.end(), [](const auto &a, const auto &b) {
        return a.second < b.second;
    });

    // Output the first 10 items
    std::cout << "First 10 items in mergeable_ranks:" << std::endl;
    for (int i = 0; i < 10 && i < sorted_bpe_ranks.size(); ++i) {
        const auto &token = sorted_bpe_ranks[i].first;
        int rank = sorted_bpe_ranks[i].second;

        // Print token and its base64 representation
        std::cout << i + 1 << ". Token: ";
        for (unsigned char c : token) {
            std::cout << c;
        }

        // // Base64 encode the token
        // std::string base64_encoded = base64_encode(token);
        // std::cout << " (base64: " << base64_encoded << "), Rank: " << rank << std::endl;

        std::cout << " Rank: " << rank << std::endl;
    }

    return 0;
}