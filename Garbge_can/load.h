#ifndef LOAD_H
#define LOAD_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <stdexcept>
#include <iterator>
#include <algorithm>
#include "tokenizer.h"

std::vector<unsigned char> base64_decode(const std::string &in);
std::string base64_encode(const std::vector<unsigned char> &in);
std::vector<std::pair<std::vector<uint8_t>, Rank>> load_tiktoken_bpe(const std::string &file_path);


#endif