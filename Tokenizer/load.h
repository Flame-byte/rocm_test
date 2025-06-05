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

std::vector<unsigned char> base64_decode(const std::string &in);
std::string base64_encode(const std::vector<unsigned char> &in);
std::map<std::vector<unsigned char>, int> load_tiktoken_bpe(const std::string &file_path);


#endif