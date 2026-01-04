#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace bpe {

using TokenContent = std::vector<uint8_t>;
using Vocabulary = std::vector<TokenContent>;
using Stats = std::map<std::pair<int, int>, int>;
using MergeRules = std::map<std::pair<int, int>, int>;

Stats get_stats(const std::vector<int> &ids);

std::vector<int> merge(const std::vector<int> &ids, std::pair<int, int> pair,
                       int new_token_id);
MergeRules train(std::vector<uint8_t> &raw_bytes, int vocab_size);

Vocabulary init_vocab();

Vocabulary build_vocab(const MergeRules &merges);

std::vector<int> encode(const std::string &text, const MergeRules &merges);

std::string decode(const std::vector<int> &ids, const Vocabulary &vocab);

std::vector<uint8_t> read_file_to_bytes(const std::string &path);

void save_model(const MergeRules &merges, const std::string &filename);

MergeRules load_model(const std::string &filename);

void print_usage(const char *program_name);

int handle_train(int argc, char *argv[]);

int handle_encode(int argc, char *argv[]);

int handle_decode(int argc, char *argv[]);

int handle_test();

} // namespace bpe
