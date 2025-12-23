/**
 * @file io.cpp
 * @brief File I/O operations for BPE tokenizer
 *
 * Handles reading files, saving models, and loading trained tokenizers.
 */

#include "bpe.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <print>
#include <string>
#include <vector>

namespace bpe {

std::vector<uint8_t> read_file_to_bytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::println(std::cerr, "Failed to open file: {}", path);
    return {};
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(size);

  if (file.read((char *)buffer.data(), size)) {
    return buffer;
  }

  return {};
}

void save_model(const MergeRules &merges, const std::string &filename) {
  std::ofstream file(filename);

  if (!file) {
    std::println(std::cerr, "Failed to open for writing: {}", filename);
    return;
  }

  // sort by token id so the file order matches 256, 257, 258..
  std::vector<std::pair<std::pair<int, int>, int>> sorted_rules;
  for (const auto &entry : merges) {
    sorted_rules.push_back(entry);
  }
  std::sort(sorted_rules.begin(), sorted_rules.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

  for (const auto &[pair, new_id] : sorted_rules) {
    file << pair.first << " " << pair.second << " " << new_id << "\n";
  }

  std::println("Saved {} merge rules to {}", merges.size(), filename);
}

MergeRules load_model(const std::string &filename) {
  MergeRules merges;
  std::ifstream file(filename);

  if (!file) {
    std::println(std::cerr, "Failed to open for reading: {}", filename);
    return merges;
  }

  int first, second, new_id;
  while (file >> first >> second >> new_id) {
    merges[{first, second}] = new_id;
  }

  std::println("Loaded {} merge rules from {}", merges.size(), filename);
  return merges;
}

} // namespace bpe
