#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

using TokenContent = std::vector<uint8_t>;
using Vocabulary = std::vector<TokenContent>;
using Stats = std::map<std::pair<int, int>, int>;
using MergeRules = std::map<std::pair<int, int>, int>;

std::vector<uint8_t> read_file_to_bytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Failed to open file: " << path << std::endl;
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

Vocabulary init_vocab() {
  Vocabulary vocab;
  vocab.reserve(256);
  for (int i = 0; i <= 255; i++) {
    vocab.push_back({(uint8_t)i});
  }
  return vocab;
}

// core logic

Stats get_stats(const std::vector<int> &ids) {
  Stats counts;
  if (ids.size() < 2)
    return counts;
  for (size_t i = 0; i < ids.size() - 1; ++i) {
    std::pair<int, int> pair = {ids[i], ids[i + 1]};
    counts[pair]++;
  }
  return counts;
}

std::vector<int> merge(const std::vector<int> &ids, std::pair<int, int> pair,
                       int new_token_id) {
  std::vector<int> new_ids;
  new_ids.reserve(ids.size());
  size_t i = 0;
  while (i < ids.size()) {
    if (i < ids.size() - 1 && ids[i] == pair.first &&
        ids[i + 1] == pair.second) {
      new_ids.push_back(new_token_id);
      i += 2;
    } else {
      new_ids.push_back(ids[i]);
      i += 1;
    }
  }
  return new_ids;
}

// training

MergeRules train(std::vector<uint8_t> &raw_bytes, int vocab_size) {
  std::vector<int> ids;
  for (uint8_t b : raw_bytes) {
    ids.push_back((int)b);
  }

  MergeRules merges;
  int num_merges = vocab_size - 256;

  std::cout << "[Training] Starting with " << ids.size()
            << " tokens. Target merges: " << num_merges << "\n";

  for (int i = 0; i < num_merges; ++i) {
    Stats stats = get_stats(ids);
    std::pair<int, int> best_pair;
    int max_count = -1;

    for (auto const &[pair, count] : stats) {
      if (count > max_count) {
        max_count = count;
        best_pair = pair;
      }
    }

    if (max_count < 1)
      break;

    int new_id = 256 + i;
    merges[best_pair] = new_id;
    std::cout << "Merge " << (i + 1) << "/" << num_merges << ": ("
              << best_pair.first << ", " << best_pair.second << ") -> "
              << new_id << " (Count: " << max_count << ")\n";

    ids = merge(ids, best_pair, new_id);
  }

  std::cout << "[Training] Final sequence length: " << ids.size() << "\n";
  return merges;
}

// encode and decode
std::vector<int> encode(const std::string &text, const MergeRules &merges) {
  std::vector<int> ids;
  for (char c : text) {
    ids.push_back((uint8_t)c);
  }

  std::vector<std::pair<std::pair<int, int>, int>> sorted_rules;
  for (const auto &entry : merges) {
    sorted_rules.push_back(entry);
  }

  std::sort(sorted_rules.begin(), sorted_rules.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

  for (const auto &[pair, new_id] : sorted_rules) {
    ids = merge(ids, pair, new_id);
  }

  return ids;
}

Vocabulary build_vocab(const MergeRules &merges) {
  Vocabulary vocab = init_vocab();

  std::vector<std::pair<std::pair<int, int>, int>> sorted_rules;
  for (const auto &entry : merges) {
    sorted_rules.push_back(entry);
  }
  std::sort(sorted_rules.begin(), sorted_rules.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

  for (const auto &[pair, new_id] : sorted_rules) {
    TokenContent new_token = vocab[pair.first];
    TokenContent right = vocab[pair.second];
    new_token.insert(new_token.end(), right.begin(), right.end());

    if (vocab.size() <= (size_t)new_id) {
      vocab.resize(new_id + 1);
    }
    vocab[new_id] = new_token;
  }
  return vocab;
}

std::string decode(const std::vector<int> &ids, const Vocabulary &vocab) {
  std::vector<uint8_t> raw_bytes;
  for (int id : ids) {
    const TokenContent &token_bytes = vocab[id];
    raw_bytes.insert(raw_bytes.end(), token_bytes.begin(), token_bytes.end());
  }
  return std::string(raw_bytes.begin(), raw_bytes.end());
}

int main() {
  std::string text = "aaabdaaabac";
  std::vector<uint8_t> input_bytes(text.begin(), text.end());

  std::cout << "Original Text: " << text << "\n";

  int target_vocab_size = 256 + 3;
  MergeRules rules = train(input_bytes, target_vocab_size);

  std::vector<int> encoded = encode(text, rules);
  std::cout << "\n[Encode] Result: ";
  for (int id : encoded)
    std::cout << id << " ";
  std::cout << "\n";

  Vocabulary vocab = build_vocab(rules);
  std::string decoded = decode(encoded, vocab);
  std::cout << "[Decode] Result: " << decoded << "\n";

  if (text == decoded) {
    std::cout << "\nSUCCESS: Round-trip verified!\n";
  } else {
    std::cout << "\nFAILURE: Decoded text does not match original.\n";
  }

  return 0;
}