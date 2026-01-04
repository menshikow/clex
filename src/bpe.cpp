#include "bpe.h"
#include <algorithm>
#include <print>

namespace bpe {

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

MergeRules train(std::vector<uint8_t> &raw_bytes, int vocab_size) {
  std::vector<int> ids;
  for (uint8_t b : raw_bytes) {
    ids.push_back((int)b);
  }

  MergeRules merges;
  int num_merges = vocab_size - 256;

  std::println("[Training] Starting with {} tokens. Target merges: {}",
               ids.size(), num_merges);

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
    std::println("Merge {}/{}: ({}, {}) -> {} (Count: {})", i + 1, num_merges,
                 best_pair.first, best_pair.second, new_id, max_count);

    ids = merge(ids, best_pair, new_id);
  }

  std::println("[Training] Final sequence length: {}\n", ids.size());

  return merges;
}

Vocabulary init_vocab() {
  Vocabulary vocab;
  vocab.reserve(256);
  for (int i = 0; i <= 255; i++) {
    vocab.push_back({(uint8_t)i});
  }
  return vocab;
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

std::string decode(const std::vector<int> &ids, const Vocabulary &vocab) {
  std::vector<uint8_t> raw_bytes;

  for (int id : ids) {
    const TokenContent &token_bytes = vocab[id];
    raw_bytes.insert(raw_bytes.end(), token_bytes.begin(), token_bytes.end());
  }

  return std::string(raw_bytes.begin(), raw_bytes.end());
}

} // namespace bpe
