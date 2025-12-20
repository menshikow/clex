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

// --- Helper Functions ---

// Reads a file into raw bytes
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

// Returns a mapping of Byte (0-255) -> Unicode Code Point
std::map<uint8_t, uint32_t> get_byte_to_token_map() {
  std::map<uint8_t, uint32_t> map;
  for (int i = 0; i <= 255; i++) {
    uint8_t b = (uint8_t)i;

    // Printable ASCII (33-126) maps to itself
    bool is_printable = (b >= 33 && b <= 126);

    if (is_printable) {
      map[b] = (uint32_t)b;
    } else {
      // Ugly bytes get shifted by 256
      map[b] = 256 + (uint32_t)b;
    }
  }
  return map;
}

// Initialize Base Vocab
Vocabulary init_vocab() {
  Vocabulary vocab;
  vocab.reserve(256);

  for (int i = 0; i <= 255; i++) {
    vocab.push_back({(uint8_t)i});
  }
  return vocab;
}

// --- Core BPE Logic ---

// Count frequency of adjacent pairs
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

// Replace all occurrences of 'pair' with 'new_token_id'
std::vector<int> merge(const std::vector<int> &ids, std::pair<int, int> pair,
                       int new_token_id) {
  std::vector<int> new_ids;
  new_ids.reserve(ids.size());

  size_t i = 0;
  while (i < ids.size()) {
    // Check if we match the pair at current position
    if (i < ids.size() - 1 && ids[i] == pair.first &&
        ids[i + 1] == pair.second) {
      new_ids.push_back(new_token_id);
      i += 2; // Skip both parts of the pair
    } else {
      new_ids.push_back(ids[i]);
      i += 1;
    }
  }
  return new_ids;
}

// --- Milestone 6: Training Loop ---

MergeRules train(std::vector<uint8_t> &raw_bytes, int vocab_size) {
  // 1. Convert raw bytes to integers (start with 0..255)
  std::vector<int> ids;
  for (uint8_t b : raw_bytes) {
    ids.push_back((int)b);
  }

  MergeRules merges;
  int num_merges = vocab_size - 256; // Base vocab is 256

  std::cout << "[Training] Starting with " << ids.size()
            << " tokens. Target merges: " << num_merges << "\n";

  for (int i = 0; i < num_merges; ++i) {
    // A. Get counts
    Stats stats = get_stats(ids);

    // B. Find the pair with max count
    std::pair<int, int> best_pair;
    int max_count = -1;

    // Simple linear scan to find max (ties broken by order of appearance)
    for (auto const &[pair, count] : stats) {
      if (count > max_count) {
        max_count = count;
        best_pair = pair;
      }
    }

    // Safety check: If no pairs exist (empty file or length 1), stop
    if (max_count < 1)
      break;

    // C. Assign new token ID
    int new_id = 256 + i;

    // D. Record the merge
    merges[best_pair] = new_id;
    std::cout << "Merge " << (i + 1) << "/" << num_merges << ": ("
              << best_pair.first << ", " << best_pair.second << ") -> "
              << new_id << " (Count: " << max_count << ")\n";

    // E. Apply the merge to the sequence
    ids = merge(ids, best_pair, new_id);
  }

  std::cout << "[Training] Final sequence length: " << ids.size() << "\n";
  return merges;
}

// --- Main ---

int main() {
  // 1. Create a dummy input with patterns: "aaabdaaabac"
  // a=97, b=98, c=99, d=100
  std::string text = "aaabdaaabac";
  std::vector<uint8_t> input_bytes(text.begin(), text.end());

  std::cout << "Text: " << text << "\n";

  // 2. Train! We want 3 new tokens (256, 257, 258)
  int target_vocab_size = 256 + 3;
  MergeRules rules = train(input_bytes, target_vocab_size);

  // 3. Print Learned Rules
  std::cout << "\nLearned Rules:\n";
  for (auto const &[pair, new_id] : rules) {
    std::cout << "  (" << pair.first << ", " << pair.second << ") -> " << new_id
              << "\n";
  }

  return 0;
}
