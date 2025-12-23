#include "bpe.h"
#include <fstream>
#include <iostream>
#include <print>
#include <string>
#include <vector>

namespace bpe {

void print_usage(const char *program_name) {
  std::println("Usage: {} <command> [options]", program_name);
  std::println("\nCommands:");
  std::println("  train <input_file> <output_model> <vocab_size>");
  std::println(
      "    Train a BPE tokenizer on input_file and save to output_model");
  std::println("    vocab_size must be >= 256");
  std::println("\n  encode <model_file> <input_file> [output_file]");
  std::println("    Encode input_file using model_file");
  std::println("    If output_file is omitted, prints token IDs to stdout");
  std::println("\n  decode <model_file> <input_file> [output_file]");
  std::println("    Decode token IDs from input_file using model_file");
  std::println("    If output_file is omitted, prints decoded text to stdout");
  std::println("\n  test");
  std::println("    Run built-in test with example text");
}

int handle_train(int argc, char *argv[]) {
  if (argc < 5) {
    std::println(std::cerr,
                 "Error: train requires input_file, output_model, and vocab_size");
    return 1;
  }

  std::string input_file = argv[2];
  std::string output_model = argv[3];
  int vocab_size = std::stoi(argv[4]);

  if (vocab_size < 256) {
    std::println(std::cerr, "Error: vocab_size must be >= 256");
    return 1;
  }

  std::vector<uint8_t> input_bytes = read_file_to_bytes(input_file);
  if (input_bytes.empty()) {
    std::println(std::cerr, "Error: Failed to read input file");
    return 1;
  }

  MergeRules rules = train(input_bytes, vocab_size);
  save_model(rules, output_model);
  return 0;
}

int handle_encode(int argc, char *argv[]) {
  if (argc < 4) {
    std::println(std::cerr, "Error: encode requires model_file and input_file");
    return 1;
  }

  std::string model_file = argv[2];
  std::string input_file = argv[3];

  MergeRules rules = load_model(model_file);
  if (rules.empty()) {
    std::println(std::cerr, "Error: Failed to load model");
    return 1;
  }

  std::vector<uint8_t> input_bytes = read_file_to_bytes(input_file);
  if (input_bytes.empty()) {
    std::println(std::cerr, "Error: Failed to read input file");
    return 1;
  }

  std::string text(input_bytes.begin(), input_bytes.end());
  std::vector<int> encoded = encode(text, rules);

  if (argc >= 5) {
    std::string output_file = argv[4];
    std::ofstream out(output_file);
    if (!out) {
      std::println(std::cerr, "Error: Failed to open output file");
      return 1;
    }
    for (size_t i = 0; i < encoded.size(); ++i) {
      out << encoded[i];
      if (i < encoded.size() - 1)
        out << " ";
    }
    out << "\n";
    std::println("Encoded {} tokens to {}", encoded.size(), output_file);
  } else {
    for (size_t i = 0; i < encoded.size(); ++i) {
      std::print("{}", encoded[i]);
      if (i < encoded.size() - 1)
        std::print(" ");
    }
    std::println("");
  }

  return 0;
}

int handle_decode(int argc, char *argv[]) {
  if (argc < 4) {
    std::println(std::cerr, "Error: decode requires model_file and input_file");
    return 1;
  }

  std::string model_file = argv[2];
  std::string input_file = argv[3];

  MergeRules rules = load_model(model_file);
  if (rules.empty()) {
    std::println(std::cerr, "Error: Failed to load model");
    return 1;
  }

  Vocabulary vocab = build_vocab(rules);

  std::ifstream in(input_file);
  if (!in) {
    std::println(std::cerr, "Error: Failed to open input file");
    return 1;
  }

  std::vector<int> ids;
  int id;
  while (in >> id) {
    ids.push_back(id);
  }

  std::string decoded = decode(ids, vocab);

  if (argc >= 5) {
    std::string output_file = argv[4];
    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
      std::println(std::cerr, "Error: Failed to open output file");
      return 1;
    }
    out.write(decoded.data(), decoded.size());
    std::println("Decoded {} tokens to {}", ids.size(), output_file);
  } else {
    std::print("{}", decoded);
  }

  return 0;
}

int handle_test() {
  std::string text = "aaabdaaabac";
  std::vector<uint8_t> input_bytes(text.begin(), text.end());

  std::println("Original Text: {}\n", text);

  int target_vocab_size = 256 + 3;
  MergeRules rules = train(input_bytes, target_vocab_size);

  std::vector<int> encoded = encode(text, rules);
  std::println("\n[Encode] Result: ");
  for (size_t i = 0; i < encoded.size(); ++i) {
    std::print("{}", encoded[i]);
    if (i < encoded.size() - 1)
      std::print(" ");
  }
  std::println("\n");

  Vocabulary vocab = build_vocab(rules);
  std::string decoded = decode(encoded, vocab);
  std::println("[Decode] Result: {}\n", decoded);

  if (text == decoded) {
    std::println("SUCCESS: Round-trip verified!\n");
    return 0;
  } else {
    std::println("FAILURE: Decoded text does not match original.\n");
    return 1;
  }
}

} // namespace bpe

