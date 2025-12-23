/**
 * @file main.cpp
 * @brief Main entry point for clex BPE tokenizer
 */

#include "bpe.h"
#include <iostream>
#include <print>
#include <string>

/**
 * @brief Main function - routes commands to appropriate handlers
 * @param argc Argument count
 * @param argv Argument vector
 * @return Exit code (0 on success, 1 on error)
 */
int main(int argc, char *argv[]) {
  using namespace bpe;

  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string command = argv[1];

  if (command == "train") {
    return handle_train(argc, argv);
  } else if (command == "encode") {
    return handle_encode(argc, argv);
  } else if (command == "decode") {
    return handle_decode(argc, argv);
  } else if (command == "test") {
    return handle_test();
  } else {
    std::println(std::cerr, "Error: Unknown command '{}'", command);
    print_usage(argv[0]);
    return 1;
  }
}
