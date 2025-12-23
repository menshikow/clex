/**
 * @file bpe.h
 * @brief Byte Pair Encoding (BPE) tokenizer API
 * 
 * This header provides the complete interface for training and using
 * a BPE tokenizer. The implementation follows the standard BPE algorithm
 * used in modern NLP systems.
 */

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace bpe {

/**
 * @brief Type aliases for BPE tokenizer components
 */
using TokenContent = std::vector<uint8_t>;  ///< Raw bytes representing a token
using Vocabulary = std::vector<TokenContent>;  ///< Complete vocabulary mapping IDs to tokens
using Stats = std::map<std::pair<int, int>, int>;  ///< Frequency statistics for token pairs
using MergeRules = std::map<std::pair<int, int>, int>;  ///< Learned merge rules (pair -> new_id)

/**
 * @name Core BPE Operations
 * @{
 */

/**
 * @brief Count frequency of adjacent token pairs
 * @param ids Sequence of token IDs
 * @return Map of token pairs to their occurrence counts
 */
Stats get_stats(const std::vector<int> &ids);

/**
 * @brief Merge all occurrences of a token pair into a new token
 * @param ids Input sequence of token IDs
 * @param pair The token pair to merge (first, second)
 * @param new_token_id The ID to assign to the merged token
 * @return New sequence with merged tokens
 */
std::vector<int> merge(const std::vector<int> &ids, std::pair<int, int> pair,
                       int new_token_id);

/**
 * @brief Train a BPE tokenizer on raw text data
 * @param raw_bytes Input text as raw bytes
 * @param vocab_size Target vocabulary size (must be >= 256)
 * @return Learned merge rules mapping token pairs to new IDs
 */
MergeRules train(std::vector<uint8_t> &raw_bytes, int vocab_size);

/** @} */

/**
 * @name Vocabulary Operations
 * @{
 */

/**
 * @brief Initialize base vocabulary with all 256 byte values
 * @return Vocabulary with single-byte tokens (IDs 0-255)
 */
Vocabulary init_vocab();

/**
 * @brief Build complete vocabulary from merge rules
 * @param merges Learned merge rules from training
 * @return Complete vocabulary mapping token IDs to byte sequences
 */
Vocabulary build_vocab(const MergeRules &merges);

/** @} */

/**
 * @name Encoding/Decoding
 * @{
 */

/**
 * @brief Encode text into token IDs using trained merge rules
 * @param text Input text to encode
 * @param merges Trained merge rules
 * @return Sequence of token IDs
 */
std::vector<int> encode(const std::string &text, const MergeRules &merges);

/**
 * @brief Decode token IDs back to original text
 * @param ids Sequence of token IDs
 * @param vocab Complete vocabulary mapping IDs to byte sequences
 * @return Reconstructed text string
 */
std::string decode(const std::vector<int> &ids, const Vocabulary &vocab);

/** @} */

/**
 * @name File I/O
 * @{
 */

/**
 * @brief Read a file into a byte vector
 * @param path File path to read
 * @return Vector of bytes, empty on error
 */
std::vector<uint8_t> read_file_to_bytes(const std::string &path);

/**
 * @brief Save trained model (merge rules) to file
 * @param merges Merge rules to save
 * @param filename Output file path
 */
void save_model(const MergeRules &merges, const std::string &filename);

/**
 * @brief Load trained model from file
 * @param filename Input file path
 * @return Merge rules, empty on error
 */
MergeRules load_model(const std::string &filename);

/** @} */

/**
 * @name CLI Interface
 * @{
 */

/**
 * @brief Print usage information
 * @param program_name Name of the executable
 */
void print_usage(const char *program_name);

/**
 * @brief Handle train command
 * @param argc Argument count
 * @param argv Argument vector
 * @return Exit code (0 on success)
 */
int handle_train(int argc, char *argv[]);

/**
 * @brief Handle encode command
 * @param argc Argument count
 * @param argv Argument vector
 * @return Exit code (0 on success)
 */
int handle_encode(int argc, char *argv[]);

/**
 * @brief Handle decode command
 * @param argc Argument count
 * @param argv Argument vector
 * @return Exit code (0 on success)
 */
int handle_decode(int argc, char *argv[]);

/**
 * @brief Run built-in test suite
 * @return Exit code (0 on success)
 */
int handle_test();

/** @} */

} // namespace bpe

