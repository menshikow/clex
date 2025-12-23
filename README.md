# clex

A Byte Pair Encoding (BPE) tokenizer implementation in C++23

## Build

```bash
mkdir build && cd build
cmake ..
make
```

The executable `clex` will be created in the `build` directory.

## Usage

### Train a Tokenizer

Train a BPE tokenizer on your text corpus:

```bash
./clex train input.txt model.txt 1000
```

- `input.txt`: Input text file to train on
- `model.txt`: Output file to save the trained model
- `1000`: Target vocabulary size (must be ≥ 256)

### Encode Text

Tokenize text using a trained model:

```bash
./clex encode model.txt input.txt [output.txt]
```

If `output.txt` is omitted, token IDs are printed to stdout.

### Decode Tokens

Convert token IDs back to text:

```bash
./clex decode model.txt tokens.txt [output.txt]
```

If `output.txt` is omitted, decoded text is printed to stdout.

### Run Tests

Verify the implementation with built-in tests:

```bash
./clex test
```

## Example

Complete workflow from training to encoding/decoding:

```bash
# Create a sample text file
echo "hello world" > input.txt

# Train a tokenizer with vocabulary size 260
./clex train input.txt model.txt 260

# Encode the text to token IDs
./clex encode model.txt input.txt tokens.txt

# Decode tokens back to original text
./clex decode model.txt tokens.txt output.txt

# Verify round-trip encoding
cat output.txt  # Should output: hello world
```

## Implementation Details

- **Training**: Iteratively merges most frequent byte pairs until target vocabulary size is reached
- **Encoding**: Applies merge rules in order to convert text to token IDs
- **Decoding**: Reconstructs text from token IDs using the vocabulary
- **Model Format**: Simple text format storing merge rules as `token1 token2 new_id`

## License

MIT License - see LICENSE file for details.
