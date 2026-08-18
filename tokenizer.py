# LLM Lab-1: Comparison of Tokenization Techniques

import re
from collections import Counter

# Sample Text Dataset

text = """
Module-1: Language AI, Language Representation: Bag-of-Words, Dense Vector Embeddings,
Types of Embedding, Encoding and Decoding Context with Attention, Self-attention.
Representational : Encoder-Only Model, Decoder-Only Models, LLM Training Paradigm and
Its Applications, Societal and Ethical Consideration.
Token & Embeddings: Introduction, LLM Tokenization, Token to LLM, Word Versus Subword
Versus Character Versus Byte Tokens, BPE, Tokenizer Properties.
"""

# 1. Word-Level Tokenization

def word_tokenization(text):
    # Extract words and punctuation separately
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

# 2. Character-Level Tokenization

def character_tokenization(text):
    # Remove spaces and tokenize each character
    return [char for char in text if not char.isspace()]

# 3. Byte-Level Tokenization

def byte_tokenization(text):
    # Convert text into UTF-8 bytes
    return list(text.encode("utf-8"))

# 4. Simple Subword-Level Tokenization using BPE

def get_words(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())

def build_bpe_vocab(text):
    words = get_words(text)
    
    # Represent each word as characters with </w> marking
    # the end of a word
    vocab = Counter()
    
    for word in words:
        chars = list(word) + ["</w>"]
        vocab[tuple(chars)] += 1
        
    return vocab

def get_pair_counts(vocab):
    pairs = Counter()
    
    for word, frequency in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += frequency

    return pairs

def merge_pair(pair, vocab):
    new_vocab = Counter()
    bigram = pair
    
    for word, frequency in vocab.items():
        new_word = []
        i = 0
        
        while i < len(word):
            if (
                i < len(word) - 1
                and word[i] == bigram[0]
                and word[i + 1] == bigram[1]
            ):
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
                
        new_vocab[tuple(new_word)] += frequency
        
    return new_vocab

def train_bpe(text, num_merges=20):
    vocab = build_bpe_vocab(text)
    merges = []
    
    for _ in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        
        if not pair_counts:
            break
            
        best_pair = pair_counts.most_common(1)[0][0]
        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)
        
    return merges

def apply_bpe_to_word(word, merges):
    tokens = list(word.lower()) + ["</w>"]
    
    for pair in merges:
        new_tokens = []
        i = 0
        
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
                
        tokens = new_tokens

    # Remove end-of-word-marker
    tokens = [token.replace("</w>", "") for token in tokens]

    return [token for token in tokens if token]

def subword_tokenization(text, merges):
    words = get_words(text)

    tokens = []

    for word in words:
        tokens.extend(apply_bpe_to_word(word, merges))

    return tokens

# Generate Tokens

word_tokens = word_tokenization(text)
character_tokens = character_tokenization(text)
byte_tokens = byte_tokenization(text)

# Train a small BPE Tokenizer
bpe_merges =train_bpe(text, num_merges=20)
subword_tokens = subword_tokenization(text, bpe_merges)

# Display Tokens

print("=" * 100)
print("*** WORD TOKENIZATION ***")
print("=" * 100)

print(word_tokens)
print("Number of tokens:", len(word_tokens))

print("\n" + "=" * 60)
print("SUBWORD TOKENIZATION (BPE)")
print("=" * 60)

print(subword_tokens)
print("Number of tokens:", len(subword_tokens))

print("\n" + "=" * 60)
print("CHARACTER TOKENIZATION")
print("=" * 60)

print(character_tokens)
print("Number of tokens:", len(character_tokens))

print("\n" + "=" * 60)
print("BYTE TOKENIZATION")
print("=" * 60)

print(byte_tokens)
print("Number of tokens:", len(byte_tokens))

# Comparison

print("\n" + "=" * 60)
print("TOKENIZATION COMPARISON")
print("=" * 60)

results = {
    "Word": len(word_tokens),
    "Subword (BPE)": len(subword_tokens),
    "Character": len(character_tokens),
    "Byte": len(byte_tokens)
}

for method, count in results.items():
    print(f"{method:20s}: {count} tokens")

# Find the method generating minimum and maximum tokens

minimum_method = min(results, key=results.get)
maximum_method = max(results, key=results.get)

print("\nMinimum tokens :", minimum_method)
print("Maximum tokens :", maximum_method)
