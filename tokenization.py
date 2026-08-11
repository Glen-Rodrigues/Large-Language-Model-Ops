# Tokenization - Word, Char and Byte
import re

from collections import Counter

# Sample plain input text or sentence
text = input("Enter the text: ")

# Word tokenization
def word_token(text):
    return re.findall(r"\w+", text)

# Char tokenization
def char_token(text):
    return [char for char in text if not char.isspace()]

# Byte Tokenization
def byte_token(text):
    return list(text.encode("utf-8"))

# Execute tokenizers
words = word_token(text)
chars = char_token(text)
bytes_list = byte_token(text)

# Word Token result
print("\n--- WORD TOKEN ---")
print("Tokens:", words)
print("No of words:", len(words))

# Char Token result
print("\n--- CHAR TOKEN ---")
print("Tokens:", chars)
print("No of characters:", len(chars))

# Byte Token result
print("\n--- BYTE TOKEN ---")
print("Tokens:", bytes_list)
print("No of bytes:", len(bytes_list))
