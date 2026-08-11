import re

# Input the text from the user
text = input("Enter the text: ")

# Count the number of words in the text
word_count = len(text.split())
print(f"Word Count: {word_count}")

# Count the number of whitespaces in the text
whitespace_count = text.count(" ")
print(f"Whitespace Count: {whitespace_count}")

# Count total number of characters
char_count = len(text)
print(f"Character Count: {char_count}")

# Count number of special characters
special_char_count = len(re.findall(r"[^a-zA-Z0-9 ]", text))
print(f"Special Character Count: {special_char_count}")

# Count the number of vowels in text
vowel_count = len(re.findall(r"[aeiou]", text, re.IGNORECASE))
print(f"Vowel Count: {vowel_count}")

# Clean text for duplicate words
cleaned_text = re.sub(r"[^\s\w]", "", text).lower()
words = cleaned_text.split()

# Display duplicate words
seen = set()
duplicates = set()

for word in words:
    if word in seen:
        duplicates.add(word)
    else:
        seen.add(word)

print(f"Duplicate words: {duplicates}")
