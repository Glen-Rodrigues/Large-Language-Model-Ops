import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# ==================================================
# 1. Input Text Corpus
# ==================================================
corpus = [
    "My cat is cute",
    "That cat is cute"
]

# ==================================================
# 2. Bag of Words Representation
# ==================================================
vector_rep = CountVectorizer()
matrix_bagofwords = vector_rep.fit_transform(corpus)

print("=" * 50)
print("BAG OF WORDS")
print("=" * 50)
print("Vocabulary:")
print(vector_rep.get_feature_names_out())
print("\nMatrix Bag of Words:")
print(matrix_bagofwords.toarray())
print("\n")

# ==================================================
# 3. Dense Vector Embedding Technique
# ==================================================
model_dense = SentenceTransformer("all-MiniLM-L6-v2")
matrix_dense = model_dense.encode(corpus)

print("=" * 50)
print("DENSE VECTOR EMBEDDINGS")
print("=" * 50)
print(f"Dense Embedding Shape -> {matrix_dense.shape}\n")
print("Dense First Sentence Embedding:")
print(matrix_dense[0])
print("\n")

# ==================================================
# 4. Comparison: Bag of Words vs. Dense Embeddings
# ==================================================
print("=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
print(f"Bag of Words dimensions               : {matrix_bagofwords.shape}")
print(f"Dense Vector Embeddings demonstration : {matrix_dense.shape}")

print("\nBag of Words Representation (1st sentence):")
print(matrix_bagofwords.toarray()[0])

print("\nDense Embeddings (1st sentence):")
print(matrix_dense[0])