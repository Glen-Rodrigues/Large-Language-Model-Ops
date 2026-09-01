from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# 1. Input Text Corpus
corpus = [
    "My cat is cute",
    "That cat is cute"
]

# 2. Bag of Words (BoW) Representation
vector_rep = CountVectorizer()
matrix_bagofwords = vector_rep.fit_transform(corpus)

# 3. Dense Vector Embedding Technique
model_dense = SentenceTransformer("all-MiniLM-L6-v2")
matrix_dense = model_dense.encode(corpus)

# OUTPUT DISPLAY

print("1. BAG OF WORDS REPRESENTATION")
print(f"Vocabulary:\n{vector_rep.get_feature_names_out()}\n")
print(f"BoW Matrix:\n{matrix_bagofwords.toarray()}\n")

print("2. DENSE VECTOR EMBEDDINGS")
print(f"Embedding Shape: {matrix_dense.shape}\n")
print(f"First Sentence Embedding:\n{matrix_dense[0]}\n")

print("3. COMPARISON SUMMARY")
print(f"Bag of Words Dimensions:      {matrix_bagofwords.shape}")
print(f"Dense Vector Dimensions:      {matrix_dense.shape}\n")
print(f"BoW First Sentence Vector:\n{matrix_bagofwords.toarray()[0]}\n")
print(f"Dense First Sentence Vector:\n{matrix_dense[0]}")
