from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Create a knowledge base
documents = [
    "Machine learning is a branch of artificial intelligence "
    "that enables computers to learn from data.",
    
    "Deep learning uses neural networks with multiple layers "
    "to learn complex patterns.",
    
    "Natural Language Processing (NLP) enables computers "
    "to understand and process human language.",
    
    "Retrieval-Augmented Generation (RAG) combines document "
    "retrieval with language generation to produce "
    "context-aware answers.",
    
    "Python is a popular programming language used in "
    "artificial intelligence and data science."
]

# Step 2: Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(documents)

# Step 3: Retrieve relevant documents
def retrieve_documents(query, top_k=2):
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(
        query_vector, document_vectors
    ).flatten()
    
    # Get the indices of the most relevant documents
    ranked_indices = similarities.argsort()[::-1]
    
    # Ignore documents with zero similarity
    results = [
        (documents[i], similarities[i])
        for i in ranked_indices
        if similarities[i] > 0
    ]
    
    return results[:top_k]

# Step 4: Generate a context-aware response
def generate_response(query):
    retrieved = retrieve_documents(query)
    
    if not retrieved:
        return "No relevant information found in the knowledge base."
        
    context = "\n".join(
        document for document, score in retrieved
    )
    
    # Simple context-based response generation
    response = (
        "Based on the retrieved information:\n"
        + context
    )
    
    return response

# Step 5: Test the RAG system
# Example1: "What is Machine Learning"
# Example2: "Use of Python for AI and ML ?"
# Example3: "Try with your own query .. -> related to AI, NL, DL, Python, etc and Check"

query = "Use of Python for AI and ML ?"
print("\nUser Query:", query)

retrieved = retrieve_documents(query)

print("\nRetrieved (RAG) Documents:")
for document, score in retrieved:
    print(f"- {document} (Similarity: {score:.3f})")
    
print("\nGenerated Response:")
print(generate_response(query))
