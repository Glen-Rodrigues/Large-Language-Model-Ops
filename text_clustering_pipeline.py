# Text Clustering using TF-IDF, Dimensionality Reduction and K-Means

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Create an example corpus
documents = [
    "The cricket team won the football match",
    "The player scored a goal in the tournament",
    "The coach trained the cricket team",
    "The tennis player won the championship",
    "The basketball team played in the final",

    "Artificial intelligence uses machine learning",
    "Python is used for machine learning applications",
    "Deep learning uses neural networks",
    "Cloud computing provides computing resources",
    "Data science analyzes large datasets",

    "Pizza and pasta are popular Italian foods",
    "The chef prepared a delicious vegetable soup",
    "Fresh fruits and vegetables are healthy",
    "Rice and curry are common food dishes",
    "The restaurant serves tasty meals"
]

# 2. Convert documents into TF-IDF vectors

vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(documents)

print("Number of documents:", len(documents))
print("Original feature dimensions:", X.shape[1])

# 3. Reduce dimensionality to 2 dimensions

svd = TruncatedSVD(n_components=2, random_state=42)

X_reduced = svd.fit_transform(X)

print("Reduced feature dimensions:", X_reduced.shape[1])

# 4. Apply K-Means clustering

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

clusters = kmeans.fit_predict(X_reduced)

# 5. Display document cluster assignments

print("\nDocument Clustering Results:\n")
for i, document in enumerate(documents):
    print("Document:", document)
    print("Cluster:", clusters[i])
    print()

# 6. Visualize the clusters

plt.figure(figsize=(10, 6))
for cluster_id in range(3):
    points = X_reduced[clusters == cluster_id]
    
    plt.scatter(
        points[:, 0],
        points[:, 1],
        label=f"Cluster {cluster_id}",
        s=100
    )

plt.title("Text Clustering using TF-IDF, SVD and K-Means")
plt.xlabel("Reduced Feature 1")
plt.ylabel("Reduced Feature 2")
plt.legend()
plt.grid(True)
plt.show()
