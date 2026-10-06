# Text Classification using Embedding Vectors, Ex: Classifying documents as Sports or Technology

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Example Corpus
documents = [
    "The cricket team won the match by five wickets",
    "The football player scored two goals",
    "The team practiced hard for the upcoming cricket tournament",
    "The tennis player won the championship",
    "The coach discussed the strategy for the football match",
    "The basketball team qualified for the final",
    
    "Artificial intelligence is transforming modern technology",
    "Machine learning models can analyze large datasets",
    "Python is widely used for artificial intelligence",
    "Cloud computing provides scalable computing resources",
    "Deep learning uses neural networks for prediction",
    "Data science combines statistics programming and machine learning"
]

labels = [
    "Sports", "Sports", "Sports",
    "Sports", "Sports", "Sports",
    "Technology", "Technology", "Technology",
    "Technology", "Technology", "Technology"
]

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    documents,
    labels,
    test_size=0.33,
    random_state=42,
    stratify=labels
)

# 3. Convert documents into embedding vectors using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
print("Number of training documents:", len(X_train))
print("Number of testing documents:", len(X_test))
print("\nEmbedding vector size:")
print(X_train_vectors.shape[1])
print("\nTraining embedding matrix:")
print(X_train_vectors.toarray())

# 4. Train a text classification model
classifier = LogisticRegression()
classifier.fit(X_train_vectors, y_train)

# 5. Predict the test documents
y_pred = classifier.predict(X_test_vectors)
print("\nActual Labels:")
print(y_test)
print("\nPredicted Labels:")
print(y_pred)

# 6. Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Test a new document
new_document = [
    "The player scored a goal in the football match"
]

new_vector = vectorizer.transform(new_document)
prediction = classifier.predict(new_vector)

print("\nNew Document:")
print(new_document[0])

print("Predicted Category:", prediction[0])
