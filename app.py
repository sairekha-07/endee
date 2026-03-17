from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading model...")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Model loaded")

with open("data.txt", "r") as f:
    documents = f.readlines()

documents = [doc.strip() for doc in documents if doc.strip() != ""]

doc_embeddings = model.encode(documents)

query = input("Enter your search: ")

query_embedding = model.encode([query])

scores = np.dot(doc_embeddings, query_embedding.T)

best_index = np.argmax(scores)

print("Best Result:", documents[best_index])