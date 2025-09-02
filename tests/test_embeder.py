import os
import sys

# Fix system path by appending the parent directory of tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from pipeline import embeddingclass

# testing out embeddingclass
gemini_embedder = embeddingclass()
query_text = "What is the capital of France?"
print(f"Embedding query: '{query_text}'")
query_embedding = gemini_embedder.embed_query(query_text)
print(f"Query embedding (first 5 dimensions): {query_embedding[:5]}")
print(f"Embedding dimension: {len(query_embedding)}")
print("-" * 50)


# 2. Test embed_documents method
document_texts = [
    "The capital of France is Paris.",
    "The Eiffel Tower is located in Paris.",
    "A dog is a man's best friend.",
    "Paris is a beautiful city.",
]

print("Embedding a list of documents...")
document_embeddings = gemini_embedder.embed_documents(document_texts)

# Print the dimensions of the embeddings
print(f"Number of document embeddings: {len(document_embeddings)}")
if document_embeddings:
    print(f"Dimension of each document embedding: {len(document_embeddings[0])}")
