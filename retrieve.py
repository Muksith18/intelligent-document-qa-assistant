import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the same embedding model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to our existing vector database
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(name="networking_book")

print("Database records:", collection.count())

# 3. Get a question from the user
question = input("\nAsk a question about your textbook: ")

# 4. Convert the question into an embedding
query_embedding = model.encode(
    question,
    normalize_embeddings=True
)

# 5. Search for the 5 most relevant chunks
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

# 6. Display the retrieved passages
print("\n========== RETRIEVED PASSAGES ==========")

for i in range(len(results["documents"][0])):
    document = results["documents"][0][i]
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print(f"\nResult {i + 1}")
    print(f"Page: {metadata['page_number']}")
    print(f"Chunk ID: {metadata['chunk_id']}")
    print(f"Cosine distance: {distance:.4f}")
    print(f"Text:\n{document}")
    print("-" * 60)
