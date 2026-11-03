
import json
import numpy as np
import chromadb

# 1. Load chunks
print("Loading chunks...")

with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# 2. Load embeddings
print("Loading embeddings...")

embeddings = np.load("embeddings.npy")

# 3. Validate data
if len(chunks) != len(embeddings):
    raise ValueError("Number of chunks and embeddings do not match!")

print("Chunks:", len(chunks))
print("Embeddings:", embeddings.shape)

# 4. Create a persistent ChromaDB database
print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="networking_book",
    metadata={"hnsw:space": "cosine"}
)

# 5. Store chunks and embeddings in batches
batch_size = 500

for start in range(0, len(chunks), batch_size):
    batch_chunks = chunks[start:start + batch_size]
    batch_embeddings = embeddings[start:start + batch_size]

    collection.upsert(
        ids=[str(chunk["chunk_id"]) for chunk in batch_chunks],
        embeddings=batch_embeddings.tolist(),
        documents=[chunk["text"] for chunk in batch_chunks],
        metadatas=[
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "source": "CN_RefereneceBook.pdf"
            }
            for chunk in batch_chunks
        ]
    )

    print(f"Stored {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

# 6. Verify database
print("\nDatabase creation completed!")
print("Total records:", collection.count())
print("Database location: ./chroma_db")
