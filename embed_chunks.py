
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Load our previously created chunks
print("Loading document chunks...")
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Total chunks:", len(chunks))

# 3. Extract text from every chunk
texts = [chunk["text"] for chunk in chunks]

# 4. Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# 5. Save embeddings
np.save("embeddings.npy", embeddings)

# 6. Verify the result
print("\nEmbedding generation completed!")
print("Embedding shape:", embeddings.shape)
print("Total vectors:", len(embeddings))
print("Dimensions per vector:", embeddings.shape[1])
print("Saved as embeddings.npy")
