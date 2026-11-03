from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID

# Use the same DOCUMENT_ID from evaluate_rag.py
DOCUMENT_ID = CANONICAL_DOCUMENT_ID

retriever = Retriever(DOCUMENT_ID)

question = "What is TCP?"

print("\n========== TCP RETRIEVAL DIAGNOSTIC ==========")

# Temporarily use a relaxed threshold for inspection.
results = retriever.retrieve(
    question,
    top_k=5,
    distance_threshold=2.0
)

for index, chunk in enumerate(results, start=1):

    print(f"\nRank: {index}")
    print("Page:", chunk["page_number"])
    print("Chunk ID:", chunk["chunk_id"])
    print("Cosine distance:", round(chunk["distance"], 4))
    print("Text:", chunk["text"][:400])

print("\n==============================================")
