from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

retriever = Retriever(DOCUMENT_ID)

question = "What is TCP?"

thresholds = [0.30, 0.33, 0.35, 0.37, 0.40]

print("\n========== THRESHOLD EXPERIMENT ==========")

for threshold in thresholds:

    results = retriever.retrieve(
        question,
        top_k=5,
        distance_threshold=threshold
    )

    print(f"\nThreshold: {threshold}")
    print("Retrieved chunks:", len(results))

    for chunk in results:
        print(
            f"Page {chunk['page_number']} | "
            f"Distance: {chunk['distance']:.4f}"
        )

print("\n==========================================")
