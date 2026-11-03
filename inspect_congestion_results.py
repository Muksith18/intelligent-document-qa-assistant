
from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

retriever = Retriever(DOCUMENT_ID)

queries = {
    "Original": "What is TCP congestion control?",
    "Expanded": (
        "TCP congestion control mechanisms, congestion window, "
        "slow start, congestion avoidance and network congestion."
    )
}

for label, query in queries.items():

    print("\n" + "=" * 60)
    print(label.upper())
    print("=" * 60)

    chunks = retriever.retrieve(
        query,
        top_k=5,
        distance_threshold=2.0
    )

    for chunk in chunks:

        if chunk["distance"] <= 0.30:

            print(
                f"\nPage: {chunk['page_number']} | "
                f"Distance: {chunk['distance']:.4f}"
            )

            print(chunk["text"][:500])
