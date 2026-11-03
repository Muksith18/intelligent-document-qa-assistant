from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

retriever = Retriever(DOCUMENT_ID)

questions = [
    "What is TCP?",
    "What is the difference between TCP and UDP?",
    "Explain the TCP three-way handshake.",
    "What is TCP congestion control?",
    "How does TCP provide reliable data transfer?"
]

thresholds = [0.30, 0.35, 0.37]

print("\n========== MULTI-QUESTION THRESHOLD TEST ==========")

for question in questions:

    print("\nQUESTION:", question)

    for threshold in thresholds:

        results = retriever.retrieve(
            question,
            top_k=5,
            distance_threshold=threshold
        )

        pages = [chunk["page_number"] for chunk in results]

        print(
            f"Threshold: {threshold:.2f} | "
            f"Retrieved: {len(results)} | "
            f"Pages: {pages}"
        )

print("\n===================================================")
