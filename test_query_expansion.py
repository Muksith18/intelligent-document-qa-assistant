
from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

retriever = Retriever(DOCUMENT_ID)

queries = {
    "Original query":
        "What is TCP?",

    "Expansion 1":
        "TCP Transmission Control Protocol",

    "Expansion 2":
        "TCP connection-oriented reliable transport-layer protocol",

    "Expansion 3":
        "What services does TCP provide? Reliable data transfer and connection-oriented communication"
}

STRICT_THRESHOLD = 0.30

print("\n========== QUERY EXPANSION EXPERIMENT ==========")

for label, query in queries.items():

    print("\n" + "=" * 60)
    print("QUERY TYPE:", label)
    print("SEARCH QUERY:", query)
    print("=" * 60)

    # Use 2.0 temporarily to inspect all five candidates.
    # We manually evaluate the original strict threshold below.
    chunks = retriever.retrieve(
        query,
        top_k=5,
        distance_threshold=2.0
    )

    accepted = 0

    for index, chunk in enumerate(chunks, start=1):

        distance = chunk["distance"]

        passed = distance <= STRICT_THRESHOLD

        if passed:
            accepted += 1

        print(f"\nRank: {index}")
        print("Page:", chunk["page_number"])
        print("Distance:", round(distance, 4))
        print(
            "Strict threshold result:",
            "ACCEPTED" if passed else "REJECTED"
        )
        print("Text:", chunk["text"][:250])

    print("\nSUMMARY")
    print("Candidates:", len(chunks))
    print("Accepted at 0.30:", accepted)

print("\n========== EXPERIMENT COMPLETED ==========")
