from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

retriever = Retriever(DOCUMENT_ID)

questions = [
    "What is TCP?",
    "What is the difference between TCP and UDP?",
    "Explain the TCP three-way handshake."
]

for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    chunks = retriever.retrieve(
        question,
        top_k=5,
        distance_threshold=0.37
    )

    for index, chunk in enumerate(chunks, start=1):

        print(f"\n--- RESULT {index} ---")
        print("Page:", chunk["page_number"])
        print("Distance:", round(chunk["distance"], 4))
        print("Text:")
        print(chunk["text"][:600])
