
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

    print("\nQUESTION:", question)

    chunks = retriever.retrieve(
        question,
        top_k=5,
        adaptive=True
    )

    print("\nFinal results:")

    for chunk in chunks:

        print(
            f"Page: {chunk['page_number']} | "
            f"Distance: {chunk['distance']:.4f}"
        )
