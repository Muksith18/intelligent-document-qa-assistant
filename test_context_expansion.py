
from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

retriever = Retriever(DOCUMENT_ID)

question = "Explain the TCP three-way handshake."

chunks = retriever.retrieve(
    question,
    top_k=5,
    distance_threshold=0.30,
    adaptive=True,
    expand_context=True,
    context_window=5
)

print("\n========== EXPANDED HANDSHAKE EVIDENCE ==========")

for chunk in chunks:

    print("\n" + "=" * 70)

    print("CHUNK:", chunk["chunk_id"])
    print("PAGE:", chunk["page_number"])
    print("CONTEXT ONLY:", chunk["is_context"])
    print("DISTANCE:", chunk["distance"])

    print("=" * 70)

    print(chunk["text"])

print("\nTotal chunks returned:", len(chunks))
