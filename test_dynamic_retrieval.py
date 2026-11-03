"""Interactive retrieval diagnostic for the canonical Phase 6 document."""

from evaluation_config import CANONICAL_DOCUMENT_ID


DOCUMENT_ID = CANONICAL_DOCUMENT_ID


def format_distance(distance):
    return "context" if distance is None else f"{distance:.4f}"


def main():
    from retrieval import Retriever

    retriever = Retriever(document_id=DOCUMENT_ID)
    results = retriever.retrieve("How does TCP establish a connection?", top_k=5)
    print("\n========== DYNAMIC RETRIEVAL ==========")
    for result in results:
        print("\nPage:", result["page_number"])
        print("Chunk ID:", result["chunk_id"])
        print("Distance:", format_distance(result.get("distance")))
        print("Original vector match:", not result.get("is_context", False))
        print("Text:", result["text"])
        print("-" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
