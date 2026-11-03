"""Manual diagnostic comparing original and expanded queries with validated labels."""

from evaluation_config import CANONICAL_DOCUMENT_ID
from evaluation_dataset import EVALUATION_DATASET, has_manually_validated_evidence


DOCUMENT_ID = CANONICAL_DOCUMENT_ID
THRESHOLD = 0.30
TOP_K = 5
EXPANSIONS = {
    "T1": "TCP connection-oriented reliable transport-layer protocol",
    "T2": "Compare TCP and UDP reliability connection-oriented connectionless delivery",
    "T3": "TCP connection establishment three-way handshake SYN SYN-ACK ACK",
}


def manually_validated_cases(dataset):
    return [case for case in dataset if has_manually_validated_evidence(case)]


def accepted_original_matches(retriever, query):
    candidates = retriever.retrieve(
        query, top_k=TOP_K, distance_threshold=2.0, expand_context=False,
    )
    return [
        chunk for chunk in candidates
        if not chunk.get("is_context", False) and chunk["distance"] <= THRESHOLD
    ]


def main():
    cases = [case for case in manually_validated_cases(EVALUATION_DATASET) if case["id"] in EXPANSIONS]
    if not cases:
        print("No manually validated labels are available for strategy comparison.")
        return 0

    from retrieval import Retriever

    retriever = Retriever(DOCUMENT_ID)
    for case in cases:
        relevant_pages = set(case["expected_evidence"]["required_pdf_pages"])
        print("\n" + "=" * 65)
        print("QUESTION:", case["question"])
        queries = {"Original": case["question"], "Expanded": EXPANSIONS[case["id"]]}
        for name, query in queries.items():
            chunks = accepted_original_matches(retriever, query)
            pages = {chunk["page_number"] for chunk in chunks}
            found = pages & relevant_pages
            precision = len(found) / len(pages) if pages else 0.0
            recall = len(found) / len(relevant_pages)
            print(f"\n{name}")
            print("Accepted original-match pages:", sorted(pages))
            print("Validated relevant pages found:", sorted(found))
            print(f"Page-label precision: {precision:.2%}")
            print(f"Page-label recall: {recall:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
