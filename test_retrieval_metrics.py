"""Manual retrieval metrics diagnostic using only validated page labels."""

from evaluation_config import CANONICAL_DOCUMENT_ID
from evaluation_dataset import EVALUATION_DATASET, has_manually_validated_evidence


DOCUMENT_ID = CANONICAL_DOCUMENT_ID
TOP_K = 5
THRESHOLD = 0.30


def manually_validated_cases(dataset):
    return [case for case in dataset if has_manually_validated_evidence(case)]


def main():
    cases = manually_validated_cases(EVALUATION_DATASET)
    if not cases:
        print("No manually validated page labels are available; retrieval metrics were not calculated.")
        return 0

    from retrieval import Retriever

    retriever = Retriever(DOCUMENT_ID)
    precisions, recalls = [], []
    print("\n========== RETRIEVAL METRICS EVALUATION ==========")
    for case in cases:
        relevant_pages = set(case["expected_evidence"]["required_pdf_pages"])
        chunks = retriever.retrieve(
            case["question"], top_k=TOP_K, distance_threshold=THRESHOLD,
            expand_context=False,
        )
        original_chunks = [chunk for chunk in chunks if not chunk.get("is_context", False)]
        retrieved_pages = {chunk["page_number"] for chunk in original_chunks}
        found_pages = retrieved_pages & relevant_pages
        precision = len(found_pages) / len(retrieved_pages) if retrieved_pages else 0.0
        recall = len(found_pages) / len(relevant_pages)
        precisions.append(precision)
        recalls.append(recall)
        print(f"\n{case['id']}: {case['question']}")
        print("Original-match pages:", sorted(retrieved_pages))
        print("Validated relevant pages found:", sorted(found_pages))
        print(f"Page-label precision: {precision:.2%}")
        print(f"Page-label recall: {recall:.2%}")

    print("\nAverage page-label precision:", f"{sum(precisions) / len(precisions):.2%}")
    print("Average page-label recall:", f"{sum(recalls) / len(recalls):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
