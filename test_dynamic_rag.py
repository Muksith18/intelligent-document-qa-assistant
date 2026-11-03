"""Interactive RAG diagnostic for the canonical Phase 6 document."""

from evaluation_config import CANONICAL_DOCUMENT_ID
from test_dynamic_retrieval import format_distance


DOCUMENT_ID = CANONICAL_DOCUMENT_ID


def main():
    from rag_engine import RAGEngine

    rag = RAGEngine(document_id=DOCUMENT_ID)
    while True:
        question = input("\nAsk your document (type exit to quit): ")
        if question.lower() == "exit":
            return 0
        result = rag.ask_question(question)
        print("\n========== ANSWER ==========")
        print(result["answer"])
        print("\n========== SOURCES ==========")
        for source in result["sources"]:
            print(
                f"Page: {source['page_number']} | Chunk: {source['chunk_id']} | "
                f"Distance: {format_distance(source.get('distance'))} | "
                f"Original vector match: {not source.get('is_context', False)}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
