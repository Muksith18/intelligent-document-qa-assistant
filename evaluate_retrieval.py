from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID


DOCUMENT_ID = CANONICAL_DOCUMENT_ID


def evaluate_query(question):

    print("\n" + "=" * 65)
    print("QUESTION:", question)
    print("=" * 65)

    retriever = Retriever(DOCUMENT_ID)

    results = retriever.retrieve(
        question,
        top_k=5
    )

    for index, chunk in enumerate(results, start=1):

        print(f"\nRANK: {index}")
        print("Page:", chunk["page_number"])
        print("Chunk ID:", chunk["chunk_id"])
        print("Cosine Distance:", round(chunk["distance"], 4))

        print("\nRetrieved Text:")
        print(chunk["text"][:400])

        print("-" * 65)


if __name__ == "__main__":

   evaluate_query(
    "Who won the FIFA World Cup in 2022?"
)
