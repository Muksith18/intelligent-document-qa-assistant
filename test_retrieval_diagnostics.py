
from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

retriever = Retriever(DOCUMENT_ID)

questions = [
    "What is TCP?",
    "What is TCP and what bandwidth does it guarantee?"
]

for question in questions:

    print("\nQUESTION:", question)

    for threshold in [0.30, 0.35, 0.37]:

        results = retriever.retrieve(
            question,
            top_k=5,
            distance_threshold=threshold
        )

        print("\nThreshold:", threshold)
        print("Retrieved chunks:", len(results))

        for chunk in results:
            print(
                "Page:", chunk["page_number"],
                "| Distance:", round(chunk["distance"], 4)
            )
            print("Text:", chunk["text"][:250])
