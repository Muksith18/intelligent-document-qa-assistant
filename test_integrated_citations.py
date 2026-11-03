
from rag_engine import RAGEngine
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

engine = RAGEngine(DOCUMENT_ID)

questions = [
    "What is the difference between TCP and UDP?",
    "Explain the TCP three-way handshake.",
    "Who won the FIFA World Cup in 2022?"
]

for question in questions:

    print("\n" + "=" * 65)
    print("QUESTION:", question)
    print("=" * 65)

    result = engine.ask_question(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nCITATION VALIDATION:")
    print(result["citation_validation"])

    print("\nINSUFFICIENT EVIDENCE:")
    print(result["insufficient_evidence"])

    print("\nTIMINGS:")
    print(result["timings"])
