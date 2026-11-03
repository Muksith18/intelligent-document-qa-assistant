
from rag_engine import RAGEngine

rag = RAGEngine()

while True:

    question = input("\nAsk your textbook (type exit to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = rag.ask_question(question)

    print("\n========== ANSWER ==========")
    print(result["answer"])

    print("\n========== SOURCES ==========")

    for source in result["sources"]:
        print(
            f"Page: {source['page_number']} | "
            f"Chunk: {source['chunk_id']} | "
            f"Distance: {source['distance']:.4f}"
        )
