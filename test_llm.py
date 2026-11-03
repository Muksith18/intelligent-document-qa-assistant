from llm import AnswerGenerator


def main():

    print("Initializing Groq Answer Generator...\n")

    generator = AnswerGenerator()

    test_chunks = [
        {
            "page_number": 10,
            "chunk_id": 1,
            "text": (
                "TCP is a connection-oriented transport layer protocol. "
                "It provides reliable data transfer between applications."
            )
        },
        {
            "page_number": 11,
            "chunk_id": 2,
            "text": (
                "TCP establishes a connection using a three-way handshake "
                "before data transmission begins."
            )
        }
    ]

    question = "What is TCP and how does it establish a connection?"

    print("\nQuestion:", question)

    answer = generator.generate(
        question,
        test_chunks
    )

    print("\n========== GENERATED ANSWER ==========")
    print(answer)
    print("=======================================")


if __name__ == "__main__":
    main()