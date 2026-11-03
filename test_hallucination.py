
from llm import AnswerGenerator


def main():

    generator = AnswerGenerator()

    test_cases = [

        {
            "name": "Supported question",

            "question": "What is TCP?",

            "chunks": [
                {
                    "page_number": 124,
                    "chunk_id": 362,
                    "text": (
                        "TCP is a connection-oriented transport "
                        "layer protocol that provides reliable "
                        "data transfer between applications."
                    )
                }
            ]
        },

        {
            "name": "Unsupported question",

            "question": (
                "What is the maximum speed of TCP "
                "according to the document?"
            ),

            "chunks": [
                {
                    "page_number": 124,
                    "chunk_id": 362,
                    "text": (
                        "TCP is a connection-oriented transport "
                        "layer protocol that provides reliable "
                        "data transfer between applications."
                    )
                }
            ]
        },

        {
            "name": "Completely unrelated question",

            "question": "Who won the FIFA World Cup in 2022?",

            "chunks": [
                {
                    "page_number": 124,
                    "chunk_id": 362,
                    "text": (
                        "TCP is a connection-oriented transport "
                        "layer protocol."
                    )
                }
            ]
        }

    ]

    for test in test_cases:

        print("\n" + "=" * 60)

        print("TEST:", test["name"])

        print("QUESTION:", test["question"])

        answer = generator.generate(
            test["question"],
            test["chunks"]
        )

        insufficient_evidence = answer.startswith(
            "INSUFFICIENT_EVIDENCE:"
        )

        print("\nANSWER:")
        print(answer)

        print("\nABSTENTION DETECTED:", insufficient_evidence)

        print("=" * 60)


if __name__ == "__main__":
    main()
