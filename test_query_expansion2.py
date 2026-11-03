
from retrieval import Retriever
from evaluation_config import CANONICAL_DOCUMENT_ID

DOCUMENT_ID = CANONICAL_DOCUMENT_ID

retriever = Retriever(DOCUMENT_ID)

test_cases = [
    {
        "name": "TCP vs UDP",
        "original": "What is the difference between TCP and UDP?",
        "expanded": (
            "Compare TCP and UDP transport-layer protocols, "
            "including reliability, connection-oriented communication, "
            "connectionless communication and data delivery."
        )
    },
    {
        "name": "TCP Three-way Handshake",
        "original": "Explain the TCP three-way handshake.",
        "expanded": (
            "TCP connection establishment three-way handshake: "
            "SYN, SYN-ACK, ACK segments exchanged between client and server."
        )
    },
    {
        "name": "TCP Congestion Control",
        "original": "What is TCP congestion control?",
        "expanded": (
            "TCP congestion control mechanisms, congestion window, "
            "slow start, congestion avoidance and network congestion."
        )
    }
]

THRESHOLD = 0.30

for test in test_cases:

    print("\n" + "=" * 65)
    print("QUESTION:", test["name"])
    print("=" * 65)

    for label in ["original", "expanded"]:

        query = test[label]

        chunks = retriever.retrieve(
            query,
            top_k=5,
            distance_threshold=2.0
        )

        accepted = [
            chunk for chunk in chunks
            if chunk["distance"] <= THRESHOLD
        ]

        print(f"\n{label.upper()} QUERY")
        print("Query:", query)
        print("Accepted at 0.30:", len(accepted))

        for chunk in accepted:
            print(
                f"Page {chunk['page_number']} | "
                f"Distance {chunk['distance']:.4f}"
            )

print("\nQUERY EXPANSION EXPERIMENT COMPLETED")
