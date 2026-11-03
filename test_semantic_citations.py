
from llm import AnswerGenerator, MODEL_NAME
from citation_verifier import CitationVerifier

generator = AnswerGenerator()

verifier = CitationVerifier(
    client=generator.client,
    model_name=MODEL_NAME
)

retrieved_chunks = [
    {
        "page_number": 201,
        "chunk_id": 1,
        "text": """
        The client process can initiate a TCP connection
        by creating a socket and specifying the server's
        IP address and port number.
        """
    },
    {
        "page_number": 295,
        "chunk_id": 2,
        "text": """
        The third stage of the TCP three-way handshake
        may carry client-to-server data in the segment payload.
        """
    },
    {
        "page_number": 296,
        "chunk_id": 3,
        "text": """
        Figure 3.39 illustrates the TCP three-way handshake.
        """
    }
]

answer = """
The client initiates a TCP connection using a socket. (Page 201)

TCP establishes a connection through SYN, SYN-ACK and ACK.
(Page 296)

The third handshake segment may carry data. (Page 295)

TCP guarantees a bandwidth of 1 Gbps. (Page 201)
"""

result = verifier.verify(
    answer,
    retrieved_chunks
)

print("\n========== VERIFICATION RESULTS ==========")

print("Overall status:", result["status"])

for index, claim in enumerate(result["claims"], start=1):

    print(f"\nClaim {index}:")
    print("Text:", claim["claim"])
    print("Cited pages:", claim["cited_pages"])
    print("Status:", claim["status"])
    print("Reason:", claim["reason"])

print("\nVerification error:", result["error"])
print("==========================================")
