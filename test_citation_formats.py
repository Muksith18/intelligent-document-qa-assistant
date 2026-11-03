
from citation_validator import validate_citations

retrieved_chunks = [
    {"page_number": 201},
    {"page_number": 327}
]

test_answers = [
    "TCP is reliable. Page 201.",
    "TCP is connection-oriented 【327】.",
    "TCP is reliable (Pages 201 and 327).",
    "TCP is reliable 【450】."
]

for answer in test_answers:

    result = validate_citations(
        answer,
        retrieved_chunks
    )

    print("\nAnswer:", answer)
    print("Cited pages:", result["cited_pages"])
    print("Valid pages:", result["valid_pages"])
    print("Invalid pages:", result["invalid_pages"])
    print("Citation valid:", result["is_valid"])
