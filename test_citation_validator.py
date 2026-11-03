
from citation_validator import validate_citations

retrieved_chunks = [
    {"page_number": 123},
    {"page_number": 201},
    {"page_number": 296}
]

test_cases = [
    {
        "name": "Valid citation",
        "answer": "TCP is reliable. (Page 123)"
    },
    {
        "name": "Invalid citation",
        "answer": "TCP is reliable. (Page 450)"
    },
    {
        "name": "Mixed citations",
        "answer": "TCP is reliable. (Pages 123 and 450)"
    },
    {
        "name": "Multiple valid citations",
        "answer": "TCP uses a handshake. (Pages 201 and 296)"
    },
    {
        "name": "No citation",
        "answer": "TCP provides reliable data transfer."
    },
    {
        "name": "Unusual Unicode whitespace",
        "answer": "TCP is reliable. (Page\u202f123)"
    }
]

for test in test_cases:

    result = validate_citations(
        test["answer"],
        retrieved_chunks
    )

    print("\n" + "=" * 55)
    print("TEST:", test["name"])
    print("Answer:", test["answer"])
    print("Cited pages:", result["cited_pages"])
    print("Valid pages:", result["valid_pages"])
    print("Invalid pages:", result["invalid_pages"])
    print("Validation:", result["is_valid"])
