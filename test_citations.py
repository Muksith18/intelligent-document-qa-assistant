
import re


def validate_citations(answer, retrieved_chunks):

    allowed_pages = {
        str(chunk["page_number"])
        for chunk in retrieved_chunks
    }

    cited_pages = re.findall(
        r"\bPage\s+(\d+)\b",
        answer,
        flags=re.IGNORECASE
    )

    invalid_citations = [
        page
        for page in cited_pages
        if page not in allowed_pages
    ]

    return {
        "cited_pages": cited_pages,
        "invalid_citations": invalid_citations,
        "valid": len(invalid_citations) == 0
    }


retrieved_chunks = [
    {"page_number": 201},
    {"page_number": 295},
    {"page_number": 296}
]


# TEST 1: Valid citation

answer_1 = (
    "TCP establishes a connection through a handshake. "
    "(Page 201)"
)


# TEST 2: Invalid citation

answer_2 = (
    "TCP establishes a connection through a handshake. "
    "(Page 450)"
)


# TEST 3: Mixed citations

answer_3 = (
    "TCP uses a three-way handshake. "
    "(Page 201, Page 450, Page 296)"
)


test_cases = [
    answer_1,
    answer_2,
    answer_3
]


for index, answer in enumerate(test_cases, start=1):

    result = validate_citations(
        answer,
        retrieved_chunks
    )

    print(f"\n========== TEST {index} ==========")

    print("Answer:", answer)

    print("Cited pages:", result["cited_pages"])

    print("Invalid citations:", result["invalid_citations"])

    print("Valid:", result["valid"])

    print("=================================")
