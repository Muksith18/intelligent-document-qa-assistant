EVIDENCE_PENDING = "PENDING_MANUAL_VALIDATION"
EVIDENCE_VALIDATED = "MANUALLY_VALIDATED"


def pending_evidence(note):
    return {
        "annotation_status": EVIDENCE_PENDING,
        "required_pdf_pages": [],
        "required_chunk_ids": [],
        "note": note,
    }


def has_manually_validated_evidence(case):
    """Return whether a case has page labels suitable for retrieval metrics."""
    evidence = case.get("expected_evidence", {})
    return (
        evidence.get("annotation_status") == EVIDENCE_VALIDATED
        and bool(evidence.get("required_pdf_pages"))
    )


EVALUATION_DATASET = [
    {
        "id": "T1",
        "category": "Direct factual",
        "question": "What is TCP?",
        "objective": "Assess whether a direct definition is grounded in retrieved evidence.",
        "expected_answer_behavior": "Answer only from supplied document evidence with PDF citations.",
        "expected_evidence": pending_evidence(
            "Annotate supporting PDF pages and chunks after manual evidence review."
        ),
        "abstention_expectation": "NOT_EXPECTED_IF_EVIDENCE_RETRIEVED",
    },
    {
        "id": "T2",
        "category": "Comparison",
        "question": "What is the difference between TCP and UDP?",
        "objective": "Assess grounded comparison without combining unsupported differences.",
        "expected_answer_behavior": "Compare only properties explicitly supported by retrieved evidence.",
        "expected_evidence": pending_evidence(
            "Annotate evidence for both protocols after manual evidence review."
        ),
        "abstention_expectation": "NOT_EXPECTED_IF_EVIDENCE_RETRIEVED",
    },
    {
        "id": "T3",
        "category": "Multi-step reasoning",
        "question": "Explain TCP three-way handshake.",
        "objective": "Assess a multi-step explanation with claim-level grounding.",
        "expected_answer_behavior": "Explain only supported handshake steps and cite each factual claim.",
        "expected_evidence": pending_evidence(
            "Annotate step-specific evidence after manual evidence review."
        ),
        "abstention_expectation": "NOT_EXPECTED_IF_EVIDENCE_RETRIEVED",
    },
    {
        "id": "T4",
        "category": "Specific detail",
        "question": "Can the third step of the TCP three-way handshake carry data?",
        "objective": "Assess retrieval and grounding for one precise protocol detail.",
        "expected_answer_behavior": "Answer the detail only when the retrieved evidence establishes it.",
        "expected_evidence": pending_evidence(
            "Annotate the supporting third-step passage after manual evidence review."
        ),
        "abstention_expectation": "NOT_EXPECTED_IF_EVIDENCE_RETRIEVED",
    },
    {
        "id": "T5",
        "category": "Partial evidence",
        "question": "What is TCP, and what exact bandwidth does it guarantee?",
        "objective": "Assess whether the system separates supported TCP facts from unsupported bandwidth claims.",
        "expected_answer_behavior": "Answer only supported TCP facts and abstain from any unsupported bandwidth guarantee.",
        "expected_evidence": pending_evidence(
            "Annotate TCP-definition evidence and confirm whether bandwidth evidence exists."
        ),
        "abstention_expectation": "PARTIALLY_EXPECTED",
    },
    {
        "id": "T6",
        "category": "Out-of-scope",
        "question": "Who won FIFA World Cup 2022?",
        "objective": "Assess refusal to answer an out-of-document question.",
        "expected_answer_behavior": "Abstain rather than use general knowledge or fabricate a cited answer.",
        "expected_evidence": pending_evidence(
            "No supporting document evidence is expected; confirm this during manual review."
        ),
        "abstention_expectation": "EXPECTED",
    },
]
