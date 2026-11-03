
import time

from retrieval import Retriever
from llm import AnswerGenerator
from citation_validator import (
    resolve_source_citations,
    validate_citations,
    validate_citation_structure,
)
from citation_verifier import CitationVerifier


class RAGEngine:

    def __init__(self, document_id):

        print("Initializing RAG Engine...")

        self.retriever = Retriever(document_id)

        self.answer_generator = AnswerGenerator()

        self.citation_verifier = CitationVerifier(
            client=self.answer_generator.client,
            model_name=self.answer_generator.model_name
        )

        print("RAG Engine is ready!")

    def create_result(
        self,
        answer,
        generated_answer=None,
        sources=None,
        fallback=False,
        citation_validation=None,
        source_citation_validation=None,
        citation_structure_validation=None,
        source_citation_map=None,
        semantic_verification=None,
        insufficient_evidence=False,
        quality_status="NOT_CHECKED",
        quality_notice=None,
        timings=None,
        error_details=None,
        abstention_reason=None,
        outcome=None,
        generation_finish_reason=None,
    ):

        return {
            "answer": answer,
            "generated_answer": generated_answer or answer,
            "sources": sources or [],
            "fallback": fallback,
            "citation_validation": citation_validation,
            "source_citation_validation": source_citation_validation,
            "citation_structure_validation": citation_structure_validation,
            "source_citation_map": source_citation_map or {},
            "semantic_verification": semantic_verification,
            "insufficient_evidence": insufficient_evidence,
            "quality_status": quality_status,
            "quality_notice": quality_notice,
            "error_details": error_details,
            "abstention_reason": abstention_reason,
            "outcome": outcome or quality_status,
            "generation_finish_reason": generation_finish_reason,
            "timings": timings or {
                "retrieval": 0,
                "generation": 0,
                "verification": 0,
                "total": 0
            }
        }

    def create_quality_notice(
        self,
        semantic_verification,
        citation_validation,
        source_citation_validation=None,
        citation_structure_validation=None,
    ):

        citation_invalid = (
            source_citation_validation
            and not source_citation_validation["is_valid"]
        ) or (
            citation_validation
            and not citation_validation["is_valid"]
        )

        if citation_structure_validation and not citation_structure_validation["is_valid"]:
            notices = []

            if citation_invalid:
                notices.append(
                    "WARNING: Citation validation failed. Some citations are "
                    "missing or do not refer to retrieved document pages."
                )

            notices.append(
                "WARNING: Citation structure validation failed. Every factual "
                "sentence and procedural step must have an immediate source-ID citation."
            )

            if semantic_verification and semantic_verification["status"] == "ERROR":
                notices.append(
                    "Semantic verification could not be completed; "
                    "the answer has not been fully verified."
                )

            return " ".join(notices)

        if citation_invalid:
            notice = (
                "WARNING: Citation validation failed. "
                "Some citations are missing or do not refer "
                "to retrieved document pages."
            )

            if semantic_verification and semantic_verification["status"] == "ERROR":
                notice += (
                    " Semantic verification could not be completed; "
                    "the answer has not been fully verified."
                )

            return notice

        if not semantic_verification:
            return (
                "WARNING: Semantic verification was not completed."
            )

        status = semantic_verification["status"]

        if status == "SUPPORTED":
            return None

        if status == "UNSUPPORTED":
            return (
                "WARNING: Semantic verification detected "
                "unsupported factual claims. "
                "Please review the answer against the source passages."
            )

        if status == "INCONCLUSIVE":
            return (
                "NOTICE: Some factual claims could not be "
                "verified confidently from the retrieved passages."
            )

        if status == "UNCITED":
            return (
                "WARNING: Some factual claims do not have "
                "adequate source citations."
            )

        if status == "ERROR":
            return (
                "NOTICE: Semantic verification could not be completed. "
                "The answer has not been fully verified."
            )

        return (
            "NOTICE: The answer verification status is unknown."
        )

    def determine_quality_status(
        self,
        semantic_verification,
        citation_validation,
        source_citation_validation=None,
        citation_structure_validation=None,
    ):

        if (
            source_citation_validation
            and not source_citation_validation["is_valid"]
        ) or (
            citation_validation
            and not citation_validation["is_valid"]
        ):
            return "CITATION_INVALID"

        if (
            citation_structure_validation
            and not citation_structure_validation["is_valid"]
        ):
            return "CITATION_STRUCTURE_INVALID"

        if not semantic_verification:
            return "NOT_CHECKED"

        if semantic_verification["status"] == "ERROR":
            return "VERIFICATION_ERROR"

        return semantic_verification["status"]

    def ask_question(self, question):

        if not question or not question.strip():

            return self.create_result(
                answer="Please enter a question.",
                quality_status="NOT_CHECKED"
            )

        # ==========================================
        # STEP 1: RETRIEVAL
        # ==========================================

        retrieval_start = time.perf_counter()

        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=5
        )

        retrieval_time = (
            time.perf_counter() - retrieval_start
        )

        # ==========================================
        # STEP 2: HANDLE EMPTY RETRIEVAL
        # ==========================================

        if not retrieved_chunks:

            answer = (
                "I couldn't find relevant information in the "
                "uploaded document to answer this question."
            )

            return self.create_result(
                answer=answer,
                generated_answer=answer,
                insufficient_evidence=True,
                quality_status="INSUFFICIENT_EVIDENCE",
                quality_notice=None,
                abstention_reason="NO_RETRIEVED_EVIDENCE",
                outcome="INSUFFICIENT_EVIDENCE",
                timings={
                    "retrieval": round(retrieval_time, 2),
                    "generation": 0,
                    "verification": 0,
                    "total": round(retrieval_time, 2)
                }
            )

        # ==========================================
        # STEP 3: ANSWER GENERATION
        # ==========================================

        generation_start = time.perf_counter()

        answer = self.answer_generator.generate(
            question,
            retrieved_chunks
        )

        generation_time = (
            time.perf_counter() - generation_start
        )

        answer = answer.strip()

        # ==========================================
        # STEP 4: HANDLE GENERATION FAILURE
        # ==========================================

        if answer.startswith((
            "AI_SERVICE_UNAVAILABLE:",
            "AI_GENERATION_FAILED:",
            "AI_GENERATION_TRUNCATED:",
        )):

            print(
                "\nAI generation did not produce a complete answer. "
                "Showing retrieved passages as a fallback."
            )

            fallback_parts = [
                "### AI generation did not produce a complete answer",
                "",
                "Here are the passages retrieved directly "
                "from your document:",
                ""
            ]

            for index, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

                fallback_parts.append(
                    f"**Passage {index} — "
                    f"Page {chunk['page_number']}**"
                )

                fallback_parts.append(chunk["text"])
                fallback_parts.append("")

            fallback_answer = "\n\n".join(fallback_parts)

            generation_error = getattr(
                self.answer_generator,
                "last_error",
                None
            ) or {
                "type": "API_ERROR",
                "message": "AI generation failed.",
                "finish_reason": None,
            }

            error_type = generation_error.get("type")

            outcome = (
                "GENERATION_TRUNCATED"
                if error_type == "TRUNCATED_RESPONSE"
                else "GENERATION_ERROR"
                if error_type in {"EMPTY_RESPONSE", "INCOMPLETE_RESPONSE"}
                else "API_ERROR"
            )

            return self.create_result(
                answer=fallback_answer,
                generated_answer=answer,
                sources=[
                    {
                        "page_number": chunk["page_number"],
                        "chunk_id": chunk["chunk_id"],
                        "distance": chunk["distance"],
                        "is_context": chunk.get("is_context", False),
                        "text": chunk["text"]
                    }
                    for chunk in retrieved_chunks
                ],
                fallback=True,
                quality_status=outcome,
                quality_notice=(
                    "NOTICE: AI generation did not produce a complete answer. "
                    "The displayed content contains retrieved "
                    "document passages, not a generated answer."
                ),
                error_details=generation_error,
                outcome=outcome,
                generation_finish_reason=getattr(
                    self.answer_generator,
                    "last_finish_reason",
                    None
                ),
                timings={
                    "retrieval": round(retrieval_time, 2),
                    "generation": round(generation_time, 2),
                    "verification": 0,
                    "total": round(
                        retrieval_time + generation_time, 2
                    )
                }
            )

        # ==========================================
        # STEP 5: HANDLE INSUFFICIENT EVIDENCE
        # ==========================================

        if answer.startswith("INSUFFICIENT_EVIDENCE:"):

            print("\nInsufficient evidence detected.")

            abstention_answer = (
                "I couldn't find sufficient information in the "
                "uploaded document to answer this question reliably."
            )

            return self.create_result(
                answer=abstention_answer,
                generated_answer=answer,
                insufficient_evidence=True,
                quality_status="INSUFFICIENT_EVIDENCE",
                abstention_reason="MODEL_ABSTAINED",
                outcome="INSUFFICIENT_EVIDENCE",
                generation_finish_reason=getattr(
                    self.answer_generator,
                    "last_finish_reason",
                    None
                ),
                timings={
                    "retrieval": round(retrieval_time, 2),
                    "generation": round(generation_time, 2),
                    "verification": 0,
                    "total": round(
                        retrieval_time + generation_time, 2
                    )
                }
            )

        # ==========================================
        # STEP 6: CITATION VALIDATION
        # ==========================================

        citation_structure_validation = validate_citation_structure(answer)

        # Diagnostic: citation structure
        print("\n========== CITATION STRUCTURE DEBUG ==========")

        print("Is valid:", citation_structure_validation["is_valid"])
        print("Claim count:", citation_structure_validation["claim_count"])

        print(
            "Trailing citation block:",
            citation_structure_validation["trailing_citation_block"]
        )

        print("\nUncited claims:")

        for claim in citation_structure_validation["uncited_claims"]:
            print("-", claim)

        print("==============================================\n")

        source_citation_validation = resolve_source_citations(
            answer,
            retrieved_chunks,
        )

        resolved_answer = source_citation_validation["answer"]

        citation_validation = validate_citations(
            resolved_answer,
            retrieved_chunks,
        )

        print("\n========== CITATION VALIDATION ==========")

        print(
            "Cited pages:",
            citation_validation["cited_pages"]
        )

        print(
            "Retrieved pages:",
            citation_validation["retrieved_pages"]
        )

        print(
            "Valid pages:",
            citation_validation["valid_pages"]
        )

        print(
            "Invalid pages:",
            citation_validation["invalid_pages"]
        )

        print(
            "Citation references valid:",
            citation_validation["is_valid"]
        )

        print("=========================================\n")

        # ==========================================
        # STEP 7: SEMANTIC VERIFICATION
        # ==========================================

        verification_start = time.perf_counter()

        semantic_verification = self.citation_verifier.verify(
            resolved_answer,
            retrieved_chunks
        )

        verification_time = (
            time.perf_counter() - verification_start
        )

        print("\n========== QUALITY REPORT ==========")

        print(
            "Semantic status:",
            semantic_verification["status"]
        )

        print(
            "Claims checked:",
            len(semantic_verification["claims"])
        )

        # Diagnostic: individual semantic claims
        print("\n========== INDIVIDUAL CLAIM RESULTS ==========")

        for index, claim in enumerate(
            semantic_verification["claims"],
            start=1
        ):

            print(f"\nClaim {index}:")
            print("Text:", claim["claim"])
            print("Cited pages:", claim["cited_pages"])
            print("Status:", claim["status"])
            print("Reason:", claim["reason"])

        print("==============================================\n")
        print("====================================\n")

        # ==========================================
        # STEP 8: DETERMINE ANSWER QUALITY
        # ==========================================

        quality_status = self.determine_quality_status(
            semantic_verification,
            citation_validation,
            source_citation_validation,
            citation_structure_validation,
        )

        quality_notice = self.create_quality_notice(
            semantic_verification,
            citation_validation,
            source_citation_validation,
            citation_structure_validation,
        )

        # Preserve the original generated answer.
        generated_answer = answer
        answer = resolved_answer

        # Add a visible notice without changing the answer.
        if quality_notice:

            answer = (
                resolved_answer
                + "\n\n---\n\n"
                + quality_notice
            )

        # ==========================================
        # STEP 9: COLLECT SOURCES
        # ==========================================

        sources = []

        for chunk in retrieved_chunks:

            sources.append({
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "distance": chunk["distance"],
                "is_context": chunk.get("is_context", False),
                "text": chunk["text"]
            })

        # ==========================================
        # STEP 10: PERFORMANCE REPORT
        # ==========================================

        total_time = (
            retrieval_time
            + generation_time
            + verification_time
        )

        timings = {
            "retrieval": round(retrieval_time, 2),
            "generation": round(generation_time, 2),
            "verification": round(verification_time, 2),
            "total": round(total_time, 2)
        }

        print("\n========== PERFORMANCE REPORT ==========")

        print("Retrieval time:", timings["retrieval"])
        print("Generation time:", timings["generation"])
        print("Verification time:", timings["verification"])
        print("Total question time:", timings["total"])

        print("Quality status:", quality_status)

        print("=========================================\n")

        # ==========================================
        # FINAL RESULT
        # ==========================================

        return self.create_result(
            answer=answer,
            generated_answer=generated_answer,
            sources=sources,
            fallback=False,
            citation_validation=citation_validation,
            source_citation_validation=source_citation_validation,
            citation_structure_validation=citation_structure_validation,
            source_citation_map=source_citation_validation["source_map"],
            semantic_verification=semantic_verification,
            insufficient_evidence=False,
            quality_status=quality_status,
            quality_notice=quality_notice,
            error_details=(
                {
                    "type": "VERIFICATION_ERROR",
                    "message": semantic_verification["error"]
                }
                if semantic_verification["status"] == "ERROR"
                else None
            ),
            outcome=quality_status,
            generation_finish_reason=getattr(
                self.answer_generator,
                "last_finish_reason",
                None
            ),
            timings=timings
        )
