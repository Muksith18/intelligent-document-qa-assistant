import sys
import types
import unittest
import re
from types import SimpleNamespace

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda **kwargs: None
sys.modules.setdefault("dotenv", dotenv_module)

groq_module = types.ModuleType("groq")
groq_module.Groq = object
sys.modules.setdefault("groq", groq_module)

embedding_model_module = types.ModuleType("embedding_model")
embedding_model_module.get_embedding_model = lambda: None
sys.modules.setdefault("embedding_model", embedding_model_module)

from citation_validator import validate_citations
from citation_verifier import CitationVerifier
from llm import AnswerGenerator
from rag_engine import RAGEngine


class FakeCompletions:
    def __init__(self, content, finish_reason="stop", error=None):
        self.content = content
        self.finish_reason = finish_reason
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                    finish_reason=self.finish_reason,
                )
            ]
        )


class FakeClient:
    def __init__(self, content, finish_reason="stop", error=None):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(content, finish_reason=finish_reason, error=error)
        )


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, top_k):
        return self.chunks


class FakeAnswerGenerator:
    def __init__(self, answer, last_error=None):
        self.answer = answer
        self.last_error = last_error

    def generate(self, question, chunks):
        return self.answer


class FakeVerifier:
    def __init__(self):
        self.received_chunks = None

    def verify(self, answer, retrieved_chunks):
        self.received_chunks = retrieved_chunks
        return {
            "status": "SUPPORTED",
            "claims": [],
            "error": None,
            "raw_response": "{}",
            "verification_time": 0,
        }


class CitationGroundingTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "page_number": 327,
                "chunk_id": 42,
                "text": "TCP provides reliable, connection-oriented delivery.",
            },
            {
                "page_number": 328,
                "chunk_id": 43,
                "text": "TCP uses acknowledgements to confirm delivery.",
            },
        ]

    def test_valid_citation_references_retrieved_pdf_page(self):
        result = validate_citations(
            "TCP is reliable.【327】",
            self.chunks,
        )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["valid_pages"], [327])
        self.assertEqual(result["invalid_pages"], [])

    def test_invalid_citation_for_unretrieved_pdf_page_is_rejected(self):
        result = validate_citations(
            "TCP is reliable.【7】",
            self.chunks,
        )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_pages"], [7])

    def test_engine_preserves_and_reports_an_invalid_generated_citation(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever([
            {**self.chunks[0], "distance": 0.12, "is_context": False}
        ])
        engine.answer_generator = FakeAnswerGenerator(
            "TCP is reliable.【7】"
        )
        engine.citation_verifier = FakeVerifier()

        result = engine.ask_question("What is TCP?")

        self.assertEqual(result["quality_status"], "CITATION_INVALID")
        self.assertEqual(result["citation_validation"]["invalid_pages"], [7])
        self.assertEqual(result["generated_answer"], "TCP is reliable.【7】")
        self.assertIn("Citation validation failed", result["answer"])

    def test_multiple_retrieved_pdf_page_citations_are_valid(self):
        result = validate_citations(
            "TCP is connection-oriented.【327】 It uses acknowledgements.【328】",
            self.chunks,
        )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["valid_pages"], [327, 328])

    def test_prompt_uses_source_ids_and_forbids_non_source_numbers(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("TCP is reliable.【327】")
        generator.model_name = "openai/gpt-oss-20b"

        generator.generate("What is TCP?", self.chunks)

        request = generator.client.chat.completions.calls[0]
        prompt = "\n".join(message["content"] for message in request["messages"])

        self.assertIn("PDF PAGE 327", prompt)
        self.assertIn("PDF PAGE 328", prompt)
        self.assertIn("SOURCE ID: [S1]", prompt)
        self.assertIn("SOURCE ID: [S2]", prompt)
        self.assertIn("Use source-ID citations only in the form [S#]", prompt)
        self.assertNotIn("Every factual claim must have a PDF-page citation", prompt)
        self.assertIn("printed textbook page numbers", prompt.lower())
        self.assertIn("never be used as citations", prompt.lower())

    def test_prompt_forbids_unsupported_examples_and_typical_use_cases(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("TCP is reliable.ã€327ã€‘")
        generator.model_name = "openai/gpt-oss-20b"

        generator.generate("Compare TCP and UDP.", self.chunks)

        request = generator.client.chat.completions.calls[0]
        prompt = "\n".join(message["content"] for message in request["messages"])

        normalized_prompt = re.sub(r"\s+", " ", prompt)
        self.assertIn(
            "Every example, typical use case, advantage, disadvantage, statistic, "
            "and technical detail must be explicitly supported",
            normalized_prompt,
        )
        self.assertIn(
            "Do not include a fact merely because it is common knowledge",
            normalized_prompt,
        )

    def test_prompt_requires_claim_level_source_id_citations_only(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("TCP is reliable. [S1]")
        generator.model_name = "openai/gpt-oss-20b"

        generator.generate("Explain TCP three-way handshake.", self.chunks)

        prompt = "\n".join(
            message["content"]
            for message in generator.client.chat.completions.calls[0]["messages"]
        )
        normalized_prompt = re.sub(r"\s+", " ", prompt)

        self.assertIn(
            "Use source-ID citations only in the form [S#]; never output PDF page citations.",
            normalized_prompt,
        )
        self.assertIn(
            "immediately after every factual sentence or numbered step",
            normalized_prompt,
        )
        self.assertIn(
            "Repeat the citation immediately after each claim, even when the same source supports multiple claims.",
            normalized_prompt,
        )

    def test_prompt_requires_narrow_supported_procedural_explanations(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("TCP establishes a connection. [S1]")
        generator.model_name = "openai/gpt-oss-20b"

        generator.generate("Explain TCP three-way handshake.", self.chunks)

        prompt = "\n".join(
            message["content"]
            for message in generator.client.chat.completions.calls[0]["messages"]
        )
        normalized_prompt = re.sub(r"\s+", " ", prompt)

        self.assertIn(
            "For procedural questions, include a step or detail only when a supplied passage explicitly establishes it.",
            normalized_prompt,
        )
        self.assertIn(
            "Omit unsupported steps or details rather than filling gaps from general knowledge.",
            normalized_prompt,
        )
        self.assertIn(
            "When the evidence supports only a partial explanation, provide that narrower explanation instead of inventing missing details.",
            normalized_prompt,
        )

    def test_normal_generation_preserves_stop_finish_reason(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("TCP is reliable.ã€327ã€‘")
        generator.model_name = "openai/gpt-oss-20b"

        answer = generator.generate("What is TCP?", self.chunks)

        request = generator.client.chat.completions.calls[0]
        self.assertEqual(answer, "TCP is reliable.ã€327ã€‘")
        self.assertEqual(generator.last_finish_reason, "stop")
        self.assertIsNone(generator.last_error)
        self.assertEqual(request["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", request)
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertFalse(request["include_reasoning"])
        self.assertNotIn("reasoning_format", request)

    def test_generation_prompt_requires_a_compact_grounded_comparison(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("TCP is reliable.ã€327ã€‘")
        generator.model_name = "openai/gpt-oss-20b"

        generator.generate("What is the difference between TCP and UDP?", self.chunks)

        request = generator.client.chat.completions.calls[0]
        prompt = "\n".join(message["content"] for message in request["messages"])
        normalized_prompt = re.sub(r"\s+", " ", prompt)
        self.assertIn(
            "For a comparison question, use a compact table with no more than four rows",
            normalized_prompt,
        )
        self.assertIn("Keep the answer under 160 words", normalized_prompt)
        self.assertIn("Do not add examples, use cases, advantages, or details", normalized_prompt)

    def test_length_finish_reason_with_partial_content_is_rejected(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("TCP is reliable.ã€327ã€‘", finish_reason="length")
        generator.model_name = "openai/gpt-oss-20b"

        answer = generator.generate("What is TCP?", self.chunks)

        self.assertTrue(answer.startswith("AI_GENERATION_TRUNCATED:"))
        self.assertEqual(generator.last_finish_reason, "length")
        self.assertEqual(generator.last_error["type"], "TRUNCATED_RESPONSE")
        self.assertEqual(generator.last_error["partial_content"], "TCP is reliable.ã€327ã€‘")

    def test_length_finish_reason_with_empty_content_is_rejected_as_truncated(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("", finish_reason="length")
        generator.model_name = "openai/gpt-oss-20b"

        answer = generator.generate("What is TCP?", self.chunks)

        self.assertTrue(answer.startswith("AI_GENERATION_TRUNCATED:"))
        self.assertEqual(generator.last_finish_reason, "length")
        self.assertEqual(generator.last_error["type"], "TRUNCATED_RESPONSE")

    def test_empty_stop_response_is_classified_separately_from_connectivity_error(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("", finish_reason="stop")
        generator.model_name = "openai/gpt-oss-20b"

        answer = generator.generate("What is TCP?", self.chunks)

        self.assertTrue(answer.startswith("AI_GENERATION_FAILED:"))
        self.assertEqual(generator.last_finish_reason, "stop")
        self.assertEqual(generator.last_error["type"], "EMPTY_RESPONSE")

    def test_rate_limit_error_preserves_rate_limit_type(self):
        rate_limit = RuntimeError("429 rate limit")
        rate_limit.status_code = 429
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("", error=rate_limit)
        generator.model_name = "openai/gpt-oss-20b"

        answer = generator.generate("What is TCP?", self.chunks)

        self.assertTrue(answer.startswith("AI_SERVICE_UNAVAILABLE:"))
        self.assertEqual(generator.last_error["type"], "RATE_LIMIT")
        self.assertIsNone(generator.last_finish_reason)

    def test_non_rate_limit_api_error_preserves_api_error_type(self):
        generator = AnswerGenerator.__new__(AnswerGenerator)
        generator.client = FakeClient("", error=RuntimeError("service unavailable"))
        generator.model_name = "openai/gpt-oss-20b"

        answer = generator.generate("What is TCP?", self.chunks)

        self.assertTrue(answer.startswith("AI_SERVICE_UNAVAILABLE:"))
        self.assertEqual(generator.last_error["type"], "API_ERROR")
        self.assertIsNone(generator.last_finish_reason)

    def test_truncated_answer_is_not_sent_to_citation_verification(self):
        verifier = FakeVerifier()
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever([{**self.chunks[0], "distance": 0.12, "is_context": False}])
        engine.answer_generator = FakeAnswerGenerator(
            "AI_GENERATION_TRUNCATED: completion reached its output limit.",
            last_error={"type": "TRUNCATED_RESPONSE", "message": "length", "finish_reason": "length"},
        )
        engine.answer_generator.last_finish_reason = "length"
        engine.citation_verifier = verifier

        result = engine.ask_question("What is TCP?")

        self.assertEqual(result["outcome"], "GENERATION_TRUNCATED")
        self.assertEqual(result["error_details"]["type"], "TRUNCATED_RESPONSE")
        self.assertEqual(result["generation_finish_reason"], "length")
        self.assertIsNone(result["citation_validation"])
        self.assertIsNone(verifier.received_chunks)
        self.assertIn("did not produce a complete answer", result["answer"])

    def test_insufficient_evidence_response_is_preserved_as_abstention(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever(self.chunks)
        engine.answer_generator = FakeAnswerGenerator(
            "INSUFFICIENT_EVIDENCE: The uploaded document does not contain "
            "sufficient information to answer this question."
        )
        engine.citation_verifier = FakeVerifier()

        result = engine.ask_question("What is the TCP checksum algorithm?")

        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["quality_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["outcome"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["abstention_reason"], "MODEL_ABSTAINED")
        self.assertIsNone(result["citation_validation"])
        self.assertIsNone(engine.citation_verifier.received_chunks)

    def test_empty_retrieval_records_a_distinct_abstention_reason(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever([])

        result = engine.ask_question("Who won FIFA World Cup 2022?")

        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["outcome"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["abstention_reason"], "NO_RETRIEVED_EVIDENCE")

    def test_api_fallback_preserves_rate_limit_details(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever([
            {**self.chunks[0], "distance": 0.12, "is_context": False}
        ])
        engine.answer_generator = FakeAnswerGenerator(
            "AI_SERVICE_UNAVAILABLE: The service could not generate an answer.",
            last_error={"type": "RATE_LIMIT", "message": "429 Too Many Requests"},
        )

        result = engine.ask_question("What is TCP?")

        self.assertTrue(result["fallback"])
        self.assertEqual(result["quality_status"], "API_ERROR")
        self.assertEqual(result["outcome"], "API_ERROR")
        self.assertEqual(result["error_details"]["type"], "RATE_LIMIT")
        self.assertFalse(result["sources"][0]["is_context"])

    def test_context_only_chunk_is_passed_to_verifier_with_pdf_page(self):
        context_only_chunk = {
            "page_number": 327,
            "chunk_id": 42,
            "text": "TCP provides reliable delivery.",
            "distance": None,
            "is_context": True,
        }
        verifier = FakeVerifier()
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever([context_only_chunk])
        engine.answer_generator = FakeAnswerGenerator(
            "TCP provides reliable delivery. [S1]"
        )
        engine.citation_verifier = verifier

        result = engine.ask_question("What does TCP provide?")

        self.assertEqual(verifier.received_chunks, [context_only_chunk])
        self.assertEqual(result["sources"][0]["page_number"], 327)
        self.assertIsNone(result["sources"][0]["distance"])
        self.assertEqual(result["quality_status"], "SUPPORTED")
        self.assertEqual(result["answer"], "TCP provides reliable delivery. 【327】")
        self.assertEqual(result["answer"], "TCP provides reliable delivery. \u3010327\u3011")

    def test_semantic_verifier_receives_pdf_page_labelled_evidence(self):
        client = FakeClient(
            '{"claims":[{"claim":"TCP provides reliable delivery.",'
            '"cited_pages":[327],"status":"SUPPORTED",'
            '"reason":"The passage states this."}]}'
        )
        verifier = CitationVerifier(client, "openai/gpt-oss-20b")

        result = verifier.verify(
            "TCP provides reliable delivery.【327】",
            [{**self.chunks[0], "distance": None, "is_context": True}],
        )

        request = client.chat.completions.calls[0]
        user_prompt = request["messages"][1]["content"]
        self.assertIn("PDF PAGE 327:", user_prompt)
        self.assertEqual(result["status"], "SUPPORTED")

    def test_semantic_verifier_reports_mixed_supported_and_unsupported_claims(self):
        client = FakeClient(
            '{"claims":[{"claim":"TCP is reliable.","cited_pages":[327],'
            '"status":"SUPPORTED","reason":"The passage states this."},'
            '{"claim":"TCP is used for gaming.","cited_pages":[327],'
            '"status":"UNSUPPORTED","reason":"The passage does not state this."}]}'
        )
        verifier = CitationVerifier(client, "openai/gpt-oss-20b")

        result = verifier.verify("TCP is reliable.ã€327ã€‘ TCP is used for gaming.ã€327ã€‘", self.chunks)

        self.assertEqual(result["status"], "UNSUPPORTED")
        self.assertEqual([claim["status"] for claim in result["claims"]], ["SUPPORTED", "UNSUPPORTED"])

    def test_semantic_verifier_rejects_procedural_details_cited_to_non_supporting_pages(self):
        claim = (
            "The handshake takes place within the transport layer and is invisible "
            "to the client and server programs."
        )
        client = FakeClient(
            '{"claims":[{"claim":"' + claim + '","cited_pages":[274,295],'
            '"status":"UNSUPPORTED","reason":"Neither cited passage establishes this detail."}]}'
        )
        verifier = CitationVerifier(client, "openai/gpt-oss-20b")
        chunks = [
            {
                "page_number": 274,
                "chunk_id": 1,
                "text": "Three segments are sent between the two hosts during connection establishment.",
            },
            {
                "page_number": 295,
                "chunk_id": 2,
                "text": "After three steps are completed, the client and server hosts can send data.",
            },
        ]

        result = verifier.verify(f"{claim} 【274】【295】", chunks)

        self.assertEqual(result["status"], "UNSUPPORTED")
        self.assertEqual(result["claims"][0]["status"], "UNSUPPORTED")
        self.assertIn("PDF PAGE 274:", client.chat.completions.calls[0]["messages"][1]["content"])
        self.assertNotIn("PDF PAGE 201:", client.chat.completions.calls[0]["messages"][1]["content"])

    def test_semantic_verifier_accepts_transport_layer_claim_when_page_201_supports_it(self):
        claim = (
            "The three-way handshake takes place within the transport layer and is "
            "completely invisible to the client and server programs."
        )
        page_201_text = (
            "The three-way handshake, which takes place within the transport layer, "
            "is completely invisible to the client and server programs."
        )
        client = FakeClient(
            '{"claims":[{"claim":"' + claim + '","cited_pages":[201],'
            '"status":"SUPPORTED","reason":"Page 201 explicitly states this."}]}'
        )
        verifier = CitationVerifier(client, "openai/gpt-oss-20b")

        result = verifier.verify(
            f"{claim} 【201】",
            [{"page_number": 201, "chunk_id": 3, "text": page_201_text}],
        )

        self.assertEqual(result["status"], "SUPPORTED")
        self.assertEqual(result["claims"][0]["cited_pages"], [201])
        verifier_prompt = client.chat.completions.calls[0]["messages"][1]["content"]
        self.assertIn("PDF PAGE 201:", verifier_prompt)
        self.assertIn(page_201_text, verifier_prompt)

    def test_semantic_verifier_reports_malformed_output_as_an_error(self):
        verifier = CitationVerifier(FakeClient("not JSON"), "openai/gpt-oss-20b")

        result = verifier.verify("TCP is reliable.ã€327ã€‘", self.chunks)

        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["claims"], [])


if __name__ == "__main__":
    unittest.main()
