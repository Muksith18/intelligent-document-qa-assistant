import sys
import types
import unittest


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

from citation_validator import (
    build_source_map,
    resolve_source_citations,
    validate_citations,
    validate_citation_structure,
)
from rag_engine import RAGEngine


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, top_k):
        return self.chunks


class FakeAnswerGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.last_error = None
        self.last_finish_reason = "stop"

    def generate(self, question, chunks):
        return self.answer


class FakeVerifier:
    def __init__(self, status="SUPPORTED", error=None):
        self.answer = None
        self.status = status
        self.error = error

    def verify(self, answer, chunks):
        self.answer = answer
        return {"status": self.status, "claims": [], "error": self.error}


class SourceCitationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunk_id": 42, "page_number": 193, "text": "TCP is reliable.", "distance": 0.1, "is_context": False},
            {"chunk_id": 43, "page_number": 193, "text": "TCP is connection-oriented.", "distance": None, "is_context": True},
            {"chunk_id": 44, "page_number": 194, "text": "UDP is connectionless.", "distance": None, "is_context": True},
        ]

    def test_source_ids_map_to_only_supplied_pdf_pages(self):
        self.assertEqual(build_source_map(self.chunks), {"S1": 193, "S2": 193, "S3": 194})

        result = resolve_source_citations("TCP is reliable. [S1]", self.chunks)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["answer"], "TCP is reliable. \u3010193\u3011")
        self.assertEqual(result["valid_source_ids"], ["S1"])

    def test_multiple_source_ids_map_to_their_actual_pages(self):
        result = resolve_source_citations("TCP [S1] UDP [S3]", self.chunks)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["answer"], "TCP \u3010193\u3011 UDP \u3010194\u3011")
        self.assertEqual(result["valid_source_ids"], ["S1", "S3"])

    def test_full_width_source_id_maps_to_the_same_pdf_page(self):
        result = resolve_source_citations("TCP 【S1】", self.chunks)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["cited_source_ids"], ["S1"])
        self.assertEqual(result["answer"], "TCP \u3010193\u3011")

    def test_mixed_bracket_formats_and_repeated_ids_are_normalized(self):
        result = resolve_source_citations("TCP [S1] UDP 【s3】 TCP again 【S1】", self.chunks)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["cited_source_ids"], ["S1", "S3"])
        self.assertEqual(result["source_citation_occurrences"], ["S1", "S3", "S1"])
        self.assertEqual(result["answer"], "TCP \u3010193\u3011 UDP \u3010194\u3011 TCP again \u3010193\u3011")

    def test_t2_full_width_source_id_regression_maps_sixth_passage_to_page_193(self):
        t2_chunks = [
            {"chunk_id": index, "page_number": 193 if index == 6 else 192, "text": "evidence"}
            for index in range(1, 7)
        ]
        answer = (
            "TCP is a connection-oriented protocol that provides a reliable byte-stream "
            "channel between two end systems, ensuring that data arrives in order and "
            "without loss【S6】. UDP is a connectionless protocol that sends independent "
            "packets from one end system to another, with no guarantees about delivery, "
            "ordering, or reliability【S6】."
        )

        result = resolve_source_citations(answer, t2_chunks)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["source_map"]["S6"], 193)
        self.assertEqual(result["source_citation_occurrences"], ["S6", "S6"])
        self.assertEqual(result["answer"].count("\u3010193\u3011"), 2)

    def test_unknown_source_id_is_rejected_and_never_mapped(self):
        result = resolve_source_citations("TCP [S99]", self.chunks)

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["unknown_source_ids"], ["S99"])
        self.assertEqual(result["answer"], "TCP [S99]")

    def test_t2_regression_resolves_to_valid_pdf_page_citations(self):
        t2_chunks = [
            {"chunk_id": index, "page_number": 193 if index == 6 else 192, "text": "evidence"}
            for index in range(1, 7)
        ]
        result = resolve_source_citations("TCP 【S6】. UDP 【S6】.", t2_chunks)

        page_validation = validate_citations(result["answer"], t2_chunks)

        self.assertEqual(result["source_citation_occurrences"], ["S6", "S6"])
        self.assertEqual(result["answer"].count("【193】"), 2)
        self.assertTrue(page_validation["is_valid"])

    def test_unknown_full_width_source_id_is_rejected_and_never_mapped(self):
        result = resolve_source_citations("TCP 【S99】", self.chunks)

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["unknown_source_ids"], ["S99"])
        self.assertEqual(result["answer"], "TCP 【S99】")

    def test_unbracketed_or_numeric_text_is_not_interpreted_as_a_source_id(self):
        answer = "Section S6, [6], (S6), and page 193 are not source citations."

        result = resolve_source_citations(answer, self.chunks)

        self.assertFalse(result["has_citations"])
        self.assertEqual(result["cited_source_ids"], [])
        self.assertEqual(result["answer"], answer)

    def test_missing_source_id_is_rejected(self):
        result = resolve_source_citations("TCP is reliable.", self.chunks)

        self.assertFalse(result["is_valid"])
        self.assertFalse(result["has_citations"])

    def test_model_generated_pdf_page_is_rejected_without_a_source_id(self):
        result = resolve_source_citations("TCP is reliable. \u3010193\u3011", self.chunks)

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["cited_source_ids"], [])
        self.assertEqual(result["answer"], "TCP is reliable. \u3010193\u3011")

    def test_t3_trailing_citations_do_not_cover_prior_claims(self):
        answer = (
            "The TCP three-way handshake establishes a reliable connection "
            "between a client and a server.\n"
            "1. The client sends a SYN segment to the server, indicating a "
            "request to open a connection.\n"
            "2. The server replies with a SYN-ACK segment, acknowledging the "
            "client's request and also requesting a connection.\n"
            "3. The client sends an ACK segment back to the server, completing "
            "the exchange.\n\n"
            "After these three segments are exchanged, the connection is "
            "established and the client and server can send data to each other. "
            "The handshake is performed entirely within the transport layer and "
            "is invisible to the application programs that use the sockets. "
            "[S12] [S13]"
        )

        result = validate_citation_structure(answer)

        self.assertFalse(result["is_valid"])
        self.assertEqual(len(result["uncited_claims"]), 5)

    def test_immediate_source_id_citation_passes_for_page_201_claim(self):
        result = validate_citation_structure(
            "The handshake takes place within the transport layer and is "
            "invisible to client and server programs. [S6]"
        )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["uncited_claims"], [])

    def test_source_id_citation_formats_survive_sentence_splitting(self):
        examples = [
            "Sentence. [S6]",
            "Sentence [S6].",
            "Sentence. 【S6】",
            "Sentence 【S6】.",
        ]

        for answer in examples:
            with self.subTest(answer=answer):
                result = validate_citation_structure(answer)
                self.assertTrue(result["is_valid"])
                self.assertEqual(result["uncited_claims"], [])

    def test_multiple_sentences_keep_their_individual_citations(self):
        result = validate_citation_structure(
            "TCP opens a connection. [S1] UDP sends datagrams. 【S2】"
        )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["claim_count"], 2)
        self.assertEqual(result["uncited_claims"], [])

    def test_abbreviation_decimal_url_and_inline_code_do_not_create_false_splits(self):
        examples = [
            "Examples include e.g. TCP and UDP. [S6]",
            "The U.S. network uses TCP. [S6]",
            "The timeout is 3.14 seconds. [S6]",
            "See https://example.com. [S6]",
            "Call `client.send(data)` to transmit it. [S6]",
        ]

        for answer in examples:
            with self.subTest(answer=answer):
                result = validate_citation_structure(answer)
                self.assertTrue(result["is_valid"])
                self.assertEqual(result["claim_count"], 1)

    def test_fenced_markdown_code_does_not_create_sentence_splits(self):
        result = validate_citation_structure(
            "Refer to this snippet. [S6]\n"
            "```python\nclient.send(data)\n```\n"
            "It sends the data. [S6]"
        )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["claim_count"], 2)

    def test_unknown_source_id_is_rejected_by_source_mapping(self):
        structure = validate_citation_structure("TCP is reliable. [S99]")
        mapping = resolve_source_citations("TCP is reliable. [S99]", self.chunks)

        self.assertTrue(structure["is_valid"])
        self.assertFalse(mapping["is_valid"])
        self.assertEqual(mapping["unknown_source_ids"], ["S99"])

    def test_page_201_claim_maps_s6_to_pdf_page_201(self):
        chunks = [
            {
                "chunk_id": index,
                "page_number": 201 if index == 6 else 199 + index,
                "text": "The three-way handshake takes place within the transport "
                "layer and is completely invisible to client and server programs.",
                "distance": 0.1,
                "is_context": False,
            }
            for index in range(1, 7)
        ]
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever(chunks)
        engine.answer_generator = FakeAnswerGenerator(
            "The three-way handshake takes place within the transport layer "
            "and is completely invisible to client and server programs. [S6]"
        )
        engine.citation_verifier = FakeVerifier()

        result = engine.ask_question("Where does the handshake take place?")

        self.assertTrue(result["citation_structure_validation"]["is_valid"])
        self.assertEqual(result["source_citation_map"]["S6"], 201)
        self.assertIn("【201】", result["answer"])
        self.assertEqual(result["citation_validation"]["valid_pages"], [201])

    def test_factual_sentence_without_source_id_fails_structure(self):
        result = validate_citation_structure("TCP is reliable.")

        self.assertFalse(result["is_valid"])
        self.assertEqual(len(result["uncited_claims"]), 1)

    def test_standalone_trailing_citation_block_is_rejected(self):
        result = validate_citation_structure(
            "The client opens the connection. [S1]\n\n[S1] [S2]"
        )

        self.assertFalse(result["is_valid"])
        self.assertTrue(result["trailing_citation_block"])

    def test_structural_failure_cannot_be_reported_as_supported(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever(self.chunks)
        engine.answer_generator = FakeAnswerGenerator(
            "TCP is reliable. TCP is connection-oriented. [S1]"
        )
        engine.citation_verifier = FakeVerifier(status="SUPPORTED")

        result = engine.ask_question("What is TCP?")

        self.assertEqual(result["quality_status"], "CITATION_STRUCTURE_INVALID")
        self.assertNotEqual(result["quality_status"], "SUPPORTED")
        self.assertFalse(result["citation_structure_validation"]["is_valid"])
        self.assertIn("Citation structure validation failed", result["answer"])

    def test_semantic_error_remains_unverified_with_warning(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever(self.chunks)
        engine.answer_generator = FakeAnswerGenerator(
            "TCP is reliable. [S1]"
        )
        engine.citation_verifier = FakeVerifier(
            status="ERROR",
            error="429 rate limit",
        )

        result = engine.ask_question("What is TCP?")

        self.assertEqual(result["quality_status"], "VERIFICATION_ERROR")
        self.assertEqual(result["semantic_verification"]["status"], "ERROR")
        self.assertIn("could not be completed", result["answer"])
        self.assertIn("not been fully verified", result["answer"])

    def test_semantic_error_warning_is_preserved_with_structural_failure(self):
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever(self.chunks)
        engine.answer_generator = FakeAnswerGenerator(
            "TCP is reliable. TCP is connection-oriented. [S1]"
        )
        engine.citation_verifier = FakeVerifier(
            status="ERROR",
            error="429 rate limit",
        )

        result = engine.ask_question("What is TCP?")

        self.assertEqual(result["quality_status"], "CITATION_STRUCTURE_INVALID")
        self.assertEqual(result["semantic_verification"]["status"], "ERROR")
        self.assertIn("Citation structure validation failed", result["answer"])
        self.assertIn("not been fully verified", result["answer"])

    def test_context_chunks_and_same_page_chunks_receive_distinct_source_ids(self):
        result = resolve_source_citations("TCP [S2] UDP [S3]", self.chunks)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["answer"], "TCP \u3010193\u3011 UDP \u3010194\u3011")

    def test_engine_verifies_mapped_pdf_page_evidence_not_raw_source_ids(self):
        verifier = FakeVerifier()
        engine = RAGEngine.__new__(RAGEngine)
        engine.retriever = FakeRetriever(self.chunks)
        engine.answer_generator = FakeAnswerGenerator("TCP is reliable. [S1]")
        engine.citation_verifier = verifier

        result = engine.ask_question("What is TCP?")

        self.assertEqual(result["generated_answer"], "TCP is reliable. [S1]")
        self.assertEqual(result["answer"], "TCP is reliable. \u3010193\u3011")
        self.assertTrue(result["source_citation_validation"]["is_valid"])
        self.assertEqual(result["source_citation_map"], {"S1": 193, "S2": 193, "S3": 194})
        self.assertTrue(result["citation_validation"]["is_valid"])
        self.assertEqual(verifier.answer, "TCP is reliable. \u3010193\u3011")


if __name__ == "__main__":
    unittest.main()
