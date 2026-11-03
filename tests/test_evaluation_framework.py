import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation_config import CANONICAL_DOCUMENT_ID
from evaluation_dataset import EVALUATION_DATASET
from evaluation_runner import EvaluationRunner


def engine_result(
    *,
    answer="TCP is reliable.【327】",
    generated_answer=None,
    sources=None,
    fallback=False,
    insufficient_evidence=False,
    citation_validation=None,
    semantic_verification=None,
    quality_status="SUPPORTED",
    error_details=None,
    abstention_reason=None,
    generation_finish_reason=None,
):
    return {
        "answer": answer,
        "generated_answer": generated_answer or answer,
        "sources": sources if sources is not None else [
            {
                "page_number": 327,
                "chunk_id": 42,
                "distance": 0.12,
                "is_context": False,
            }
        ],
        "fallback": fallback,
        "insufficient_evidence": insufficient_evidence,
        "citation_validation": citation_validation or {
            "cited_pages": [327],
            "valid_pages": [327],
            "invalid_pages": [],
            "is_valid": True,
        },
        "semantic_verification": semantic_verification or {
            "status": "SUPPORTED",
            "claims": [{"claim": "TCP is reliable.", "status": "SUPPORTED"}],
            "error": None,
        },
        "quality_status": quality_status,
        "error_details": error_details,
        "abstention_reason": abstention_reason,
        "generation_finish_reason": generation_finish_reason,
        "timings": {
            "retrieval": 0.1,
            "generation": 0.2,
            "verification": 0.3,
            "total": 0.6,
        },
    }


class FakeEngine:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.answer_generator = SimpleNamespace(model_name="openai/gpt-oss-20b")

    def ask_question(self, question):
        self.calls.append(question)
        response = self.responses.get(question, engine_result())
        if isinstance(response, Exception):
            raise response
        return response


class EvaluationFrameworkTests(unittest.TestCase):
    def make_runner(self, responses=None, **kwargs):
        self.engine = FakeEngine(responses or {})
        return EvaluationRunner(
            document_id=CANONICAL_DOCUMENT_ID,
            engine_factory=lambda document_id: self.engine,
            **kwargs,
        )

    def test_canonical_dataset_contains_exactly_the_six_phase_cases(self):
        self.assertEqual([case["id"] for case in EVALUATION_DATASET], [
            "T1", "T2", "T3", "T4", "T5", "T6"
        ])
        self.assertEqual(EVALUATION_DATASET[3]["question"],
                         "Can the third step of the TCP three-way handshake carry data?")
        self.assertEqual(EVALUATION_DATASET[4]["question"],
                         "What is TCP, and what exact bandwidth does it guarantee?")
        self.assertEqual(EVALUATION_DATASET[5]["question"],
                         "Who won FIFA World Cup 2022?")
        self.assertTrue(all(
            case["expected_evidence"]["annotation_status"] == "PENDING_MANUAL_VALIDATION"
            for case in EVALUATION_DATASET
        ))

    def test_runs_a_single_selected_case(self):
        runner = self.make_runner()

        report = runner.run(test_id="T2")

        self.assertEqual([item["test_id"] for item in report["results"]], ["T2"])
        self.assertEqual(self.engine.calls, [EVALUATION_DATASET[1]["question"]])
        self.assertEqual(report["summary"]["total_cases"], 1)

    def test_runs_all_six_cases_sequentially(self):
        runner = self.make_runner()

        report = runner.run()

        self.assertEqual(len(report["results"]), 6)
        self.assertEqual(report["summary"]["total_cases"], 6)
        self.assertEqual(len(self.engine.calls), 6)

    def test_optional_cooldown_runs_only_between_live_cases(self):
        delays = []
        runner = self.make_runner(sleep_fn=delays.append)

        runner.run(cooldown_enabled=True, cooldown_seconds=65)

        self.assertEqual(delays, [65, 65, 65, 65, 65])

    def test_records_valid_citation_and_semantic_status_independently(self):
        runner = self.make_runner()

        result = runner.run(test_id="T1")["results"][0]

        self.assertTrue(result["citation_validation_status"])
        self.assertEqual(result["semantic_verification_status"], "SUPPORTED")
        self.assertEqual(result["final_evaluation_outcome"], "SUPPORTED")

    def test_invalid_citation_overrides_supported_verifier_status(self):
        invalid = engine_result(citation_validation={
            "cited_pages": [7],
            "valid_pages": [],
            "invalid_pages": [7],
            "is_valid": False,
        })
        runner = self.make_runner({EVALUATION_DATASET[0]["question"]: invalid})

        result = runner.run(test_id="T1")["results"][0]

        self.assertEqual(result["final_evaluation_outcome"], "CITATION_INVALID")
        self.assertEqual(result["invalid_citation_pages"], [7])
        self.assertEqual(result["semantic_verification_status"], "SUPPORTED")

    def test_context_only_chunk_with_no_distance_is_safe_and_partitioned(self):
        result = engine_result(sources=[
            {"page_number": 327, "chunk_id": 42, "distance": 0.12, "is_context": False},
            {"page_number": 328, "chunk_id": 43, "distance": None, "is_context": True},
        ])
        runner = self.make_runner({EVALUATION_DATASET[0]["question"]: result})

        record = runner.run(test_id="T1")["results"][0]

        self.assertEqual(record["original_matching_chunks"][0]["chunk_id"], 42)
        self.assertEqual(record["context_expanded_chunks"][0]["chunk_id"], 43)
        self.assertIsNone(record["context_expanded_chunks"][0]["distance"])
        self.assertEqual(record["retrieved_pdf_pages"], [327, 328])

    def test_empty_retrieval_is_classified_as_insufficient_evidence(self):
        empty = engine_result(
            answer="No evidence.",
            sources=[],
            insufficient_evidence=True,
            citation_validation=None,
            semantic_verification=None,
            quality_status="INSUFFICIENT_EVIDENCE",
            abstention_reason="NO_RETRIEVED_EVIDENCE",
        )
        runner = self.make_runner({EVALUATION_DATASET[5]["question"]: empty})

        record = runner.run(test_id="T6")["results"][0]

        self.assertEqual(record["final_evaluation_outcome"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(record["abstention_reason"], "NO_RETRIEVED_EVIDENCE")

    def test_model_abstention_is_distinguished_from_empty_retrieval(self):
        abstention = engine_result(
            answer="I could not answer reliably.",
            generated_answer="INSUFFICIENT_EVIDENCE: insufficient support",
            insufficient_evidence=True,
            citation_validation=None,
            semantic_verification=None,
            quality_status="INSUFFICIENT_EVIDENCE",
            abstention_reason="MODEL_ABSTAINED",
        )
        runner = self.make_runner({EVALUATION_DATASET[4]["question"]: abstention})

        record = runner.run(test_id="T5")["results"][0]

        self.assertEqual(record["final_evaluation_outcome"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(record["abstention_reason"], "MODEL_ABSTAINED")

    def test_api_failure_and_fallback_are_not_counted_as_supported(self):
        failure = engine_result(
            answer="Retrieved fallback passages.",
            generated_answer="AI_SERVICE_UNAVAILABLE: service failed",
            fallback=True,
            citation_validation=None,
            semantic_verification=None,
            quality_status="API_ERROR",
            error_details={"type": "API_ERROR", "message": "service failed"},
        )
        runner = self.make_runner({EVALUATION_DATASET[0]["question"]: failure})

        record = runner.run(test_id="T1")["results"][0]

        self.assertTrue(record["fallback"])
        self.assertEqual(record["final_evaluation_outcome"], "API_ERROR")
        self.assertNotEqual(record["final_evaluation_outcome"], "SUPPORTED")

    def test_rate_limit_is_preserved_as_a_distinct_api_error_detail(self):
        failure = engine_result(
            fallback=True,
            citation_validation=None,
            semantic_verification=None,
            quality_status="API_ERROR",
            error_details={"type": "RATE_LIMIT", "message": "429 Too Many Requests"},
        )
        runner = self.make_runner({EVALUATION_DATASET[0]["question"]: failure})

        record = runner.run(test_id="T1")["results"][0]

        self.assertEqual(record["final_evaluation_outcome"], "API_ERROR")
        self.assertEqual(record["error_details"]["type"], "RATE_LIMIT")

    def test_truncated_generation_is_not_an_api_connectivity_success_and_preserves_finish_reason(self):
        truncated = engine_result(
            answer="Retrieved fallback passages.",
            generated_answer="AI_GENERATION_TRUNCATED: completion reached its output limit.",
            fallback=True,
            citation_validation=None,
            semantic_verification=None,
            quality_status="GENERATION_TRUNCATED",
            error_details={"type": "TRUNCATED_RESPONSE", "message": "length", "finish_reason": "length"},
            generation_finish_reason="length",
        )
        runner = self.make_runner({EVALUATION_DATASET[1]["question"]: truncated})

        record = runner.run(test_id="T2")["results"][0]

        self.assertEqual(record["final_evaluation_outcome"], "GENERATION_TRUNCATED")
        self.assertEqual(record["generation_finish_reason"], "length")
        self.assertNotEqual(record["final_evaluation_outcome"], "SUPPORTED")

    def test_saved_json_report_preserves_truncation_metadata(self):
        truncated = engine_result(
            fallback=True,
            citation_validation=None,
            semantic_verification=None,
            error_details={"type": "TRUNCATED_RESPONSE", "message": "length", "finish_reason": "length"},
            generation_finish_reason="length",
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_runner(
                {EVALUATION_DATASET[1]["question"]: truncated},
                results_directory=Path(directory),
            )
            report = runner.run(test_id="T2", save=True)
            saved = json.loads(Path(report["result_path"]).read_text(encoding="utf-8"))

        record = saved["results"][0]
        self.assertEqual(record["generation_finish_reason"], "length")
        self.assertEqual(record["error_details"]["finish_reason"], "length")
        self.assertEqual(record["final_evaluation_outcome"], "GENERATION_TRUNCATED")

    def test_stop_policy_saves_partial_results_after_an_engine_error(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_runner(
                {EVALUATION_DATASET[0]["question"]: RuntimeError("429 rate limit")},
                results_directory=Path(directory),
            )

            report = runner.run(on_api_error="stop", save=True)

            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(report["results"][0]["final_evaluation_outcome"], "API_ERROR")
            self.assertEqual(report["results"][0]["error_details"]["type"], "RATE_LIMIT")
            self.assertTrue(Path(report["result_path"]).is_file())

    def test_semantic_verification_failure_is_distinguished(self):
        failure = engine_result(
            semantic_verification={"status": "ERROR", "claims": [], "error": "schema failure"},
            quality_status="VERIFICATION_ERROR",
            error_details={"type": "VERIFICATION_ERROR", "message": "schema failure"},
        )
        runner = self.make_runner({EVALUATION_DATASET[0]["question"]: failure})

        record = runner.run(test_id="T1")["results"][0]

        self.assertEqual(record["final_evaluation_outcome"], "VERIFICATION_ERROR")
        self.assertEqual(record["error_details"]["message"], "schema failure")

    def test_json_report_is_saved_without_overwriting_previous_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.make_runner(results_directory=Path(directory))

            first = runner.run(test_id="T1", save=True)
            second = runner.run(test_id="T1", save=True)

            first_path = Path(first["result_path"])
            second_path = Path(second["result_path"])
            self.assertTrue(first_path.is_file())
            self.assertTrue(second_path.is_file())
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(json.loads(first_path.read_text(encoding="utf-8"))["document_id"],
                             CANONICAL_DOCUMENT_ID)

    def test_summary_counts_outcomes_correctly(self):
        responses = {
            EVALUATION_DATASET[0]["question"]: engine_result(),
            EVALUATION_DATASET[1]["question"]: engine_result(citation_validation={
                "cited_pages": [9], "valid_pages": [], "invalid_pages": [9], "is_valid": False
            }),
            EVALUATION_DATASET[2]["question"]: engine_result(
                semantic_verification={"status": "ERROR", "claims": [], "error": "bad JSON"},
                quality_status="VERIFICATION_ERROR",
            ),
            EVALUATION_DATASET[3]["question"]: engine_result(
                fallback=True,
                citation_validation=None,
                semantic_verification=None,
                quality_status="API_ERROR",
                error_details={"type": "RATE_LIMIT", "message": "429"},
            ),
            EVALUATION_DATASET[4]["question"]: engine_result(
                insufficient_evidence=True,
                citation_validation=None,
                semantic_verification=None,
                quality_status="INSUFFICIENT_EVIDENCE",
                abstention_reason="MODEL_ABSTAINED",
            ),
            EVALUATION_DATASET[5]["question"]: engine_result(
                semantic_verification={"status": "INCONCLUSIVE", "claims": [], "error": None},
                quality_status="INCONCLUSIVE",
            ),
        }
        runner = self.make_runner(responses)

        summary = runner.run()["summary"]

        self.assertEqual(summary["outcome_counts"], {
            "API_ERROR": 1,
            "CITATION_INVALID": 1,
            "INCONCLUSIVE": 1,
            "INSUFFICIENT_EVIDENCE": 1,
            "SUPPORTED": 1,
            "VERIFICATION_ERROR": 1,
        })


if __name__ == "__main__":
    unittest.main()
