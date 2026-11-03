import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evaluation_config import (
    CANONICAL_DOCUMENT_ID,
    DEFAULT_COOLDOWN_SECONDS,
    EVALUATION_RESULTS_DIRECTORY,
)
from evaluation_dataset import EVALUATION_DATASET


OUTCOME_ORDER = (
    "SUPPORTED",
    "UNSUPPORTED",
    "INCONCLUSIVE",
    "INSUFFICIENT_EVIDENCE",
    "CITATION_INVALID",
    "API_ERROR",
    "VERIFICATION_ERROR",
    "GENERATION_ERROR",
    "GENERATION_TRUNCATED",
    "NOT_CHECKED",
)


class EvaluationRunner:
    def __init__(
        self,
        document_id=CANONICAL_DOCUMENT_ID,
        dataset=EVALUATION_DATASET,
        engine_factory=None,
        results_directory=EVALUATION_RESULTS_DIRECTORY,
        sleep_fn=time.sleep,
    ):
        self.document_id = document_id
        self.dataset = dataset
        self.engine_factory = engine_factory
        self.results_directory = Path(results_directory)
        self.sleep_fn = sleep_fn

    def select_cases(self, test_id=None):
        if test_id is None:
            return list(self.dataset)

        selected = [case for case in self.dataset if case["id"] == test_id]
        if not selected:
            raise ValueError(f"Unknown evaluation test ID: {test_id}")
        return selected

    @staticmethod
    def _timestamp():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _is_rate_limit(error):
        text = str(error).lower()
        status_code = getattr(error, "status_code", None)
        return status_code == 429 or "429" in text or "rate limit" in text

    def _exception_record(self, case, error, model_name):
        error_type = "RATE_LIMIT" if self._is_rate_limit(error) else "API_ERROR"
        return {
            "test_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "document_id": self.document_id,
            "model_name": model_name,
            "timestamp": self._timestamp(),
            "generated_answer": None,
            "displayed_answer": None,
            "fallback": False,
            "insufficient_evidence": False,
            "abstention_reason": None,
            "generation_finish_reason": None,
            "retrieved_chunk_ids": [],
            "retrieved_pdf_pages": [],
            "original_matching_chunks": [],
            "context_expanded_chunks": [],
            "cited_pages": [],
            "valid_citation_pages": [],
            "invalid_citation_pages": [],
            "citation_validation_status": None,
            "source_citation_validation_status": None,
            "source_citation_occurrences": [],
            "cited_source_ids": [],
            "valid_source_ids": [],
            "unknown_source_ids": [],
            "source_citation_map": {},
            "semantic_verification_status": "NOT_CHECKED",
            "claim_verification_results": [],
            "retrieval_time": 0,
            "generation_time": 0,
            "verification_time": 0,
            "total_processing_time": 0,
            "error_details": {"type": error_type, "message": str(error)},
            "final_evaluation_outcome": "API_ERROR",
        }

    @staticmethod
    def _classify(
        result,
        citation_validation,
        semantic_verification,
        source_citation_validation=None,
    ):
        error_type = (result.get("error_details") or {}).get("type")
        if error_type == "TRUNCATED_RESPONSE":
            return "GENERATION_TRUNCATED"
        if error_type in {"EMPTY_RESPONSE", "INCOMPLETE_RESPONSE"}:
            return "GENERATION_ERROR"
        if result.get("fallback") or error_type in {"API_ERROR", "RATE_LIMIT"}:
            return "API_ERROR"
        if result.get("insufficient_evidence"):
            return "INSUFFICIENT_EVIDENCE"
        if (
            source_citation_validation
            and not source_citation_validation.get("is_valid", False)
        ):
            return "CITATION_INVALID"
        if citation_validation and not citation_validation.get("is_valid", False):
            return "CITATION_INVALID"

        semantic_status = (semantic_verification or {}).get("status")
        if semantic_status == "ERROR":
            return "VERIFICATION_ERROR"
        if semantic_status in {"SUPPORTED", "UNSUPPORTED", "INCONCLUSIVE"}:
            return semantic_status
        return "NOT_CHECKED"

    def _result_record(self, case, result, model_name):
        sources = result.get("sources") or []
        citation_validation = result.get("citation_validation")
        source_citation_validation = result.get("source_citation_validation")
        semantic_verification = result.get("semantic_verification")
        timings = result.get("timings") or {}

        original_sources = [
            source for source in sources if not source.get("is_context", False)
        ]
        context_sources = [
            source for source in sources if source.get("is_context", False)
        ]
        error_details = result.get("error_details")
        if not error_details and (semantic_verification or {}).get("status") == "ERROR":
            error_details = {
                "type": "VERIFICATION_ERROR",
                "message": semantic_verification.get("error"),
            }

        return {
            "test_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "document_id": self.document_id,
            "model_name": model_name,
            "timestamp": self._timestamp(),
            "generated_answer": result.get("generated_answer"),
            "displayed_answer": result.get("answer"),
            "fallback": bool(result.get("fallback")),
            "insufficient_evidence": bool(result.get("insufficient_evidence")),
            "abstention_reason": result.get("abstention_reason"),
            "generation_finish_reason": result.get("generation_finish_reason"),
            "retrieved_chunk_ids": [source.get("chunk_id") for source in sources],
            "retrieved_pdf_pages": sorted({source.get("page_number") for source in sources}),
            "original_matching_chunks": [dict(source) for source in original_sources],
            "context_expanded_chunks": [dict(source) for source in context_sources],
            "cited_pages": (citation_validation or {}).get("cited_pages", []),
            "valid_citation_pages": (citation_validation or {}).get("valid_pages", []),
            "invalid_citation_pages": (citation_validation or {}).get("invalid_pages", []),
            "cited_source_ids": (source_citation_validation or {}).get("cited_source_ids", []),
            "source_citation_occurrences": (source_citation_validation or {}).get(
                "source_citation_occurrences", []
            ),
            "valid_source_ids": (source_citation_validation or {}).get("valid_source_ids", []),
            "unknown_source_ids": (source_citation_validation or {}).get("unknown_source_ids", []),
            "source_citation_map": result.get("source_citation_map", {}),
            "citation_validation_status": (
                (citation_validation or {}).get("is_valid")
                if citation_validation is not None else None
            ),
            "source_citation_validation_status": (
                (source_citation_validation or {}).get("is_valid")
                if source_citation_validation is not None else None
            ),
            "semantic_verification_status": (
                (semantic_verification or {}).get("status", "NOT_CHECKED")
            ),
            "claim_verification_results": (semantic_verification or {}).get("claims", []),
            "retrieval_time": timings.get("retrieval", 0),
            "generation_time": timings.get("generation", 0),
            "verification_time": timings.get("verification", 0),
            "total_processing_time": timings.get("total", 0),
            "error_details": error_details,
            "final_evaluation_outcome": self._classify(
                result,
                citation_validation,
                semantic_verification,
                source_citation_validation,
            ),
        }

    @staticmethod
    def _summary(records):
        counts = Counter(record["final_evaluation_outcome"] for record in records)
        return {
            "total_cases": len(records),
            "outcome_counts": dict(sorted(counts.items())),
            "average_total_processing_time": round(
                sum(record["total_processing_time"] for record in records) / len(records), 2
            ) if records else 0,
        }

    def _save(self, report):
        self.results_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        path = self.results_directory / f"rag_evaluation_{timestamp}.json"
        suffix = 1
        while path.exists():
            path = self.results_directory / f"rag_evaluation_{timestamp}_{suffix}.json"
            suffix += 1
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def run(
        self,
        test_id=None,
        cooldown_enabled=False,
        cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
        on_api_error="continue",
        save=False,
    ):
        if on_api_error not in {"continue", "stop"}:
            raise ValueError("on_api_error must be 'continue' or 'stop'")

        cases = self.select_cases(test_id)
        if self.engine_factory is None:
            from rag_engine import RAGEngine
            engine = RAGEngine(self.document_id)
        else:
            engine = self.engine_factory(self.document_id)
        model_name = getattr(getattr(engine, "answer_generator", None), "model_name", "unknown")
        records = []

        for index, case in enumerate(cases):
            try:
                result = engine.ask_question(case["question"])
                record = self._result_record(case, result, model_name)
            except Exception as error:
                record = self._exception_record(case, error, model_name)

            records.append(record)

            if record["final_evaluation_outcome"] == "API_ERROR" and on_api_error == "stop":
                break
            if cooldown_enabled and index < len(cases) - 1:
                self.sleep_fn(cooldown_seconds)

        report = {
            "timestamp": self._timestamp(),
            "document_id": self.document_id,
            "model_name": model_name,
            "results": records,
            "summary": self._summary(records),
        }
        if save:
            report["result_path"] = str(self._save(report))
        return report


def format_terminal_summary(report):
    lines = [
        "\n========== RAG EVALUATION SUMMARY ==========",
        f"Document ID: {report['document_id']}",
        f"Model: {report['model_name']}",
    ]
    for record in report["results"]:
        lines.append(
            f"{record['test_id']} | {record['category']} | "
            f"{record['final_evaluation_outcome']} | "
            f"{record['total_processing_time']}s"
        )
    lines.append(f"Outcome counts: {report['summary']['outcome_counts']}")
    if report.get("result_path"):
        lines.append(f"Saved: {report['result_path']}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Phase 6 RAG evaluation suite.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--test", help="Run one case, for example T2.")
    selection.add_argument("--all", action="store_true", help="Run all six cases (default).")
    parser.add_argument("--cooldown", action="store_true", help="Wait between live evaluations.")
    parser.add_argument("--cooldown-seconds", type=int, default=DEFAULT_COOLDOWN_SECONDS)
    parser.add_argument("--on-api-error", choices=("continue", "stop"), default="continue")
    parser.add_argument("--results-directory", type=Path, default=EVALUATION_RESULTS_DIRECTORY)
    args = parser.parse_args(argv)

    runner = EvaluationRunner(results_directory=args.results_directory)
    report = runner.run(
        test_id=args.test,
        cooldown_enabled=args.cooldown,
        cooldown_seconds=args.cooldown_seconds,
        on_api_error=args.on_api_error,
        save=True,
    )
    print(format_terminal_summary(report))


if __name__ == "__main__":
    main()
