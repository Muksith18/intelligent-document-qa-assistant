import importlib
import sys
import types
import unittest
from unittest.mock import patch

from evaluation_config import CANONICAL_DOCUMENT_ID
from evaluation_dataset import EVIDENCE_PENDING, EVIDENCE_VALIDATED


class DiagnosticScriptTests(unittest.TestCase):
    def import_without_runtime_dependencies(self, module_name):
        retrieval = types.ModuleType("retrieval")
        retrieval.Retriever = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostics must not create a retriever at import time")
        )
        rag_engine = types.ModuleType("rag_engine")
        rag_engine.RAGEngine = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostics must not create an engine at import time")
        )
        ingestion = types.ModuleType("ingestion")
        ingestion.DocumentIngestion = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostics must not ingest a document at import time")
        )
        with patch.dict(sys.modules, {
            "retrieval": retrieval,
            "rag_engine": rag_engine,
            "ingestion": ingestion,
        }):
            sys.modules.pop(module_name, None)
            module = importlib.import_module(module_name)
        sys.modules.pop(module_name, None)
        return module

    def test_supported_diagnostics_import_without_database_or_api_work(self):
        for module_name in (
            "test_retrieval_metrics",
            "compare_retrieval_strategies",
            "test_dynamic_retrieval",
            "test_dynamic_rag",
        ):
            with self.subTest(module=module_name):
                module = self.import_without_runtime_dependencies(module_name)
                self.assertEqual(module.DOCUMENT_ID, CANONICAL_DOCUMENT_ID)
                self.assertTrue(callable(module.main))

    def test_metrics_diagnostics_only_accept_manually_validated_labels(self):
        module = self.import_without_runtime_dependencies("test_retrieval_metrics")
        dataset = [
            {
                "id": "validated",
                "expected_evidence": {
                    "annotation_status": EVIDENCE_VALIDATED,
                    "required_pdf_pages": [193],
                },
            },
            {
                "id": "pending",
                "expected_evidence": {
                    "annotation_status": EVIDENCE_PENDING,
                    "required_pdf_pages": [],
                },
            },
        ]

        self.assertEqual(
            [case["id"] for case in module.manually_validated_cases(dataset)],
            ["validated"],
        )

    def test_context_only_distances_are_safe_in_interactive_diagnostics(self):
        module = self.import_without_runtime_dependencies("test_dynamic_retrieval")

        self.assertEqual(module.format_distance(None), "context")
        self.assertEqual(module.format_distance(0.25), "0.2500")

    def test_legacy_ingestion_script_is_safe_to_import_and_marked_legacy(self):
        module = self.import_without_runtime_dependencies("test_ingestion")

        self.assertTrue(module.LEGACY_DIAGNOSTIC)
        self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
