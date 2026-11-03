# Phase 6 Completion Report — RAG Quality Engineering

## Status

**Conditionally complete.** The offline quality framework, citation controls,
error classification, and audit tooling are implemented and verified. Phase 6
is not fully proven live: only T1 has a supported live outcome, T2 was
correctly rejected as semantically unsupported before this prompt tightening,
and T3–T6 remain pending. All six ground-truth evidence annotations remain
`PENDING_MANUAL_VALIDATION`; no pages or chunks were invented.

## Objectives and architecture

The project retains its Python, PyPDF, MiniLM-L6-v2, ChromaDB cosine-distance,
Groq `openai/gpt-oss-20b`, and Streamlit architecture. The evaluation flow is:

1. retrieve original vector matches, then context-expand around one anchor;
2. generate an answer from PDF-page-labelled passages;
3. validate literal citations against retrieved PDF pages;
4. semantically verify individual claims against retrieved passages;
5. classify and persist the outcome in a timestamped JSON report.

No retrieval threshold, fallback threshold, context window, embedding model,
distance metric, ChromaDB configuration, or Groq model was changed.

## Files inspected

`llm.py`, `retrieval.py`, `rag_engine.py`, `citation_validator.py`,
`citation_verifier.py`, `evaluation_config.py`, `evaluation_dataset.py`,
`evaluation_runner.py`, `eval_rag.py`, `evaluate_rag.py`, the files under
`tests/`, and all Phase 6 retrieval/evaluation diagnostic scripts were
inspected. Existing live report inspected:
`evaluation_results/rag_evaluation_20261101T155007_815215Z.json`.

## Changes made

- `llm.py`: strengthened evidence instructions. Examples, typical uses,
  advantages, disadvantages, statistics, and technical details now require
  explicit passage support; missing requested detail must be acknowledged
  rather than completed from common knowledge.
- `retrieval.py`: all original vector candidates now explicitly carry
  `is_context=False`, including when context expansion is disabled.
- `evaluation_dataset.py`: introduced clear `PENDING_MANUAL_VALIDATION` and
  `MANUALLY_VALIDATED` statuses plus a helper that allows metrics only for
  validated labels.
- `test_retrieval_metrics.py` and `compare_retrieval_strategies.py`: repaired
  obsolete `relevant_pages`/`must_abstain` accesses; made imports side-effect
  free; report page-label metrics only from original vector hits and only after
  manual evidence validation. They correctly skip now because no labels are
  validated.
- `test_dynamic_retrieval.py` and `test_dynamic_rag.py`: migrated to the
  canonical document ID, avoid database/API work on import, and safely display
  context-only (`distance=None`) chunks.
- `test_ingestion.py`: explicitly marked a legacy manual fixture and made it
  import-safe. Its historical `networking_test_001` remains intentionally
  isolated, never a supported Phase 6 evaluation document.
- `pytest.ini`: restricts pytest collection to `tests/`, preventing legacy
  manual scripts from issuing database, ingestion, or API work during tests.
- `tests/test_citation_grounding.py`: added prompt-grounding and semantic
  mixed/malformed-output coverage.
- `tests/test_retrieval_quality.py`: added fully mocked threshold, fallback,
  provenance, de-duplication, `distance=None`, and start-boundary checks.
- `tests/test_diagnostic_scripts.py`: added import-safety, canonical-ID,
  pending-label, legacy-fixture, and context-display checks.

## Bugs found and fixed

1. **T2 overreach:** page 193 supports TCP/UDP connection and reliability
   properties, but not web/file-transfer or streaming/gaming examples. Citation
   syntax was valid; semantic grounding was not. The verifier remained
   unchanged and correctly rejected the unsupported claims.
2. **Stale metrics schema:** two diagnostics used removed dataset keys and
   could raise `KeyError`.
3. **Context provenance gap:** raw non-expanded retrieval had no explicit
   original-match flag.
4. **Strategy diagnostic bug:** it mixed context chunks with `distance=None`
   into numeric comparisons, and its integer expansion-map keys did not match
   canonical string test IDs.
5. **Unsafe legacy script imports:** several selected diagnostics performed
   work at import time; the supported tools are now import-safe, and the old
   ingestion fixture is marked legacy.

## Verification

The following commands were run without live Groq requests or document
ingestion:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q .
python eval_rag.py --help
python evaluate_rag.py --help
python test_retrieval_metrics.py
python compare_retrieval_strategies.py
```

Results: **36 tests run, 36 passed, 0 failures/errors**. Compilation passed.
Both CLI help commands passed. The two metrics diagnostics exited safely with
the expected pending-label message. The JSON schema audit confirmed that
`rag_evaluation_20261101T155007_815215Z.json` has every required per-case
field. Unit tests use fakes/mocks; no Groq call was made in this completion
task.

## Methodology

Retrieval metrics are not reported as Precision@K/Recall@K until a human has
provided validated relevance labels. Once labels exist, the diagnostics compute
page-label precision/recall only from original threshold-accepted vector hits,
not context-expanded neighbors. Cosine distance is never described as a
confidence probability.

Citation validation extracts page citations and accepts them only when the
page occurs in retrieved evidence. Semantic verification is independent: it
checks each claim against the page-cited passages, and an LLM verdict remains a
quality signal rather than absolute proof.

## Live evidence

- T1 was previously reported live as citation-valid and semantically
  supported (six claims, 3.98 s).
- T2's saved report is citation-valid but `UNSUPPORTED`: four core comparison
  claims were supported, while the unsupported use-case examples were caught.
- T3–T6 have no successful recorded live evaluation yet.

## Known limitations and remaining manual work

- Prompt strengthening reduces unsupported generation but cannot guarantee an
  LLM will obey; citation validation and semantic verification remain required.
- Semantic verification is another model judgment and can fail, be
  inconclusive, or disagree with human review.
- The canonical dataset has no manually validated page/chunk labels yet, so
  retrieval quality cannot honestly be scored.
- The existing T2 report predates the tightened prompt; it is audit evidence,
  not proof of the new prompt's live behavior.
- T3–T6, and T2 after this change, require controlled live runs.

## Recommended Phase 7 preparation

Manually validate evidence annotations first. Then run one live case at a
time with cooldown and preserve every JSON report. Review claim-level failures
before changing retrieval or prompting again.

Suggested next command (a live API call):

```powershell
python eval_rag.py --test T2 --on-api-error stop
```

Use `--cooldown --cooldown-seconds 65` when running the full suite after the
single-case result is reviewed.
