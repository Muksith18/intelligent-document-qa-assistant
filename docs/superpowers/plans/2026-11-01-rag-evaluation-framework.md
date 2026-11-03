# RAG Evaluation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one safe, persistent, mock-testable Phase 6 evaluation workflow around the existing RAG engine.

**Architecture:** A small configuration module owns the active document ID. A canonical six-case dataset stores only behavioral expectations and marks evidence annotation as pending. A dependency-injected runner normalizes existing engine results into auditable JSON; it does not call or reconfigure retrieval itself.

**Tech Stack:** Python standard library (`argparse`, `json`, `pathlib`, `unittest`) and the existing RAG engine.

**Spec:** User-provided “PHASE 6 — TASK 1: Build a Reliable RAG Evaluation Framework”.

## Global Constraints

- Do not change retrieval thresholds, context-expansion logic, embeddings, ChromaDB configuration, Groq model, or citation-grounding behavior.
- Do not run live Groq evaluation; all automated tests must use fakes.
- Do not invent evidence page numbers or chunk IDs.
- Save one new JSON result file per evaluation run.

## Review Focus

- A rate-limit fallback must be recorded as `API_ERROR`, never as a supported answer.
- Context-only chunks with `distance=None` must be persisted and summarized safely.
- Citation-valid and semantically-supported outcomes must remain independent.
- Partial results must be written if a later case fails or a stop policy is selected.
- Canonical dataset questions must exactly match T1–T6 and have no fabricated evidence labels.

---

### Task 1: Canonical configuration and dataset

**Files:**
- Create: `evaluation_config.py`
- Modify: `evaluation_dataset.py`
- Test: `tests/test_evaluation_framework.py`

**Interfaces:**
- Produces `CANONICAL_DOCUMENT_ID`, `DEFAULT_COOLDOWN_SECONDS`, `EVALUATION_RESULTS_DIRECTORY`, and six `EVALUATION_DATASET` mappings.

- [ ] Write tests asserting the ID and the six exact IDs/questions.
- [ ] Implement the configuration and dataset with pending-manual-review evidence markers.
- [ ] Run the isolated unit test module.

### Task 2: Persisted evaluation runner

**Files:**
- Create: `evaluation_runner.py`
- Modify: `eval_rag.py`, `evaluate_rag.py`, `.gitignore`
- Test: `tests/test_evaluation_framework.py`

**Interfaces:**
- Consumes an engine factory with `ask_question(question) -> dict`.
- Produces `EvaluationRunner.run(...) -> dict` and a unique JSON report path.

- [ ] Write failing tests for one-case/all-case execution, source partitioning, outcome classification, cooldown, persistence, and summary counts.
- [ ] Implement result normalization, safe sequential orchestration, optional cooldown, partial persistence, and CLI forwarding.
- [ ] Run isolated unit tests.

### Task 3: Production result metadata and retrieval diagnostics

**Files:**
- Modify: `llm.py`, `rag_engine.py`, `test_retrieval_metrics.py`
- Test: `tests/test_evaluation_framework.py`, `tests/test_citation_grounding.py`

**Interfaces:**
- Extends existing result dictionaries with `error_details`, `abstention_reason`, `outcome`, and `is_context` source metadata.

- [ ] Write failing fake-engine tests for no retrieval, model abstention, generic API failure, rate-limit fallback, semantic-verifier error, and citation invalidity.
- [ ] Implement only metadata/error classification needed to distinguish outcomes; retain existing visible responses.
- [ ] Make retrieval metrics use canonical configuration and original matching chunks only (`expand_context=False`).
- [ ] Run both isolated test modules and compile checks.
