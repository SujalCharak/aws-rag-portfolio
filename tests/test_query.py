"""
Tests for the query side pure logic, plus an import smoke test.

The smoke test earns its place: retrieval and generation were pulled out
of app.py into their own modules so the evaluation harness could import
the same code the Lambda runs. That refactor is exactly the kind that
passes every unit test and then fails at cold start on a bad import
path, because nothing else in the suite actually imports the handler.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "query"))

from conftest import load_lambda_module  # noqa: E402
from generation import build_context  # noqa: E402
from retrieval import aggregate_to_documents  # noqa: E402


def _hit(doc_id, distance, chunk_index=0, chunk_text="text"):
    return {
        "key": f"{doc_id}#chunk_{chunk_index:03d}",
        "distance": distance,
        "metadata": {
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
        },
    }


def test_handler_module_imports_cleanly():
    """
    Catches a broken import chain (app -> retrieval/generation ->
    clients) that would otherwise only show up as a Lambda cold start
    failure after deploying.

    Loaded by path rather than with `import app`: the ingest handler is
    also called app.py, and a plain import here returns whichever of the
    two was imported first in the session. Since both export `handler`,
    this assertion would have passed against the wrong module.
    """
    app = load_lambda_module("query_app", "src", "query", "app.py")

    assert callable(app.handler)
    # Specific to the query handler, so the test fails loudly rather
    # than passing against the ingest module if the loading breaks.
    assert app.GENERATION_MODEL_ID


def test_documents_rank_by_their_best_chunk_not_their_chunk_count():
    """
    A three chunk document should not outrank a one chunk document that
    matched better. Summing chunk scores instead of taking the best is
    the usual way this goes wrong, and it quietly biases retrieval
    toward long documents.
    """
    hits = [
        _hit("long", 0.40, chunk_index=0),
        _hit("long", 0.45, chunk_index=1),
        _hit("long", 0.50, chunk_index=2),
        _hit("short", 0.30),
    ]

    assert aggregate_to_documents(hits, k=10) == ["short", "long"]


def test_best_chunk_wins_even_when_it_appears_late():
    """
    The minimum has to be taken across every occurrence of a document,
    not the first one encountered.
    """
    hits = [
        _hit("a", 0.90, chunk_index=0),
        _hit("b", 0.20),
        _hit("a", 0.10, chunk_index=1),
    ]

    assert aggregate_to_documents(hits, k=10) == ["a", "b"]


def test_documents_are_deduplicated():
    hits = [_hit("a", 0.1, chunk_index=i) for i in range(5)]

    assert aggregate_to_documents(hits, k=10) == ["a"]


def test_result_is_truncated_to_k_documents():
    hits = [_hit(f"doc{i}", i / 100) for i in range(20)]

    result = aggregate_to_documents(hits, k=3)

    assert result == ["doc0", "doc1", "doc2"]


def test_hits_missing_doc_id_or_distance_are_skipped():
    """
    Defensive rather than theoretical: a vector written before the
    metadata schema settled would have no doc_id, and letting that raise
    a KeyError would kill a 300 query evaluation run partway through.
    """
    hits = [
        {"distance": 0.1, "metadata": {}},
        {"metadata": {"doc_id": "no_distance"}},
        _hit("good", 0.2),
    ]

    assert aggregate_to_documents(hits, k=10) == ["good"]


def test_empty_hits_produce_empty_ranking():
    assert aggregate_to_documents([], k=10) == []


def test_context_is_numbered_for_attribution():
    """
    The numbering gives the model a way to attribute claims to
    passages, and gives the groundedness judge the same labels to check
    those attributions against.
    """
    hits = [_hit("a", 0.1, chunk_text="first"), _hit("b", 0.2, chunk_text="second")]

    context = build_context(hits)

    assert context == "[1] first\n\n[2] second"
