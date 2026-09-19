"""
Unit tests for the pure logic in the ingest function: chunking, document
parsing, key naming, and batch boundaries.

None of these touch AWS. They run in under a second without credentials,
which is what makes them suitable for CI on every push. The parts that do
talk to AWS are covered by actually running a load and checking the DLQ
depth and index count, not by mocking every boto3 call, which mostly
tests that the mock was written correctly.
"""

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "ingest"))

from conftest import load_lambda_module  # noqa: E402

# Loaded by path rather than with `import app`, because the query
# handler is also called app.py and the two collide in a shared test
# session. See load_lambda_module in conftest.py.
app = load_lambda_module("ingest_app", "src", "ingest", "app.py")

build_vectors = app.build_vectors
chunk_text = app.chunk_text
clean_doc_id = app.clean_doc_id
parse_documents = app.parse_documents
write_vectors = app.write_vectors


# --- chunking -----------------------------------------------------------

def test_chunk_text_splits_long_input():
    text = " ".join(f"word{i}" for i in range(700))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) == 3
    # Overlap means consecutive chunks share words rather than cutting
    # clean at 300 and 600.
    assert chunks[0].split()[-1] == chunks[1].split()[49]


def test_chunk_text_short_input_returns_one_chunk():
    assert len(chunk_text("just a few words here", chunk_size=300, overlap=50)) == 1


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


# --- document ids -------------------------------------------------------

def test_clean_doc_id_strips_extension_and_path():
    assert clean_doc_id("uploads/my paper (final).pdf") == "my_paper__final_"


def test_clean_doc_id_is_deterministic():
    # Same input, same output, every time. This is what makes vector keys
    # built from doc_id safe to write more than once.
    assert clean_doc_id("uploads/doc.txt") == clean_doc_id("uploads/doc.txt")


# --- parsing ------------------------------------------------------------

def test_parse_documents_plain_text_is_one_document():
    docs = parse_documents("notes.txt", "some plain text")
    assert docs == [("notes", "some plain text")]


def test_parse_documents_jsonl_yields_many():
    raw = "\n".join([
        json.dumps({"doc_id": "d1", "title": "First", "text": "body one"}),
        json.dumps({"doc_id": "d2", "title": "Second", "text": "body two"}),
    ])
    docs = parse_documents("corpus/shard_000.jsonl", raw)
    assert len(docs) == 2
    assert docs[0][0] == "d1"
    # Title is prepended to the body before chunking.
    assert docs[0][1].startswith("First")
    assert "body one" in docs[0][1]


def test_parse_documents_jsonl_skips_blank_lines():
    raw = json.dumps({"doc_id": "d1", "text": "only one"}) + "\n\n   \n"
    assert len(parse_documents("s.jsonl", raw)) == 1


def test_parse_documents_jsonl_skips_empty_text():
    raw = "\n".join([
        json.dumps({"doc_id": "d1", "text": "kept"}),
        json.dumps({"doc_id": "d2", "text": "   "}),
    ])
    assert [d[0] for d in parse_documents("s.jsonl", raw)] == ["d1"]


def test_parse_documents_malformed_json_raises():
    # Failing loudly is deliberate. A partially imported corpus still
    # answers questions, just wrongly, and the eval numbers look fine,
    # so a silent skip here would be the worst outcome.
    try:
        parse_documents("s.jsonl", "{not valid json")
    except ValueError as exc:
        assert "line 1" in str(exc)
    else:
        raise AssertionError("expected ValueError on malformed JSON")


def test_parse_documents_handles_missing_title():
    raw = json.dumps({"doc_id": "d1", "text": "body only"})
    docs = parse_documents("s.jsonl", raw)
    assert docs == [("d1", "body only")]


# --- vector construction and batching -----------------------------------

def test_build_vectors_uses_deterministic_zero_padded_keys():
    long_text = " ".join(f"word{i}" for i in range(700))
    with patch.object(app, "embed_text", return_value=[0.0] * 1024):
        vectors = build_vectors("d1", long_text, "corpus/shard_000.jsonl")
    assert [v["key"] for v in vectors] == [
        "d1#chunk_000", "d1#chunk_001", "d1#chunk_002",
    ]
    assert vectors[0]["metadata"]["doc_id"] == "d1"
    assert vectors[0]["metadata"]["source_key"] == "corpus/shard_000.jsonl"


def test_write_vectors_splits_into_batches_within_api_limit():
    # 450 vectors at a batch size of 200 should be three requests of
    # 200, 200, 50 - never a single request over the API's 500 ceiling.
    vectors = [{"key": f"k{i}"} for i in range(450)]
    calls = []

    def fake_put(**kwargs):
        calls.append(len(kwargs["vectors"]))

    with patch.object(app.s3vectors, "put_vectors", side_effect=fake_put):
        written = write_vectors(vectors)

    assert calls == [200, 200, 50]
    assert written == 450


def test_write_vectors_empty_input_makes_no_calls():
    calls = []
    with patch.object(app.s3vectors, "put_vectors", side_effect=lambda **k: calls.append(1)):
        assert write_vectors([]) == 0
    assert calls == []


def test_write_vectors_exact_batch_multiple():
    vectors = [{"key": f"k{i}"} for i in range(400)]
    calls = []
    with patch.object(app.s3vectors, "put_vectors", side_effect=lambda **k: calls.append(len(k["vectors"]))):
        write_vectors(vectors)
    # Exactly two full batches, no trailing empty request.
    assert calls == [200, 200]
