"""
Unit tests for the pure logic in the ingest function: chunking and key
naming. These don't touch AWS at all, which is the point, they run in
under a second and don't need credentials, so they're what a CI pipeline
runs on every push. The AWS-touching parts (embedding, writing vectors)
get tested separately, later, against a real deployed stack.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "ingest"))

from app import chunk_text, clean_doc_id  # noqa: E402


def test_chunk_text_splits_long_input():
    text = " ".join(f"word{i}" for i in range(700))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) == 3
    # Overlap means chunk boundaries share words, not cut clean at 300/600.
    assert chunks[0].split()[-1] == chunks[1].split()[49]


def test_chunk_text_short_input_returns_one_chunk():
    chunks = chunk_text("just a few words here", chunk_size=300, overlap=50)
    assert len(chunks) == 1


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_clean_doc_id_strips_extension_and_path():
    assert clean_doc_id("uploads/my paper (final).pdf") == "my_paper__final_"


def test_clean_doc_id_is_deterministic():
    # Same input, same output, every time. This is what makes the vector
    # keys built from doc_id safe to write more than once (see app.py's
    # docstring on idempotency).
    assert clean_doc_id("uploads/doc.txt") == clean_doc_id("uploads/doc.txt")
