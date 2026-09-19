"""
Tests for the corpus prep script.

The important one here is test_doc_id_normalization_matches_lambda. The
prep script and the ingest Lambda each normalize document ids, in
separate codebases that never import each other. If those two ever
disagree, nothing crashes: documents ingest fine, queries run fine, and
the evaluation harness reports near zero precision because the ids it
compares no longer line up. That is a painful bug to find by reading
metrics, and a trivial one to catch here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "ingest"))

from app import clean_doc_id  # noqa: E402
from prepare_scifact import estimate_chunks, normalize_doc_id, shard  # noqa: E402


def test_doc_id_normalization_matches_lambda():
    cases = [
        "4983",
        "PMC-4711.v2",
        "doc with spaces",
        "path/to/doc.txt",
        "weird:chars*here",
        "already_clean-id",
    ]
    for case in cases:
        assert normalize_doc_id(case) == clean_doc_id(case), case


def test_shard_splits_evenly():
    items = list(range(10))
    assert [len(s) for s in shard(items, 4)] == [4, 4, 2]


def test_shard_exact_multiple_has_no_trailing_empty():
    assert [len(s) for s in shard(list(range(8)), 4)] == [4, 4]


def test_shard_empty_input():
    assert list(shard([], 4)) == []


def test_estimate_chunks_short_doc_is_one_chunk():
    docs = [{"text": " ".join(["w"] * 100)}]
    assert estimate_chunks(docs) == 1


def test_estimate_chunks_matches_actual_chunker():
    # The estimate is allowed to be approximate, but it should agree with
    # the real chunker on ordinary inputs, otherwise the cost and volume
    # numbers printed before a load are misleading.
    from app import chunk_text

    for word_count in (50, 300, 301, 550, 700, 1200):
        text = " ".join(f"w{i}" for i in range(word_count))
        assert estimate_chunks([{"text": text}]) == len(chunk_text(text)), word_count
