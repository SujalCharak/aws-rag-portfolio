"""
Tests for Reciprocal Rank Fusion.

Pure arithmetic over rank positions, so it is worth pinning down
precisely. The interesting property is the one in
test_agreement_beats_a_single_strong_opinion: fusion is supposed to
reward documents both systems liked over documents only one system
ranked first. If that inverts, the hybrid run would score well for the
wrong reason and nothing else in the pipeline would notice.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from fuse_runs import reciprocal_rank_fusion  # noqa: E402


def test_single_run_preserves_its_own_order():
    run = {"q1": ["a", "b", "c"]}

    fused = reciprocal_rank_fusion([run])

    assert fused["q1"] == ["a", "b", "c"]


def test_agreement_beats_a_single_strong_opinion():
    """
    'a' is second in both runs. 'x' and 'b' are each first in one run
    and absent from the other. Two second places should outrank one
    first place, which is the property that makes fusion useful rather
    than just an average of opinions.
    """
    run_dense = {"q1": ["x", "a"]}
    run_lexical = {"q1": ["b", "a"]}

    fused = reciprocal_rank_fusion([run_dense, run_lexical])

    assert fused["q1"][0] == "a"


def test_queries_are_unioned_not_intersected():
    """
    A query one system returned nothing for should still be scored on
    what the other found. Intersecting would silently shrink the
    evaluated query set and flatter whichever system failed.
    """
    run_dense = {"q1": ["a"]}
    run_lexical = {"q2": ["b"]}

    fused = reciprocal_rank_fusion([run_dense, run_lexical])

    assert set(fused) == {"q1", "q2"}
    assert fused["q1"] == ["a"]
    assert fused["q2"] == ["b"]


def test_result_is_truncated_to_depth():
    run = {"q1": [f"doc{i}" for i in range(20)]}

    fused = reciprocal_rank_fusion([run], depth=5)

    assert fused["q1"] == ["doc0", "doc1", "doc2", "doc3", "doc4"]


def test_k_controls_whether_one_top_rank_outweighs_broad_agreement():
    """
    'solo' is first in one run and absent from the other. 'agreed' is
    third in both. Which should win is exactly what k decides:

      solo   = 1 / (k + 1)
      agreed = 2 / (k + 3)

    At k = 0 the first position dominates and solo wins. At the
    conventional k = 60 the rank differences flatten, and appearing in
    both runs matters more than topping one of them.

    This is the knob's whole purpose, so it is worth a test that fails
    if the constant stops being applied.
    """
    run_a = {"q1": ["solo", "a1", "agreed"]}
    run_b = {"q1": ["b1", "b2", "agreed"]}

    sharp = reciprocal_rank_fusion([run_a, run_b], k=0)["q1"]
    flat = reciprocal_rank_fusion([run_a, run_b], k=60)["q1"]

    assert sharp.index("solo") < sharp.index("agreed")
    assert flat.index("agreed") < flat.index("solo")
