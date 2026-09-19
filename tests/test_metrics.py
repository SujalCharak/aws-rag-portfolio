"""
Tests for the retrieval metrics.

Two independent checks, because these functions produce every number
that ends up in RESULTS.md. A silent error in nDCG would not crash
anything, it would just publish wrong figures with confident formatting.

1. Hand computed values for a small ranking, so the arithmetic is
   pinned to something a reader can verify with a calculator.
2. A cross check against pytrec_eval, the Python binding for trec_eval,
   which is the reference implementation behind the published BEIR
   numbers. Agreement there is what makes these results comparable to
   anyone else's rather than merely self consistent.

The pytrec_eval check skips cleanly when the package is not installed,
so CI stays fast and free of a compiled dependency while the check still
runs locally where it matters.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from metrics import (  # noqa: E402
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Relevant documents sit at ranks 2 and 4 of a five document ranking.
RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d"}


def test_precision_divides_by_k_not_by_results_returned():
    assert precision_at_k(RANKED, RELEVANT, 1) == 0.0
    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.4)

    # Only two documents retrieved, both relevant, but k is 5. trec_eval
    # scores this 0.4, not 1.0: a short perfect list is not a perfect
    # top five. Getting this wrong inflates every precision number.
    assert precision_at_k(["b", "d"], RELEVANT, 5) == pytest.approx(0.4)


def test_recall_counts_against_all_relevant_documents():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.5)
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(1.0)


def test_reciprocal_rank_uses_first_relevant_document():
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.5)
    assert reciprocal_rank(["b", "a"], RELEVANT) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "y", "z"], RELEVANT) == 0.0


def test_ndcg_matches_hand_computed_values():
    # DCG@5  = 1/log2(3) + 1/log2(5) = 0.6309298 + 0.4306766 = 1.0616063
    # IDCG@5 = 1/log2(2) + 1/log2(3) = 1.0       + 0.6309298 = 1.6309298
    # nDCG@5 = 1.0616063 / 1.6309298
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.6509209, abs=1e-6)

    # DCG@3  = 1/log2(3) only, since the second relevant document sits
    # at rank 4 and is cut off. IDCG@3 is unchanged: the ideal ranking
    # still fits both relevant documents inside the top three.
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.3868528, abs=1e-6)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["b", "d", "a"], RELEVANT, 3) == pytest.approx(1.0)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved():
    assert ndcg_at_k(["x", "y"], RELEVANT, 2) == 0.0


def test_ndcg_is_zero_rather_than_undefined_when_no_labels_exist():
    # Guards a division by zero that would otherwise surface as a crash
    # partway through a 300 query run.
    assert ndcg_at_k(RANKED, set(), 5) == 0.0


def test_evaluate_run_skips_queries_with_no_relevant_documents():
    run = {
        "q1": ["b", "a", "c"],
        "q2": ["a", "b", "c"],
        "q3": ["a", "b", "c"],
    }
    qrels = {
        "q1": {"b"},
        "q2": {"c"},
        # q3 deliberately absent: an unlabelled query.
    }

    results = evaluate_run(run, qrels, k_values=(3,))

    # Two queries scored, not three. Scoring the unlabelled one as zero
    # would drag the mean down in proportion to how incomplete the
    # labels are, which measures the dataset rather than the system.
    assert results["num_queries"] == 2
    assert results["recall@3"] == pytest.approx(1.0)
    assert results["MRR"] == pytest.approx((1.0 + 1 / 3) / 2)


def test_evaluate_run_returns_empty_result_for_no_scorable_queries():
    results = evaluate_run({"q1": ["a"]}, {})
    assert results["num_queries"] == 0


def test_matches_trec_eval_reference_implementation():
    """
    The check that makes these numbers comparable to published results
    rather than merely internally consistent.
    """
    pytrec_eval = pytest.importorskip(
        "pytrec_eval",
        reason="pytrec_eval not installed; run pip install pytrec_eval-terrier to enable",
    )

    qrels_trec = {"q1": {"b": 1, "d": 1}}
    # Descending scores encode the same rank order as RANKED.
    run_trec = {"q1": {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0}}

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_trec, {"ndcg_cut.3", "ndcg_cut.5", "recall.3", "recall.5", "P.3", "P.5", "recip_rank"}
    )
    reference = evaluator.evaluate(run_trec)["q1"]

    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(reference["P_3"], abs=1e-9)
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(reference["P_5"], abs=1e-9)
    assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(reference["recall_3"], abs=1e-9)
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(reference["recall_5"], abs=1e-9)
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(reference["recip_rank"], abs=1e-9)
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(reference["ndcg_cut_3"], abs=1e-9)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(reference["ndcg_cut_5"], abs=1e-9)
