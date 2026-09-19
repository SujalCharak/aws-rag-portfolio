"""
Retrieval metrics: precision@k, recall@k, MRR, nDCG@k.

These are implemented here rather than pulled from a library so the
arithmetic is inspectable and explainable. That is only defensible if
the implementation is correct, so tests/test_metrics.py checks it two
ways: against hand computed values, and against pytrec_eval, the Python
binding for trec_eval, which is the reference implementation the BEIR
published numbers come from. Matching trec_eval is what makes the
numbers in RESULTS.md comparable to anyone else's.

All of this assumes binary relevance, which is what SciFact's
qrels provide: a document either supports the claim or is not labelled.

That assumption is load bearing for nDCG in particular. trec_eval
computes gain as 2^rel - 1, while the plainer formulation uses rel
directly. For binary relevance the two agree exactly, since 2^1 - 1 = 1,
which is why the implementation below can be this simple and still match
the reference. On graded relevance they would diverge, and this code
would need to say which convention it follows.
"""

import math


def precision_at_k(ranked_ids, relevant_ids, k):
    """
    Fraction of the top k results that are relevant.

    Divides by k even when fewer than k documents were retrieved, which
    is what trec_eval does: retrieving three results and getting all
    three right is not the same as a perfect top ten.

    Worth knowing when reading this number on SciFact: the dataset
    averages 1.13 relevant documents per query, so precision@10 is
    capped near 0.11 no matter how good the system is. It is reported
    for completeness, not because it discriminates.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids, relevant_ids, k):
    """Fraction of all relevant documents that appear in the top k."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    """
    One over the rank of the first relevant document, or zero if none
    appears. Computed over the whole ranked list rather than a cutoff,
    matching trec_eval's recip_rank.

    This is the most informative single number on a dataset with about
    one relevant document per query: it asks how far down the list the
    user has to read before finding the thing they came for.
    """
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / position
    return 0.0


def dcg_at_k(ranked_ids, relevant_ids, k):
    """
    Discounted cumulative gain: relevant documents contribute gain that
    shrinks logarithmically with how far down the list they sit.
    """
    return sum(
        1.0 / math.log2(position + 2)
        for position, doc_id in enumerate(ranked_ids[:k])
        if doc_id in relevant_ids
    )


def ideal_dcg_at_k(num_relevant, k):
    """DCG of the best possible ranking: every relevant document first."""
    return sum(1.0 / math.log2(position + 2) for position in range(min(num_relevant, k)))


def ndcg_at_k(ranked_ids, relevant_ids, k):
    """DCG normalised against the best achievable ranking, so 1.0 is perfect."""
    ideal = ideal_dcg_at_k(len(relevant_ids), k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant_ids, k) / ideal


def evaluate_run(run, qrels, k_values=(1, 3, 5, 10)):
    """
    Score a full run and return mean metrics across queries.

    run:   {query_id: [doc_id, ...]} ranked best first
    qrels: {query_id: set_of_relevant_doc_ids}

    Every retrieval system in this project, dense, BM25, and the fused
    hybrid, is scored by this one function. A baseline scored by
    different code is not a baseline, it is a second experiment.

    Queries with no relevant documents in the qrels are skipped rather
    than scored as zero, matching trec_eval. Scoring them as zero would
    silently drag every mean down in proportion to how incomplete the
    labels are, which says nothing about the system.
    """
    scored_query_ids = [qid for qid in run if qrels.get(qid)]

    results = {"num_queries": len(scored_query_ids)}
    if not scored_query_ids:
        return results

    for k in k_values:
        results[f"P@{k}"] = _mean(
            precision_at_k(run[qid], qrels[qid], k) for qid in scored_query_ids
        )
        results[f"recall@{k}"] = _mean(
            recall_at_k(run[qid], qrels[qid], k) for qid in scored_query_ids
        )
        results[f"nDCG@{k}"] = _mean(
            ndcg_at_k(run[qid], qrels[qid], k) for qid in scored_query_ids
        )

    results["MRR"] = _mean(reciprocal_rank(run[qid], qrels[qid]) for qid in scored_query_ids)
    return results


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0
