"""
Reciprocal Rank Fusion of two or more run files.

If the dense retriever and BM25 make different mistakes, combining their
rankings should beat either alone. That is the whole premise of hybrid
retrieval, and it is worth measuring rather than assuming, because it
only holds when the two systems are genuinely complementary. If fusion
does not help here, that is also worth knowing before anyone builds
infrastructure to serve it.

RRF scores a document by summing 1 / (k + rank) across the runs it
appears in. Two properties make it the usual choice:

- It needs only ranks, not scores. Cosine distances and BM25 scores live
  on incomparable scales, and normalising them against each other
  requires choices that quietly determine the outcome. Ranks sidestep
  that entirely.
- The constant k, conventionally 60, damps the influence of the very top
  positions, so one system being confidently wrong cannot dominate the
  fused ranking.

Honest scope: this fuses at evaluation time from two separately produced
run files. It measures whether the signals complement each other, which
is the question worth answering first. It is not a hybrid retriever
serving live traffic, and the writeup should say so.

Usage:
    python eval/fuse_runs.py --runs data/eval/run_vector.json data/eval/run_bm25.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

RRF_K = 60


def reciprocal_rank_fusion(runs, k=RRF_K, depth=10):
    """
    runs: list of {query_id: [doc_id, ...]} ranked best first.
    Returns a single fused run.

    Queries are unioned across runs rather than intersected. A query one
    system returned nothing for should still be scored on what the other
    found, not dropped.
    """
    query_ids = set()
    for run in runs:
        query_ids.update(run.keys())

    fused = {}
    for query_id in query_ids:
        scores = defaultdict(float)
        for run in runs:
            for rank, doc_id in enumerate(run.get(query_id, []), start=1):
                scores[doc_id] += 1.0 / (k + rank)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        fused[query_id] = [doc_id for doc_id, _ in ranked[:depth]]

    return fused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["data/eval/run_vector.json", "data/eval/run_bm25.json"],
    )
    parser.add_argument("--out", default="data/eval/run_hybrid_rrf.json")
    parser.add_argument("--k", type=int, default=RRF_K)
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()

    runs = []
    for path in args.runs:
        runs.append(json.loads(Path(path).read_text()))
        print(f"Loaded {path}")

    fused = reciprocal_rank_fusion(runs, k=args.k, depth=args.depth)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fused, indent=2))
    print(f"Wrote fused run for {len(fused)} queries to {out_path}")


if __name__ == "__main__":
    main()
