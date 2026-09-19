"""
Run the deployed vector retrieval over the SciFact queries and write a
ranked run file.

This calls the same retrieval code the query Lambda runs, imported from
src/query/, rather than a local reimplementation. See src/query/retrieval.py
for why that matters.

Output is a run file, {query_id: [doc_id, ...]} ranked best first, not a
set of metrics. Scoring happens in score_runs.py, which scores every
system, dense, BM25, and hybrid, through one code path. A baseline
scored by different code is not a baseline.

Raw chunk hits are cached alongside the run. Retrieving is the slow,
billable part; changing the document aggregation depth or rule is not.
Caching means the aggregation can be reworked and rescored immediately
instead of paying for 300 more embedding calls each time.

Usage:
    python eval/run_retrieval_eval.py
    python eval/run_retrieval_eval.py --limit 10          # quick smoke test
    python eval/run_retrieval_eval.py --from-cache        # rescore, no AWS calls
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Set before boto3 builds any client. Boto3 reads these environment
# variables for retry behaviour, so the harness gets patient, self
# throttling retries against Bedrock's rate limits without the Lambda
# inheriting them. The Lambda wants the opposite: it sits behind an API
# Gateway timeout, so failing fast and letting the caller retry beats
# backing off for thirty seconds inside a request nobody is waiting on
# any more.
os.environ.setdefault("AWS_RETRY_MODE", "adaptive")
os.environ.setdefault("AWS_MAX_ATTEMPTS", "10")

import boto3  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "query"))

from retrieval import aggregate_to_documents, embed_text, search_chunks  # noqa: E402


def resolve_vector_bucket():
    """
    Derive the vector bucket name from the caller's account rather than
    hardcoding it. Keeps an account id out of a committed file, and
    means this runs against whichever account is configured without
    editing the script.
    """
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"rag-portfolio-vectors-{account_id}"


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def retrieve_all(queries, chunk_depth, vector_bucket, index_name):
    """
    Embed and retrieve for every query, sequentially.

    Sequential on purpose. The account's Bedrock throughput is the
    binding constraint here, and 300 queries at roughly half a second
    each is a few minutes, paid once because the results are cached.
    Parallelising it would trade a couple of saved minutes for the same
    throttling failures that made the corpus load painful.
    """
    hits_by_query = {}
    started = time.time()

    for i, query in enumerate(queries, start=1):
        query_vector = embed_text(query["text"])
        hits = search_chunks(
            query_vector,
            top_k=chunk_depth,
            vector_bucket=vector_bucket,
            index_name=index_name,
        )
        # Keep only what scoring and inspection need. The full response
        # carries the embedding vectors back, which would make the cache
        # hundreds of megabytes for no benefit.
        hits_by_query[query["query_id"]] = [
            {
                "doc_id": hit["metadata"]["doc_id"],
                "chunk_index": hit["metadata"].get("chunk_index"),
                "distance": hit.get("distance"),
                "metadata": {"doc_id": hit["metadata"]["doc_id"]},
            }
            for hit in hits
        ]

        if i % 25 == 0 or i == len(queries):
            elapsed = time.time() - started
            print(f"  {i}/{len(queries)} queries, {elapsed:.0f}s elapsed", flush=True)

    return hits_by_query


def build_run(hits_by_query, doc_depth):
    return {
        query_id: aggregate_to_documents(hits, k=doc_depth)
        for query_id, hits in hits_by_query.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--out", default="data/eval/run_vector.json")
    parser.add_argument("--cache", default="data/eval/vector_hits.json")
    parser.add_argument(
        "--chunk-depth",
        type=int,
        default=50,
        help=(
            "chunks to retrieve per query before collapsing to documents. "
            "Deeper than the reported document depth on purpose: several "
            "chunks can belong to one document, so retrieving 10 chunks "
            "can yield fewer than 10 distinct documents and quietly cap "
            "recall@10 below what the system can actually do."
        ),
    )
    parser.add_argument("--doc-depth", type=int, default=10)
    parser.add_argument("--index-name", default="documents-index")
    parser.add_argument("--vector-bucket", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="rebuild the run from cached hits without calling AWS",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache)

    if args.from_cache:
        if not cache_path.exists():
            sys.exit(f"No cache at {cache_path}. Run without --from-cache first.")
        hits_by_query = json.loads(cache_path.read_text())
        print(f"Loaded cached hits for {len(hits_by_query)} queries")
    else:
        queries = read_jsonl(args.queries)
        if args.limit:
            queries = queries[: args.limit]

        vector_bucket = args.vector_bucket or resolve_vector_bucket()
        print(f"Retrieving for {len(queries)} queries from {vector_bucket}/{args.index_name}")

        hits_by_query = retrieve_all(queries, args.chunk_depth, vector_bucket, args.index_name)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(hits_by_query))
        print(f"Cached raw hits to {cache_path}")

    run = build_run(hits_by_query, args.doc_depth)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(run, indent=2))

    empty = sum(1 for ranked in run.values() if not ranked)
    print(f"Wrote run for {len(run)} queries to {out_path}")
    if empty:
        print(f"Warning: {empty} queries returned nothing at all")


if __name__ == "__main__":
    main()
