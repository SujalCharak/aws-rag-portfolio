"""
Download SciFact and prepare it for ingestion and evaluation.

Run this on a machine with normal internet access. It writes three things:

  data/corpus/shard_XXX.jsonl   documents, sharded, ready to upload to S3
  data/eval/queries.jsonl       test queries
  data/eval/qrels.jsonl         human relevance labels for those queries

Usage:
    pip install ir_datasets
    python scripts/prepare_scifact.py
    python scripts/prepare_scifact.py --docs-per-shard 250 --out data

Why SciFact rather than a corpus of our own:

The evaluation harness needs queries with known correct answers. Writing
those by hand is slow and biased: questions written while looking at a
document tend to reuse its wording, which flatters vector search for the
wrong reason. SciFact ships with human relevance judgments, produced
independently of any retrieval system, and published baselines exist to
compare against. Around 5,000 scientific abstracts, small enough to
ingest for pennies.

Why sharding, rather than one file per document:

One S3 object per document means one Lambda invocation per document.
Thousands of concurrent invocations against one Bedrock quota throttles
into mass failure. Shards mean one invocation handles many documents,
which is both how batch pipelines are normally shaped and what keeps the
request rate survivable.
"""

import argparse
import json
import re
from pathlib import Path


def normalize_doc_id(value):
    """Match the ingest Lambda's document id normalization exactly.

    This matters more than it looks. src/ingest/app.py rewrites document
    ids when it builds vector keys, stripping anything that isn't
    alphanumeric, underscore, or dash. The relevance labels have to go
    through the same transformation, or evaluation silently compares
    normalized retrieved ids against raw labelled ids, matches nothing,
    and reports precision of zero on a system that is actually working.

    SciFact ids happen to be numeric, so nothing changes for this corpus
    in practice, but relying on that would break the moment the harness
    is pointed at a corpus with ids like "PMC-4711.v2".

    Kept as a copy rather than an import because this script runs on a
    laptop and the Lambda code ships to AWS; wiring a shared package
    between them costs more than it saves at this size. The test suite
    asserts the two stay in agreement.
    """
    base = str(value).rsplit("/", 1)[-1]
    base = re.sub(r"\.[^.]+$", "", base)
    return re.sub(r"[^a-zA-Z0-9_-]", "_", base)


def shard(items, per_shard):
    """Split a list into chunks of at most per_shard items."""
    for start in range(0, len(items), per_shard):
        yield items[start:start + per_shard]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def estimate_chunks(documents, chunk_size=300, overlap=50):
    """Rough count of how many chunks the corpus will produce.

    Mirrors the ingest function's sliding window: each chunk advances by
    (chunk_size - overlap) words. Only used to print an estimate, so the
    arithmetic being approximate is fine.
    """
    stride = chunk_size - overlap
    total = 0
    for doc in documents:
        words = len(doc["text"].split())
        total += 1 if words <= chunk_size else 1 + -(-(words - chunk_size) // stride)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="output directory")
    parser.add_argument(
        "--docs-per-shard",
        type=int,
        default=250,
        help=(
            "documents per shard file. Smaller shards mean more parallelism "
            "but more Lambda invocations; larger shards risk the 15 minute "
            "function timeout. 250 keeps each shard well under both."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="beir/scifact/test",
        help="ir_datasets identifier",
    )
    args = parser.parse_args()

    import ir_datasets

    out = Path(args.out)
    print(f"Loading {args.dataset} (first run downloads and caches it)...")
    dataset = ir_datasets.load(args.dataset)

    documents = [
        {
            "doc_id": normalize_doc_id(doc.doc_id),
            "title": (doc.title or "").strip(),
            "text": (doc.text or "").strip(),
        }
        for doc in dataset.docs_iter()
    ]
    documents = [d for d in documents if d["text"]]

    shards = list(shard(documents, args.docs_per_shard))
    for i, batch in enumerate(shards):
        write_jsonl(out / "corpus" / f"shard_{i:03d}.jsonl", batch)

    queries = [
        {"query_id": str(q.query_id), "text": q.text.strip()}
        for q in dataset.queries_iter()
    ]
    write_jsonl(out / "eval" / "queries.jsonl", queries)

    # Only positive judgments are kept. BEIR qrels use relevance > 0 to
    # mean relevant; rows at 0 are explicit negatives, which the ranking
    # metrics do not need.
    qrels = [
        {
            "query_id": str(qrel.query_id),
            "doc_id": normalize_doc_id(qrel.doc_id),
            "relevance": int(qrel.relevance),
        }
        for qrel in dataset.qrels_iter()
        if int(qrel.relevance) > 0
    ]
    write_jsonl(out / "eval" / "qrels.jsonl", qrels)

    labelled_queries = {q["query_id"] for q in qrels}
    estimated_chunks = estimate_chunks(documents)

    manifest = {
        "dataset": args.dataset,
        "documents": len(documents),
        "shards": len(shards),
        "docs_per_shard": args.docs_per_shard,
        "queries": len(queries),
        "queries_with_labels": len(labelled_queries),
        "qrels": len(qrels),
        "estimated_chunks": estimated_chunks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    # Titan Text Embeddings V2 is priced per 1000 input tokens. Assuming
    # roughly 1.3 tokens per word, this is a ballpark, not a bill.
    tokens = sum(len(d["text"].split()) for d in documents) * 1.3
    print(f"\nRough embedding cost: ${tokens / 1000 * 0.00002:.4f}")
    print(f"Wrote {len(shards)} shards to {out / 'corpus'}")
    print(f"Eval files in {out / 'eval'}")


if __name__ == "__main__":
    main()
