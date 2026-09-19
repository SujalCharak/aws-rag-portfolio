"""
BM25 lexical baseline over the same corpus, written as a run file.

The point of a baseline is to answer "is the expensive thing worth it".
Embeddings, a vector index, and a per query model call are a lot of
machinery to justify, and BM25 is decades old, runs locally, costs
nothing, and is genuinely strong on this dataset. Published BEIR results
put BM25 around 0.665 nDCG@10 on SciFact, ahead of a fair number of
dense retrievers, because scientific claims reuse the exact terminology
of the abstracts that support them, which is the case lexical matching
handles best.

If the dense retriever loses here, that is a result worth reporting, not
a bug to tune away.

Two details make this a controlled comparison rather than two loosely
related experiments:

- Text preparation is imported from the ingest Lambda, not reimplemented.
  The same parse_documents and chunk_text that produced the embedded
  text produce the text BM25 indexes, so the two systems see byte
  identical input.
- Document aggregation reuses aggregate_to_documents from the query
  path, by expressing BM25 scores as negated distances. Both systems
  therefore collapse units to documents by exactly the same rule.

Usage:
    pip install rank_bm25
    python eval/run_bm25_baseline.py
    python eval/run_bm25_baseline.py --granularity document
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

# The ingest module fails fast on missing configuration at import time,
# which is right for a Lambda and inconvenient here, where only its pure
# text functions are wanted. Placeholders satisfy the import; nothing in
# this script makes an AWS call.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("VECTOR_BUCKET_NAME", "unused-no-aws-calls-here")
os.environ.setdefault("VECTOR_INDEX_NAME", "unused-no-aws-calls-here")
os.environ.setdefault("EMBEDDING_MODEL_ID", "unused-no-aws-calls-here")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "query"))

from retrieval import aggregate_to_documents  # noqa: E402


def _load_by_path(module_name, path):
    """
    Load a module from an explicit path under a chosen name.

    Needed because both Lambda handlers are called app.py. src/query is
    on sys.path above so retrieval.py can resolve its own imports, which
    means a plain `from app import ...` here would return the query
    handler rather than the ingest one, and then fail on a missing
    environment variable that has nothing to do with this script.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ingest = _load_by_path("ingest_app", REPO_ROOT / "src" / "ingest" / "app.py")
chunk_text = _ingest.chunk_text
parse_documents = _ingest.parse_documents

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """
    Lowercase and split on non alphanumeric characters.

    No stemming and no stopword removal, which makes this a slightly
    weaker BM25 than the Anserini and Lucene setups behind most
    published numbers. Worth stating rather than quietly claiming parity
    with them: the comparison against the dense retriever here is sound,
    since both see the same text, but treating this score as
    interchangeable with a published BM25 figure would not be.
    """
    return TOKEN_PATTERN.findall(text.lower())


def load_corpus(corpus_dir):
    """
    Rebuild (doc_id, text) pairs by running the shard files back through
    the ingest function's own parser, so the text matches what was
    embedded rather than merely resembling it.
    """
    documents = []
    for shard_path in sorted(Path(corpus_dir).glob("*.jsonl")):
        raw_text = shard_path.read_text(encoding="utf-8")
        documents.extend(parse_documents(shard_path.name, raw_text))
    return documents


def build_units(documents, granularity):
    """
    Return (doc_id, unit_text) pairs at the requested granularity.

    Chunk granularity matches the dense retriever exactly, which is the
    controlled comparison. Document granularity matches how published
    BEIR BM25 numbers are computed, which is the comparable one. They
    answer different questions, so both are worth running.
    """
    if granularity == "document":
        return [(doc_id, text) for doc_id, text in documents]

    units = []
    for doc_id, text in documents:
        for chunk in chunk_text(text):
            units.append((doc_id, chunk))
    return units


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--granularity", choices=("chunk", "document"), default="chunk")
    parser.add_argument("--doc-depth", type=int, default=10)
    parser.add_argument(
        "--unit-depth",
        type=int,
        default=50,
        help="units scored before collapsing to documents, matching the dense run's chunk depth",
    )
    args = parser.parse_args()

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit("rank_bm25 is not installed. Run: pip install rank_bm25")

    out_path = Path(
        args.out or f"data/eval/run_bm25{'_wholedoc' if args.granularity == 'document' else ''}.json"
    )

    documents = load_corpus(args.corpus)
    units = build_units(documents, args.granularity)
    print(f"Loaded {len(documents)} documents, indexing {len(units)} {args.granularity} units")

    bm25 = BM25Okapi([tokenize(text) for _, text in units])
    unit_doc_ids = [doc_id for doc_id, _ in units]

    queries = read_jsonl(args.queries)
    run = {}

    for i, query in enumerate(queries, start=1):
        scores = bm25.get_scores(tokenize(query["text"]))

        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        top_indices = top_indices[: args.unit_depth]

        # BM25 scores rank higher is better, while the vector path ranks
        # lower distance is better. Negating lets both go through the
        # same aggregation function, so the two systems provably collapse
        # units to documents by the same rule rather than by two
        # implementations that are meant to agree.
        pseudo_hits = [
            {"distance": -float(scores[idx]), "metadata": {"doc_id": unit_doc_ids[idx]}}
            for idx in top_indices
        ]
        run[query["query_id"]] = aggregate_to_documents(pseudo_hits, k=args.doc_depth)

        if i % 50 == 0 or i == len(queries):
            print(f"  {i}/{len(queries)} queries scored", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(run, indent=2))
    print(f"Wrote run for {len(run)} queries to {out_path}")


if __name__ == "__main__":
    main()
