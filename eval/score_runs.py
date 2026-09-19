"""
Score every run file through one scorer and print the comparison table.

Each retrieval system writes a run file, {query_id: [doc_id, ...]}, and
this reads all of them. Scoring lives here rather than inside each
system's script so that dense, BM25, and hybrid are measured by
identical code. A baseline scored by its own slightly different
implementation is not a comparison, it is two experiments printed next
to each other.

Usage:
    python eval/score_runs.py
    python eval/score_runs.py --runs data/eval/run_vector.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import evaluate_run  # noqa: E402

# Shown in the table. The full metric set goes to JSON; this is the
# subset worth reading side by side.
TABLE_COLUMNS = ["nDCG@10", "MRR", "recall@5", "recall@10", "P@1", "P@10"]


def load_qrels(path):
    """Read qrels into {query_id: set_of_relevant_doc_ids}."""
    relevant = defaultdict(set)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("relevance", 0)) > 0:
                relevant[row["query_id"]].add(row["doc_id"])
    return dict(relevant)


def system_name(path):
    return Path(path).stem.replace("run_", "")


def format_table(scored):
    header = "| System | " + " | ".join(TABLE_COLUMNS) + " | Queries |"
    divider = "|---" * (len(TABLE_COLUMNS) + 2) + "|"
    rows = [header, divider]

    for name, results in scored.items():
        cells = [f"{results.get(column, 0.0):.4f}" for column in TABLE_COLUMNS]
        rows.append(f"| {name} | " + " | ".join(cells) + f" | {results['num_queries']} |")

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", default="data/eval/qrels.jsonl")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="run files to score. Defaults to every data/eval/run_*.json",
    )
    parser.add_argument("--out", default="data/eval/results.json")
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    print(f"Loaded labels for {len(qrels)} queries\n")

    run_paths = args.runs or sorted(str(p) for p in Path("data/eval").glob("run_*.json"))
    if not run_paths:
        sys.exit("No run files found. Run a retrieval script first.")

    scored = {}
    for path in run_paths:
        run = json.loads(Path(path).read_text())
        scored[system_name(path)] = evaluate_run(run, qrels)

    table = format_table(scored)
    print(table)

    Path(args.out).write_text(json.dumps(scored, indent=2))
    print(f"\nFull metrics written to {args.out}")

    _print_caveats(qrels)


def _print_caveats(qrels):
    """
    Print the context needed to read these numbers correctly, next to
    the numbers themselves rather than buried in a document nobody
    opens alongside them.
    """
    if not qrels:
        return

    avg_relevant = sum(len(docs) for docs in qrels.values()) / len(qrels)
    print(
        f"\nThis dataset averages {avg_relevant:.2f} relevant documents per query, "
        f"so P@10 is capped near {avg_relevant / 10:.2f}."
    )
    print("Low precision at depth is arithmetic here, not a failure. MRR and recall carry the signal.")


if __name__ == "__main__":
    main()
