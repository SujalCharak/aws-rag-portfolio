"""
Compare the model judge's verdicts against the hand labels and report
Cohen's kappa.

This is the step that decides whether the groundedness number is worth
reporting at all. An unvalidated LLM judge produces a percentage with no
established relationship to anything, and the percentage looks identical
whether the judge is careful or is answering "grounded" reflexively.

Usage:
    python eval/score_judge.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agreement import cohen_kappa, confusion_matrix  # noqa: E402
from groundedness import VALID_LABELS  # noqa: E402


def read_labels(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            row["id"]: row["human_label"].strip().lower()
            for row in csv.DictReader(handle)
            if row.get("human_label", "").strip()
        }


def interpret(kappa):
    """Conventional reading bands, stated as convention rather than law."""
    for threshold, description in (
        (0.8, "near perfect"),
        (0.6, "substantial"),
        (0.4, "moderate"),
        (0.2, "fair"),
    ):
        if kappa >= threshold:
            return description
    return "negligible"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="data/eval/groundedness_samples.json")
    parser.add_argument("--labels", default="data/eval/labels.csv")
    parser.add_argument("--out", default="data/eval/judge_validation.json")
    args = parser.parse_args()

    samples = {s["id"]: s for s in json.loads(Path(args.samples).read_text())}
    human_labels = read_labels(args.labels)

    if not human_labels:
        sys.exit(f"No labels filled in yet. Add them to {args.labels}.")

    invalid = {
        item_id: label for item_id, label in human_labels.items() if label not in VALID_LABELS
    }
    if invalid:
        sys.exit(f"Unrecognised labels: {invalid}. Use one of {VALID_LABELS}.")

    # Only items with both a human label and a parseable judge verdict
    # can be compared. Everything dropped here is counted and reported
    # rather than quietly excluded, since a judge that frequently
    # returns unreadable output is itself a finding.
    paired_ids = [
        item_id for item_id in human_labels
        if item_id in samples and samples[item_id].get("judge_label")
    ]

    judge = [samples[item_id]["judge_label"] for item_id in paired_ids]
    human = [human_labels[item_id] for item_id in paired_ids]

    kappa = cohen_kappa(judge, human)
    raw_agreement = sum(1 for a, b in zip(judge, human) if a == b) / len(paired_ids)

    print(f"Compared {len(paired_ids)} items\n")
    print(f"Raw agreement   {raw_agreement:.3f}")
    print(f"Cohen's kappa   {kappa:.3f}  ({interpret(kappa)})")

    print("\nConfusion matrix, rows are the judge, columns are you:\n")
    matrix = confusion_matrix(judge, human, VALID_LABELS)
    header = " " * 12 + "".join(f"{name:>12}" for name in VALID_LABELS)
    print(header)
    for judge_label in VALID_LABELS:
        row = "".join(f"{matrix[judge_label][h]:>12}" for h in VALID_LABELS)
        print(f"{judge_label:<12}{row}")

    unlabelled = len(samples) - len(human_labels)
    unparseable = sum(1 for s in samples.values() if not s.get("judge_label"))
    if unlabelled:
        print(f"\n{unlabelled} samples not yet labelled by hand")
    if unparseable:
        print(f"{unparseable} judge replies were unparseable and excluded")

    # Groundedness rate among answers that actually asserted something.
    # Abstentions are excluded rather than counted as grounded: an
    # answer that declines to answer makes no unsupported claims and so
    # is trivially grounded, which would let a system that always says
    # "I don't know" report a perfect score.
    asserted = [label for label in human if label != "abstained"]
    if asserted:
        grounded_rate = sum(1 for label in asserted if label == "grounded") / len(asserted)
        print(
            f"\nGrounded rate (your labels, excluding {len(human) - len(asserted)} "
            f"abstentions): {grounded_rate:.3f}"
        )

    results = {
        "items_compared": len(paired_ids),
        "raw_agreement": raw_agreement,
        "cohens_kappa": kappa,
        "interpretation": interpret(kappa),
        "confusion_matrix": matrix,
        "unparseable_judge_replies": unparseable,
        "unlabelled_samples": unlabelled,
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
