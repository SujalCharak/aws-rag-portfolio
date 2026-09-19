"""
Generate answers for a sample of queries, judge whether each is grounded
in its retrieved context, and produce the materials for hand labelling
the same sample.

Retrieval and generation are imported from src/query/, so the answers
being judged are produced by the deployed prompt and model rather than
an approximation of them. Retrieval depth here is the query Lambda's
TOP_K, not the deeper depth used for ranking metrics: the point is to
judge the context the system actually generates from.

Three files come out:

  groundedness_samples.json  everything, including the judge's verdicts
  label_sheet.md             readable items for hand labelling
  labels.csv                 a blank column to fill in

The label sheet deliberately omits the judge's verdicts. Showing a human
the model's answer before they label anchors them to it, and the
resulting agreement score would partly measure that anchoring rather
than the judge's quality. The verdicts are joined back in by
score_judge.py after the labels exist.

Usage:
    python eval/run_groundedness.py --sample-size 50
    python eval/run_groundedness.py --judge-model us.anthropic.claude-...
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("AWS_RETRY_MODE", "adaptive")
os.environ.setdefault("AWS_MAX_ATTEMPTS", "10")

import boto3  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "query"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generation import build_context, generate_answer  # noqa: E402
from groundedness import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    RUBRIC,
    build_judge_prompt,
    parse_verdict,
)
from retrieval import embed_text, search_chunks  # noqa: E402

PRODUCTION_TOP_K = 5


def resolve_judge_model(explicit=None):
    """
    Find a Claude model to judge with, rather than hardcoding an id.

    Model identifiers change, and newer Claude models on Bedrock are
    often callable only through an inference profile id rather than the
    bare model id, so a hardcoded default tends to fail with a
    ValidationException that does not explain itself. Discovering the id
    at runtime avoids both problems.
    """
    if explicit:
        return explicit

    bedrock = boto3.client("bedrock")

    try:
        profiles = bedrock.list_inference_profiles().get("inferenceProfileSummaries", [])
        sonnet_profiles = [
            p for p in profiles if "sonnet" in p.get("inferenceProfileId", "").lower()
        ]
        if sonnet_profiles:
            chosen = sorted(sonnet_profiles, key=lambda p: p["inferenceProfileId"])[-1]
            return chosen["inferenceProfileId"]
    except Exception:
        # Falls through to the foundation model listing below. Not every
        # account or region exposes inference profiles.
        pass

    models = bedrock.list_foundation_models(byProvider="anthropic").get("modelSummaries", [])
    on_demand_sonnets = [
        m for m in models
        if "sonnet" in m["modelId"].lower()
        and "ON_DEMAND" in m.get("inferenceTypesSupported", [])
    ]
    if on_demand_sonnets:
        return sorted(on_demand_sonnets, key=lambda m: m["modelId"])[-1]["modelId"]

    sys.exit(
        "Could not find a Claude Sonnet model to judge with. List what is available:\n"
        "  aws bedrock list-inference-profiles --query 'inferenceProfileSummaries[].inferenceProfileId'\n"
        "  aws bedrock list-foundation-models --by-provider anthropic "
        "--query 'modelSummaries[].modelId'\n"
        "Then pass one with --judge-model."
    )


def judge_answer(question, context, answer, judge_model):
    response = boto3.client("bedrock-runtime").converse(
        modelId=judge_model,
        system=[{"text": JUDGE_SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": build_judge_prompt(question, context, answer)}]}],
        # Temperature zero so a rerun produces the same verdicts. A
        # judge that returns different labels on identical input cannot
        # be validated against a fixed set of human labels.
        inferenceConfig={"maxTokens": 256, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_vector_bucket():
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"rag-portfolio-vectors-{account_id}"


def write_label_sheet(path, samples):
    """Readable items plus the rubric, with no judge verdicts anywhere."""
    lines = [
        "# Groundedness labelling sheet",
        "",
        "Read each item and write its label in `labels.csv` next to the matching id.",
        "",
        "The model judge was given these exact instructions, and nothing else,",
        "so that the agreement score measures the judge rather than a difference",
        "in what the two of you were asked to do.",
        "",
        "```",
        RUBRIC,
        "```",
        "",
        "---",
        "",
    ]

    for sample in samples:
        lines.extend([
            f"## {sample['id']}",
            "",
            f"**Question:** {sample['question']}",
            "",
            f"**Answer:** {sample['answer']}",
            "",
            "<details><summary>Context passages</summary>",
            "",
            "```",
            sample["context"],
            "```",
            "",
            "</details>",
            "",
            "---",
            "",
        ])

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_blank_labels(path, samples):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "human_label"])
        for sample in samples:
            writer.writerow([sample["id"], ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/eval/queries.jsonl")
    parser.add_argument("--out-dir", default="data/eval")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="fixed so the sample is reproducible; a different sample would need relabelling",
    )
    parser.add_argument("--index-name", default="documents-index")
    parser.add_argument("--vector-bucket", default=None)
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = read_jsonl(args.queries)
    rng = random.Random(args.seed)
    sampled = rng.sample(queries, min(args.sample_size, len(queries)))

    vector_bucket = args.vector_bucket or resolve_vector_bucket()
    judge_model = resolve_judge_model(args.judge_model)
    print(f"Judging with {judge_model}")
    print(f"Generating and judging {len(sampled)} answers\n")

    samples = []
    unparseable = 0
    started = time.time()

    for i, query in enumerate(sampled, start=1):
        query_vector = embed_text(query["text"])
        hits = search_chunks(
            query_vector,
            top_k=PRODUCTION_TOP_K,
            vector_bucket=vector_bucket,
            index_name=args.index_name,
        )

        context = build_context(hits)
        answer = generate_answer(query["text"], hits)
        judge_reply = judge_answer(query["text"], context, answer, judge_model)
        label, reason = parse_verdict(judge_reply)

        if label is None:
            unparseable += 1

        samples.append({
            "id": query["query_id"],
            "question": query["text"],
            "answer": answer,
            "context": context,
            "retrieved_doc_ids": [hit["metadata"]["doc_id"] for hit in hits],
            "judge_label": label,
            "judge_reason": reason,
            "judge_raw": judge_reply,
        })

        if i % 10 == 0 or i == len(sampled):
            print(f"  {i}/{len(sampled)}, {time.time() - started:.0f}s elapsed", flush=True)

    (out_dir / "groundedness_samples.json").write_text(json.dumps(samples, indent=2))
    write_label_sheet(out_dir / "label_sheet.md", samples)
    write_blank_labels(out_dir / "labels.csv", samples)

    judged = [s["judge_label"] for s in samples if s["judge_label"]]
    grounded = sum(1 for label in judged if label == "grounded")
    abstained = sum(1 for label in judged if label == "abstained")

    print(f"\nJudge: {grounded} grounded, {abstained} abstained, "
          f"{len(judged) - grounded - abstained} ungrounded")
    if unparseable:
        print(f"Warning: {unparseable} judge replies could not be parsed, recorded as null")

    print(f"\nWrote {out_dir / 'groundedness_samples.json'}")
    print(f"Label these by hand: open {out_dir / 'label_sheet.md'}, "
          f"fill in {out_dir / 'labels.csv'}")
    print("The sheet deliberately does not show the judge's verdicts. Do not read")
    print("groundedness_samples.json before labelling, it would anchor your labels.")


if __name__ == "__main__":
    main()
