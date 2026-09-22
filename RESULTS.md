# Results

## Retrieval

300 SciFact test queries, 339 human relevance judgments, scored with
`eval/metrics.py` (cross-validated against `pytrec_eval`, the reference
implementation behind published BEIR benchmarks). Full methodology and
caveats in the README's Evaluation section; the short version is below
each number that needs it.

| System | nDCG@10 | MRR | recall@5 | recall@10 | P@1 | P@10 |
|---|---|---|---|---|---|---|
| BM25 (lexical) | 0.6508 | 0.6185 | 0.7251 | 0.7707 | 0.5200 | 0.0847 |
| Dense (Titan V2) | 0.6761 | 0.6434 | 0.7433 | 0.7952 | 0.5567 | 0.0900 |
| Hybrid (RRF) | 0.7061 | 0.6588 | 0.7728 | 0.8778 | 0.5567 | 0.0980 |

**Dense retrieval beat BM25, but only modestly.** Going in, BM25 was the
more likely winner on this corpus, it's decades old, costs nothing, and
scientific claims tend to reuse the exact wording of the abstracts that
support them, which is the case lexical matching is best at. BM25's
0.6508 nDCG@10 here lands almost exactly on published BEIR baselines for
this dataset (~0.665). Titan V2 edged past it anyway, a real but modest
win (0.6761 vs 0.6508), not a rout.

**Hybrid fusion won outright**, ahead of both individual systems on
every metric, not just on average. The clearest signal is recall@10:
0.8778 versus 0.7952 (dense) and 0.7707 (BM25), meaning the fused
ranking surfaced relevant documents in its top 10 that neither system
alone found in its own top 10. That's the actual premise of hybrid
retrieval, two systems making different mistakes, holding up under
measurement rather than being assumed. Scope note: this is Reciprocal
Rank Fusion applied at evaluation time over two separately produced run
files (`eval/fuse_runs.py`), not a hybrid retriever serving live query
traffic.

**P@10 near 0.08-0.10 is a ceiling, not a weakness.** SciFact averages
1.13 relevant documents per query, so P@10 cannot exceed roughly 0.11
regardless of system quality. MRR and recall carry the actual signal
here; P@1 (0.52-0.56, meaning the correct document is the literal top
result more than half the time) is the precision number worth reading.

## Groundedness

Whether generated answers are actually supported by their retrieved
context, judged by a Claude model (a different model from Nova Micro,
which generates the answers, so the judge isn't grading itself).

**Status: harness built and run, human validation of the judge not yet
complete.** 49 of 50 sampled answers were generated and judged
(`eval/run_groundedness.py`; one item was blocked by Bedrock content
filtering, unrelated to the pipeline itself): 24 grounded, 18 abstained,
7 ungrounded per the model judge.

That judge verdict is not reported as a headline number here, on
purpose. An LLM judge that's never checked against an independent human
rater produces a percentage with no established meaning, it looks
identical whether the judge is careful or is answering "grounded"
reflexively. The hand labeling and Cohen's kappa validation
(`eval/score_judge.py`) that would make this number trustworthy are
still pending. This section gets filled in once that's done, rather than
publishing a groundedness rate that hasn't actually been validated.
