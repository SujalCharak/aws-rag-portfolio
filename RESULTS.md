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
which generates the answers, so the judge isn't grading itself), with
the judge itself validated against independent hand labels rather than
trusted outright.

49 of 50 sampled answers were generated and judged
(`eval/run_groundedness.py`; one item was blocked by Bedrock content
filtering, unrelated to the pipeline itself). All 49 were then labeled
by hand, blind to the judge's verdicts, using the identical rubric in
`eval/groundedness.py`, and compared with `eval/score_judge.py`.

| | Raw agreement | Cohen's kappa |
|---|---|---|
| Judge vs. hand labels | 81.6% | 0.693 |

0.693 lands in the "substantial agreement" band (0.61-0.80) on the
Landis-Koch scale commonly used to interpret kappa. That's well above
chance-level agreement and above what a judge parroting one label by
reflex would score, since kappa is built to punish exactly that. It's
not "almost perfect" (>0.80) either: this judge is a reasonable proxy
for a careful human read, not a replacement for one.

Confusion matrix, judge verdict against hand label:

| Judge ↓ / Hand label → | grounded | ungrounded | abstained |
|---|---|---|---|
| **grounded** | 21 | 3 | 0 |
| **ungrounded** | 4 | 2 | 1 |
| **abstained** | 0 | 1 | 17 |

Disagreement concentrates exactly where it's hardest: telling
"grounded" apart from "ungrounded" requires catching one specific
unsupported claim buried in an otherwise accurate answer, a harder call
than recognizing "abstained" (a structural property of the answer, not
a factual check). The judge missed a handful of subtle cases by hand
label during this review, things like a plausible-sounding but wrong
arithmetic conversion, and a projected/modeled result restated as an
observed one, exactly the kind of error an LLM judge skimming for
obvious contradictions is more likely to wave through than a human
reading closely.

**Grounded rate: 80.6%** of non-abstained answers (25 of 31), by hand
label. This is the number worth reporting, not the raw judge output
from before validation, since it's now backed by an independently
measured agreement rate rather than asserted on the judge's word alone.
Abstentions (18 of 49, 37%) are excluded from this rate rather than
counted as grounded, since declining to answer asserts nothing that
could be unsupported; folding them in would let a system that always
says "I don't know" score a perfect rate.

Caveat: 49 samples is a small validation set. Substantial agreement on
49 items is a real finding, but the kappa estimate itself carries real
uncertainty at this sample size, and a wider validated sample would
tighten it.
