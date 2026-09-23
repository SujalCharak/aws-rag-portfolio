# Document Retrieval Pipeline with Evaluation Harness

A document ingestion and retrieval pipeline on AWS: S3, Lambda, S3 Vectors,
and Bedrock, with an evaluation harness on top that measures retrieval and
generation quality directly, precision, recall, MRR, and groundedness,
rather than assuming a plausible looking answer means the system works.

Status: core pipeline built and verified end to end, including CI/CD
via GitHub Actions with OIDC. Retrieval evaluated against SciFact (300
queries): hybrid retrieval reaches 0.706 nDCG@10, ahead of both a dense
only and a BM25 only baseline. Groundedness judge validated against
independent hand labels: Cohen's kappa 0.693 (substantial agreement).
Full numbers and caveats in [RESULTS.md](RESULTS.md).

The AWS stack itself is currently torn down (`sam delete`) between demos
to avoid an idle unauthenticated endpoint and to keep cost at zero. See
Deploying below to bring it back up.

## Architecture

```
S3 upload --> Ingest Lambda --> chunk + embed (Bedrock Titan V2) --> S3 Vectors index
                                                                            |
HTTP POST /query --> Query Lambda --> embed query --> search index --------+
                                          |
                                    generate answer (Bedrock Nova Micro)
```

Everything above is declared in `template.yaml` and deployed as one
CloudFormation stack via AWS SAM. No console clicking for any of it.

## Why these choices

- **S3 Vectors, not OpenSearch Serverless.** OpenSearch Serverless bills
  for compute by the hour even when idle, easy to burn through free tier
  credits by accident. S3 Vectors is pay per storage and query, effectively
  free at this project's scale (low cents/month for the corpus size here).
  Tradeoff: no hybrid search, no HNSW tuning, addressed by running a BM25
  baseline in the eval harness (day 3) rather than pretending vector search
  alone is sufficient.
- **Hand-rolled chunking and embedding, not Bedrock Knowledge Bases.**
  Knowledge Bases would do this in a few console clicks, but that hides the
  exact decisions (chunk size, overlap) the eval harness exists to measure.
- **Nova Micro for generation, not Claude.** Same setup cost either way
  since Bedrock now auto-enables serverless models on first use, Nova Micro
  chosen for lower cost during iteration. Swapping to a Claude model later
  is a one line parameter change (see `GenerationModelId` in template.yaml),
  since generation calls go through Bedrock's unified `converse()` API.

## Repo layout

```
template.yaml                 SAM template: all application infrastructure
ci/github-oidc.yaml           Bootstrap stack: GitHub Actions deploy role
.github/workflows/ci.yml      Tests and lint on every push, deploy from main
src/ingest/app.py             Lambda: parse, chunk, embed, write vectors
src/query/app.py              Lambda: HTTP boundary for the query path
src/query/retrieval.py        Embed, search, collapse chunks to documents
src/query/generation.py       Prompt and answer generation
eval/metrics.py               precision@k, recall@k, MRR, nDCG@k
eval/run_retrieval_eval.py    Dense retrieval over the labelled query set
eval/run_bm25_baseline.py     Lexical baseline over the same corpus
eval/fuse_runs.py             Reciprocal Rank Fusion of two runs
eval/score_runs.py            Scores every run through one scorer
eval/run_groundedness.py      Generate, judge, and build labelling sheets
eval/score_judge.py           Cohen's kappa: judge against hand labels
scripts/prepare_scifact.py    Download and shard the evaluation corpus
scripts/count_vectors.py      Count vectors actually stored in the index
tests/                        Unit tests for pure logic (no AWS calls)
```

## Loading a corpus

Documents are ingested as JSONL shards rather than one file per
document. One S3 object carries many documents, so one upload is one
Lambda invocation instead of thousands. Uncapped, per document
invocations would scale Lambda to whatever the account allows and
throttle the shared Bedrock embedding quota into mass failure; sharding
plus a reserved concurrency limit makes the load survivable.

```bash
pip install ir_datasets
python scripts/prepare_scifact.py          # writes data/corpus and data/eval
aws s3 cp data/corpus/ s3://<documents-bucket>/corpus/ --recursive
```

Failed ingests land in an SQS dead letter queue rather than disappearing.
S3 invokes Lambda asynchronously, so without one, a failed document is
retried twice and then silently dropped, and a retrieval system quietly
missing part of its corpus still returns plausible answers. Check the
queue depth after a load:

```bash
aws sqs get-queue-attributes \
  --queue-url <IngestDeadLetterQueueUrl from stack outputs> \
  --attribute-names ApproximateNumberOfMessages
```

A non-zero DLQ count after a load doesn't necessarily mean documents are
missing, it depends on whether the corpus made it in. On the SciFact load,
the account's Lambda concurrency limit (10, a new account default well
below the usual 1000) meant 21 near simultaneous shard uploads had to
compete for 10 execution slots. Some invocation attempts sat throttled
long enough to exceed the retry budget (`EventInvokeConfig`: 2 retries, 1
hour max age) and landed in the DLQ, having never actually run. S3's
at-least-once delivery meant a second delivery of the same event usually
found a slot and succeeded, which is why every shard still showed up in
the ingest logs. The fix in that case isn't retrying harder, it's raising
the account's concurrency limit (Service Quotas) or reserving a smaller
`IngestConcurrencyLimit` so fewer invocations contend at once. Either way,
the right way to confirm a load actually succeeded is a direct vector
count against the index (`scripts/count_vectors.py`), not DLQ depth alone.

## CI/CD

GitHub Actions runs the unit tests, `sam validate --lint`, and `sam
build` on every push and pull request, then deploys from `main`.

Deploys authenticate through GitHub's OIDC provider, so no AWS
credentials are stored in GitHub. The workflow requests a short lived
token describing the run, AWS validates it against a trust policy scoped
to this exact repository and ref, and returns credentials that expire in
an hour. Setup is a one time bootstrap stack:

```bash
aws cloudformation deploy \
  --template-file ci/github-oidc.yaml \
  --stack-name rag-portfolio-ci \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides GitHubOrg=<your-github-username>
```

Then set the resulting role ARN as an Actions variable named
`AWS_DEPLOY_ROLE_ARN`.

## Deploying

Requires the AWS CLI and SAM CLI installed, and an IAM user (not root)
with credentials configured locally.

```bash
aws configure                  # paste your IAM user's access key and secret
sam build                      # packages code + dependencies
sam deploy --guided            # first deploy: asks stack name, region, confirms changes
```

`sam deploy --guided` writes your answers to `samconfig.toml` (gitignored,
account specific). After the first guided run, plain `sam deploy` reuses it.

## Evaluation

Results and their caveats are in [RESULTS.md](RESULTS.md). This section
covers how to reproduce them.

Retrieval quality is measured against SciFact's human relevance
judgments rather than inferred from answers looking plausible.

```bash
python eval/run_retrieval_eval.py      # dense retrieval, caches raw hits
python eval/run_bm25_baseline.py       # lexical baseline, same corpus
python eval/fuse_runs.py               # reciprocal rank fusion of the two
python eval/score_runs.py              # scores every run, prints the table
```

Each retrieval system writes a run file and `score_runs.py` scores all
of them through one code path, so the baseline is a comparison rather
than a second experiment. The retrieval code itself is imported from
`src/query/`, the same modules the deployed Lambda runs, so the numbers
describe the deployed system rather than a reimplementation of it.

Three things about these numbers are worth stating before reading them:

- **Precision at depth is capped by the dataset.** SciFact averages 1.13
  relevant documents per query, so P@10 cannot exceed about 0.11 no
  matter how good retrieval is. MRR and recall carry the signal here.
- **BM25 is a serious baseline on this corpus**, not a straw man.
  Published BEIR results put it around 0.665 nDCG@10 on SciFact, ahead
  of many dense retrievers, because scientific claims reuse the exact
  terminology of the abstracts supporting them.
- **The hybrid row is evaluation time fusion**, combining two separately
  produced run files. It measures whether the signals complement each
  other. It is not a hybrid retriever serving live traffic.

### Groundedness

Whether generated answers are actually supported by their retrieved
context, with the judge itself validated rather than trusted.

```bash
python eval/run_groundedness.py --sample-size 50
# fill in data/eval/labels.csv by hand, reading data/eval/label_sheet.md
python eval/score_judge.py
```

An LLM judge that is never checked against a human produces a percentage
with no established meaning, and it looks the same whether the judge is
careful or is answering "grounded" reflexively. So the same sample is
labelled by hand and the two sets of labels are compared with Cohen's
kappa, which corrects for agreement that would happen by chance. A judge
that always says "grounded" and is right 80% of the time scores 0.

Three details keep that comparison honest. The judge is a different
model from the one generating the answers, so it is not grading itself.
The labelling sheet omits the judge's verdicts, so reading them cannot
anchor the human labels. And both sides are given the identical rubric,
defined once in `eval/groundedness.py`, so the score measures the judge
rather than a difference in instructions.

Abstentions are counted separately rather than as grounded. An answer
that declines to answer asserts nothing unsupported and so is trivially
grounded, which would let a system that always says "I don't know"
report a perfect score.

## Cost

Designed to stay inside AWS's $100-200 free plan credit. Embedding and
generation costs are fractions of a cent per document at this corpus size.
The one component that can run up a real bill if misconfigured is a vector
store billed by idle compute, which is exactly why S3 Vectors was chosen
over OpenSearch Serverless.
