# Document Retrieval Pipeline with Evaluation Harness

A document ingestion and retrieval pipeline on AWS: S3, Lambda, S3 Vectors,
and Bedrock, with an evaluation harness on top that measures retrieval and
generation quality directly, precision, recall, MRR, and groundedness,
rather than assuming a plausible looking answer means the system works.

Status: core pipeline deployed and verified end to end (ingest and query
both working against a live stack). SciFact corpus loaded (5183 documents,
5820 chunks indexed). Evaluation harness in progress.

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
src/query/app.py              Lambda: retrieve, generate, return with citations
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

Not yet built. Day 3 adds: a labelled query set against a public corpus
(SciFact), precision at k / recall at k / MRR / nDCG, a BM25 baseline for
comparison, and groundedness checking on generated answers with the
judge itself validated against hand labels.

## Cost

Designed to stay inside AWS's $100-200 free plan credit. Embedding and
generation costs are fractions of a cent per document at this corpus size.
The one component that can run up a real bill if misconfigured is a vector
store billed by idle compute, which is exactly why S3 Vectors was chosen
over OpenSearch Serverless.
