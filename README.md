# Document Retrieval Pipeline with Evaluation Harness

A document ingestion and retrieval pipeline on AWS: S3, Lambda, S3 Vectors,
and Bedrock, with an evaluation harness on top that measures retrieval and
generation quality directly, precision, recall, MRR, and groundedness,
rather than assuming a plausible looking answer means the system works.

Status: core pipeline deployed and verified end to end (ingest and query
both working against a live stack). Evaluation harness in progress.

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
template.yaml           SAM/CloudFormation template: all infrastructure
src/ingest/app.py       Lambda: chunk, embed, write to vector index
src/query/app.py        Lambda: retrieve, generate, return with citations
tests/                  Unit tests for pure logic (no AWS calls)
```

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
