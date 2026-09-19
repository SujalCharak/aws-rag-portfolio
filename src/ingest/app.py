"""
Ingest Lambda: triggered whenever a file lands in the documents bucket.

Flow: read the uploaded file -> split each document into overlapping
chunks -> embed each chunk with Bedrock -> write the vectors to the
S3 Vectors index in batches.

Two input formats are supported, decided by file extension:

- .jsonl  One JSON document per line: {"doc_id", "title", "text"}.
          This is the format bulk loads use. One S3 object carries many
          documents, so one upload is one Lambda invocation rather than
          thousands.
- anything else
          The whole file is treated as a single plain text document,
          which keeps ad hoc single file testing easy.

Why the JSONL shape matters, since it drives most of this file:

Uploading one object per document would mean one S3 event and one Lambda
invocation per document. At corpus scale that is thousands of concurrent
invocations, each hammering the same Bedrock endpoint, which has a
requests per minute quota far below what Lambda will happily generate.
The load throttles itself into mass failure. Batching documents into
shards, and capping Lambda concurrency in template.yaml, turns that into
a slower load that actually finishes. It is also just how batch data
pipelines are normally shaped: files of records, not a file per record.

Other design points:

- Vector keys are deterministic: "{doc_id}#chunk_003", never a random
  UUID. S3 delivers events at least once, not exactly once, and failed
  async invocations are retried. A deterministic key means a duplicate
  delivery overwrites the same vector instead of creating a second copy.
  Safe to run twice is called idempotency, and it is the first thing
  worth checking in any event driven pipeline.

- The Bedrock client uses adaptive retry mode. Standard retries back off
  on a fixed schedule; adaptive mode additionally rate limits the client
  itself when it starts seeing throttles, which is the correct behavior
  when many Lambdas share one account level quota.

- put_vectors is batched. The API accepts up to 500 vectors and 20 MiB
  per request; this uses 200, which keeps payloads comfortably inside
  the size limit while cutting request count by two orders of magnitude
  versus writing one vector at a time.
"""

import json
import os
import re

import boto3
from botocore.config import Config

# Adaptive mode adds client side rate limiting on top of retries, which
# is what you want when concurrent Lambdas share one Bedrock quota.
# Without this, throttled calls raise immediately and the whole shard
# fails, sending a few hundred perfectly good documents to the DLQ.
_bedrock_config = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    read_timeout=30,
)

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime", config=_bedrock_config)
s3vectors = boto3.client("s3vectors")

VECTOR_BUCKET_NAME = os.environ["VECTOR_BUCKET_NAME"]
VECTOR_INDEX_NAME = os.environ["VECTOR_INDEX_NAME"]
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]

CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50

# API maximum is 500 vectors / 20 MiB per request. At 1024 dimensions a
# vector serializes to roughly 12 KB, plus up to a couple KB of chunk
# text in metadata, so 200 lands near 3 MB per request: well clear of
# the ceiling, with no meaningful gain from pushing closer to it.
PUT_VECTORS_BATCH_SIZE = 200


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split text into overlapping word windows.

    Word count stands in for token count here. It is not exact, but it is
    close enough for sizing chunks and avoids shipping a tokenizer into
    the deployment package for this one purpose. The overlap means a fact
    sitting on a chunk boundary still appears whole in at least one chunk.

    Chunk size and overlap are the two knobs the evaluation harness exists
    to measure. Treat these numbers as a starting point, not a finding.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def clean_doc_id(value):
    """Normalize an identifier into something safe to use in a vector key."""
    base = value.rsplit("/", 1)[-1]
    base = re.sub(r"\.[^.]+$", "", base)
    return re.sub(r"[^a-zA-Z0-9_-]", "_", base)


def parse_documents(key, raw_text):
    """Turn a raw S3 object into a list of (doc_id, text) pairs.

    JSONL files yield many documents, anything else yields exactly one.
    Blank lines are skipped. A malformed JSON line raises, which fails
    the whole shard and sends it to the DLQ rather than silently
    importing a partial corpus. For a retrieval system, quietly missing
    documents is worse than a loud failure: the system still answers,
    just wrongly, and the evaluation numbers look fine.
    """
    if not key.lower().endswith(".jsonl"):
        return [(clean_doc_id(key), raw_text)]

    documents = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} line {line_number} is not valid JSON") from exc

        doc_id = clean_doc_id(str(record.get("doc_id") or f"line_{line_number}"))
        # Title and body are concatenated before chunking. For abstracts
        # the title carries real signal and is short enough that keeping
        # it with the text costs nothing.
        title = (record.get("title") or "").strip()
        body = (record.get("text") or "").strip()
        text = f"{title}\n\n{body}".strip() if title else body
        if text:
            documents.append((doc_id, text))
    return documents


def embed_text(text):
    """Call Bedrock Titan Text Embeddings V2 and return a float list.

    Titan V2 emits 256, 512, or 1024 dimensions from the same model,
    chosen per request. 1024 is the highest quality it offers; the
    tradeoff against smaller vectors is storage and query cost, which at
    this corpus size is a rounding error either way.
    """
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def write_vectors(vectors):
    """Write vectors to the index in batches, returning the count written."""
    written = 0
    for start in range(0, len(vectors), PUT_VECTORS_BATCH_SIZE):
        batch = vectors[start:start + PUT_VECTORS_BATCH_SIZE]
        # put_vectors upserts by key, so a replayed event overwrites
        # rather than duplicating. That is the idempotency property from
        # the module docstring, actually exercised.
        s3vectors.put_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=VECTOR_INDEX_NAME,
            vectors=batch,
        )
        written += len(batch)
    return written


def build_vectors(doc_id, text, source_key):
    """Chunk one document, embed each chunk, return vector records."""
    vectors = []
    for i, chunk in enumerate(chunk_text(text)):
        vectors.append({
            "key": f"{doc_id}#chunk_{i:03d}",
            "data": {"float32": embed_text(chunk)},
            "metadata": {
                "doc_id": doc_id,
                "source_key": source_key,
                "chunk_index": i,
                # Non-filterable: returned with search results but not
                # usable in a query filter. Declared as non-filterable in
                # template.yaml. Storing the text here means the query
                # path never has to go back to S3 for it.
                "chunk_text": chunk,
            },
        })
    return vectors


def handler(event, context):
    results = []

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        obj = s3.get_object(Bucket=bucket, Key=key)
        raw_text = obj["Body"].read().decode("utf-8", errors="ignore")

        documents = parse_documents(key, raw_text)

        vectors = []
        for doc_id, text in documents:
            vectors.extend(build_vectors(doc_id, text, key))

        written = write_vectors(vectors)

        result = {
            "event": "shard_ingested",
            "source_key": key,
            "documents": len(documents),
            "chunks_written": written,
        }
        results.append(result)
        # One structured line per shard. JSON rather than free text so
        # CloudWatch Logs Insights can query these directly, which is how
        # you answer "how many documents landed" without reading logs by
        # eye.
        print(json.dumps(result))

    return {"statusCode": 200, "body": json.dumps({"results": results})}
