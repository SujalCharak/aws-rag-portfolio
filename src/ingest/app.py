"""
Ingest Lambda: triggered whenever a file lands in the documents bucket.

Flow: read the uploaded file -> split it into overlapping chunks ->
embed each chunk with Bedrock -> write the vectors to the S3 Vectors index.

Design choices worth understanding, not just copying:

1. Chunking is word based with overlap. A fixed word count avoids splitting
   mid sentence too badly, and the overlap means a fact sitting right on a
   chunk boundary still appears whole in at least one chunk. This is a
   starting point, not the final answer. Day 3's eval harness is what
   actually tells you whether a different chunk size retrieves better.

2. Vector keys are deterministic: "{doc_id}#chunk_003", not a random UUID.
   S3 events can fire more than once for the same upload (S3 guarantees
   at least once delivery, not exactly once). A deterministic key means a
   duplicate event overwrites the same vector instead of creating a copy.
   That property, safe to run twice, is called idempotency, and it's the
   first thing an interviewer checks for in an event driven pipeline.

3. Bedrock model invocation uses raw JSON request and response bodies,
   not a Bedrock specific SDK method. Every foundation model on Bedrock
   speaks its own JSON shape, and invoke_model is the one operation that
   works for all of them. You pass in the model specific body, you get
   the model specific body back.
"""

import json
import os
import re
import uuid

import boto3

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")
s3vectors = boto3.client("s3vectors")

VECTOR_BUCKET_NAME = os.environ["VECTOR_BUCKET_NAME"]
VECTOR_INDEX_NAME = os.environ["VECTOR_INDEX_NAME"]
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]

CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split text into overlapping word windows.

    Word count is a proxy for token count, not exact, but close enough
    for chunk sizing and avoids adding a tokenizer dependency for this step.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def embed_text(text):
    """Call Bedrock Titan Text Embeddings V2 and return a float list.

    Titan V2 supports 256, 512, or 1024 dimensional output from one model,
    you choose at request time. We use 1024 for the best retrieval quality
    the model offers; the tradeoff against 256 or 512 is storage and query
    cost, which at this project's scale is a rounding error either way.
    """
    body = json.dumps({
        "inputText": text,
        "dimensions": 1024,
        "normalize": True,
    })
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    return payload["embedding"]


def clean_doc_id(key):
    """Turn an S3 object key into a safe, readable vector key prefix."""
    base = key.rsplit("/", 1)[-1]
    base = re.sub(r"\.[^.]+$", "", base)
    return re.sub(r"[^a-zA-Z0-9_-]", "_", base)


def handler(event, context):
    processed = []

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        obj = s3.get_object(Bucket=bucket, Key=key)
        raw_text = obj["Body"].read().decode("utf-8", errors="ignore")

        doc_id = clean_doc_id(key)
        chunks = chunk_text(raw_text)

        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk)
            vectors.append({
                "key": f"{doc_id}#chunk_{i:03d}",
                "data": {"float32": embedding},
                "metadata": {
                    "doc_id": doc_id,
                    "source_key": key,
                    "chunk_index": i,
                    # Non-filterable: retrievable alongside search results,
                    # but can't be used in a query filter. Declared as such
                    # in template.yaml's MetadataConfiguration.
                    "chunk_text": chunk,
                },
            })

        if vectors:
            # put_vectors upserts by key, so re-running this on the same
            # document overwrites the old vectors rather than duplicating
            # them. That's the idempotency property from point 2 above,
            # actually exercised.
            s3vectors.put_vectors(
                vectorBucketName=VECTOR_BUCKET_NAME,
                indexName=VECTOR_INDEX_NAME,
                vectors=vectors,
            )

        processed.append({"doc_id": doc_id, "chunks_written": len(vectors)})
        print(json.dumps({
            "event": "document_ingested",
            "doc_id": doc_id,
            "source_key": key,
            "chunks_written": len(vectors),
        }))

    return {"statusCode": 200, "body": json.dumps({"processed": processed})}
