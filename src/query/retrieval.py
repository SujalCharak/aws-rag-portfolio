"""
Retrieval logic shared by the query Lambda and the evaluation harness.

This module exists so the harness measures the same code that serves
requests. If the harness reimplemented retrieval, the numbers in
RESULTS.md would describe the harness rather than the deployed system,
and the two would drift apart silently every time either side changed.
That gap is the first thing worth asking about any reported eval number,
so it is worth closing by construction rather than by discipline.
"""

import json
import os

from clients import bedrock_client, s3vectors_client

DEFAULT_TOP_K = 5
EMBEDDING_DIMENSIONS = 1024


def embed_text(text, model_id=None):
    """
    Embed a single string with the configured Bedrock embedding model.

    Embeddings go through invoke_model with a model specific JSON body
    rather than through converse(). Embedding models have no shared
    response shape across providers, so Bedrock offers no unified API
    for them, unlike chat models.
    """
    model_id = model_id or os.environ["EMBEDDING_MODEL_ID"]
    body = json.dumps({
        "inputText": text,
        "dimensions": EMBEDDING_DIMENSIONS,
        "normalize": True,
    })
    response = bedrock_client().invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    return payload["embedding"]


def search_chunks(query_vector, top_k=DEFAULT_TOP_K, vector_bucket=None, index_name=None):
    """
    Return the top_k nearest chunks, as stored. One element per chunk,
    not per document: a single document can occupy several of these
    slots. Callers that need document level results run the output
    through aggregate_to_documents below.
    """
    vector_bucket = vector_bucket or os.environ["VECTOR_BUCKET_NAME"]
    index_name = index_name or os.environ["VECTOR_INDEX_NAME"]

    response = s3vectors_client().query_vectors(
        vectorBucketName=vector_bucket,
        indexName=index_name,
        queryVector={"float32": query_vector},
        topK=top_k,
        returnMetadata=True,
        returnDistance=True,
    )
    return response.get("vectors", [])


def aggregate_to_documents(chunk_hits, k=10):
    """
    Collapse chunk level hits into a ranked list of document ids.

    This reconciles a mismatch that has to be resolved before anything
    can be scored: the index stores chunks, but relevance labels are per
    document. Retrieving 50 chunks might surface only 30 distinct
    documents, so "the top 10 results" is ambiguous until the rule for
    collapsing them is stated.

    The rule here is max pooling: a document ranks by its single best
    matching chunk. S3 Vectors returns cosine *distance*, where smaller
    is more similar, so the best chunk is the minimum distance one.

    Why max pooling rather than the alternatives:

    - Summing chunk scores rewards long documents simply for having more
      chunks and therefore more chances to match. That is a retrieval
      bug wearing a scoring choice as a disguise.
    - Averaging punishes a document whose one highly relevant passage
      sits among unrelated ones, which is the ordinary shape of a long
      document and exactly the case chunking exists to handle.

    On this corpus the choice barely moves the numbers, because SciFact
    abstracts are short and only about 12% of them split into more than
    one chunk. It still has to be made explicitly, and it would matter a
    great deal on longer documents.
    """
    best_distance = {}
    for hit in chunk_hits:
        metadata = hit.get("metadata") or {}
        doc_id = metadata.get("doc_id")
        distance = hit.get("distance")
        if doc_id is None or distance is None:
            continue
        if doc_id not in best_distance or distance < best_distance[doc_id]:
            best_distance[doc_id] = distance

    ranked = sorted(best_distance.items(), key=lambda item: item[1])
    return [doc_id for doc_id, _ in ranked[:k]]
