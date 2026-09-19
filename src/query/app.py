"""
Query Lambda: takes a question over HTTP, retrieves relevant chunks from
the S3 Vectors index, and generates an answer grounded in those chunks.

The retrieval and generation steps themselves live in retrieval.py and
generation.py, which the evaluation harness imports directly. This file
is the HTTP boundary and nothing else: parse the request, call the same
code the harness measures, shape the response. Keeping it that thin is
what lets the reported eval numbers describe this endpoint rather than a
parallel implementation that resembles it.
"""

import json
import os

from generation import generate_answer
from retrieval import embed_text, search_chunks

# Read at import time, on purpose. A misconfigured Lambda should fail
# loudly at cold start rather than limp along with a None and fail
# confusingly three calls later. The modules above resolve the same
# variables lazily so they stay importable for unit tests in CI, where
# no AWS configuration exists; this is the place that insists on them.
VECTOR_BUCKET_NAME = os.environ["VECTOR_BUCKET_NAME"]
VECTOR_INDEX_NAME = os.environ["VECTOR_INDEX_NAME"]
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
GENERATION_MODEL_ID = os.environ["GENERATION_MODEL_ID"]

TOP_K = 5


def handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
        question = body.get("question", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return _response(400, {"error": "Request body must be JSON with a 'question' field."})

    if not question:
        return _response(400, {"error": "Missing 'question' in request body."})

    query_vector = embed_text(question)
    retrieved = search_chunks(query_vector, top_k=TOP_K)

    if not retrieved:
        answer = "No documents have been ingested yet, so there's nothing to search."
    else:
        answer = generate_answer(question, retrieved)

    # Returning the retrieved chunks alongside the answer, not just the
    # answer text, is deliberate: the evaluation harness needs the actual
    # retrieved doc_ids and distances to score precision, recall, and
    # MRR against labelled queries. An API that only returns prose can't
    # be evaluated this way after the fact.
    result = {
        "question": question,
        "answer": answer,
        "retrieved": [
            {
                "doc_id": v["metadata"]["doc_id"],
                "chunk_index": v["metadata"]["chunk_index"],
                "distance": v.get("distance"),
            }
            for v in retrieved
        ],
    }
    print(json.dumps({"event": "query_answered", "question": question, "num_retrieved": len(retrieved)}))
    return _response(200, result)


def _response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body_dict),
    }
