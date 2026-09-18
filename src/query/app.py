"""
Query Lambda: takes a question over HTTP, retrieves relevant chunks from
the S3 Vectors index, and generates an answer grounded in those chunks.

Two different Bedrock call styles show up here on purpose, and the
difference between them is worth knowing:

- Embeddings use invoke_model with a raw, model specific JSON body.
  Embedding models don't have a shared response shape across providers,
  so there's no unified API for them.
- Generation uses converse(), Bedrock's unified chat API. Any Bedrock
  chat model, Nova, Claude, Llama, whatever, speaks the same
  messages in, message out shape through converse(). Swapping
  GENERATION_MODEL_ID from Nova Micro to a Claude model later is a one
  line environment variable change, no code change, precisely because
  this call is unified and the embedding call is not.
"""

import json
import os

import boto3

bedrock = boto3.client("bedrock-runtime")
s3vectors = boto3.client("s3vectors")

VECTOR_BUCKET_NAME = os.environ["VECTOR_BUCKET_NAME"]
VECTOR_INDEX_NAME = os.environ["VECTOR_INDEX_NAME"]
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
GENERATION_MODEL_ID = os.environ["GENERATION_MODEL_ID"]

TOP_K = 5

SYSTEM_PROMPT = (
    "Answer the question using only the context provided below. "
    "If the context does not contain the answer, say you don't know "
    "rather than guessing. Keep the answer to two or three sentences."
)


def embed_text(text):
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


def retrieve(query_vector, top_k=TOP_K):
    response = s3vectors.query_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=VECTOR_INDEX_NAME,
        queryVector={"float32": query_vector},
        topK=top_k,
        returnMetadata=True,
        returnDistance=True,
    )
    return response.get("vectors", [])


def generate_answer(question, retrieved):
    context = "\n\n".join(
        f"[{i+1}] {v['metadata']['chunk_text']}"
        for i, v in enumerate(retrieved)
    )
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    response = bedrock.converse(
        modelId=GENERATION_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"]


def handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
        question = body.get("question", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return _response(400, {"error": "Request body must be JSON with a 'question' field."})

    if not question:
        return _response(400, {"error": "Missing 'question' in request body."})

    query_vector = embed_text(question)
    retrieved = retrieve(query_vector)

    if not retrieved:
        answer = "No documents have been ingested yet, so there's nothing to search."
    else:
        answer = generate_answer(question, retrieved)

    # Returning the retrieved chunks alongside the answer, not just the
    # answer text, is deliberate: the day 3 evaluation harness needs the
    # actual retrieved doc_ids and distances to score precision, recall,
    # and MRR against labelled queries. An API that only returns prose
    # can't be evaluated this way after the fact.
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
