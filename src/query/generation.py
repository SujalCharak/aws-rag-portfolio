"""
Answer generation, shared by the query Lambda and the evaluation harness.

Shared for the same reason retrieval is: the groundedness evaluation
measures whether generated answers are supported by their retrieved
context, and that verdict is only meaningful if the answers were
produced by the deployed prompt and model. An eval that quietly uses a
different system prompt is measuring a system nobody is running.

Generation uses converse(), Bedrock's unified chat API. Any Bedrock chat
model, Nova, Claude, Llama, speaks the same messages in, message out
shape through it. Swapping GENERATION_MODEL_ID from Nova Micro to a
Claude model is an environment variable change and no code change,
precisely because this call is unified while the embedding call in
retrieval.py is not.
"""

import os

from clients import bedrock_client

SYSTEM_PROMPT = (
    "Answer the question using only the context provided below. "
    "If the context does not contain the answer, say you don't know "
    "rather than guessing. Keep the answer to two or three sentences."
)


def build_context(retrieved):
    """
    Format retrieved chunks as numbered context blocks.

    The numbering is not decoration. It gives the model a way to
    attribute claims to specific passages, and it gives the
    groundedness judge the same labels to check those attributions
    against.
    """
    return "\n\n".join(
        f"[{i + 1}] {hit['metadata']['chunk_text']}"
        for i, hit in enumerate(retrieved)
    )


def generate_answer(question, retrieved, model_id=None):
    model_id = model_id or os.environ["GENERATION_MODEL_ID"]
    user_message = f"Context:\n{build_context(retrieved)}\n\nQuestion: {question}"

    response = bedrock_client().converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        # Temperature zero because this sits under an evaluation harness.
        # Sampling would mean a rerun produces different answers and
        # therefore different groundedness scores, making it impossible
        # to tell an actual regression from noise.
        inferenceConfig={"maxTokens": 512, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"]
