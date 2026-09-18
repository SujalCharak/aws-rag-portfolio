"""
pytest reads this file before collecting any test module. The Lambda
handlers read required config from os.environ at import time and raise
immediately if a variable is missing, that's deliberate, a Lambda that's
misconfigured should fail loud at cold start, not limp along with a None
and fail confusingly three calls later.

The cost of that fail-fast design is that even importing app.py for a
pure logic test (chunk_text, which touches no AWS config at all) requires
these variables to exist. Setting harmless placeholders here, once, for
the whole test session, keeps app.py's production behavior honest while
still letting the pure functions be tested in isolation.
"""

import os

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("VECTOR_BUCKET_NAME", "test-vector-bucket")
os.environ.setdefault("VECTOR_INDEX_NAME", "test-index")
os.environ.setdefault("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
os.environ.setdefault("GENERATION_MODEL_ID", "amazon.nova-micro-v1:0")
