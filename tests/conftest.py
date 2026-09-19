"""
pytest reads this file before collecting any test module. The Lambda
handlers read required config from os.environ at import time and raise
immediately if a variable is missing, that's deliberate, a Lambda that's
misconfigured should fail loud at cold start, not limp along with a None
and fail confusingly three calls later.

The cost of that fail-fast design is that even importing a handler for a
pure logic test requires these variables to exist. Setting harmless
placeholders here, once, for the whole test session, keeps the handlers'
production behavior honest while still letting the pure functions be
tested in isolation.
"""

import importlib.util
import os

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("VECTOR_BUCKET_NAME", "test-vector-bucket")
os.environ.setdefault("VECTOR_INDEX_NAME", "test-index")
os.environ.setdefault("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
os.environ.setdefault("GENERATION_MODEL_ID", "amazon.nova-micro-v1:0")


def load_lambda_module(module_name, *path_parts):
    """
    Load a Lambda handler from an explicit file path, under a name of
    our choosing.

    Both handlers are called app.py, which is the SAM convention and
    fine in production, where each function is packaged alone. In a test
    session they collide: sys.modules keys on the module name, so
    whichever `import app` runs first wins and every later one silently
    receives the wrong module. That failure is particularly nasty
    because both handlers export a callable named `handler`, so a test
    asserting the query handler imports cleanly will pass while actually
    holding the ingest handler.

    Loading by path under distinct names removes the collision, and with
    it the dependence on test collection order.
    """
    path = os.path.join(os.path.dirname(__file__), "..", *path_parts)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
