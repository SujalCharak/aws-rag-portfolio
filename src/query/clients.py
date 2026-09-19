"""
Lazily constructed boto3 clients, shared by the retrieval and generation
modules.

Two reasons this is its own module rather than a client per file.

One client per service, reused. Boto3 clients are relatively expensive
to build and are safe to reuse, which is why the Lambda idiom is to
create them once at module scope and let them live across warm
invocations. Retrieval and generation both call Bedrock, and they should
share one client rather than each holding their own.

Lazily, not at import time. The pure functions in retrieval.py are unit
tested in CI, where no AWS credentials or region exist. A boto3.client
call at module scope raises NoRegionError on import in that environment
and takes the whole test run down with it, so construction is deferred
to first use.
"""

import boto3

_bedrock = None
_s3vectors = None


def bedrock_client():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime")
    return _bedrock


def s3vectors_client():
    global _s3vectors
    if _s3vectors is None:
        _s3vectors = boto3.client("s3vectors")
    return _s3vectors
