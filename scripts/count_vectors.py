"""
Counts vectors actually present in the S3 Vectors index, by paginating
list_vectors rather than trusting CloudWatch log lines. Log lines say
what the ingest function believed it wrote; this says what is actually
stored, which is the only number worth trusting after a bulk load.

Usage:
    python scripts/count_vectors.py
"""

import argparse

import boto3

DEFAULT_INDEX_NAME = "documents-index"


def resolve_vector_bucket():
    """
    Derive the bucket name from the caller's account rather than
    hardcoding it, so this runs against whichever account is configured
    and keeps an account id out of a committed file.
    """
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"rag-portfolio-vectors-{account_id}"


def count_vectors(vector_bucket, index_name):
    client = boto3.client("s3vectors")
    count = 0
    next_token = None

    while True:
        kwargs = {
            "vectorBucketName": vector_bucket,
            "indexName": index_name,
            "maxResults": 500,
        }
        if next_token:
            kwargs["nextToken"] = next_token

        response = client.list_vectors(**kwargs)
        count += len(response.get("vectors", []))
        next_token = response.get("nextToken")

        if not next_token:
            break

    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-bucket", default=None)
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    args = parser.parse_args()

    vector_bucket = args.vector_bucket or resolve_vector_bucket()
    total = count_vectors(vector_bucket, args.index_name)
    print(f"Total vectors in {vector_bucket}/{args.index_name}: {total}")


if __name__ == "__main__":
    main()
