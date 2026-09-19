"""
Counts vectors actually present in the S3 Vectors index, by paginating
list_vectors rather than trusting CloudWatch log lines. Log lines say what
the Lambda claimed to write; this says what's actually stored, which is
the only number worth trusting after a bulk load.

Usage:
    python scripts/count_vectors.py
"""

import boto3

VECTOR_BUCKET_NAME = "rag-portfolio-vectors-075818203465"
INDEX_NAME = "documents-index"


def main():
    client = boto3.client("s3vectors")
    count = 0
    next_token = None

    while True:
        kwargs = {
            "vectorBucketName": VECTOR_BUCKET_NAME,
            "indexName": INDEX_NAME,
            "maxResults": 500,
        }
        if next_token:
            kwargs["nextToken"] = next_token

        resp = client.list_vectors(**kwargs)
        batch = resp.get("vectors", [])
        count += len(batch)
        next_token = resp.get("nextToken")

        if not next_token:
            break

    print(f"Total vectors in index: {count}")


if __name__ == "__main__":
    main()
