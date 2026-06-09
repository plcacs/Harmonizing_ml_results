from typing import Any

# === Third-party dependency: boto3 ===
def client(*args, **kwargs) -> Any: ...

# === Third-party dependency: botocore.response ===
class StreamingBody(IOBase):
    def __init__(self, raw_stream, content_length) -> Any: ...

# === Third-party dependency: moto ===
# Used symbols: mock_aws

# === Internal dependency: prefect ===
flow: Any

# === Third-party dependency: prefect_aws.lambda_function ===
class LambdaFunction(Block):
    ...

# === Third-party dependency: pydantic_core ===
# Used symbols: from_json, to_json

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark