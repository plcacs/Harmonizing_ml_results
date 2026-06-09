from typing import Any

# === Third-party dependency: botocore ===
# Used symbols: exceptions

# === Third-party dependency: botocore.exceptions ===
class ClientError(Exception): ...

# === Third-party dependency: botocore.loaders ===
def create_loader(search_path_string = ...) -> Any: ...

# === Third-party dependency: botocore.utils ===
def datetime2timestamp(dt, default_timezone = ...) -> Any: ...

# === Third-party dependency: botocore.vendored.requests ===
# Used symbols: ConnectionError

# === Unresolved dependency: botocore.vendored.requests.exceptions ===
# Used unresolved symbols: ReadTimeout

# === Internal dependency: chalice.constants ===
DEFAULT_STAGE_NAME = 'dev'
MAX_LAMBDA_DEPLOYMENT_SIZE = 50 * 1024 ** 2

# === Internal dependency: chalice.vendored.botocore.regions ===
class EndpointResolver(BaseEndpointResolver):
    def __init__(self, endpoint_data): ...

# === Third-party dependency: mypy_extensions ===
TypedDict: _TypedDictMeta