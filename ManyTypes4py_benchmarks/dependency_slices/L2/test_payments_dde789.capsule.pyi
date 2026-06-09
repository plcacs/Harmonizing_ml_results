from typing import Any

# === Third-party dependency: eth_utils ===
# Used symbols: decode_hex, to_checksum_address, to_hex

# === Third-party dependency: grequests ===
def map(requests, stream = ..., size = ..., exception_handler = ..., gtimeout = ...) -> Any: ...
get: partial
post: partial

# === Third-party dependency: pytest ===
# Used symbols: mark

# === Internal dependency: raiden.constants ===
UINT64_MAX: Any

# === Internal dependency: raiden.settings ===
DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS: BlockTimeout

# === Internal dependency: raiden.tests.integration.api.rest.utils ===
def get_json_response(response) -> Any: ...
def assert_response_with_error(response, status_code) -> Any: ...
def assert_proper_response(response, status_code = ...) -> Any: ...
def assert_payment_secret_and_hash(response, payment) -> Any: ...
def assert_payment_conflict(responses) -> Any: ...
def api_url_for(api_server, endpoint, **kwargs) -> Any: ...

# === Internal dependency: raiden.tests.utils.detect_failure ===
def raise_on_failure(test_function: Callable) -> Callable: ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_secret(i: int = ...) -> Secret: ...
def make_secret_hash(i: int = ...) -> SecretHash: ...
def make_secret_with_hash(i: int = ...) -> Tuple[Secret, SecretHash]: ...

# === Internal dependency: raiden.tests.utils.transfer ===
def watch_for_unlock_failures(*apps) -> Any: ...

# === Internal dependency: raiden.utils.secrethash ===
def sha256_secrethash(secret: Secret) -> SecretHash: ...

# === Internal dependency: raiden.utils.typing ===
BlockTimeout: NewType
Secret: NewType