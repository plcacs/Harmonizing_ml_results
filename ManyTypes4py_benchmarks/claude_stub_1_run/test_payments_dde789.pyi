```pyi
from http import HTTPStatus
from typing import Any

DEFAULT_AMOUNT: str
DEFAULT_ID: str

def test_api_payments_target_error(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any,
) -> None: ...

def test_api_payments(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    deposit: Any,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_without_pfs(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    deposit: Any,
) -> None: ...

def test_api_payments_without_pfs_failure(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
) -> None: ...

def test_api_payments_secret_hash_errors(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_secret_no_hash(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_hash_no_secret(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_post_without_required_params(
    api_server_test_instance: Any,
    token_addresses: Any,
) -> None: ...

def test_api_payments_with_resolver(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_secret_and_hash(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_conflicts(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_lock_timeout(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_invalid_input(
    api_server_test_instance: Any,
    raiden_network: Any,
    token_addresses: Any,
    pfs_mock: Any,
) -> None: ...
```