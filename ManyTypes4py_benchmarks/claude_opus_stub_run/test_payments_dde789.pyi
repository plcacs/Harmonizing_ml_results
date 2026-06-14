from typing import Any

from raiden.api.rest import APIServer
from raiden.raiden_service import RaidenService
from raiden.utils.typing import List

DEFAULT_AMOUNT: str
DEFAULT_ID: str

def test_api_payments_target_error(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    pfs_mock: Any,
) -> None: ...

def test_api_payments(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    deposit: int,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_without_pfs(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    deposit: int,
) -> None: ...

def test_api_payments_without_pfs_failure(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
) -> None: ...

def test_api_payments_secret_hash_errors(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_secret_no_hash(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_hash_no_secret(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_post_without_required_params(
    api_server_test_instance: APIServer,
    token_addresses: List[bytes],
) -> None: ...

def test_api_payments_with_resolver(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_secret_and_hash(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_conflicts(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_lock_timeout(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_invalid_input(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[bytes],
    pfs_mock: Any,
) -> None: ...