from http import HTTPStatus
from typing import Any, Dict, List, Tuple, Union
from raiden.api.rest import APIServer
from raiden.raiden_service import RaidenService
from raiden.utils.typing import Secret

DEFAULT_AMOUNT: str = '200'
DEFAULT_ID: str = '42'

def test_api_payments_target_error(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

def test_api_payments(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    deposit: int,
    pfs_mock: Any,
) -> None: ...

def test_api_payments_without_pfs(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    deposit: int,
) -> None: ...

def test_api_payments_without_pfs_failure(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
) -> None: ...

def test_api_payments_secret_hash_errors(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_secret_no_hash(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_hash_no_secret(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_post_without_required_params(
    api_server_test_instance: APIServer,
    token_addresses: List[str],
) -> None: ...

def test_api_payments_with_resolver(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_secret_and_hash(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_conflicts(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_lock_timeout(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

def test_api_payments_with_invalid_input(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...