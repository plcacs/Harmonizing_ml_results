from typing import Any, Dict, List, Optional, Tuple, Union
from http import HTTPStatus
import grequests
import pytest
from eth_utils import decode_hex, to_checksum_address, to_hex
from raiden.api.rest import APIServer
from raiden.constants import UINT64_MAX
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.tests.integration.api.rest.utils import (
    api_url_for,
    assert_payment_conflict,
    assert_payment_secret_and_hash,
    assert_proper_response,
    assert_response_with_error,
    get_json_response,
)
from raiden.tests.utils import factories
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.transfer import watch_for_unlock_failures
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.typing import Secret

DEFAULT_AMOUNT: str = ...
DEFAULT_ID: str = ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_target_error(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    deposit: int,
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_without_pfs(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    deposit: int,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_without_pfs_failure(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_secret_hash_errors(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_with_secret_no_hash(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_with_hash_no_secret(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_post_without_required_params(
    api_server_test_instance: APIServer,
    token_addresses: List[str],
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("resolver_ports", [[None, 8000]])
@pytest.mark.parametrize("enable_rest_api", [True])
@pytest.mark.usefixtures("resolvers")
def test_api_payments_with_resolver(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_with_secret_and_hash(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_conflicts(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
@pytest.mark.parametrize("deposit", [1000])
def test_api_payments_with_lock_timeout(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("enable_rest_api", [True])
def test_api_payments_with_invalid_input(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[str],
    pfs_mock: Any,
) -> None: ...