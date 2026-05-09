from http import HTTPStatus
from typing import List, Optional, Tuple, Union
from eth_utils import ChecksumAddress
from raiden.api.rest import APIServer
from raiden.raiden_service import RaidenService
from raiden.settings import PFS_mock
from raiden.tests.utils.transfer import watch_for_unlock_failures

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_target_error(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], pfs_mock: PFS_mock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], deposit: int, pfs_mock: PFS_mock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_without_pfs(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], deposit: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_without_pfs_failure(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_secret_hash_errors(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], pfs_mock: PFS_mock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_with_secret_no_hash(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], pfs_mock: PFS_mock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_with_hash_no_secret(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], pfs_mock: PFS_mock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_post_without_required_params(api_server_test_instance: APIServer, token_addresses: List[bytes]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('resolver_ports', [[None, 8000]])
@pytest.mark.parametrize('enable_rest_api', [True])
@pytest.mark.usefixtures('resolvers')
def test_api_payments_with_resolver(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], pfs_mock: PFS_mock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_with_secret_and_hash(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], pfs_mock: PFS_mock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_conflicts(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], pfs_mock: PFS_mock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
@pytest.mark.parametrize('deposit', [1000])
def test_api_payments_with_lock_timeout(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], pfs_mock: PFS_mock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_payments_with_invalid_input(api_server_test_instance: APIServer, raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[bytes], pfs_mock: PFS_mock) -> None:
    ...