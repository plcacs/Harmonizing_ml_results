from grequests import Response
from http import HTTPStatus
from raiden.raiden_service import RaidenService
from raiden.tests.utils.client import BurnEth
from raiden.tests.utils.detect_failure import RaiseOnFailure
from raiden.tests.utils.events import CheckDictNestedAttrs
from raiden.transfer.state import ChannelState
from raiden.utils.typing import List, TokenAmount
from raiden_contracts.constants import TEST_SETTLE_TIMEOUT_MIN, TEST_SETTLE_TIMEOUT_MAX

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_status_channel_nonexistant(api_server_test_instance, token_addresses: List[str]) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_open_and_deposit(api_server_test_instance, token_addresses: List[str], reveal_timeout: int) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_open_and_deposit_race(api_server_test_instance, raiden_network: List[RaidenService], token_addresses: List[str], token_network_registry_address: str, retry_timeout: float) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_open_close_and_settle(api_server_test_instance, token_addresses: List[str], reveal_timeout: int) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_close_insufficient_eth(api_server_test_instance, token_addresses: List[str], reveal_timeout: int) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_open_channel_invalid_input(api_server_test_instance, token_addresses: List[str], reveal_timeout: int) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_state_change_errors(api_server_test_instance, token_addresses: List[str], reveal_timeout: int) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('deposit', [1000])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_withdraw(api_server_test_instance, raiden_network: List[RaidenService], token_addresses: List[str], pfs_mock: PFSProxy) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('deposit', [1000])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_withdraw_with_offline_partner(api_server_test_instance, raiden_network: List[RaidenService], token_addresses: List[str]) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('deposit', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_set_reveal_timeout(api_server_test_instance, raiden_network: List[RaidenService], token_addresses: List[str], settle_timeout: int) -> None:
    ...

@RaiseOnFailure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('deposit', [DEPOSIT_FOR_TEST_API_DEPOSIT_LIMIT])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_channel_deposit_limit(api_server_test_instance, proxy_manager: ProxyManager, token_network_registry_address: str, token_addresses: List[str], reveal_timeout: int) -> None:
    ...