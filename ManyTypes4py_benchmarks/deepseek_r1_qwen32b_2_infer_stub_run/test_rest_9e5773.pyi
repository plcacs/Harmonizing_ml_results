from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from eth_utils import ChecksumAddress as TokenAddress, ChecksumAddress as TargetAddress
from grequests import Response
from http import HTTPStatus
from raiden.api.rest import APIServer
from raiden.raiden_service import RaidenService
from raiden.settings import BlockNumber, PaymentAmount, PaymentID
from raiden.tests.integration.fixtures.smartcontracts import RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.transfer.state import ChannelState
from raiden.utils.typing import FeeAmount
from unittest.mock import Mock

class CustomException(Exception):
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_payload_with_invalid_addresses(api_server_test_instance: APIServer) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_payload_with_address_invalid_chars(api_server_test_instance: APIServer) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_payload_with_address_invalid_length(api_server_test_instance: APIServer) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_payload_with_address_not_eip55(api_server_test_instance: APIServer) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_query_our_address(api_server_test_instance: APIServer) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_get_raiden_version(api_server_test_instance: APIServer) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_get_node_settings(api_server_test_instance: APIServer) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_get_contract_infos(api_server_test_instance: APIServer) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_get_channel_list(api_server_test_instance: APIServer, token_addresses: List[TokenAddress], reveal_timeout: int) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('number_of_tokens', [2])
@pytest.mark.parametrize('environment_type', [Environment.PRODUCTION])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_tokens(api_server_test_instance: APIServer, blockchain_services: BlockchainServices, token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_query_partners_by_token(api_server_test_instance: APIServer, blockchain_services: BlockchainServices, token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_timestamp_format(api_server_test_instance: APIServer, raiden_network: List[RaidenService], token_addresses: List[TokenAddress], pfs_mock: PFSMock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_tokens', [0])
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('environment_type', [Environment.DEVELOPMENT])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_get_token_network_for_token(api_server_test_instance: APIServer, token_amount: int, token_addresses: List[TokenAddress], raiden_network: List[RaidenService], contract_manager: Any, retry_timeout: float, unregistered_token: str) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('number_of_tokens', [1])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_get_connections_info(raiden_network: List[RaidenService], api_server_test_instance: APIServer, token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [3])
@pytest.mark.parametrize('enable_rest_api', [True])
@pytest.mark.parametrize('number_of_tokens', [2])
def test_payment_events_endpoints(api_server_test_instance: APIServer, raiden_network: List[RaidenService], token_addresses: List[TokenAddress], pfs_mock: PFSMock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_channel_events_raiden(api_server_test_instance: APIServer, raiden_network: List[RaidenService], token_addresses: List[TokenAddress], pfs_mock: PFSMock) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [3])
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('enable_rest_api', [True])
@patch('raiden.message_handler.decrypt_secret', side_effect=InvalidSecret)
def test_pending_transfers_endpoint(decrypt_patch: Mock, raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('number_of_tokens', [1])
@pytest.mark.parametrize('token_contract_name', [CONTRACT_CUSTOM_TOKEN])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_testnet_token_mint(api_server_test_instance: APIServer, token_addresses: List[TokenAddress]) -> None:
    ...

@raise_on_failure
@pytest.mark.skip(reason='Skipped for now, please re-enable later')
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_udc_api(api_server_test_instance: APIServer, retry_timeout: float) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_udc_api_with_invalid_parameters(api_server_test_instance: APIServer) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
@pytest.mark.parametrize('user_deposit_address', [None])
def test_no_udc_configured(api_server_test_instance: APIServer, retry_timeout: float) -> None:
    ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_shutdown(api_server_test_instance: APIServer) -> None:
    ...