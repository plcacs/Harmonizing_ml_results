import datetime
import json
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Set, Union

from eth_utils import ChecksumAddress
from flask import Response
from grequests import Request, Response as GResponse
from pytest import Mark
from raiden.api.rest import APIServer
from raiden.settings import BlockNumber, FeeAmount, PaymentAmount, PaymentID, TargetAddress, TokenAddress
from raiden.tests.integration.fixtures.smartcontracts import RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT
from raiden.transfer.state import ChannelState
from raiden.utils.typing import BlockNumber, FeeAmount, List, PaymentAmount, PaymentID, TargetAddress, TokenAddress

class CustomException(Exception):
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_payload_with_invalid_addresses(api_server_test_instance: APIServer) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_payload_with_address_invalid_chars(api_server_test_instance: APIServer) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_payload_with_address_invalid_length(api_server_test_instance: APIServer) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_payload_with_address_not_eip55(api_server_test_instance: APIServer) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_api_query_our_address(api_server_test_instance: APIServer) -> None:
    ...

@Mark.parametrize('enable_rest_api', [True])
def test_api_get_raiden_version(api_server_test_instance: APIServer) -> None:
    ...

@Mark.parametrize('enable_rest_api', [True])
def test_api_get_node_settings(api_server_test_instance: APIServer) -> None:
    ...

@Mark.parametrize('enable_rest_api', [True])
def test_api_get_contract_infos(api_server_test_instance: APIServer) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_api_get_channel_list(api_server_test_instance: APIServer, token_addresses: List[TokenAddress], reveal_timeout: int) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('number_of_tokens', [2])
@Mark.parametrize('environment_type', [Environment.PRODUCTION])
@Mark.parametrize('enable_rest_api', [True])
def test_api_tokens(api_server_test_instance: APIServer, blockchain_services: Any, token_addresses: List[TokenAddress]) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_query_partners_by_token(api_server_test_instance: APIServer, blockchain_services: Any, token_addresses: List[TokenAddress]) -> None:
    ...

@Mark.parametrize('number_of_nodes', [2])
@Mark.parametrize('enable_rest_api', [True])
def test_api_timestamp_format(api_server_test_instance: APIServer, raiden_network: Any, token_addresses: List[TokenAddress], pfs_mock: Any) -> None:
    ...

@Mark.parametrize('number_of_tokens', [0])
@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('environment_type', [Environment.DEVELOPMENT])
@Mark.parametrize('enable_rest_api', [True])
def test_get_token_network_for_token(api_server_test_instance: APIServer, token_amount: int, token_addresses: List[TokenAddress], raiden_network: Any, contract_manager: Any, retry_timeout: float, unregistered_token: TokenAddress) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('number_of_tokens', [1])
@Mark.parametrize('enable_rest_api', [True])
def test_get_connections_info(raiden_network: Any, api_server_test_instance: APIServer, token_addresses: List[TokenAddress]) -> None:
    ...

@Mark.parametrize('number_of_nodes', [3])
@Mark.parametrize('enable_rest_api', [True])
@Mark.parametrize('number_of_tokens', [2])
def test_payment_events_endpoints(api_server_test_instance: APIServer, raiden_network: Any, token_addresses: List[TokenAddress], pfs_mock: Any) -> None:
    ...

@Mark.parametrize('number_of_nodes', [2])
@Mark.parametrize('enable_rest_api', [True])
def test_channel_events_raiden(api_server_test_instance: APIServer, raiden_network: Any, token_addresses: List[TokenAddress], pfs_mock: Any) -> None:
    ...

@Mark.parametrize('number_of_nodes', [3])
@Mark.parametrize('channels_per_node', [CHAIN])
@Mark.parametrize('enable_rest_api', [True])
def test_pending_transfers_endpoint(raiden_network: Any, token_addresses: List[TokenAddress]) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('number_of_tokens', [1])
@Mark.parametrize('token_contract_name', [CONTRACT_CUSTOM_TOKEN])
@Mark.parametrize('enable_rest_api', [True])
def test_api_testnet_token_mint(api_server_test_instance: APIServer, token_addresses: List[TokenAddress]) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_udc_api(api_server_test_instance: APIServer, retry_timeout: float) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_udc_api_with_invalid_parameters(api_server_test_instance: APIServer) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
@Mark.parametrize('user_deposit_address', [None])
def test_no_udc_configured(api_server_test_instance: APIServer, retry_timeout: float) -> None:
    ...

@Mark.parametrize('number_of_nodes', [1])
@Mark.parametrize('channels_per_node', [0])
@Mark.parametrize('enable_rest_api', [True])
def test_shutdown(api_server_test_instance: APIServer) -> None:
    ...