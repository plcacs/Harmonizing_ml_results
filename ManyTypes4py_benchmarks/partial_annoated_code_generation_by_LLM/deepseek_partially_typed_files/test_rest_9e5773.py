import datetime
import json
from http import HTTPStatus
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Optional, Union, cast

import gevent
import grequests
import pytest
from eth_utils import is_checksum_address, to_checksum_address, to_hex
from flask import url_for
from raiden.api.python import RaidenAPI
from raiden.api.rest import APIServer
from raiden.constants import BLOCK_ID_LATEST, Environment
from raiden.exceptions import InvalidSecret
from raiden.messages.transfers import LockedTransfer, Unlock
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS, INTERNAL_ROUTING_DEFAULT_FEE_PERC
from raiden.tests.integration.api.rest.utils import api_url_for, assert_proper_response, assert_response_with_code, assert_response_with_error, get_json_response
from raiden.tests.integration.api.utils import prepare_api_server
from raiden.tests.integration.fixtures.smartcontracts import RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT
from raiden.tests.utils import factories
from raiden.tests.utils.client import burn_eth
from raiden.tests.utils.detect_failure import expect_failure, raise_on_failure
from raiden.tests.utils.events import must_have_event, must_have_events
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.tests.utils.transfer import block_offset_timeout, create_route_state_for_route, watch_for_unlock_failures
from raiden.transfer import views
from raiden.transfer.mediated_transfer.initiator import calculate_fee_margin
from raiden.transfer.state import ChannelState
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.system import get_system_spec
from raiden.utils.typing import BlockNumber, FeeAmount, List, PaymentAmount, PaymentID, TargetAddress, TokenAddress
from raiden.waiting import TransferWaitResult, wait_for_block, wait_for_received_transfer_result, wait_for_token_network
from raiden_contracts.constants import CONTRACT_CUSTOM_TOKEN, CONTRACTS_VERSION

DEPOSIT_FOR_TEST_API_DEPOSIT_LIMIT: int = RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT + 2

class CustomException(Exception):
    pass

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_payload_with_invalid_addresses(api_server_test_instance: APIServer) -> None:
    """Addresses require leading 0x in the payload."""
    invalid_address: str = '61c808d82a3ac53231750dadc13c777b59310bd9'
    channel_data_obj: Dict[str, str] = {'partner_address': invalid_address, 'token_address': '0xEA674fdDe714fd979de3EdF0F56AA9716B898ec8', 'settle_timeout': '10'}
    request: grequests.AsyncRequest = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response: grequests.Response = request.send().response
    assert_response_with_error(response, HTTPStatus.BAD_REQUEST)
    url_without_prefix: str = 'http://localhost:{port}/api/v1/channels/ea674fdde714fd979de3edf0f56aa9716b898ec8'.format(port=api_server_test_instance.config.port)
    request = grequests.patch(url_without_prefix, json=dict(state=ChannelState.STATE_SETTLED.value))
    response = request.send().response
    assert_response_with_code(response, HTTPStatus.NOT_FOUND)

@pytest.mark.xfail(strict=True, reason='Crashed app also crashes on teardown', raises=CustomException)
@expect_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_crash_on_unhandled_exception(api_server_test_instance: APIServer) -> None:
    """Crash when an unhandled exception happens on APIServer."""

    @api_server_test_instance.flask_app.route('/error_endpoint', methods=['GET'])
    def error_endpoint() -> None:
        raise CustomException('This is an unhandled error')
    with api_server_test_instance.flask_app.app_context():
        url: str = url_for('error_endpoint')
    request: grequests.AsyncRequest = grequests.get(url)
    request.send()
    api_server_test_instance.greenlet.get(timeout=10)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_payload_with_address_invalid_chars(api_server_test_instance: APIServer) -> None:
    """Addresses cannot have invalid characters in it."""
    invalid_address: str = '0x61c808d82a3ac53231750dadc13c777b59310bdg'
    channel_data_obj: Dict[str, str] = {'partner_address': invalid_address, 'token_address': '0xEA674fdDe714fd979de3EdF0F56AA9716B898ec8', 'settle_timeout': '10'}
    request: grequests.AsyncRequest = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response: grequests.Response = request.send().response
    assert_response_with_error(response, HTTPStatus.BAD_REQUEST)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_payload_with_address_invalid_length(api_server_test_instance: APIServer) -> None:
    """Encoded addresses must have the right length."""
    invalid_address: str = '0x61c808d82a3ac53231750dadc13c777b59310b'
    channel_data_obj: Dict[str, str] = {'partner_address': invalid_address, 'token_address': '0xEA674fdDe714fd979de3EdF0F56AA9716B898ec8', 'settle_timeout': '10'}
    request: grequests.AsyncRequest = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response: grequests.Response = request.send().response
    assert_response_with_error(response, HTTPStatus.BAD_REQUEST)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_payload_with_address_not_eip55(api_server_test_instance: APIServer) -> None:
    """Provided addresses must be EIP55 encoded."""
    invalid_address: str = '0xf696209d2ca35e6c88e5b99b7cda3abf316bed69'
    channel_data_obj: Dict[str, str] = {'partner_address': invalid_address, 'token_address': '0xEA674fdDe714fd979de3EdF0F56AA9716B898ec8', 'settle_timeout': '90'}
    request: grequests.AsyncRequest = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response: grequests.Response = request.send().response
    assert_response_with_error(response, HTTPStatus.BAD_REQUEST)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_query_our_address(api_server_test_instance: APIServer) -> None:
    request: grequests.AsyncRequest = grequests.get(api_url_for(api_server_test_instance, 'addressresource'))
    response: grequests.Response = request.send().response
    assert_proper_response(response)
    our_address: str = api_server_test_instance.rest_api.raiden_api.address
    assert get_json_response(response) == {'our_address': to_checksum_address(our_address)}

@raise_on_failure
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_get_raiden_version(api_server_test_instance: APIServer) -> None:
    request: grequests.AsyncRequest = grequests.get(api_url_for(api_server_test_instance, 'versionresource'))
    response: grequests.Response = request.send().response
    assert_proper_response(response)
    raiden_version: str = get_system_spec()['raiden']
    assert get_json_response(response) == {'version': raiden_version}

@raise_on_failure
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_get_node_settings(api_server_test_instance: APIServer) -> None:
    request: grequests.AsyncRequest = grequests.get(api_url_for(api_server_test_instance, 'nodesettingsresource'))
    response: grequests.Response = request.send().response
    assert_proper_response(response)
    pfs_config: Any = api_server_test_instance.rest_api.raiden_api.raiden.config.pfs_config
    assert get_json_response(response) == {'pathfinding_service_address': pfs_config and pfs_config.info.url}

@raise_on_failure
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_get_contract_infos(api_server_test_instance: APIServer) -> None:
    request: grequests.AsyncRequest = grequests.get(api_url_for(api_server_test_instance, 'contractsresource'))
    response: grequests.Response = request.send().response
    assert_proper_response(response)
    json_response: Dict[str, Any] = get_json_response(response)
    assert json_response['contracts_version'] == CONTRACTS_VERSION
    for contract_name in ['token_network_registry_address', 'secret_registry_address', 'service_registry_address', 'user_deposit_address', 'monitoring_service_address', 'one_to_n_address']:
        address: Optional[str] = json_response[contract_name]
        if address is not None:
            assert is_checksum_address(address)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_get_channel_list(api_server_test_instance: APIServer, token_addresses: List[TokenAddress], reveal_timeout: int) -> None:
    partner_address: str = '0x61C808D82A3Ac53231750daDc13c777b59310bD9'
    request: grequests.AsyncRequest = grequests.get(api_url_for(api_server_test_instance, 'channelsresource'))
    response: grequests.Response = request.send().response
    assert_proper_response(response, HTTPStatus.OK)
    json_response: List[Dict[str, Any]] = get_json_response(response)
    assert json_response == []
    token_address: TokenAddress = token_addresses[0]
    settle_timeout: int = 1650
    channel_data_obj: Dict[str, str] = {'partner_address': partner_address, 'token_address': to_checksum_address(token_address), 'settle_timeout': str(settle_timeout), 'reveal_timeout': str(reveal_timeout)}
    request = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response = request.send().response
    assert_proper_response(response, HTTPStatus.CREATED)
    request = grequests.get(api_url_for(api_server_test_instance, 'channelsresource'))
    response = request.send().response
    assert_proper_response(response, HTTPStatus.OK)
    json_response = get_json_response(response)
    channel_info: Dict[str, Any] = json_response[0]
    assert channel_info['partner_address'] == partner_address
    assert channel_info['token_address'] == to_checksum_address(token_address)
    assert channel_info['total_deposit'] == '0'
    assert 'token_network_address' in channel_info

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('number_of_tokens', [2])
@pytest.mark.parametrize('environment_type', [Environment.PRODUCTION])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_tokens(api_server_test_instance: APIServer, blockchain_services: Any, token_addresses: List[TokenAddress]) -> None:
    partner_address: str = '0x61C808D82A3Ac53231750daDc13c777b59310bD9'
    token_address1: TokenAddress = token_addresses[0]
    token_address2: TokenAddress = token_addresses[1]
    settle_timeout: int = 1650
    channel_data_obj: Dict[str, str] = {'partner_address': partner_address, 'token_address': to_checksum_address(token_address1), 'settle_timeout': str(settle_timeout)}
    request: grequests.AsyncRequest = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response: grequests.Response = request.send().response
    assert_proper_response(response, HTTPStatus.CREATED)
    settle_timeout = 1650
    channel_data_obj = {'partner_address': partner_address, 'token_address': to_checksum_address(token_address2), 'settle_timeout': str(settle_timeout)}
    request = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response = request.send().response
    assert_proper_response(response, HTTPStatus.CREATED)
    request = grequests.get(api_url_for(api_server_test_instance, 'tokensresource'))
    response = request.send().response
    assert_proper_response(response)
    json_response: List[str] = get_json_response(response)
    expected_response: List[str] = [to_checksum_address(token_address1), to_checksum_address(token_address2)]
    assert set(json_response) == set(expected_response)

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [1])
@pytest.mark.parametrize('channels_per_node', [0])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_query_partners_by_token(api_server_test_instance: APIServer, blockchain_services: Any, token_addresses: List[TokenAddress]) -> None:
    first_partner_address: str = '0x61C808D82A3Ac53231750daDc13c777b59310bD9'
    second_partner_address: str = '0x29FA6cf0Cce24582a9B20DB94Be4B6E017896038'
    token_address: TokenAddress = token_addresses[0]
    settle_timeout: int = 1650
    channel_data_obj: Dict[str, str] = {'partner_address': first_partner_address, 'token_address': to_checksum_address(token_address), 'settle_timeout': str(settle_timeout)}
    request: grequests.AsyncRequest = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response: grequests.Response = request.send().response
    assert_proper_response(response, HTTPStatus.CREATED)
    json_response: Dict[str, Any] = get_json_response(response)
    channel_data_obj['partner_address'] = second_partner_address
    request = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response = request.send().response
    assert_proper_response(response, HTTPStatus.CREATED)
    json_response = get_json_response(response)
    channel_data_obj['partner_address'] = '0xb07937AbA15304FBBB0Bf6454a9377a76E3dD39E'
    channel_data_obj['token_address'] = to_checksum_address(token_address)
    request = grequests.put(api_url_for(api_server_test_instance, 'channelsresource'), json=channel_data_obj)
    response = request.send().response
    assert_proper_response(response, HTTPStatus.CREATED)
    request = grequests.get(api_url_for(api_server_test_instance, 'partnersresourcebytokenaddress', token_address=to_checksum_address(token_address)))
    response = request.send().response
    assert_proper_response(response)
    json_response = get_json_response(response)
    expected_response: List[Dict[str, str]] = [{'partner_address': first_partner_address, 'channel': '/api/v1/channels/{}/{}'.format(to_checksum_address(token_address), to_checksum_address(first_partner_address))}, {'partner_address': second_partner_address, 'channel': '/api/v1/channels/{}/{}'.format(to_checksum_address(token_address), to_checksum_address(second_partner_address))}]
    assert all((r in json_response for r in expected_response))

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('enable_rest_api', [True])
def test_api_timestamp_format(api_server_test_instance: APIServer, raiden_network: List[RaidenService], token_addresses: List[TokenAddress