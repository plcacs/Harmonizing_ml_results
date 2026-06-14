import datetime
import json
from http import HTTPStatus
from unittest.mock import Mock, patch
from typing import Any, List

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
from raiden.tests.integration.api.rest.utils import (
    api_url_for,
    assert_proper_response,
    assert_response_with_code,
    assert_response_with_error,
    get_json_response,
)
from raiden.tests.integration.api.utils import prepare_api_server
from raiden.tests.integration.fixtures.smartcontracts import RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT
from raiden.tests.utils import factories
from raiden.tests.utils.client import burn_eth
from raiden.tests.utils.detect_failure import expect_failure, raise_on_failure
from raiden.tests.utils.events import must_have_event, must_have_events
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.tests.utils.transfer import (
    block_offset_timeout,
    create_route_state_for_route,
    watch_for_unlock_failures,
)
from raiden.transfer import views
from raiden.transfer.mediated_transfer.initiator import calculate_fee_margin
from raiden.transfer.state import ChannelState
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.system import get_system_spec
from raiden.utils.typing import (
    BlockNumber,
    FeeAmount,
    PaymentAmount,
    PaymentID,
    TargetAddress,
    TokenAddress,
)
from raiden.waiting import (
    TransferWaitResult,
    wait_for_block,
    wait_for_received_transfer_result,
    wait_for_token_network,
)
from raiden_contracts.constants import CONTRACT_CUSTOM_TOKEN, CONTRACTS_VERSION

DEPOSIT_FOR_TEST_API_DEPOSIT_LIMIT: int = ...

class CustomException(Exception): ...

def test_payload_with_invalid_addresses(api_server_test_instance: APIServer) -> None: ...
def test_crash_on_unhandled_exception(api_server_test_instance: APIServer) -> None: ...
def test_payload_with_address_invalid_chars(api_server_test_instance: APIServer) -> None: ...
def test_payload_with_address_invalid_length(api_server_test_instance: APIServer) -> None: ...
def test_payload_with_address_not_eip55(api_server_test_instance: APIServer) -> None: ...
def test_api_query_our_address(api_server_test_instance: APIServer) -> None: ...
def test_api_get_raiden_version(api_server_test_instance: APIServer) -> None: ...
def test_api_get_node_settings(api_server_test_instance: APIServer) -> None: ...
def test_api_get_contract_infos(api_server_test_instance: APIServer) -> None: ...
def test_api_get_channel_list(api_server_test_instance: APIServer, token_addresses: List[TokenAddress], reveal_timeout: int) -> None: ...
def test_api_tokens(api_server_test_instance: APIServer, blockchain_services: Any, token_addresses: List[TokenAddress]) -> None: ...
def test_query_partners_by_token(api_server_test_instance: APIServer, blockchain_services: Any, token_addresses: List[TokenAddress]) -> None: ...
def test_api_timestamp_format(api_server_test_instance: APIServer, raiden_network: List[RaidenService], token_addresses: List[TokenAddress], pfs_mock: Any) -> None: ...
def test_get_token_network_for_token(api_server_test_instance: APIServer, token_amount: int, token_addresses: List[TokenAddress], raiden_network: List[RaidenService], contract_manager: Any, retry_timeout: float, unregistered_token: TokenAddress) -> None: ...
def test_get_connections_info(raiden_network: List[RaidenService], api_server_test_instance: APIServer, token_addresses: List[TokenAddress]) -> None: ...
def test_payment_events_endpoints(api_server_test_instance: APIServer, raiden_network: List[RaidenService], token_addresses: List[TokenAddress], pfs_mock: Any) -> None: ...
def test_channel_events_raiden(api_server_test_instance: APIServer, raiden_network: List[RaidenService], token_addresses: List[TokenAddress], pfs_mock: Any) -> None: ...
def test_pending_transfers_endpoint(decrypt_patch: Any, raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None: ...
def test_api_testnet_token_mint(api_server_test_instance: APIServer, token_addresses: List[TokenAddress]) -> None: ...
def test_udc_api(api_server_test_instance: APIServer, retry_timeout: float) -> None: ...
def test_udc_api_with_invalid_parameters(api_server_test_instance: APIServer) -> None: ...
def test_no_udc_configured(api_server_test_instance: APIServer, retry_timeout: float) -> None: ...
def test_shutdown(api_server_test_instance: APIServer) -> None: ...