from typing import Any, Dict, List, Optional, Union
from http import HTTPStatus
import gevent
import grequests
import pytest
from eth_utils import HexAddress, HexStr
from raiden.api.rest import APIServer
from raiden.constants import BlockIdentifier
from raiden.raiden_service import RaidenService
from raiden.tests.integration.api.rest.test_rest import DepositAmount
from raiden.tests.integration.api.rest.utils import APIResponse
from raiden.tests.utils.client import RPCClient
from raiden.tests.utils.detect_failure import FailureDetector
from raiden.tests.utils.events import EventChecker
from raiden.transfer import views
from raiden.transfer.state import ChannelState
from raiden.utils.typing import Address, TokenAmount, TokenNetworkAddress
from raiden.waiting import WaitResult
from raiden_contracts.constants import SettleTimeout

# Module-level fixtures inferred from pytest decorators
api_server_test_instance: APIServer
token_addresses: List[Address]
reveal_timeout: int
raiden_network: List[RaidenService]
token_network_registry_address: Address
retry_timeout: float
proxy_manager: Any
pfs_mock: Any
deposit: int
settle_timeout: int

def test_api_channel_status_channel_nonexistant(
    api_server_test_instance: APIServer,
    token_addresses: List[Address]
) -> None: ...

def test_api_channel_open_and_deposit(
    api_server_test_instance: APIServer,
    token_addresses: List[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_open_and_deposit_race(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[Address],
    reveal_timeout: int,
    token_network_registry_address: Address,
    retry_timeout: float
) -> None: ...

def test_api_channel_open_close_and_settle(
    api_server_test_instance: APIServer,
    token_addresses: List[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_close_insufficient_eth(
    api_server_test_instance: APIServer,
    token_addresses: List[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_open_channel_invalid_input(
    api_server_test_instance: APIServer,
    token_addresses: List[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_state_change_errors(
    api_server_test_instance: APIServer,
    token_addresses: List[Address],
    reveal_timeout: int
) -> None: ...

def test_api_channel_withdraw(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[Address],
    pfs_mock: Any
) -> None: ...

def test_api_channel_withdraw_with_offline_partner(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[Address]
) -> None: ...

def test_api_channel_set_reveal_timeout(
    api_server_test_instance: APIServer,
    raiden_network: List[RaidenService],
    token_addresses: List[Address],
    settle_timeout: int
) -> None: ...

def test_api_channel_deposit_limit(
    api_server_test_instance: APIServer,
    proxy_manager: Any,
    token_network_registry_address: Address,
    token_addresses: List[Address],
    reveal_timeout: int
) -> None: ...