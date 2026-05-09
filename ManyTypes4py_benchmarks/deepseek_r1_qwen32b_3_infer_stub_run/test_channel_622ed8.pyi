from http import HTTPStatus
from typing import List, Optional, Tuple
from uuid import UUID

from eth_utils import ChecksumAddress
from grequests import Response
from raiden.api.rest import APIServer
from raiden.raiden_service import RaidenService
from raiden.tests.utils.events import DictAttrs
from raiden.transfer.state import ChannelState
from raiden.utils.typing import TokenAmount
from raiden_contracts.constants import TEST_SETTLE_TIMEOUT_MAX, TEST_SETTLE_TIMEOUT_MIN

def test_api_channel_status_channel_nonexistant(
    api_server_test_instance: APIServer,
    token_addresses: List[ChecksumAddress],
) -> None:
    ...

def test_api_channel_open_and_deposit(
    api_server_test_instance: APIServer,
    token_addresses: List[ChecksumAddress],
    reveal_timeout: int,
) -> None:
    ...

def test_api_channel_open_and_deposit_race(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, ...],
    token_addresses: List[ChecksumAddress],
    reveal_timeout: int,
    token_network_registry_address: ChecksumAddress,
    retry_timeout: float,
) -> None:
    ...

def test_api_channel_open_close_and_settle(
    api_server_test_instance: APIServer,
    token_addresses: List[ChecksumAddress],
    reveal_timeout: int,
) -> None:
    ...

def test_api_channel_close_insufficient_eth(
    api_server_test_instance: APIServer,
    token_addresses: List[ChecksumAddress],
    reveal_timeout: int,
) -> None:
    ...

def test_api_channel_open_channel_invalid_input(
    api_server_test_instance: APIServer,
    token_addresses: List[ChecksumAddress],
    reveal_timeout: int,
) -> None:
    ...

def test_api_channel_state_change_errors(
    api_server_test_instance: APIServer,
    token_addresses: List[ChecksumAddress],
    reveal_timeout: int,
) -> None:
    ...

def test_api_channel_withdraw(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, ...],
    token_addresses: List[ChecksumAddress],
    pfs_mock: object,
) -> None:
    ...

def test_api_channel_withdraw_with_offline_partner(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, ...],
    token_addresses: List[ChecksumAddress],
) -> None:
    ...

def test_api_channel_set_reveal_timeout(
    api_server_test_instance: APIServer,
    raiden_network: Tuple[RaidenService, ...],
    token_addresses: List[ChecksumAddress],
    settle_timeout: int,
) -> None:
    ...

def test_api_channel_deposit_limit(
    api_server_test_instance: APIServer,
    proxy_manager: object,
    token_network_registry_address: ChecksumAddress,
    token_addresses: List[ChecksumAddress],
    reveal_timeout: int,
) -> None:
    ...