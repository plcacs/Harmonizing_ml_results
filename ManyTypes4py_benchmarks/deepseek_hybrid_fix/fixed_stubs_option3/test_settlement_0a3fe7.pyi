import random
from unittest.mock import patch, Mock
import gevent
import pytest
from eth_utils import keccak, HexBytes
from gevent import Timeout, Greenlet
from raiden import waiting
from raiden.api.python import RaidenAPI
from raiden.constants import BLOCK_ID_LATEST, EMPTY_SIGNATURE, UINT64_MAX
from raiden.exceptions import InvalidSecret, RaidenUnrecoverableError
from raiden.messages.transfers import LockedTransfer, LockExpired, RevealSecret, Unlock
from raiden.messages.withdraw import WithdrawExpired
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_RETRY_TIMEOUT
from raiden.storage.restore import channel_state_until_state_change
from raiden.storage.sqlite import HIGH_STATECHANGE_ULID, RANGE_ALL_STATE_CHANGES, StateChangeID
from raiden.tests.utils import factories
from raiden.tests.utils.client import burn_eth
from raiden.tests.utils.detect_failure import expect_failure, raise_on_failure, FailureDetector
from raiden.tests.utils.events import raiden_state_changes_search_for_item, search_for_item, StateChangeSearchResult
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.tests.utils.transfer import assert_synced_channel_state, block_offset_timeout, create_route_state_for_route, get_channelstate, transfer
from raiden.transfer import channel, views
from raiden.transfer.events import SendWithdrawConfirmation
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.state import NettingChannelState
from raiden.transfer.state_change import ContractReceiveChannelBatchUnlock, ContractReceiveChannelClosed, ContractReceiveChannelSettled, ContractReceiveChannelWithdraw
from raiden.utils.formatting import to_checksum_address
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.typing import Address, Balance, BlockNumber, BlockTimeout as BlockOffset, List, MessageID, PaymentAmount, PaymentID, Secret, SecretRegistryAddress, TargetAddress, TokenAddress, TokenAmount, TokenNetworkAddress, WithdrawAmount
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from _pytest.logging import LogCaptureFixture

MSG_BLOCKCHAIN_EVENTS: str

def wait_for_batch_unlock(app: RaidenService, token_network_address: TokenNetworkAddress, receiver: Address, sender: Address) -> None: ...

def is_channel_registered(node_app: RaidenService, partner_app: RaidenService, canonical_identifier: CanonicalIdentifier) -> bool: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_settle_is_automatically_called(raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[TokenAddress]) -> None: ...

@pytest.mark.flaky
@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_coop_settle_is_automatically_called(raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[TokenAddress]) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_coop_settle_fails_with_pending_lock(raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[TokenAddress]) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_lock_expiry(raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[TokenAddress], deposit: TokenAmount) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_batch_unlock(raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[TokenAddress], secret_registry_address: SecretRegistryAddress, deposit: TokenAmount) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('reveal_timeout', [8])
def test_register_secret(raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[TokenAddress], secret_registry_address: SecretRegistryAddress) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_channel_withdraw(raiden_network: Tuple[RaidenService, RaidenService], token_addresses: List[TokenAddress], deposit: TokenAmount, retry_timeout: float, pfs_mock: Mock) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_channel_withdraw_expired(raiden_network: Tuple[RaidenService, RaidenService], network_wait: float, number_of_nodes: int, token_addresses: List[TokenAddress], deposit: TokenAmount, retry_timeout: float, pfs_mock: Mock) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [CHAIN])
def test_settled_lock(token_addresses: List[TokenAddress], raiden_network: Tuple[RaidenService, RaidenService], deposit: TokenAmount, retry_timeout: float) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('channels_per_node', [1])
def test_automatic_secret_registration(raiden_chain: Tuple[RaidenService, RaidenService], token_addresses: List[TokenAddress]) -> None: ...

@raise_on_failure
@pytest.mark.xfail(reason='test incomplete')
@pytest.mark.parametrize('number_of_nodes', [3])
def test_start_end_attack(token_addresses: List[TokenAddress], raiden_chain: Tuple[RaidenService, RaidenService, RaidenService], deposit: TokenAmount) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_automatic_dispute(raiden_network: Tuple[RaidenService, RaidenService], deposit: TokenAmount, token_addresses: List[TokenAddress]) -> None: ...

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_batch_unlock_after_restart(raiden_network: Tuple[RaidenService, RaidenService], restart_node: Callable[[RaidenService], RaidenService], token_addresses: List[TokenAddress], deposit: TokenAmount) -> None: ...

@expect_failure
@pytest.mark.parametrize('number_of_nodes', (2,))
@pytest.mark.parametrize('channels_per_node', (1,))
def test_handle_insufficient_eth(raiden_network: Tuple[RaidenService, RaidenService], restart_node: Callable[[RaidenService], RaidenService], token_addresses: List[TokenAddress], caplog: LogCaptureFixture) -> None: ...