import pytest
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import Mock
from eth_utils import HexBytes
from gevent import Greenlet, Timeout
from raiden.api.python import RaidenAPI
from raiden.constants import EMPTY_SIGNATURE
from raiden.messages.transfers import LockedTransfer, LockExpired, RevealSecret, Unlock
from raiden.messages.withdraw import WithdrawExpired
from raiden.raiden_service import RaidenService
from raiden.storage.sqlite import StateChangeID
from raiden.tests.utils.detect_failure import FailureDetector
from raiden.tests.utils.events import StateChangeSearchResult
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.transfer.events import SendWithdrawConfirmation
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.state import NettingChannelState
from raiden.transfer.state_change import (
    ContractReceiveChannelBatchUnlock,
    ContractReceiveChannelClosed,
    ContractReceiveChannelSettled,
    ContractReceiveChannelWithdraw,
)
from raiden.utils.typing import (
    Address,
    Balance,
    BlockNumber,
    BlockTimeout,
    MessageID,
    PaymentAmount,
    PaymentID,
    Secret,
    SecretRegistryAddress,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
    WithdrawAmount,
)

MSG_BLOCKCHAIN_EVENTS: str = ...

def wait_for_batch_unlock(
    app: RaidenService,
    token_network_address: TokenNetworkAddress,
    receiver: Address,
    sender: Address,
) -> None: ...

def is_channel_registered(
    node_app: RaidenService,
    partner_app: RaidenService,
    canonical_identifier: CanonicalIdentifier,
) -> bool: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
def test_settle_is_automatically_called(
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

@pytest.mark.flaky
@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
def test_coop_settle_is_automatically_called(
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
def test_coop_settle_fails_with_pending_lock(
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
def test_lock_expiry(
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
def test_batch_unlock(
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[TokenAddress],
    secret_registry_address: SecretRegistryAddress,
    deposit: TokenAmount,
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("reveal_timeout", [8])
def test_register_secret(
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[TokenAddress],
    secret_registry_address: SecretRegistryAddress,
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
def test_channel_withdraw(
    raiden_network: Tuple[RaidenService, RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
    retry_timeout: float,
    pfs_mock: Mock,
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
def test_channel_withdraw_expired(
    raiden_network: Tuple[RaidenService, RaidenService],
    network_wait: float,
    number_of_nodes: int,
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
    retry_timeout: float,
    pfs_mock: Mock,
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("channels_per_node", [1])
def test_settled_lock(
    token_addresses: List[TokenAddress],
    raiden_network: Tuple[RaidenService, RaidenService],
    deposit: TokenAmount,
    retry_timeout: float,
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
@pytest.mark.parametrize("channels_per_node", [1])
def test_automatic_secret_registration(
    raiden_chain: Tuple[RaidenService, RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

@FailureDetector
@pytest.mark.xfail(reason="test incomplete")
@pytest.mark.parametrize("number_of_nodes", [3])
def test_start_end_attack(
    token_addresses: List[TokenAddress],
    raiden_chain: Tuple[RaidenService, RaidenService, RaidenService],
    deposit: TokenAmount,
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
def test_automatic_dispute(
    raiden_network: Tuple[RaidenService, RaidenService],
    deposit: TokenAmount,
    token_addresses: List[TokenAddress],
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", [2])
def test_batch_unlock_after_restart(
    raiden_network: Tuple[RaidenService, RaidenService],
    restart_node: Callable[[RaidenService], RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
) -> None: ...

@FailureDetector
@pytest.mark.parametrize("number_of_nodes", (2,))
@pytest.mark.parametrize("channels_per_node", (1,))
def test_handle_insufficient_eth(
    raiden_network: Tuple[RaidenService, RaidenService],
    restart_node: Callable[[RaidenService], RaidenService],
    token_addresses: List[TokenAddress],
    caplog: pytest.LogCaptureFixture,
) -> None: ...