import random
from typing import List
from unittest.mock import patch

import gevent
import pytest
from eth_utils import keccak
from gevent import Timeout

from raiden import waiting
from raiden.api.python import RaidenAPI
from raiden.constants import BLOCK_ID_LATEST, EMPTY_SIGNATURE, UINT64_MAX
from raiden.exceptions import InvalidSecret, RaidenUnrecoverableError
from raiden.messages.transfers import LockedTransfer, LockExpired, RevealSecret, Unlock
from raiden.messages.withdraw import WithdrawExpired
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_RETRY_TIMEOUT
from raiden.storage.restore import channel_state_until_state_change
from raiden.storage.sqlite import HIGH_STATECHANGE_ULID, RANGE_ALL_STATE_CHANGES
from raiden.tests.utils import factories
from raiden.tests.utils.client import burn_eth
from raiden.tests.utils.detect_failure import expect_failure, raise_on_failure
from raiden.tests.utils.events import raiden_state_changes_search_for_item, search_for_item
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.tests.utils.transfer import (
    assert_synced_channel_state,
    block_offset_timeout,
    create_route_state_for_route,
    get_channelstate,
    transfer,
)
from raiden.transfer import channel, views
from raiden.transfer.events import SendWithdrawConfirmation
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.state import NettingChannelState
from raiden.transfer.state_change import (
    ContractReceiveChannelBatchUnlock,
    ContractReceiveChannelClosed,
    ContractReceiveChannelSettled,
    ContractReceiveChannelWithdraw,
)
from raiden.utils.formatting import to_checksum_address
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.typing import (
    Address,
    Balance,
    BlockNumber,
    BlockTimeout as BlockOffset,
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

MSG_BLOCKCHAIN_EVENTS: str

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

def test_settle_is_automatically_called(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

def test_coop_settle_is_automatically_called(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

def test_coop_settle_fails_with_pending_lock(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

def test_lock_expiry(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
) -> None: ...

def test_batch_unlock(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    secret_registry_address: SecretRegistryAddress,
    deposit: TokenAmount,
) -> None: ...

def test_register_secret(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    secret_registry_address: SecretRegistryAddress,
) -> None: ...

def test_channel_withdraw(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
    retry_timeout: float,
    pfs_mock: object,
) -> None: ...

def test_channel_withdraw_expired(
    raiden_network: List[RaidenService],
    network_wait: float,
    number_of_nodes: int,
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
    retry_timeout: float,
    pfs_mock: object,
) -> None: ...

def test_settled_lock(
    token_addresses: List[TokenAddress],
    raiden_network: List[RaidenService],
    deposit: TokenAmount,
    retry_timeout: float,
) -> None: ...

def test_automatic_secret_registration(
    raiden_chain: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None: ...

def test_start_end_attack(
    token_addresses: List[TokenAddress],
    raiden_chain: List[RaidenService],
    deposit: TokenAmount,
) -> None: ...

def test_automatic_dispute(
    raiden_network: List[RaidenService],
    deposit: TokenAmount,
    token_addresses: List[TokenAddress],
) -> None: ...

def test_batch_unlock_after_restart(
    raiden_network: List[RaidenService],
    restart_node: object,
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
) -> None: ...

def test_handle_insufficient_eth(
    raiden_network: List[RaidenService],
    restart_node: object,
    token_addresses: List[TokenAddress],
    caplog: pytest.LogCaptureFixture,
) -> None: ...