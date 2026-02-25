import random
from unittest.mock import patch
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Callable, TypeVar, Generic

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

MSG_BLOCKCHAIN_EVENTS = "Waiting for blockchain events requires a running node and alarm task."

def wait_for_batch_unlock(
    app: RaidenService,
    token_network_address: TokenNetworkAddress,
    receiver: Address,
    sender: Address,
) -> None:
    unlock_event: Optional[ContractReceiveChannelBatchUnlock] = None
    while not unlock_event:
        gevent.sleep(1)

        assert app.wal, MSG_BLOCKCHAIN_EVENTS
        assert app.alarm, MSG_BLOCKCHAIN_EVENTS

        state_changes = app.wal.storage.get_statechanges_by_range(RANGE_ALL_STATE_CHANGES)

        unlock_event = search_for_item(
            state_changes,
            ContractReceiveChannelBatchUnlock,
            {
                "token_network_address": token_network_address,
                "receiver": receiver,
                "sender": sender,
            },
        )

def is_channel_registered(
    node_app: RaidenService, 
    partner_app: RaidenService, 
    canonical_identifier: CanonicalIdentifier
) -> bool:
    token_network = views.get_token_network_by_address(
        chain_state=views.state_from_raiden(node_app),
        token_network_address=canonical_identifier.token_network_address,
    )
    assert token_network

    is_in_channelid_map = (
        canonical_identifier.channel_identifier in token_network.channelidentifiers_to_channels
    )
    is_in_partner_map = (
        canonical_identifier.channel_identifier
        in token_network.partneraddresses_to_channelidentifiers[partner_app.address]
    )

    return is_in_channelid_map and is_in_partner_map

@raise_on_failure
@pytest.mark.parametrize("number_of_nodes", [2])
def test_settle_is_automatically_called(
    raiden_network: List[RaidenService], 
    token_addresses: List[TokenAddress]
) -> None:
    app0, app1 = raiden_network
    registry_address = app0.default_registry.address
    token_address = token_addresses[0]
    token_network_address = views.get_token_network_address_by_token_address(
        views.state_from_raiden(app0), app0.default_registry.address, token_address
    )
    assert token_network_address
    token_network = views.get_token_network_by_address(
        views.state_from_raiden(app0), token_network_address
    )
    assert token_network

    channel_identifier = get_channelstate(app0, app1, token_network_address).identifier

    assert channel_identifier in token_network.partneraddresses_to_channelidentifiers[app1.address]

    RaidenAPI(app1).channel_close(registry_address, token_address, app0.address, coop_settle=False)

    waiting.wait_for_close(
        app0,
        registry_address,
        token_address,
        [channel_identifier],
        app0.alarm.sleep_time,
    )

    channel_state = views.get_channelstate_for(
        views.state_from_raiden(app0), registry_address, token_address, app1.address
    )
    assert channel_state
    assert channel_state.close_transaction
    assert channel_state.close_transaction.finished_block_number

    waiting.wait_for_settle(
        app0,
        registry_address,
        token_address,
        [channel_identifier],
        app0.alarm.sleep_time,
    )

    token_network = views.get_token_network_by_address(
        views.state_from_raiden(app0), token_network_address
    )
    assert token_network

    assert (
        channel_identifier
        not in token_network.partneraddresses_to_channelidentifiers[app1.address]
    )

    assert app0.wal, MSG_BLOCKCHAIN_EVENTS
    assert app0.alarm, MSG_BLOCKCHAIN_EVENTS
    state_changes = app0.wal.storage.get_statechanges_by_range(RANGE_ALL_STATE_CHANGES)

    assert search_for_item(
        state_changes,
        ContractReceiveChannelClosed,
        {
            "token_network_address": token_network_address,
            "channel_identifier": channel_identifier,
            "transaction_from": app1.address,
            "block_number": channel_state.close_transaction.finished_block_number,
        },
    )

    assert search_for_item(
        state_changes,
        ContractReceiveChannelSettled,
        {"token_network_address": token_network_address, "channel_identifier": channel_identifier},
    )

# ... (rest of the type annotations follow the same pattern)
