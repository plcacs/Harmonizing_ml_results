import random
from unittest.mock import patch
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable

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

MSG_BLOCKCHAIN_EVENTS = 'Waiting for blockchain events requires a running node and alarm task.'

def wait_for_batch_unlock(
    app: RaidenService,
    token_network_address: TokenNetworkAddress,
    receiver: Address,
    sender: Address,
) -> None:
    unlock_event = None
    while not unlock_event:
        gevent.sleep(1)
        assert app.wal, MSG_BLOCKCHAIN_EVENTS
        assert app.alarm, MSG_BLOCKCHAIN_EVENTS
        state_changes = app.wal.storage.get_statechanges_by_range(RANGE_ALL_STATE_CHANGES)
        unlock_event = search_for_item(
            state_changes,
            ContractReceiveChannelBatchUnlock,
            {
                'token_network_address': token_network_address,
                'receiver': receiver,
                'sender': sender,
            },
        )

def is_channel_registered(
    node_app: RaidenService,
    partner_app: RaidenService,
    canonical_identifier: CanonicalIdentifier,
) -> bool:
    """True if the `node_app` has a channel with `partner_app` in its state."""
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
@pytest.mark.parametrize('number_of_nodes', [2])
def test_settle_is_automatically_called(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None:
    """Settle is automatically called by one of the nodes."""
    app0, app1 = raiden_network
    registry_address = app0.default_registry.address
    token_address = token_addresses[0]
    token_network_address = views.get_token_network_address_by_token_address(
        views.state_from_raiden(app0),
        app0.default_registry.address,
        token_address,
    )
    assert token_network_address
    token_network = views.get_token_network_by_address(
        views.state_from_raiden(app0),
        token_network_address,
    )
    assert token_network
    channel_identifier = get_channelstate(app0, app1, token_network_address).identifier
    assert channel_identifier in token_network.partneraddresses_to_channelidentifiers[app1.address]
    RaidenAPI(app1).channel_close(
        registry_address,
        token_address,
        app0.address,
        coop_settle=False,
    )
    waiting.wait_for_close(
        app0,
        registry_address,
        token_address,
        [channel_identifier],
        app0.alarm.sleep_time,
    )
    channel_state = views.get_channelstate_for(
        views.state_from_raiden(app0),
        registry_address,
        token_address,
        app1.address,
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
        views.state_from_raiden(app0),
        token_network_address,
    )
    assert token_network
    assert channel_identifier not in token_network.partneraddresses_to_channelidentifiers[app1.address]
    assert app0.wal, MSG_BLOCKCHAIN_EVENTS
    assert app0.alarm, MSG_BLOCKCHAIN_EVENTS
    state_changes = app0.wal.storage.get_statechanges_by_range(RANGE_ALL_STATE_CHANGES)
    assert search_for_item(
        state_changes,
        ContractReceiveChannelClosed,
        {
            'token_network_address': token_network_address,
            'channel_identifier': channel_identifier,
            'transaction_from': app1.address,
            'block_number': channel_state.close_transaction.finished_block_number,
        },
    )
    assert search_for_item(
        state_changes,
        ContractReceiveChannelSettled,
        {
            'token_network_address': token_network_address,
            'channel_identifier': channel_identifier,
        },
    )

@pytest.mark.flaky
@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_coop_settle_is_automatically_called(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None:
    """Settle is automatically called by one of the nodes."""
    app0, app1 = raiden_network
    registry_address = app0.default_registry.address
    token_address = token_addresses[0]
    token_network_address = views.get_token_network_address_by_token_address(
        views.state_from_raiden(app0),
        app0.default_registry.address,
        token_address,
    )
    assert token_network_address, 'token_network_address must not be None'
    token_network = views.get_token_network_by_address(
        views.state_from_raiden(app0),
        token_network_address,
    )
    assert token_network, 'token_network must exist'
    channel_identifier = get_channelstate(app0, app1, token_network_address).identifier
    channel_state = views.get_channelstate_for(
        views.state_from_raiden(app0),
        registry_address,
        token_address,
        app1.address,
    )
    assert channel_state is not None, 'channel_state must exist'
    app_to_expected_balances = {
        app0: channel_state.our_state.contract_balance + 10,
        app1: channel_state.partner_state.contract_balance - 10,
    }
    payment_id = factories.make_payment_id()
    secret = factories.make_secret(0)
    secrethash = sha256_secrethash(secret)
    RaidenAPI(app1).transfer_and_wait(
        app0.default_registry.address,
        token_address,
        target=TargetAddress(app0.address),
        amount=PaymentAmount(10),
        identifier=payment_id,
        secret=secret,
        secrethash=secrethash,
    )
    waiting.wait_for_received_transfer_result(
        app0,
        payment_id,
        PaymentAmount(10),
        1.0,
        secrethash,
    )
    RaidenAPI(app0).channel_close(
        registry_address,
        token_address,
        app1.address,
        coop_settle=True,
    )
    channel_state = views.get_channelstate_for(
        views.state_from_raiden(app0),
        registry_address,
        token_address,
        app1.address,
    )
    assert channel_state is None, 'channel_state must have been deleted after channel is settled'
    token_network = views.get_token_network_by_address(
        views.state_from_raiden(app0),
        token_network_address,
    )
    assert token_network
    assert channel_identifier not in token_network.partneraddresses_to_channelidentifiers[app1.address]
    assert app0.wal, MSG_BLOCKCHAIN_EVENTS
    assert app0.alarm, MSG_BLOCKCHAIN_EVENTS
    state_changes = app0.wal.storage.get_statechanges_by_range(RANGE_ALL_STATE_CHANGES)
    for app, partner_app in ((app0, app1), (app1, app0)):
        assert app.wal, MSG_BLOCKCHAIN_EVENTS
        assert app.alarm, MSG_BLOCKCHAIN_EVENTS
        state_changes = app.wal.storage.get_statechanges_by_range(RANGE_ALL_STATE_CHANGES)
        assert not search_for_item(
            state_changes,
            ContractReceiveChannelClosed,
            {
                'token_network_address': token_network_address,
                'channel_identifier': channel_identifier,
            },
        )
        assert search_for_item(
            state_changes,
            ContractReceiveChannelWithdraw,
            {
                'token_network_address': token_network_address,
                'channel_identifier': channel_identifier,
            },
        )
        assert search_for_item(
            state_changes,
            ContractReceiveChannelSettled,
            {
                'token_network_address': token_network_address,
                'channel_identifier': channel_identifier,
                'our_transferred_amount': app_to_expected_balances[app],
                'partner_transferred_amount': app_to_expected_balances[partner_app],
            },
        )

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_coop_settle_fails_with_pending_lock(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
) -> None:
    """Settle is automatically called by one of the nodes."""
    app0, app1 = raiden_network
    registry_address = app0.default_registry.address
    token_address = token_addresses[0]
    token_network_address = views.get_token_network_address_by_token_address(
        views.state_from_raiden(app0),
        app0.default_registry.address,
        token_address,
    )
    assert token_network_address, 'token_network_address must not be None'
    channel_identifier = get_channelstate(app0, app1, token_network_address).identifier
    hold_event_handler = app1.raiden_event_handler
    msg = 'hold event handler necessary to control messages'
    assert isinstance(hold_event_handler, HoldRaidenEventHandler), msg
    secret = factories.make_secret()
    secrethash = sha256_secrethash(secret)
    RaidenAPI(app1).transfer_and_wait(
        app0.default_registry.address,
        token_address,
        target=TargetAddress(app0.address),
        amount=PaymentAmount(10),
        secret=secret,
        secrethash=secrethash,
    )
    hold_event_handler.hold_unlock_for(secrethash=secrethash)
    RaidenAPI(app0).channel_close(
        registry_address,
        token_address,
        app1.address,
        coop_settle=True,
    )
    channel_state = views.get_channelstate_for(
        views.state_from_raiden(app0),
        registry_address,
        token_address,
        app1.address,
    )
    assert app0.wal, MSG_BLOCKCHAIN_EVENTS
    assert app0.alarm, MSG_BLOCKCHAIN_EVENTS
    state_changes = app0.wal.storage.get_statechanges_by_range(RANGE_ALL_STATE_CHANGES)
    assert channel_state is not None, 'Channel state should not be deleted'
    assert search_for_item(
        state_changes,
        ContractReceiveChannelClosed,
        {
            'token_network_address': token_network_address,
            'channel_identifier': channel_identifier,
        },
    )
    assert not search_for_item(
        state_changes,
        ContractReceiveChannelSettled,
        {
            'token_network_address': token_network_address,
            'channel_identifier': channel_identifier,
        },
    )

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', [2])
def test_lock_expiry(
    raiden_network: List[RaidenService],
    token_addresses: List[TokenAddress],
    deposit: TokenAmount,
) -> None:
    """Test lock expiry and removal."""
    alice_app, bob_app = raiden_network
    token_address = token_addresses[0]
    token_network_address = views.get_token_network_address_by_token_address(
        views.state_from_raiden(alice_app),
        alice_app.default_registry.address,
        token_address,
    )
    assert token_network_address
    hold_event_handler = bob_app.raiden_event_handler
    wait_message_handler = bob_app.message_handler
    msg = 'hold event handler necessary to control messages'
    assert isinstance(hold_event_handler, HoldRaidenEventHandler), msg
    assert isinstance(wait_message_handler, WaitForMessage), msg
    token_network = views.get_token_network_by_address(
        views.state_from_raiden(alice_app),
        token_network_address,
    )
    assert token_network
    channel_state = get_channelstate(alice_app, bob_app, token_network_address)
    channel_identifier = channel_state.identifier
    assert channel_identifier in token_network.partneraddresses_to_channelidentifiers[bob_app.address]
    alice_to_bob_amount = PaymentAmount(10)
    identifier = factories.make_payment_id()
    target = TargetAddress(bob_app.address)
    transfer_1_secret = factories.make_secret(0)
    transfer_1_secrethash = sha256_secrethash(transfer_1_secret)
    transfer_2_secret = factories.make_secret(1)
    transfer_2_secrethash = sha256_secrethash(transfer_2_secret)
    hold_event_handler.hold_secretrequest_for(secrethash=transfer_1_secrethash)
    transfer1_received = wait_message_handler.wait_for_message(
        LockedTransfer,
        {'lock': {'secrethash': transfer_1_secrethash}},
    )
    transfer2_received = wait_message_handler.wait_for_message(
        LockedTransfer,
        {'lock': {'secrethash': transfer_2_secrethash}},
    )
    remove_expired_lock_received = wait_message_handler.wait_for_message(
        LockExpired,
        {'secrethash': transfer_1_secrethash},
    )
    with patch('raiden.message_handler.decrypt_secret', side_effect=InvalidSecret):
        alice_app.mediated_transfer_async(
            token_network_address=token_network_address,
            amount=alice_to_bob_amount,
            target=target,
            identifier=identifier,
            secret=transfer_1_secret,
            route_states=[create_route_state_for_route([alice_app, bob_app], token_address)],
        )
        transfer1_received.wait()
    alice_bob_channel_state = get_channelstate(alice_app, bob_app, token_network_address)
    lock = channel.get_lock(alice_bob_channel_state.our_state, transfer_1_secrethash)
    assert lock
    assert_synced_channel_state(
        token_network_address,
        alice_app,
        Balance(deposit),
        [lock],
        bob_app,
        Balance(deposit),
        [],
    )
    alice_channel_state = get_channelstate(alice_app, bob_app, token_network_address)
    assert transfer_1_secrethash in alice_channel_state.our_state.secrethashes_to_lockedlocks
    bob_channel_state = get_channelstate(bob_app, alice_app, token_network_address)
    assert transfer_1_secrethash in bob_channel_state.partner_state.secrethashes_to_lockedlocks
    alice_chain_state = views.state_from_raiden(alice_app)
    assert transfer_1_secrethash in alice_chain_state.payment_mapping.secrethashes_to_task
    remove_expired_lock_received.wait()
    alice_channel_state = get_channelstate(alice_app, bob_app, token_network_address)
    assert transfer_1_secrethash not in alice_channel_state.our_state.secrethashes_to_lockedlocks
    bob_channel_state = get_channelstate(bob_app, alice_app, token_network_address)
    assert transfer_1_secrethash not in bob_channel_state.partner_state.secrethashes_to_lockedlocks
    alice_chain_state = views.state_from_raiden(alice_app)
    assert transfer_1_secrethash not in alice_chain_state.payment_mapping.secrethashes_to_task
    alice_to_bob_amount = PaymentAmount(10)
   