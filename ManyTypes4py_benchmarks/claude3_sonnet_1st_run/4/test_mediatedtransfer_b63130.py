from typing import Dict, List, Optional, Tuple, Union, cast
from unittest.mock import Mock, patch
import gevent
import pytest
from eth_utils.crypto import keccak
from raiden.constants import MAXIMUM_PENDING_TRANSFERS
from raiden.exceptions import InvalidSecret, RaidenUnrecoverableError
from raiden.message_handler import MessageHandler
from raiden.messages.transfers import LockedTransfer, RevealSecret, SecretRequest
from raiden.network.pathfinding import PFSConfig, PFSInfo, PFSProxy
from raiden.raiden_service import RaidenService
from raiden.settings import DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS
from raiden.storage.sqlite import RANGE_ALL_STATE_CHANGES
from raiden.tests.utils import factories
from raiden.tests.utils.detect_failure import raise_on_failure
from raiden.tests.utils.events import search_for_item
from raiden.tests.utils.factories import make_secret
from raiden.tests.utils.mediation_fees import get_amount_for_sending_before_and_after_fees
from raiden.tests.utils.network import CHAIN
from raiden.tests.utils.protocol import HoldRaidenEventHandler, WaitForMessage
from raiden.tests.utils.transfer import TransferState, assert_succeeding_transfer_invariants, assert_synced_channel_state, block_timeout_for_transfer_by_secrethash, create_route_state_for_route, transfer, transfer_and_assert_path, wait_assert
from raiden.transfer import views
from raiden.transfer.events import EventPaymentSentFailed, EventPaymentSentSuccess
from raiden.transfer.mediated_transfer.events import SendSecretRequest
from raiden.transfer.mediated_transfer.initiator import calculate_fee_margin
from raiden.transfer.mediated_transfer.mediation_fee import FeeScheduleState
from raiden.transfer.mediated_transfer.state_change import ActionInitMediator, ActionInitTarget
from raiden.transfer.mediated_transfer.tasks import InitiatorTask
from raiden.utils.secrethash import sha256_secrethash
from raiden.utils.typing import Address, BlockExpiration, BlockNumber, BlockTimeout, FeeAmount, PaymentAmount, PaymentID, ProportionalFeeAmount, Secret, TargetAddress, TokenAddress, TokenAmount
from raiden.waiting import wait_for_block

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [2])
def test_transfer_with_secret(raiden_network: List[RaidenService], number_of_nodes: int, deposit: int, token_addresses: List[TokenAddress], network_wait: int) -> None:
    app0, app1 = raiden_network
    token_address = token_addresses[0]
    chain_state = views.state_from_raiden(app0)
    token_network_registry_address = app0.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address)
    amount = PaymentAmount(10)
    secret_hash = transfer(initiator_app=app0, target_app=app1, token_address=token_address, amount=amount, transfer_state=TransferState.LOCKED, identifier=PaymentID(1), timeout=network_wait * number_of_nodes, routes=[[app0, app1]])
    assert isinstance(app1.raiden_event_handler, HoldRaidenEventHandler)
    app1.raiden_event_handler.hold(SendSecretRequest, {'secrethash': secret_hash})
    with gevent.Timeout(20):
        wait_assert(assert_succeeding_transfer_invariants, token_network_address, app0, deposit - amount, [], app1, deposit + amount, [])

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [3])
def test_mediated_transfer(raiden_network: List[RaidenService], number_of_nodes: int, deposit: int, token_addresses: List[TokenAddress], network_wait: int) -> None:
    app0, app1, app2 = raiden_network
    token_address = token_addresses[0]
    chain_state = views.state_from_raiden(app0)
    token_network_registry_address = app0.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address)
    amount = PaymentAmount(10)
    secrethash = transfer(initiator_app=app0, target_app=app2, token_address=token_address, amount=amount, identifier=PaymentID(1), timeout=network_wait * number_of_nodes, routes=[[app0, app1, app2]])
    with block_timeout_for_transfer_by_secrethash(app1, secrethash):
        wait_assert(assert_succeeding_transfer_invariants, token_network_address, app0, deposit - amount, [], app1, deposit + amount, [])
    with block_timeout_for_transfer_by_secrethash(app1, secrethash):
        wait_assert(assert_succeeding_transfer_invariants, token_network_address, app1, deposit - amount, [], app2, deposit + amount, [])

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [1])
def test_locked_transfer_secret_registered_onchain(raiden_network: List[RaidenService], token_addresses: List[TokenAddress], secret_registry_address: Address, retry_timeout: int) -> None:
    app0 = raiden_network[0]
    token_address = token_addresses[0]
    chain_state = views.state_from_raiden(app0)
    token_network_registry_address = app0.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address)
    assert token_network_address, 'token must be registered by the fixtures.'
    amount = PaymentAmount(1)
    target = factories.UNIT_TRANSFER_INITIATOR
    identifier = PaymentID(1)
    transfer_secret = make_secret()
    secret_registry_proxy = app0.proxy_manager.secret_registry(secret_registry_address, block_identifier=chain_state.block_hash)
    secret_registry_proxy.register_secret(secret=transfer_secret)
    block_number = app0.get_block_number()
    wait_for_block(raiden=app0, block_number=BlockNumber(block_number + DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS), retry_timeout=retry_timeout)
    with pytest.raises(RaidenUnrecoverableError):
        app0.mediated_transfer_async(token_network_address=token_network_address, amount=amount, target=TargetAddress(target), identifier=identifier, secret=transfer_secret)
    expiration = BlockExpiration(9999)
    locked_transfer = factories.create(factories.LockedTransferProperties(amount=TokenAmount(amount), target=TargetAddress(app0.address), expiration=expiration, secret=transfer_secret))
    message_handler = MessageHandler()
    message_handler.handle_message_lockedtransfer(app0, locked_transfer)
    assert app0.wal, 'test apps must be started by the fixtures.'
    state_changes = app0.wal.storage.get_statechanges_by_range(RANGE_ALL_STATE_CHANGES)
    transfer_statechange_dispatched = search_for_item(state_changes, ActionInitMediator, {}) or search_for_item(state_changes, ActionInitTarget, {})
    assert not transfer_statechange_dispatched

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [3])
def test_mediated_transfer_with_entire_deposit(raiden_network: List[RaidenService], number_of_nodes: int, token_addresses: List[TokenAddress], deposit: int, network_wait: int) -> None:
    app0, app1, app2 = raiden_network
    token_address = token_addresses[0]
    chain_state = views.state_from_raiden(app0)
    token_network_registry_address = app0.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address)
    assert token_network_address

    def calc_fees(mediator_app: RaidenService, from_app: RaidenService, to_app: RaidenService, amount: PaymentAmount) -> Tuple[PaymentAmount, List[FeeAmount]]:
        assert token_network_address
        from_channel_state = views.get_channelstate_by_token_network_and_partner(chain_state=views.state_from_raiden(mediator_app), token_network_address=token_network_address, partner_address=from_app.address)
        assert from_channel_state
        to_channel_state = views.get_channelstate_by_token_network_and_partner(chain_state=views.state_from_raiden(mediator_app), token_network_address=token_network_address, partner_address=to_app.address)
        assert to_channel_state
        return get_amount_for_sending_before_and_after_fees(amount_to_leave_initiator=amount, channels=[(from_channel_state, to_channel_state)])
    fee_calculation1 = calc_fees(app1, app0, app2, PaymentAmount(200))
    secrethash = transfer_and_assert_path(path=raiden_network, token_address=token_address, amount=fee_calculation1.amount_to_send, identifier=PaymentID(1), timeout=network_wait * number_of_nodes, fee_estimate=FeeAmount(sum(fee_calculation1.mediation_fees)))
    with block_timeout_for_transfer_by_secrethash(app1, secrethash):
        wait_assert(func=assert_succeeding_transfer_invariants, token_network_address=token_network_address, app0=app0, balance0=0, pending_locks0=[], app1=app1, balance1=deposit * 2, pending_locks1=[])
    with block_timeout_for_transfer_by_secrethash(app2, secrethash):
        wait_assert(func=assert_succeeding_transfer_invariants, token_network_address=token_network_address, app0=app1, balance0=3, pending_locks0=[], app1=app2, balance1=deposit * 2 - sum(fee_calculation1.mediation_fees), pending_locks1=[])
    fee_calculation2 = calc_fees(app1, app2, app0, PaymentAmount(deposit * 2 - sum(fee_calculation1.mediation_fees)))
    transfer_and_assert_path(path=list(raiden_network[::-1]), token_address=token_address, amount=fee_calculation2.amount_to_send, identifier=PaymentID(2), timeout=network_wait * number_of_nodes, fee_estimate=FeeAmount(sum(fee_calculation2.mediation_fees)))
    with block_timeout_for_transfer_by_secrethash(app1, secrethash):
        wait_assert(func=assert_succeeding_transfer_invariants, token_network_address=token_network_address, app0=app0, balance0=2 * deposit - sum(fee_calculation1.mediation_fees) - sum(fee_calculation2.mediation_fees), pending_locks0=[], app1=app1, balance1=6, pending_locks1=[])
    with block_timeout_for_transfer_by_secrethash(app2, secrethash):
        wait_assert(func=assert_succeeding_transfer_invariants, token_network_address=token_network_address, app0=app1, balance0=deposit * 2, pending_locks0=[], app1=app2, balance1=0, pending_locks1=[])

@pytest.mark.skip(reason='flaky, see https://github.com/raiden-network/raiden/issues/4694')
@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [3])
def test_mediated_transfer_messages_out_of_order(raiden_network: List[RaidenService], deposit: int, token_addresses: List[TokenAddress], network_wait: int) -> None:
    """Raiden must properly handle repeated locked transfer messages."""
    app0, app1, app2 = raiden_network
    app1_wait_for_message = WaitForMessage()
    app2_wait_for_message = WaitForMessage()
    app1.message_handler = app1_wait_for_message
    app2.message_handler = app2_wait_for_message
    secret = factories.make_secret(0)
    secrethash = sha256_secrethash(secret)
    app1_mediatedtransfer = app1_wait_for_message.wait_for_message(LockedTransfer, {'lock': {'secrethash': secrethash}})
    app2_mediatedtransfer = app2_wait_for_message.wait_for_message(LockedTransfer, {'lock': {'secrethash': secrethash}})
    app1_revealsecret = app1_wait_for_message.wait_for_message(RevealSecret, {'secret': secret})
    app2_revealsecret = app2_wait_for_message.wait_for_message(RevealSecret, {'secret': secret})
    token_address = token_addresses[0]
    chain_state = views.state_from_raiden(app0)
    token_network_registry_address = app0.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address)
    assert token_network_address, 'token must be registered by the fixtures.'
    amount = PaymentAmount(10)
    identifier = PaymentID(1)
    transfer_received = app0.mediated_transfer_async(token_network_address=token_network_address, amount=amount, target=TargetAddress(app2.address), identifier=identifier, secret=secret)
    app2_revealsecret.get(timeout=network_wait)
    mediated_transfer_msg = app2_mediatedtransfer.get_nowait()
    app2.message_handler.handle_message_lockedtransfer(app2, mediated_transfer_msg)
    app1_revealsecret.get(timeout=network_wait)
    app1.message_handler.handle_message_lockedtransfer(app1, app1_mediatedtransfer.get_nowait())
    transfer_received.payment_done.wait()
    with block_timeout_for_transfer_by_secrethash(app1, secrethash):
        wait_assert(assert_succeeding_transfer_invariants, token_network_address, app0, deposit - amount, [], app1, deposit + amount, [])
    with block_timeout_for_transfer_by_secrethash(app2, secrethash):
        wait_assert(assert_succeeding_transfer_invariants, token_network_address, app1, deposit - amount, [], app2, deposit + amount, [])

@raise_on_failure
@pytest.mark.parametrize('number_of_nodes', (3,))
@pytest.mark.parametrize('channels_per_node', (CHAIN,))
def test_mediated_transfer_calls_pfs(raiden_chain: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    app0, app1, app2 = raiden_chain
    token_address = token_addresses[0]
    chain_state = views.state_from_raiden(app0)
    token_network_registry_address = app0.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address)
    assert token_network_address, "Fixture token_addresses don't have correspoding token_network"
    with patch.object(app0.pfs_proxy, 'query_paths', return_value=([], None)) as patched:
        app0.mediated_transfer_async(token_network_address=token_network_address, amount=PaymentAmount(10), target=TargetAddress(app1.address), identifier=PaymentID(1), secret=Secret(b'1' * 32))
        assert not patched.called
    pfs_config = PFSConfig(info=PFSInfo(url='mock-address', chain_id=app0.rpc_client.chain_id, token_network_registry_address=token_network_registry_address, user_deposit_address=factories.make_address(), payment_address=factories.make_address(), confirmed_block_number=chain_state.block_number, message='', operator='', version='', price=TokenAmount(0), matrix_server='http://matrix.example.com'), maximum_fee=TokenAmount(100), iou_timeout=BlockTimeout(100), max_paths=5)
    app0.pfs_proxy = PFSProxy(pfs_config)
    with patch.object(app0.pfs_proxy, 'query_paths', return_value=([], None)) as patched:
        app0.mediated_transfer_async(token_network_address=token_network_address, amount=PaymentAmount(11), target=TargetAddress(app2.address), identifier=PaymentID(2), secret=Secret(b'2' * 32))
        assert patched.call_count == 1
        locked_transfer = factories.create(factories.LockedTransferProperties(amount=TokenAmount(5), initiator=factories.HOP1, target=TargetAddress(app2.address), sender=Address(factories.HOP1), pkey=factories.HOP1_KEY, token=token_address, canonical_identifier=factories.make_canonical_identifier(token_network_address=token_network_address)))
        app0.on_messages([locked_transfer])
        assert patched.call_count == 1

@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [3])
@patch('raiden.message_handler.decrypt_secret', side_effect=InvalidSecret)
def test_mediated_transfer_with_node_consuming_more_than_allocated_fee(decrypt_patch: Mock, raiden_network: List[RaidenService], number_of_nodes: int, deposit: int, token_addresses: List[TokenAddress], network_wait: int) -> None:
    """
    Tests a mediator node consuming more fees than allocated.
    Which means that the initiator will not reveal the secret
    to the target.
    """
    app0, app1, app2 = raiden_network
    token_address = token_addresses[0]
    chain_state = views.state_from_raiden(app0)
    token_network_registry_address = app0.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address)
    assert token_network_address
    amount = PaymentAmount(100)
    fee = FeeAmount(5)
    fee_margin = calculate_fee_margin(amount, fee)
    app1_app2_channel_state = views.get_channelstate_by_token_network_and_partner(chain_state=views.state_from_raiden(app1), token_network_address=token_network_address, partner_address=app2.address)
    assert app1_app2_channel_state
    app1_app2_channel_state.fee_schedule = FeeScheduleState(flat=FeeAmount(fee * 2))
    secret = factories.make_secret(0)
    secrethash = sha256_secrethash(secret)
    wait_message_handler = WaitForMessage()
    app0.message_handler = wait_message_handler
    secret_request_received = wait_message_handler.wait_for_message(SecretRequest, {'secrethash': secrethash})
    app0.mediated_transfer_async(token_network_address=token_network_address, amount=amount, target=TargetAddress(app2.address), identifier=PaymentID(1), secret=secret, route_states=[create_route_state_for_route(apps=raiden_network, token_address=token_address, fee_estimate=fee)])
    app0_app1_channel_state = views.get_channelstate_by_token_network_and_partner(chain_state=views.state_from_raiden(app0), token_network_address=token_network_address, partner_address=app1.address)
    assert app0_app1_channel_state
    msg = 'App0 should have the transfer in secrethashes_to_lockedlocks'
    assert secrethash in app0_app1_channel_state.our_state.secrethashes_to_lockedlocks, msg
    msg = 'App0 should have locked the amount + fee'
    lock_amount = app0_app1_channel_state.our_state.secrethashes_to_lockedlocks[secrethash].amount
    assert lock_amount == amount + fee + fee_margin, msg
    secret_request_received.wait()
    app0_chain_state = views.state_from_raiden(app0)
    initiator_task = cast(InitiatorTask, app0_chain_state.payment_mapping.secrethashes_to_task[secrethash])
    msg = 'App0 should have never revealed the secret'
    transfer_state = initiator_task.manager_state.initiator_transfers[secrethash].transfer_state
    assert transfer_state != 'transfer_secret_revealed', msg

@raise_on_failure
@pytest.mark.parametrize('case_no', range(8))
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [4])
def test_mediated_transfer_with_fees(raiden_network: List[RaidenService], number_of_nodes: int, deposit: int, token_addresses: List[TokenAddress], network_wait: int, case_no: int) -> None:
    """
    Test mediation with a variety of fee schedules
    """
    apps = raiden_network
    token_address = token_addresses[0]
    chain_state = views.state_from_raiden(apps[0])
    token_network_registry_address = apps[0].default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address)
    assert token_network_address

    def set_fee_schedule(app: RaidenService, other_app: RaidenService, fee_schedule: FeeScheduleState) -> None:
        assert token_network_address
        channel_state = views.get_channelstate_by_token_network_and_partner(chain_state=views.state_from_raiden(app), token_network_address=token_network_address, partner_address=other_app.address)
        assert channel_state
        channel_state.fee_schedule = fee_schedule

    def assert_balances(expected_transferred_amounts: List[int]) -> None:
        assert token_network_address
        for i, transferred_amount in enumerate(expected_transferred_amounts):
            assert_synced_channel_state(token_network_address=token_network_address, app0=apps[i], balance0=deposit - transferred_amount, pending_locks0=[], app1=apps[i + 1], balance1=deposit + transferred_amount, pending_locks1=[])
    amount = PaymentAmount(35)
    fee_without_margin = FeeAmount(20)
    fee = FeeAmount(fee_without_margin + calculate_fee_margin(amount, fee_without_margin))
    no_fees = FeeScheduleState(flat=FeeAmount(0), proportional=ProportionalFeeAmount(0), imbalance_penalty=None)
    no_fees_no_cap = FeeScheduleState(flat=FeeAmount(0), proportional=ProportionalFeeAmount(0), imbalance_penalty=None, cap_fees=False)
    cases: List[Dict[str, Union[List[Optional[FeeScheduleState]], List[int]]]] = [dict(fee_schedules=[no_fees, no_fees, no_fees], incoming_fee_schedules=[no_fees, no_fees, no_fees], expected_transferred_amounts=[amount + fee, amount + fee, amount + fee]), dict(fee_schedules=[no_fees, FeeScheduleState(flat=fee), no_fees], incoming_fee_schedules=[no_fees, no_fees, no_fees], expected_transferred_amounts=[amount + fee, amount, amount]), dict(fee_schedules=[no_fees, FeeScheduleState(proportional=ProportionalFeeAmount(int(200000.0))), no_fees], incoming_fee_schedules=[no_fees, no_fees, no_fees], expected_transferred_amounts=[amount + fee, amount + fee - 9, amount + fee - 9]), dict(fee_schedules=[no_fees, FeeScheduleState(proportional=ProportionalFeeAmount(int(200000.0))), FeeScheduleState(proportional=ProportionalFeeAmount(int(200000.0)))], incoming_fee_schedules=[no_fees, no_fees, no_fees], expected_transferred_amounts=[amount + fee, amount + fee - 9, amount + fee - 9 - 8]), dict(fee_schedules=[no_fees, FeeScheduleState(imbalance_penalty=[(0, 200), (1000, 0)]), no_fees], incoming_fee_schedules=[no_fees, no_fees, no_fees], expected_transferred_amounts=[amount + fee, amount + fee - (amount + fee) // 5 + 2, amount + fee - (amount + fee) // 5 + 2]), dict(fee_schedules=[no_fees, no_fees, no_fees], incoming_fee_schedules=[FeeScheduleState(imbalance_penalty=[(0, 0), (1000, 200)]), None, None], expected_transferred_amounts=[amount + fee, amount + fee - (amount + fee) // 5, amount + fee - (amount + fee) // 5]), dict(fee_schedules=[no_fees, FeeScheduleState(cap_fees=False, imbalance_penalty=[(0, 0), (1000, 50)]), no_fees], incoming_fee_schedules=[no_fees_no_cap, no_fees, no_fees], expected_transferred_amounts=[amount + fee, amount + fee + 3, amount + fee + 3]), dict(fee_schedules=[no_fees, FeeScheduleState(cap_fees=True, imbalance_penalty=[(0, 0), (1000, 50)]), no_fees], incoming_fee_schedules=[no_fees, no_fees, no_fees], expected_transferred_amounts=[amount + fee, amount + fee, amount + fee])]
    case = cases[case_no]
    for i, fee_schedule in enumerate(case.get('fee_schedules', [])):
        if fee_schedule:
            set_fee_schedule(apps[i], apps[i + 1], fee_schedule)
    for i, fee_schedule in enumerate(case.get('incoming_fee_schedules', [])):
        if fee_schedule:
            set_fee_schedule(apps[i + 1], apps[i], fee_schedule)
    disable_max_mediation_fee_patch = patch('raiden.transfer.mediated_transfer.initiator.MAX_MEDIATION_FEE_PERC', new=10000)
    with disable_max_mediation_fee_patch:
        transfer_and_assert_path(path=raiden_network, token_address=token_address, amount=amount, identifier=PaymentID(2), fee_estimate=fee_without_margin)
    assert_balances(case['expected_transferred_amounts'])

@pytest.mark.flaky
@raise_on_failure
@pytest.mark.parametrize('channels_per_node', [CHAIN])
@pytest.mark.parametrize('number_of_nodes', [2])
@pytest.mark.parametrize('settle_timeout', [1000])
def test_max_locks_reached(raiden_network: List[RaidenService], token_addresses: List[TokenAddress]) -> None:
    """
    Test that once the limit of concurrent transfers is reached for a channel, no new
    transfers can be started.
    """
    app0, app1 = raiden_network
    app0.pfs_proxy.set_services((app0, app1))
    app1.pfs_proxy.set_services((app0, app1))
    token_address = token_addresses[0]
    token_network_registry_address = app0.default_registry.address
    token_network_address = views.get_token_network_address_by_token_address(views.state_from_raiden(app0), token_network_registry_address, token_address)
    assert token_network_address is not None
    hold_event_handler_app1 = app1.raiden_event_handler
    assert isinstance(hold_event_handler_app1, HoldRaidenEventHandler)
    channel_state = views.get_channelstate_for(views.state_from_raiden(app0), token_network_registry_address, token_address, app1.address)
    assert channel_state is not None
    statuses = []
    secrethashes = []
    for i in range(MAXIMUM_PENDING_TRANSFERS):
        secret = Secret(keccak(i))
        secrethash = sha256_secrethash(secret)
        secrethashes.append(secrethash)
        result = hold_event_handler_app1.hold_secretreveal_for(secrethash=secrethash)
        status = app0.mediated_transfer_async(token_network_address=token_network_address, amount=PaymentAmount(1), target=TargetAddress(app1.address), identifier=PaymentID(i), secret=secret, lock_timeout=channel_state.settle_timeout).payment_done
        statuses.append(status)
        result.wait()
    channel_state = views.get_channelstate_for(views.state_from_raiden(app0), token_network_registry_address, token_address, app1.address)
    assert channel_state is not None
    locks = channel_state.our_state.pending_locks.locks
    assert len(locks) == MAXIMUM_PENDING_TRANSFERS
    payment_id = 1111
    secret = Secret(keccak(payment_id))
    secrethash = sha256_secrethash(secret)
    failed_status = app0.mediated_transfer_async(token_network_address=token_network_address, amount=PaymentAmount(1), target=TargetAddress(app1.address), identifier=PaymentID(payment_id), secret=secret)
    failed_status.payment_done.wait()
    result = failed_status.payment_done.value
    assert isinstance(result, EventPaymentSentFailed)
    assert result.reason == 'You have no suitable channel to initiate this payment.'
    for secrethash in secrethashes:
        hold_event_handler_app1.release_secretreveal_for(app1, secrethash=secrethash)
    gevent.joinall(statuses, timeout=20, raise_error=True)
    for status in statuses:
        assert isinstance(status.value, EventPaymentSentSuccess)
