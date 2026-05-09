import random
import pytest
from raiden.tests.unit.channel_state.utils import assert_partner_state, create_channel_from_models, create_model
from raiden.tests.utils.events import search_for_item
from raiden.tests.utils.factories import make_block_hash
from raiden.transfer import channel
from raiden.transfer.events import ContractSendChannelCoopSettle, EventInvalidReceivedWithdrawRequest, SendProcessed, SendWithdrawConfirmation, SendWithdrawRequest
from raiden.transfer.state import CoopSettleState, PendingWithdrawState, message_identifier_from_prng
from raiden.transfer.state_change import ActionChannelCoopSettle, ReceiveWithdrawConfirmation, ReceiveWithdrawRequest
from raiden.utils.packing import pack_withdraw
from raiden.utils.signer import LocalSigner

def _make_receive_coop_settle_withdraw_confirmation(
    channel_state: channel.ChannelState, 
    pseudo_random_generator: random.Random, 
    signer: LocalSigner, 
    nonce: int, 
    total_withdraw: int, 
    withdraw_expiration: int
) -> ReceiveWithdrawConfirmation:
    ...

def _make_receive_coop_settle_withdraw_request(
    channel_state: channel.ChannelState, 
    pseudo_random_generator: random.Random, 
    signer: LocalSigner, 
    nonce: int, 
    total_withdraw: int, 
    withdraw_expiration: int
) -> ReceiveWithdrawRequest:
    ...

def _assert_coop_settle_state(
    our_state: channel.ChannelState, 
    partner_state: channel.ChannelState, 
    our_initial_model: create_model.Model, 
    our_current_nonce: int, 
    coop_settle_expiration: int, 
    initiated_coop_settle: bool = False
) -> None:
    ...

def test_initiate() -> None:
    ...

def test_receive_request_while_pending_transfers() -> None:
    ...

def test_receive_request() -> None:
    ...

def test_contract_event() -> None:
    ...

def test_receive_initiator_confirmation_has_no_effect() -> None:
    ...

@pytest.mark.skip(reason='Test yet to be implemented')
def test_clean_state_after_successful_onchain_coop_settle() -> None:
    ...
