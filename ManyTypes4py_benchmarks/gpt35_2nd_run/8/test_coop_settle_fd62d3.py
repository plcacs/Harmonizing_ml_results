from raiden.transfer.state import NettingChannelState
from raiden.transfer.state_change import StateChange
from raiden.transfer.events import Event, SendWithdrawRequest, EventInvalidReceivedWithdrawRequest, SendWithdrawConfirmation, SendProcessed, ContractSendChannelCoopSettle
from raiden.transfer.state import CoopSettleState, PendingWithdrawState
from raiden.transfer.state_change import ActionChannelCoopSettle, ReceiveWithdrawConfirmation, ReceiveWithdrawRequest
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.utils import message_identifier_from_prng
from raiden.utils.signer import Signer
from raiden.utils.typing import Address, BlockHash, BlockNumber, ChannelID, Nonce, TokenAmount
from raiden.messages import LockedTransfer
from raiden.tests.utils.factories import make_block_hash
from raiden.tests.unit.channel_state.utils import assert_partner_state, create_channel_from_models, create_model
from raiden.utils.packing import pack_withdraw

def _make_receive_coop_settle_withdraw_confirmation(channel_state: NettingChannelState, pseudo_random_generator: random.Random, signer: Signer, nonce: Nonce, total_withdraw: TokenAmount, withdraw_expiration: BlockNumber) -> ReceiveWithdrawConfirmation:
    ...

def _make_receive_coop_settle_withdraw_request(channel_state: NettingChannelState, pseudo_random_generator: random.Random, signer: Signer, nonce: Nonce, total_withdraw: TokenAmount, withdraw_expiration: BlockNumber) -> ReceiveWithdrawRequest:
    ...

def _assert_coop_settle_state(our_state: NettingChannelState, partner_state: NettingChannelState, our_initial_model: Model, our_current_nonce: Nonce, coop_settle_expiration: BlockNumber, initiated_coop_settle: bool = False) -> None:
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
