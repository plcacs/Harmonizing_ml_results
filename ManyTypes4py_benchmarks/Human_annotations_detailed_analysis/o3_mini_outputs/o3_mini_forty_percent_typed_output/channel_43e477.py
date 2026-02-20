#!/usr/bin/env python3
# pylint: disable=too-many-lines,too-many-branches,too-many-statements
from enum import Enum
import random
from functools import singledispatch
from typing import Any, Dict, List, Optional, Tuple, Union

from eth_utils import keccak

# Assume the following types are imported from the Raiden project’s typing and state modules
# (In the actual code these would be imported appropriately)
Address = str
AddressMetadata = Dict[str, Any]
Balance = int
BlockExpiration = int
BlockHash = str
BlockNumber = int
BlockTimeout = int
ChainID = int
ChannelID = int
LockedAmount = int
Nonce = int
PaymentID = int
PaymentWithFeeAmount = int
Secret = bytes
SecretHash = bytes
Signature = bytes
TokenAmount = int
WithdrawAmount = int
EncodedData = bytes

# Placeholder types for domain-specific objects
class ChannelState(Enum):
    STATE_OPENED = "opened"
    STATE_CLOSED = "closed"
    STATE_SETTLED = "settled"
    STATE_SETTLING = "settling"
    STATE_UNUSABLE = "unusable"

class SuccessOrError:
    def __init__(self, as_error_message: str = "") -> None:
        self.as_error_message = as_error_message
        self.ok = as_error_message == ""

    def __bool__(self) -> bool:
        return self.ok

class PendingLocksState:
    def __init__(self, locks: List[EncodedData]) -> None:
        self.locks = locks

class FeeScheduleState:
    def __init__(self, cap_fees: Any, flat: Any, proportional: Any, imbalance_penalty: Any) -> None:
        self.cap_fees = cap_fees
        self.flat = flat
        self.proportional = proportional
        self.imbalance_penalty = imbalance_penalty

# Domain-specific state types (placeholders)
class NettingChannelState:
    def __init__(self) -> None:
        self.identifier: ChannelID = 0
        self.token_address: Address = ""
        self.chain_id: ChainID = 0
        self.settle_timeout: int = 0
        self.reveal_timeout: int = 0
        self.close_transaction: Optional[Any] = None
        self.settle_transaction: Optional[Any] = None
        self.update_transaction: Optional[Any] = None
        self.our_state: NettingChannelEndState = NettingChannelEndState()
        self.partner_state: NettingChannelEndState = NettingChannelEndState()
        self.fee_schedule: FeeScheduleState = FeeScheduleState(None, None, None, 0)
        self.partner_total_withdraw: WithdrawAmount = 0
        self.our_total_withdraw: WithdrawAmount = 0
        self.chain_id = 1
        self.token_network_address: Address = ""
        self.canonical_identifier: Any = None

class NettingChannelEndState:
    def __init__(self) -> None:
        self.address: Address = ""
        self.balance_proof: Optional['BalanceProofSignedState'] = None
        self.nonce: Nonce = 0
        self.pending_locks: PendingLocksState = PendingLocksState([])
        self.secrethashes_to_lockedlocks: Dict[SecretHash, 'HashTimeLockState'] = {}
        self.secrethashes_to_unlockedlocks: Dict[SecretHash, 'UnlockPartialProofState'] = {}
        self.secrethashes_to_onchain_unlockedlocks: Dict[SecretHash, 'UnlockPartialProofState'] = {}
        self.withdraws_pending: Dict[WithdrawAmount, 'PendingWithdrawState'] = {}
        self.withdraws_expired: List['ExpiredWithdrawState'] = []
        self.offchain_total_withdraw: WithdrawAmount = 0
        self.onchain_total_withdraw: WithdrawAmount = 0
        self.contract_balance: Balance = 0
        self.initiated_coop_settle: Optional['CoopSettleState'] = None
        self.onchain_locksroot: Any = None

class BalanceProofSignedState:
    def __init__(self, nonce: Nonce = 0, transferred_amount: TokenAmount = 0, locked_amount: LockedAmount = 0,
                 locksroot: Any = None, canonical_identifier: Any = None, signature: Optional[Signature] = None) -> None:
        self.nonce = nonce
        self.transferred_amount = transferred_amount
        self.locked_amount = locked_amount
        self.locksroot = locksroot
        self.canonical_identifier = canonical_identifier
        self.signature = signature
        self.sender: Address = ""

class BalanceProofUnsignedState:
    def __init__(self, nonce: Nonce, transferred_amount: TokenAmount, locked_amount: LockedAmount,
                 locksroot: Any, canonical_identifier: Any) -> None:
        self.nonce = nonce
        self.transferred_amount = transferred_amount
        self.locked_amount = locked_amount
        self.locksroot = locksroot
        self.canonical_identifier = canonical_identifier

class HashTimeLockState:
    def __init__(self, amount: TokenAmount = 0, expiration: BlockExpiration = 0, secrethash: SecretHash = b"", encoded: Optional[bytes] = None) -> None:
        self.amount = amount
        self.expiration = expiration
        self.secrethash = secrethash
        self.encoded: bytes = encoded if encoded is not None else b""

class UnlockPartialProofState:
    def __init__(self, lock: HashTimeLockState, secret: Secret) -> None:
        self.lock = lock
        self.secret = secret
        self.encoded: bytes = lock.encoded

class PendingWithdrawState:
    def __init__(self, total_withdraw: WithdrawAmount, nonce: Nonce, expiration: BlockExpiration,
                 recipient_metadata: AddressMetadata) -> None:
        self.total_withdraw = total_withdraw
        self.nonce = nonce
        self.expiration = expiration
        self.recipient_metadata = recipient_metadata

class ExpiredWithdrawState:
    def __init__(self, total_withdraw: WithdrawAmount, expiration: BlockExpiration, nonce: Nonce,
                 recipient_metadata: AddressMetadata) -> None:
        self.total_withdraw = total_withdraw
        self.expiration = expiration
        self.nonce = nonce
        self.recipient_metadata = recipient_metadata

class CoopSettleState:
    def __init__(self, total_withdraw_initiator: WithdrawAmount, total_withdraw_partner: WithdrawAmount,
                 expiration: BlockExpiration, partner_signature_request: Optional[Signature] = None,
                 partner_signature_confirmation: Optional[Signature] = None) -> None:
        self.total_withdraw_initiator = total_withdraw_initiator
        self.total_withdraw_partner = total_withdraw_partner
        self.expiration = expiration
        self.partner_signature_request = partner_signature_request
        self.partner_signature_confirmation = partner_signature_confirmation
        self.transaction: Optional[Any] = None

# Message / event types (placeholders)
class Event:
    pass

class SendLockedTransfer(Event):
    def __init__(self, recipient: Address,
                 recipient_metadata: Optional[AddressMetadata],
                 message_identifier: int,
                 transfer: Any,
                 canonical_identifier: Any) -> None:
        self.recipient = recipient
        self.recipient_metadata = recipient_metadata
        self.message_identifier = message_identifier
        self.transfer = transfer
        self.canonical_identifier = canonical_identifier

class SendUnlock(Event):
    def __init__(self, recipient: Address,
                 recipient_metadata: Optional[AddressMetadata],
                 message_identifier: int,
                 payment_identifier: PaymentID,
                 token_address: Address,
                 secret: Secret,
                 balance_proof: BalanceProofUnsignedState,
                 canonical_identifier: Any) -> None:
        self.recipient = recipient
        self.recipient_metadata = recipient_metadata
        self.message_identifier = message_identifier
        self.payment_identifier = payment_identifier
        self.token_address = token_address
        self.secret = secret
        self.balance_proof = balance_proof
        self.canonical_identifier = canonical_identifier

class SendWithdrawRequest(Event):
    def __init__(self, canonical_identifier: Any,
                 recipient: Address,
                 message_identifier: int,
                 total_withdraw: WithdrawAmount,
                 participant: Address,
                 nonce: Nonce,
                 expiration: BlockExpiration,
                 recipient_metadata: AddressMetadata,
                 coop_settle: bool) -> None:
        self.canonical_identifier = canonical_identifier
        self.recipient = recipient
        self.message_identifier = message_identifier
        self.total_withdraw = total_withdraw
        self.participant = participant
        self.nonce = nonce
        self.expiration = expiration
        self.recipient_metadata = recipient_metadata
        self.coop_settle = coop_settle

class SendWithdrawExpired(Event):
    def __init__(self, recipient: Address,
                 recipient_metadata: AddressMetadata,
                 canonical_identifier: Any,
                 message_identifier: int,
                 total_withdraw: WithdrawAmount,
                 participant: Address,
                 expiration: BlockExpiration,
                 nonce: Nonce) -> None:
        self.recipient = recipient
        self.recipient_metadata = recipient_metadata
        self.canonical_identifier = canonical_identifier
        self.message_identifier = message_identifier
        self.total_withdraw = total_withdraw
        self.participant = participant
        self.expiration = expiration
        self.nonce = nonce

class SendWithdrawConfirmation(Event):
    def __init__(self, canonical_identifier: Any,
                 recipient: Address,
                 recipient_metadata: Optional[AddressMetadata],
                 message_identifier: int,
                 total_withdraw: WithdrawAmount,
                 participant: Address,
                 nonce: Nonce,
                 expiration: BlockExpiration) -> None:
        self.canonical_identifier = canonical_identifier
        self.recipient = recipient
        self.recipient_metadata = recipient_metadata
        self.message_identifier = message_identifier
        self.total_withdraw = total_withdraw
        self.participant = participant
        self.nonce = nonce
        self.expiration = expiration

class SendProcessed(Event):
    def __init__(self, recipient: Address,
                 recipient_metadata: Optional[AddressMetadata],
                 message_identifier: int,
                 canonical_identifier: Any) -> None:
        self.recipient = recipient
        self.recipient_metadata = recipient_metadata
        self.message_identifier = message_identifier
        self.canonical_identifier = canonical_identifier

class SendLockExpired(Event):
    def __init__(self, recipient: Address,
                 recipient_metadata: Optional[AddressMetadata],
                 canonical_identifier: Any,
                 message_identifier: int,
                 balance_proof: BalanceProofUnsignedState,
                 secrethash: SecretHash) -> None:
        self.recipient = recipient
        self.recipient_metadata = recipient_metadata
        self.canonical_identifier = canonical_identifier
        self.message_identifier = message_identifier
        self.balance_proof = balance_proof
        self.secrethash = secrethash

class EventInvalidActionCoopSettle(Event):
    def __init__(self, attempted_withdraw: WithdrawAmount, reason: str) -> None:
        self.attempted_withdraw = attempted_withdraw
        self.reason = reason

class EventInvalidActionSetRevealTimeout(Event):
    def __init__(self, reveal_timeout: int, reason: str) -> None:
        self.reveal_timeout = reveal_timeout
        self.reason = reason

class EventInvalidActionWithdraw(Event):
    def __init__(self, attempted_withdraw: WithdrawAmount, reason: str) -> None:
        self.attempted_withdraw = attempted_withdraw
        self.reason = reason

class EventInvalidReceivedLockedTransfer(Event):
    def __init__(self, payment_identifier: PaymentID, reason: str) -> None:
        self.payment_identifier = payment_identifier
        self.reason = reason

class EventInvalidReceivedLockExpired(Event):
    def __init__(self, secrethash: SecretHash, reason: str) -> None:
        self.secrethash = secrethash
        self.reason = reason

class EventInvalidReceivedTransferRefund(Event):
    def __init__(self, payment_identifier: PaymentID, reason: str) -> None:
        self.payment_identifier = payment_identifier
        self.reason = reason

class EventInvalidReceivedUnlock(Event):
    def __init__(self, secrethash: SecretHash, reason: str) -> None:
        self.secrethash = secrethash
        self.reason = reason

class EventInvalidReceivedWithdraw(Event):
    def __init__(self, attempted_withdraw: WithdrawAmount, reason: str) -> None:
        self.attempted_withdraw = attempted_withdraw
        self.reason = reason

class EventInvalidReceivedWithdrawExpired(Event):
    def __init__(self, attempted_withdraw: WithdrawAmount, reason: str) -> None:
        self.attempted_withdraw = attempted_withdraw
        self.reason = reason

class ContractSendChannelClose(Event):
    def __init__(self, canonical_identifier: Any,
                 balance_proof: Optional[BalanceProofSignedState],
                 triggered_by_block_hash: BlockHash) -> None:
        self.canonical_identifier = canonical_identifier
        self.balance_proof = balance_proof
        self.triggered_by_block_hash = triggered_by_block_hash

class ContractSendChannelCoopSettle(Event):
    def __init__(self, canonical_identifier: Any,
                 our_total_withdraw: WithdrawAmount,
                 partner_total_withdraw: WithdrawAmount,
                 expiration: BlockExpiration,
                 signature_our_withdraw: Signature,
                 signature_partner_withdraw: Signature,
                 triggered_by_block_hash: BlockHash) -> None:
        self.canonical_identifier = canonical_identifier
        self.our_total_withdraw = our_total_withdraw
        self.partner_total_withdraw = partner_total_withdraw
        self.expiration = expiration
        self.signature_our_withdraw = signature_our_withdraw
        self.signature_partner_withdraw = signature_partner_withdraw
        self.triggered_by_block_hash = triggered_by_block_hash

class ContractSendChannelSettle(Event):
    def __init__(self, canonical_identifier: Any, triggered_by_block_hash: BlockHash) -> None:
        self.canonical_identifier = canonical_identifier
        self.triggered_by_block_hash = triggered_by_block_hash

class ContractSendChannelUpdateTransfer(Event):
    def __init__(self, expiration: BlockExpiration,
                 balance_proof: BalanceProofSignedState,
                 triggered_by_block_hash: BlockHash) -> None:
        self.expiration = expiration
        self.balance_proof = balance_proof
        self.triggered_by_block_hash = triggered_by_block_hash

class ContractSendChannelBatchUnlock(Event):
    def __init__(self, canonical_identifier: Any, sender: Address, triggered_by_block_hash: BlockHash) -> None:
        self.canonical_identifier = canonical_identifier
        self.sender = sender
        self.triggered_by_block_hash = triggered_by_block_hash

class ContractReceiveChannelSettled(Event):
    def __init__(self, channel_identifier: ChannelID,
                 our_onchain_locksroot: Any,
                 partner_onchain_locksroot: Any,
                 our_transferred_amount: TokenAmount,
                 partner_transferred_amount: TokenAmount,
                 block_hash: BlockHash) -> None:
        self.channel_identifier = channel_identifier
        self.our_onchain_locksroot = our_onchain_locksroot
        self.partner_onchain_locksroot = partner_onchain_locksroot
        self.our_transferred_amount = our_transferred_amount
        self.partner_transferred_amount = partner_transferred_amount
        self.block_hash = block_hash

class ContractReceiveChannelWithdraw(Event):
    def __init__(self, participant: Address, total_withdraw: WithdrawAmount, fee_config: Any) -> None:
        self.participant = participant
        self.total_withdraw = total_withdraw
        self.fee_config = fee_config

class ContractReceiveChannelDeposit(Event):
    pass

class ContractReceiveChannelBatchUnlock(Event):
    pass

class ContractReceiveChannelClosed(Event):
    pass

# State change types (placeholders)
class StateChange:
    pass

class ActionChannelClose(StateChange):
    def __init__(self, channel_identifier: ChannelID) -> None:
        self.channel_identifier = channel_identifier

class ActionChannelCoopSettle(StateChange):
    def __init__(self, recipient_metadata: AddressMetadata) -> None:
        self.recipient_metadata = recipient_metadata

class ActionChannelWithdraw(StateChange):
    def __init__(self, total_withdraw: WithdrawAmount, recipient_metadata: AddressMetadata) -> None:
        self.total_withdraw = total_withdraw
        self.recipient_metadata = recipient_metadata

class ActionChannelSetRevealTimeout(StateChange):
    def __init__(self, reveal_timeout: int) -> None:
        self.reveal_timeout = reveal_timeout

class ReceiveUnlock(StateChange):
    def __init__(self, balance_proof: BalanceProofSignedState, message_identifier: int, secrethash: SecretHash) -> None:
        self.balance_proof = balance_proof
        self.message_identifier = message_identifier
        self.secrethash = secrethash

class ReceiveWithdrawRequest(StateChange):
    def __init__(self, canonical_identifier: Any, total_withdraw: WithdrawAmount, sender: Address, nonce: Nonce,
                 expiration: BlockExpiration, sender_metadata: AddressMetadata, coop_settle: bool, signature: Signature, message_identifier: int) -> None:
        self.canonical_identifier = canonical_identifier
        self.total_withdraw = total_withdraw
        self.sender = sender
        self.nonce = nonce
        self.expiration = expiration
        self.sender_metadata = sender_metadata
        self.coop_settle = coop_settle
        self.signature = signature
        self.message_identifier = message_identifier

class ReceiveWithdrawConfirmation(StateChange):
    def __init__(self, canonical_identifier: Any, total_withdraw: WithdrawAmount, sender: Address, nonce: Nonce,
                 expiration: BlockExpiration, signature: Signature, message_identifier: int) -> None:
        self.canonical_identifier = canonical_identifier
        self.total_withdraw = total_withdraw
        self.sender = sender
        self.nonce = nonce
        self.expiration = expiration
        self.signature = signature
        self.message_identifier = message_identifier

class ReceiveWithdrawExpired(StateChange):
    def __init__(self, total_withdraw: WithdrawAmount, nonce: Nonce, message_identifier: int) -> None:
        self.total_withdraw = total_withdraw
        self.nonce = nonce
        self.message_identifier = message_identifier

class ReceiveLockExpired(StateChange):
    def __init__(self, balance_proof: BalanceProofSignedState, secrethash: SecretHash, message_identifier: int) -> None:
        self.balance_proof = balance_proof
        self.secrethash = secrethash
        self.message_identifier = message_identifier

class ReceiveTransferRefund(StateChange):
    def __init__(self, transfer: Any) -> None:
        self.transfer = transfer

class Block(StateChange):
    def __init__(self, block_number: BlockNumber, block_hash: BlockHash) -> None:
        self.block_number = block_number
        self.block_hash = block_hash

# Transition result type
class TransitionResult:
    def __init__(self, new_state: Optional[NettingChannelState], events: List[Event]) -> None:
        self.new_state = new_state
        self.events = events

# Fee calculation placeholder
def calculate_imbalance_fees(channel_capacity: TokenAmount, proportional_imbalance_fee: Any) -> Any:
    return 0

def get_proportional_imbalance_fee(token_address: Address) -> Any:
    return 0

# Utility functions placeholders
def pack_withdraw(canonical_identifier: Any, participant: Address, total_withdraw: WithdrawAmount, expiration_block: BlockExpiration) -> bytes:
    return b""

def pack_balance_proof(nonce: Nonce, balance_hash: bytes, additional_hash: bytes, canonical_identifier: Any) -> bytes:
    return b""

def hash_balance_data(transferred_amount: TokenAmount, locked_amount: LockedAmount, locksroot: Any) -> bytes:
    return b""

def to_checksum_address(address: Address) -> Address:
    return address

def message_identifier_from_prng(prng: random.Random) -> int:
    return prng.randint(0, 2**31 - 1)

def is_valid_signature(*, data: bytes, signature: Signature, sender_address: Address) -> SuccessOrError:
    # Placeholder signature verification
    return SuccessOrError()

# Function implementations with complete type annotations

def get_safe_initial_expiration(block_number: BlockNumber, reveal_timeout: BlockTimeout, lock_timeout: Optional[BlockTimeout] = None) -> BlockExpiration:
    if lock_timeout is not None:
        expiration: BlockExpiration = block_number + lock_timeout
    else:
        expiration = block_number + reveal_timeout * 2
    return expiration

def get_sender_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    return expiration + 10

def get_receiver_expiration_threshold(expiration: BlockExpiration) -> BlockExpiration:
    return expiration + 5

def is_channel_usable_for_mediation(channel_state: NettingChannelState, transfer_amount: PaymentWithFeeAmount, lock_timeout: BlockTimeout) -> bool:
    return True

def is_channel_usable_for_new_transfer(channel_state: NettingChannelState, transfer_amount: PaymentWithFeeAmount, lock_timeout: BlockTimeout) -> ChannelState:
    return ChannelState.STATE_OPENED

def is_lock_pending(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_lockedlocks

def is_lock_locked(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_lockedlocks

def is_lock_expired(end_state: NettingChannelEndState, lock: HashTimeLockState, block_number: BlockNumber, lock_expiration_threshold: BlockExpiration) -> SuccessOrError:
    if block_number > lock_expiration_threshold:
        return SuccessOrError()
    return SuccessOrError("Lock not expired")

def is_transfer_expired(transfer: BalanceProofSignedState, affected_channel: NettingChannelState, block_number: BlockNumber) -> bool:
    return False

def is_withdraw_expired(block_number: BlockNumber, expiration_threshold: BlockExpiration) -> bool:
    return block_number >= expiration_threshold

def is_secret_known(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_unlockedlocks

def is_secret_known_offchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_unlockedlocks

def is_secret_known_onchain(end_state: NettingChannelEndState, secrethash: SecretHash) -> bool:
    return secrethash in end_state.secrethashes_to_onchain_unlockedlocks

def is_valid_channel_total_withdraw(channel_total_withdraw: TokenAmount) -> bool:
    return channel_total_withdraw >= 0

def is_valid_withdraw(withdraw_request: Any) -> SuccessOrError:
    packed = pack_withdraw(withdraw_request.canonical_identifier, withdraw_request.participant, withdraw_request.total_withdraw, withdraw_request.expiration)
    return is_valid_signature(data=packed, signature=withdraw_request.signature, sender_address=withdraw_request.sender)

def is_valid_balanceproof_signature(balance_proof: BalanceProofSignedState, sender_address: Address) -> SuccessOrError:
    data_that_was_signed = pack_balance_proof(nonce=balance_proof.nonce,
                                              balance_hash=hash_balance_data(balance_proof.transferred_amount, balance_proof.locked_amount, balance_proof.locksroot),
                                              additional_hash=b'',
                                              canonical_identifier=balance_proof.canonical_identifier)
    return is_valid_signature(data=data_that_was_signed, signature=balance_proof.signature, sender_address=sender_address)

def is_balance_proof_usable_onchain(received_balance_proof: BalanceProofSignedState, channel_state: NettingChannelState, sender_state: NettingChannelEndState) -> SuccessOrError:
    return SuccessOrError()

def is_valid_lockedtransfer(transfer_state: BalanceProofSignedState, channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    return valid_lockedtransfer_check(channel_state, sender_state, receiver_state, "LockedTransfer", transfer_state, HashTimeLockState())

def is_valid_lock_expired(state_change: ReceiveLockExpired, channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState, block_number: BlockNumber) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    return (True, None, PendingLocksState([]))

def valid_lockedtransfer_check(channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState, message_name: str, received_balance_proof: BalanceProofSignedState, lock: HashTimeLockState) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    return (True, None, PendingLocksState([]))

def refund_transfer_matches_transfer(refund_transfer: BalanceProofSignedState, transfer: Any) -> bool:
    return True

def is_valid_refund(refund: ReceiveTransferRefund, channel_state: NettingChannelState, sender_state: NettingChannelEndState, receiver_state: NettingChannelEndState, received_transfer: Any) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    return (True, "", PendingLocksState([]))

def is_valid_unlock(unlock: ReceiveUnlock, channel_state: NettingChannelState, sender_state: NettingChannelEndState) -> Tuple[bool, Optional[str], Optional[PendingLocksState]]:
    return (True, None, PendingLocksState([]))

def is_valid_total_withdraw(channel_state: NettingChannelState, our_total_withdraw: WithdrawAmount, allow_zero: bool = False) -> SuccessOrError:
    return SuccessOrError()

def is_valid_action_coopsettle(channel_state: NettingChannelState, coop_settle: ActionChannelCoopSettle, total_withdraw: WithdrawAmount) -> SuccessOrError:
    return SuccessOrError()

def is_valid_withdraw_request(channel_state: NettingChannelState, withdraw_request: Any) -> SuccessOrError:
    return SuccessOrError()

def is_valid_withdraw_confirmation(channel_state: NettingChannelState, received_withdraw: Any) -> SuccessOrError:
    return SuccessOrError()

def is_valid_withdraw_expired(channel_state: NettingChannelState, state_change: ReceiveWithdrawExpired, withdraw_state: PendingWithdrawState, block_number: BlockNumber) -> SuccessOrError:
    return SuccessOrError()

def get_amount_unclaimed_onchain(end_state: NettingChannelEndState) -> TokenAmount:
    return 0

def get_amount_locked(end_state: NettingChannelEndState) -> LockedAmount:
    total_pending: LockedAmount = sum(lock.amount for lock in end_state.secrethashes_to_lockedlocks.values())
    total_unclaimed = sum(partial.lock.amount for partial in end_state.secrethashes_to_unlockedlocks.values())
    total_unclaimed_onchain = get_amount_unclaimed_onchain(end_state)
    result = total_pending + total_unclaimed + total_unclaimed_onchain
    return result

def get_capacity(channel_state: NettingChannelState) -> TokenAmount:
    return channel_state.our_total_withdraw + channel_state.partner_total_withdraw

def get_balance(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> Balance:
    return _get_balance(sender, receiver, subtract_withdraws=True)

def _get_balance(sender: NettingChannelEndState, receiver: NettingChannelEndState, subtract_withdraws: bool = True) -> Balance:
    sender_transferred_amount = sender.balance_proof.transferred_amount if sender.balance_proof else 0
    receiver_transferred_amount = receiver.balance_proof.transferred_amount if receiver.balance_proof else 0
    max_withdraw = max(sender.offchain_total_withdraw, sender.onchain_total_withdraw)
    withdraw = max_withdraw if subtract_withdraws else 0
    return sender.contract_balance - withdraw - sender_transferred_amount + receiver_transferred_amount

def get_max_withdraw_amount(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> WithdrawAmount:
    return _get_balance(sender, receiver, subtract_withdraws=False)

def get_current_balanceproof(end_state: NettingChannelEndState) -> Tuple[Any, Nonce, TokenAmount, LockedAmount]:
    if end_state.balance_proof:
        locksroot = end_state.balance_proof.locksroot
        nonce = end_state.nonce
        transferred_amount = end_state.balance_proof.transferred_amount
        locked_amount = get_amount_locked(end_state)
    else:
        locksroot = b"\x00"  # placeholder for LOCKSROOT_OF_NO_LOCKS
        nonce = 0
        transferred_amount = 0
        locked_amount = 0
    return (locksroot, nonce, transferred_amount, locked_amount)

def get_current_nonce(end_state: NettingChannelEndState) -> Nonce:
    return end_state.nonce

def get_distributable(sender: NettingChannelEndState, receiver: NettingChannelEndState) -> TokenAmount:
    _, _, transferred_amount, locked_amount = get_current_balanceproof(sender)
    distributable = get_balance(sender, receiver) - get_amount_locked(sender)
    overflow_limit = max(2**256 - transferred_amount - locked_amount, 0)
    return min(overflow_limit, distributable)

def get_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> Optional[HashTimeLockState]:
    lock = end_state.secrethashes_to_lockedlocks.get(secrethash)
    if not lock:
        partial_unlock = end_state.secrethashes_to_unlockedlocks.get(secrethash)
        if not partial_unlock:
            partial_unlock = end_state.secrethashes_to_onchain_unlockedlocks.get(secrethash)
        if partial_unlock:
            lock = partial_unlock.lock
    return lock

def lock_exists_in_either_channel_side(channel_state: NettingChannelState, secrethash: SecretHash) -> bool:
    lock = get_lock(channel_state.our_state, secrethash)
    if not lock:
        lock = get_lock(channel_state.partner_state, secrethash)
    return lock is not None

def get_next_nonce(end_state: NettingChannelEndState) -> Nonce:
    return end_state.nonce + 1

def get_number_of_pending_transfers(channel_end_state: NettingChannelEndState) -> int:
    return len(channel_end_state.pending_locks.locks)

def get_status(channel_state: NettingChannelState) -> ChannelState:
    if channel_state.settle_transaction:
        finished_successfully = (channel_state.settle_transaction.result == "SUCCESS")
        running = channel_state.settle_transaction.finished_block_number is None
        if finished_successfully:
            return ChannelState.STATE_SETTLED
        elif running:
            return ChannelState.STATE_SETTLING
        else:
            return ChannelState.STATE_UNUSABLE
    elif channel_state.close_transaction:
        finished_successfully = (channel_state.close_transaction.result == "SUCCESS")
        running = channel_state.close_transaction.finished_block_number is None
        if finished_successfully:
            return ChannelState.STATE_CLOSED
        elif running:
            return ChannelState.STATE_CLOSING
        else:
            return ChannelState.STATE_UNUSABLE
    else:
        return ChannelState.STATE_OPENED

def _del_unclaimed_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> None:
    if secrethash in end_state.secrethashes_to_lockedlocks:
        del end_state.secrethashes_to_lockedlocks[secrethash]
    if secrethash in end_state.secrethashes_to_unlockedlocks:
        del end_state.secrethashes_to_unlockedlocks[secrethash]

def _del_lock(end_state: NettingChannelEndState, secrethash: SecretHash) -> None:
    assert is_lock_pending(end_state, secrethash), "Lock must be pending"
    _del_unclaimed_lock(end_state, secrethash)
    if secrethash in end_state.secrethashes_to_onchain_unlockedlocks:
        del end_state.secrethashes_to_onchain_unlockedlocks[secrethash]

def set_closed(channel_state: NettingChannelState, block_number: BlockNumber) -> None:
    if not channel_state.close_transaction:
        channel_state.close_transaction = type("TxStatus", (), {"finished_block_number": block_number, "result": "SUCCESS"})
    elif not channel_state.close_transaction.finished_block_number:
        channel_state.close_transaction.finished_block_number = block_number
        channel_state.close_transaction.result = "SUCCESS"

def set_settled(channel_state: NettingChannelState, block_number: BlockNumber) -> None:
    if not channel_state.settle_transaction:
        channel_state.settle_transaction = type("TxStatus", (), {"finished_block_number": block_number, "result": "SUCCESS"})
    elif not channel_state.settle_transaction.finished_block_number:
        channel_state.settle_transaction.finished_block_number = block_number
        channel_state.settle_transaction.result = "SUCCESS"

def set_coop_settled(channel_end_state: NettingChannelEndState, block_number: BlockNumber) -> None:
    assert channel_end_state.initiated_coop_settle is not None, "This should only be called on a state where a CoopSettle is initiated"
    if not channel_end_state.initiated_coop_settle.transaction:
        channel_end_state.initiated_coop_settle.transaction = type("TxStatus", (), {"finished_block_number": block_number, "result": "SUCCESS"})
    elif not channel_end_state.initiated_coop_settle.transaction.finished_block_number:
        channel_end_state.initiated_coop_settle.transaction.finished_block_number = block_number
        channel_end_state.initiated_coop_settle.transaction.result = "SUCCESS"

def update_contract_balance(end_state: NettingChannelEndState, contract_balance: Balance) -> None:
    if contract_balance > end_state.contract_balance:
        end_state.contract_balance = contract_balance

def compute_locks_with(locks: PendingLocksState, lock: HashTimeLockState) -> Optional[PendingLocksState]:
    if lock.encoded not in locks.locks:
        new_locks = list(locks.locks)
        new_locks.append(lock.encoded)
        return PendingLocksState(new_locks)
    else:
        return None

def compute_locks_without(locks: PendingLocksState, lock_encoded: EncodedData) -> Optional[PendingLocksState]:
    if lock_encoded in locks.locks:
        new_locks = list(locks.locks)
        new_locks.remove(lock_encoded)
        return PendingLocksState(new_locks)
    else:
        return None

def compute_locksroot(locks: PendingLocksState) -> bytes:
    return keccak(b"".join(locks.locks))

def create_sendlockedtransfer(channel_state: NettingChannelState,
                              initiator: Address,
                              target: Address,
                              amount: TokenAmount,
                              message_identifier: int,
                              payment_identifier: PaymentID,
                              expiration: BlockExpiration,
                              secret: Secret,
                              secrethash: SecretHash,
                              route_states: Any,
                              recipient_metadata: Optional[AddressMetadata] = None,
                              previous_metadata: Optional[Dict[str, Any]] = None) -> Tuple[SendLockedTransfer, PendingLocksState]:
    our_state = channel_state.our_state
    partner_state = channel_state.partner_state
    our_balance_proof = our_state.balance_proof
    assert amount <= get_distributable(our_state, partner_state), "caller must make sure there is enough balance"
    assert get_status(channel_state) == ChannelState.STATE_OPENED, "caller must make sure the channel is open"
    lock = HashTimeLockState(amount=amount, expiration=expiration, secrethash=secrethash)
    pending_locks = compute_locks_with(our_state.pending_locks, lock)
    assert pending_locks, "lock is already registered"
    locksroot = compute_locksroot(pending_locks)
    transferred_amount = our_balance_proof.transferred_amount if our_balance_proof else 0
    assert transferred_amount + amount <= 2**256 - 1, "caller must make sure the result wont overflow"
    token = channel_state.token_address
    locked_amount = get_amount_locked(our_state) + amount
    nonce = get_next_nonce(our_state)
    balance_proof = BalanceProofUnsignedState(nonce=nonce,
                                               transferred_amount=transferred_amount,
                                               locked_amount=locked_amount,
                                               locksroot=locksroot,
                                               canonical_identifier=channel_state.canonical_identifier)
    locked_transfer = type("LockedTransferUnsignedState", (), {
        "payment_identifier": payment_identifier,
        "token": token,
        "balance_proof": balance_proof,
        "secret": secret,
        "lock": lock,
        "initiator": initiator,
        "target": target,
        "route_states": route_states,
        "metadata": previous_metadata
    })
    recipient = partner_state.address
    if recipient_metadata is None:
        recipient_metadata = {}  # Placeholder for get_address_metadata(recipient, route_states)
    lockedtransfer = SendLockedTransfer(recipient=recipient,
                                        recipient_metadata=recipient_metadata,
                                        message_identifier=message_identifier,
                                        transfer=locked_transfer,
                                        canonical_identifier=channel_state.canonical_identifier)
    return lockedtransfer, pending_locks

def create_unlock(channel_state: NettingChannelState,
                  message_identifier: int,
                  payment_identifier: PaymentID,
                  secret: Secret,
                  lock: HashTimeLockState,
                  block_number: BlockNumber,
                  recipient_metadata: Optional[AddressMetadata] = None) -> SendUnlock:
    our_state = channel_state.our_state
    assert is_lock_pending(our_state, lock.secrethash), "caller must make sure the lock is known"
    assert get_status(channel_state) == ChannelState.STATE_OPENED, "caller must make sure the channel is open"
    expired = is_lock_expired(our_state, lock, block_number, lock.expiration)
    assert not expired, "caller must make sure the lock is not expired"
    our_balance_proof = our_state.balance_proof
    assert our_balance_proof is not None, "the lock is pending, it must be in the pending locks"
    transferred_amount = lock.amount + our_balance_proof.transferred_amount
    pending_locks = compute_locks_without(our_state.pending_locks, lock.encoded)
    assert pending_locks is not None, "the lock is pending, it must be in the pending locks"
    locksroot = compute_locksroot(pending_locks)
    token_address = channel_state.token_address
    recipient = channel_state.partner_state.address
    locked_amount = get_amount_locked(our_state) - lock.amount
    nonce = get_next_nonce(our_state)
    channel_state.our_state.nonce = nonce
    balance_proof = BalanceProofUnsignedState(nonce=nonce,
                                               transferred_amount=transferred_amount,
                                               locked_amount=locked_amount,
                                               locksroot=locksroot,
                                               canonical_identifier=channel_state.canonical_identifier)
    unlock_lock = SendUnlock(recipient=recipient,
                             recipient_metadata=recipient_metadata,
                             message_identifier=message_identifier,
                             payment_identifier=payment_identifier,
                             token_address=token_address,
                             secret=secret,
                             balance_proof=balance_proof,
                             canonical_identifier=channel_state.canonical_identifier)
    return unlock_lock

def send_lockedtransfer(channel_state: NettingChannelState,
                        initiator: Address,
                        target: Address,
                        amount: TokenAmount,
                        message_identifier: int,
                        payment_identifier: PaymentID,
                        expiration: BlockExpiration,
                        secret: Secret,
                        secrethash: SecretHash,
                        route_states: Any,
                        recipient_metadata: Optional[AddressMetadata] = None,
                        previous_metadata: Optional[Dict[str, Any]] = None) -> SendLockedTransfer:
    send_locked_transfer_event, pending_locks = create_sendlockedtransfer(channel_state=channel_state,
                                                                          initiator=initiator,
                                                                          target=target,
                                                                          amount=amount,
                                                                          message_identifier=message_identifier,
                                                                          payment_identifier=payment_identifier,
                                                                          expiration=expiration,
                                                                          secret=secret,
                                                                          secrethash=secrethash,
                                                                          route_states=route_states,
                                                                          recipient_metadata=recipient_metadata,
                                                                          previous_metadata=previous_metadata)
    transfer = send_locked_transfer_event.transfer
    lock = transfer.lock
    channel_state.our_state.balance_proof = transfer.balance_proof
    channel_state.our_state.nonce = transfer.balance_proof.nonce
    channel_state.our_state.pending_locks = pending_locks
    channel_state.our_state.secrethashes_to_lockedlocks[lock.secrethash] = lock
    return send_locked_transfer_event

def send_unlock(channel_state: NettingChannelState,
                message_identifier: int,
                payment_identifier: PaymentID,
                secret: Secret,
                secrethash: SecretHash,
                block_number: BlockNumber,
                recipient_metadata: Optional[AddressMetadata] = None) -> SendUnlock:
    lock = get_lock(channel_state.our_state, secrethash)
    assert lock, "caller must ensure the lock exists"
    unlock_msg = create_unlock(channel_state,
                               message_identifier,
                               payment_identifier,
                               secret,
                               lock,
                               block_number,
                               recipient_metadata)
    channel_state.our_state.balance_proof = unlock_msg.balance_proof
    channel_state.our_state.pending_locks = compute_locks_without(channel_state.our_state.pending_locks, lock.encoded)  # type: ignore
    _del_lock(channel_state.our_state, lock.secrethash)
    return unlock_msg

def events_for_close(channel_state: NettingChannelState, block_number: BlockNumber, block_hash: BlockHash) -> List[Event]:
    events: List[Event] = []
    if get_status(channel_state) in [ChannelState.STATE_OPENED, ChannelState.STATE_CLOSING]:
        channel_state.close_transaction = type("TxStatus", (), {"finished_block_number": None})
        balance_proof = channel_state.partner_state.balance_proof
        close_event = ContractSendChannelClose(canonical_identifier=channel_state.canonical_identifier,
                                                 balance_proof=balance_proof,
                                                 triggered_by_block_hash=block_hash)
        events.append(close_event)
    return events

def send_withdraw_request(channel_state: NettingChannelState,
                          total_withdraw: WithdrawAmount,
                          expiration: BlockExpiration,
                          pseudo_random_generator: random.Random,
                          recipient_metadata: AddressMetadata,
                          coop_settle: bool = False) -> List[Event]:
    events: List[Event] = []
    if get_status(channel_state) not in [ChannelState.STATE_OPENED, ChannelState.STATE_CLOSING]:
        return events
    nonce = get_next_nonce(channel_state.our_state)
    withdraw_state = PendingWithdrawState(total_withdraw=total_withdraw, nonce=nonce, expiration=expiration, recipient_metadata=recipient_metadata)
    channel_state.our_state.nonce = nonce
    channel_state.our_state.withdraws_pending[withdraw_state.total_withdraw] = withdraw_state
    withdraw_event = SendWithdrawRequest(canonical_identifier=channel_state.canonical_identifier,
                                           recipient=channel_state.partner_state.address,
                                           message_identifier=message_identifier_from_prng(pseudo_random_generator),
                                           total_withdraw=withdraw_state.total_withdraw,
                                           participant=channel_state.our_state.address,
                                           nonce=channel_state.our_state.nonce,
                                           expiration=withdraw_state.expiration,
                                           recipient_metadata=recipient_metadata,
                                           coop_settle=coop_settle)
    events.append(withdraw_event)
    return events

def create_sendexpiredlock(sender_end_state: NettingChannelEndState,
                           locked_lock: HashTimeLockState,
                           pseudo_random_generator: random.Random,
                           chain_id: ChainID,
                           token_network_address: Address,
                           channel_identifier: ChannelID,
                           recipient: Address,
                           recipient_metadata: Optional[AddressMetadata] = None) -> Tuple[Optional[SendLockExpired], Optional[PendingLocksState]]:
    locked_amount = get_amount_locked(sender_end_state)
    balance_proof = sender_end_state.balance_proof
    updated_locked_amount = locked_amount - locked_lock.amount
    assert balance_proof is not None, "there should be a balance proof because a lock is expiring"
    transferred_amount = balance_proof.transferred_amount
    pending_locks = compute_locks_without(sender_end_state.pending_locks, locked_lock.encoded)
    if not pending_locks:
        return (None, None)
    nonce = get_next_nonce(sender_end_state)
    locksroot = compute_locksroot(pending_locks)
    new_balance_proof = BalanceProofUnsignedState(nonce=nonce,
                                                  transferred_amount=transferred_amount,
                                                  locked_amount=updated_locked_amount,
                                                  locksroot=locksroot,
                                                  canonical_identifier={"chain_id": chain_id, "token_network_address": token_network_address, "channel_identifier": channel_identifier})
    send_lock_expired = SendLockExpired(recipient=recipient,
                                        recipient_metadata=recipient_metadata,
                                        canonical_identifier=new_balance_proof.canonical_identifier,
                                        message_identifier=message_identifier_from_prng(pseudo_random_generator),
                                        balance_proof=new_balance_proof,
                                        secrethash=locked_lock.secrethash)
    return send_lock_expired, pending_locks

def send_lock_expired(channel_state: NettingChannelState,
                      locked_lock: HashTimeLockState,
                      pseudo_random_generator: random.Random,
                      recipient_metadata: Optional[AddressMetadata] = None) -> List[SendLockExpired]:
    assert get_status(channel_state) == ChannelState.STATE_OPENED, "caller must make sure the channel is open"
    send_lock_expired_msg, pending_locks = create_sendexpiredlock(sender_end_state=channel_state.our_state,
                                                                  locked_lock=locked_lock,
                                                                  pseudo_random_generator=pseudo_random_generator,
                                                                  chain_id=channel_state.chain_id,
                                                                  token_network_address=channel_state.token_network_address,
                                                                  channel_identifier=channel_state.identifier,
                                                                  recipient=channel_state.partner_state.address,
                                                                  recipient_metadata=recipient_metadata)
    if send_lock_expired_msg:
        assert pending_locks is not None, "create_sendexpiredlock should return both message and pending locks"
        channel_state.our_state.pending_locks = pending_locks
        channel_state.our_state.balance_proof = send_lock_expired_msg.balance_proof
        channel_state.our_state.nonce = send_lock_expired_msg.balance_proof.nonce
        _del_unclaimed_lock(channel_state.our_state, locked_lock.secrethash)
        return [send_lock_expired_msg]
    return []

def events_for_expired_withdraws(channel_state: NettingChannelState,
                                 block_number: BlockNumber,
                                 pseudo_random_generator: random.Random) -> List[SendWithdrawExpired]:
    events: List[SendWithdrawExpired] = []
    for withdraw_state in list(channel_state.our_state.withdraws_pending.values()):
        withdraw_expired_flag = is_withdraw_expired(block_number, get_sender_expiration_threshold(withdraw_state.expiration))
        if not withdraw_expired_flag:
            break
        nonce = get_next_nonce(channel_state.our_state)
        channel_state.our_state.nonce = nonce
        coop_settle = channel_state.our_state.initiated_coop_settle
        if coop_settle is not None:
            if coop_settle.total_withdraw_initiator == withdraw_state.total_withdraw and coop_settle.expiration == withdraw_state.expiration:
                channel_state.our_state.initiated_coop_settle = None
        channel_state.our_state.withdraws_expired.append(ExpiredWithdrawState(withdraw_state.total_withdraw, withdraw_state.expiration, withdraw_state.nonce, withdraw_state.recipient_metadata))
        del channel_state.our_state.withdraws_pending[withdraw_state.total_withdraw]
        events.append(SendWithdrawExpired(recipient=channel_state.partner_state.address,
                                          recipient_metadata=withdraw_state.recipient_metadata,
                                          canonical_identifier=channel_state.canonical_identifier,
                                          message_identifier=message_identifier_from_prng(pseudo_random_generator),
                                          total_withdraw=withdraw_state.total_withdraw,
                                          participant=channel_state.our_state.address,
                                          expiration=withdraw_state.expiration,
                                          nonce=nonce))
    return events

def register_secret_endstate(end_state: NettingChannelEndState, secret: Secret, secrethash: SecretHash) -> None:
    if is_lock_locked(end_state, secrethash):
        pending_lock = end_state.secrethashes_to_lockedlocks[secrethash]
        del end_state.secrethashes_to_lockedlocks[secrethash]
        end_state.secrethashes_to_unlockedlocks[secrethash] = UnlockPartialProofState(pending_lock, secret)

def register_onchain_secret_endstate(end_state: NettingChannelEndState,
                                     secret: Secret,
                                     secrethash: SecretHash,
                                     secret_reveal_block_number: BlockNumber,
                                     delete_lock: bool = True) -> None:
    pending_lock: Optional[HashTimeLockState] = None
    if is_lock_locked(end_state, secrethash):
        pending_lock = end_state.secrethashes_to_lockedlocks[secrethash]
    if secrethash in end_state.secrethashes_to_unlockedlocks:
        pending_lock = end_state.secrethashes_to_unlockedlocks[secrethash].lock
    if pending_lock:
        if pending_lock.expiration < secret_reveal_block_number:
            return
        if delete_lock:
            _del_lock(end_state, secrethash)
        end_state.secrethashes_to_onchain_unlockedlocks[secrethash] = UnlockPartialProofState(pending_lock, secret)

def register_offchain_secret(channel_state: NettingChannelState, secret: Secret, secrethash: SecretHash) -> None:
    our_state = channel_state.our_state
    partner_state = channel_state.partner_state
    register_secret_endstate(our_state, secret, secrethash)
    register_secret_endstate(partner_state, secret, secrethash)

def register_onchain_secret(channel_state: NettingChannelState,
                            secret: Secret,
                            secrethash: SecretHash,
                            secret_reveal_block_number: BlockNumber,
                            delete_lock: bool = True) -> None:
    our_state = channel_state.our_state
    partner_state = channel_state.partner_state
    register_onchain_secret_endstate(our_state, secret, secrethash, secret_reveal_block_number, delete_lock)
    register_onchain_secret_endstate(partner_state, secret, secrethash, secret_reveal_block_number, delete_lock)

@singledispatch
def handle_state_transitions(action: StateChange,
                             channel_state: NettingChannelState,
                             block_number: BlockNumber,
                             block_hash: BlockHash,
                             pseudo_random_generator: random.Random) -> TransitionResult:
    return TransitionResult(channel_state, [])

@handle_state_transitions.register
def _handle_action_close(action: ActionChannelClose,
                         channel_state: NettingChannelState,
                         block_number: BlockNumber,
                         block_hash: BlockHash,
                         **kwargs: Any) -> TransitionResult:
    assert channel_state.identifier == action.channel_identifier, "caller must make sure the ids match"
    events = events_for_close(channel_state=channel_state, block_number=block_number, block_hash=block_hash)
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_action_coop_settle(action: ActionChannelCoopSettle,
                               channel_state: NettingChannelState,
                               pseudo_random_generator: random.Random,
                               block_number: BlockNumber,
                               **kwargs: Any) -> TransitionResult:
    events: List[Event] = []
    our_max_total_withdraw = get_max_withdraw_amount(channel_state.our_state, channel_state.partner_state)
    partner_max_total_withdraw = get_max_withdraw_amount(channel_state.partner_state, channel_state.our_state)
    valid_coop_settle = is_valid_action_coopsettle(channel_state, action, our_max_total_withdraw)
    if valid_coop_settle:
        expiration = get_safe_initial_expiration(block_number=block_number, reveal_timeout=channel_state.reveal_timeout)
        coop_settle = CoopSettleState(total_withdraw_initiator=our_max_total_withdraw,
                                      total_withdraw_partner=partner_max_total_withdraw,
                                      expiration=expiration)
        channel_state.our_state.initiated_coop_settle = coop_settle
        events = send_withdraw_request(channel_state=channel_state,
                                       total_withdraw=coop_settle.total_withdraw_initiator,
                                       expiration=coop_settle.expiration,
                                       pseudo_random_generator=pseudo_random_generator,
                                       recipient_metadata=action.recipient_metadata,
                                       coop_settle=True)
    else:
        error_msg = valid_coop_settle.as_error_message
        assert error_msg, "is_valid_action_coopsettle should return error msg if not valid"
        events = [EventInvalidActionCoopSettle(attempted_withdraw=our_max_total_withdraw, reason=error_msg)]
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_action_withdraw(action: ActionChannelWithdraw,
                            channel_state: NettingChannelState,
                            pseudo_random_generator: random.Random,
                            block_number: BlockNumber,
                            **kwargs: Any) -> TransitionResult:
    events: List[Event] = []
    is_valid_withdraw_flag = is_valid_total_withdraw(channel_state, action.total_withdraw)
    if is_valid_withdraw_flag:
        expiration = get_safe_initial_expiration(block_number=block_number, reveal_timeout=channel_state.reveal_timeout)
        events = send_withdraw_request(channel_state=channel_state,
                                       total_withdraw=action.total_withdraw,
                                       expiration=expiration,
                                       pseudo_random_generator=pseudo_random_generator,
                                       recipient_metadata=action.recipient_metadata)
    else:
        error_msg = is_valid_withdraw_flag.as_error_message
        assert error_msg, "is_valid_total_withdraw should return error msg if not valid"
        events = [EventInvalidActionWithdraw(attempted_withdraw=action.total_withdraw, reason=error_msg)]
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_action_set_reveal_timeout(action: ActionChannelSetRevealTimeout,
                                      channel_state: NettingChannelState,
                                      **kwargs: Any) -> TransitionResult:
    events: List[Event] = []
    is_valid_reveal_timeout = (action.reveal_timeout >= 7 and channel_state.settle_timeout >= action.reveal_timeout * 2)
    if is_valid_reveal_timeout:
        channel_state.reveal_timeout = action.reveal_timeout
    else:
        error_msg = "Settle timeout should be at least twice as large as reveal timeout"
        events = [EventInvalidActionSetRevealTimeout(reveal_timeout=action.reveal_timeout, reason=error_msg)]
    return TransitionResult(channel_state, events)

def events_for_coop_settle(channel_state: NettingChannelState,
                           coop_settle_state: CoopSettleState,
                           block_number: BlockNumber,
                           block_hash: BlockHash) -> List[Event]:
    if (coop_settle_state.partner_signature_request is not None and
        coop_settle_state.partner_signature_confirmation is not None and
        coop_settle_state.expiration >= block_number - channel_state.reveal_timeout):
        assert channel_state.our_state.initiated_coop_settle is not None, "CoopSettleState should be present in our state if we initiated it"
        send_coop_settle = ContractSendChannelCoopSettle(canonical_identifier=channel_state.canonical_identifier,
                                                         our_total_withdraw=coop_settle_state.total_withdraw_initiator,
                                                         partner_total_withdraw=coop_settle_state.total_withdraw_partner,
                                                         expiration=coop_settle_state.expiration,
                                                         signature_our_withdraw=coop_settle_state.partner_signature_confirmation,
                                                         signature_partner_withdraw=coop_settle_state.partner_signature_request,
                                                         triggered_by_block_hash=block_hash)
        channel_state.our_state.initiated_coop_settle.transaction = type("TxStatus", (), {"finished_block_number": block_number})
        return [send_coop_settle]
    return []

@handle_state_transitions.register
def _handle_receive_withdraw_request(action: ReceiveWithdrawRequest,
                                     channel_state: NettingChannelState,
                                     block_number: BlockNumber,
                                     block_hash: BlockHash,
                                     pseudo_random_generator: random.Random) -> TransitionResult:
    events: List[Event] = []
    def transition_result_invalid(success_or_error: SuccessOrError) -> TransitionResult:
        error_msg = success_or_error.as_error_message
        assert error_msg, "is_valid_withdraw_request should return error msg if not valid"
        invalid_withdraw_request = EventInvalidReceivedWithdrawRequest(attempted_withdraw=action.total_withdraw, reason=error_msg)
        return TransitionResult(channel_state, [invalid_withdraw_request])
    is_valid_flag = is_valid_withdraw_request(channel_state=channel_state, withdraw_request=action)
    if not is_valid_flag:
        return transition_result_invalid(is_valid_flag)
    withdraw_state = PendingWithdrawState(total_withdraw=action.total_withdraw,
                                          nonce=action.nonce,
                                          expiration=action.expiration,
                                          recipient_metadata=action.sender_metadata)
    channel_state.partner_state.withdraws_pending[withdraw_state.total_withdraw] = withdraw_state
    channel_state.partner_state.nonce = action.nonce
    our_initiated_coop_settle = channel_state.our_state.initiated_coop_settle
    if our_initiated_coop_settle is not None or action.coop_settle:
        partner_max_total_withdraw = get_max_withdraw_amount(channel_state.partner_state, channel_state.our_state)
        if partner_max_total_withdraw != action.total_withdraw:
            return transition_result_invalid(SuccessOrError("Partner did not withdraw with maximum balance (should=%s)." % str(partner_max_total_withdraw)))
        if get_number_of_pending_transfers(channel_state.partner_state) > 0:
            return transition_result_invalid(SuccessOrError("Partner has pending transfers."))
        if our_initiated_coop_settle is not None:
            if our_initiated_coop_settle.expiration != action.expiration:
                return transition_result_invalid(SuccessOrError("Partner requested withdraw while we initiated a coop-settle: Partner's withdraw has differing expiration."))
            assert our_initiated_coop_settle.total_withdraw_partner == action.total_withdraw, "The expected total withdraw of the partner doesn't match the withdraw-request"
            our_initiated_coop_settle.partner_signature_request = action.signature
            coop_settle_events = events_for_coop_settle(channel_state, our_initiated_coop_settle, block_number, block_hash)
            events.extend(coop_settle_events)
        else:
            our_max_total_withdraw = get_max_withdraw_amount(channel_state.our_state, channel_state.partner_state)
            if get_number_of_pending_transfers(channel_state.our_state) > 0:
                return transition_result_invalid(SuccessOrError("Partner initiated coop-settle, but we have pending transfers."))
            partner_initiated_coop_settle = CoopSettleState(total_withdraw_initiator=action.total_withdraw,
                                                            total_withdraw_partner=our_max_total_withdraw,
                                                            expiration=action.expiration,
                                                            partner_signature_request=action.signature)
            channel_state.partner_state.initiated_coop_settle = partner_initiated_coop_settle
            events = send_withdraw_request(channel_state=channel_state,
                                           expiration=partner_initiated_coop_settle.expiration,
                                           total_withdraw=our_max_total_withdraw,
                                           pseudo_random_generator=pseudo_random_generator,
                                           recipient_metadata=action.sender_metadata,
                                           coop_settle=False)
            events.extend(events)
    channel_state.our_state.nonce = get_next_nonce(channel_state.our_state)
    send_withdraw = SendWithdrawConfirmation(canonical_identifier=channel_state.canonical_identifier,
                                             recipient=channel_state.partner_state.address,
                                             recipient_metadata=action.sender_metadata,
                                             message_identifier=action.message_identifier,
                                             total_withdraw=action.total_withdraw,
                                             participant=channel_state.partner_state.address,
                                             nonce=channel_state.our_state.nonce,
                                             expiration=withdraw_state.expiration)
    events.append(send_withdraw)
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_receive_withdraw_confirmation(action: ReceiveWithdrawConfirmation,
                                          channel_state: NettingChannelState,
                                          block_number: BlockNumber,
                                          block_hash: BlockHash,
                                          **kwargs: Any) -> TransitionResult:
    is_valid_flag = is_valid_withdraw_confirmation(channel_state=channel_state, received_withdraw=action)
    withdraw_state = channel_state.our_state.withdraws_pending.get(action.total_withdraw)
    recipient_metadata: Optional[AddressMetadata] = withdraw_state.recipient_metadata if withdraw_state is not None else None
    events: List[Event]
    if is_valid_flag:
        channel_state.partner_state.nonce = action.nonce
        events = [SendProcessed(recipient=channel_state.partner_state.address,
                                  recipient_metadata=recipient_metadata,
                                  message_identifier=action.message_identifier,
                                  canonical_identifier="CANONICAL_IDENTIFIER_UNORDERED_QUEUE")]
        our_initiated_coop_settle = channel_state.our_state.initiated_coop_settle
        partner_initiated_coop_settle = channel_state.partner_state.initiated_coop_settle
        if our_initiated_coop_settle is not None:
            assert partner_initiated_coop_settle is None, "Only one party can initiate a coop settle"
            our_initiated_coop_settle.partner_signature_confirmation = action.signature
            coop_settle_events = events_for_coop_settle(channel_state, our_initiated_coop_settle, block_number, block_hash)
            events.extend(coop_settle_events)
        if our_initiated_coop_settle is None and partner_initiated_coop_settle is None:
            if action.expiration >= block_number - channel_state.reveal_timeout:
                withdraw_on_chain = ContractSendChannelWithdraw(canonical_identifier=action.canonical_identifier,
                                                                 total_withdraw=action.total_withdraw,
                                                                 partner_signature=action.signature,
                                                                 expiration=action.expiration,
                                                                 triggered_by_block_hash=block_hash)
                events.append(withdraw_on_chain)
    else:
        error_msg = is_valid_flag.as_error_message
        assert error_msg, "is_valid_withdraw_confirmation should return error msg if not valid"
        invalid_withdraw = EventInvalidReceivedWithdraw(attempted_withdraw=action.total_withdraw, reason=error_msg)
        events = [invalid_withdraw]
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_receive_withdraw_expired(action: ReceiveWithdrawExpired,
                                     channel_state: NettingChannelState,
                                     block_number: BlockNumber,
                                     **kwargs: Any) -> TransitionResult:
    events: List[Event] = []
    withdraw_state = channel_state.partner_state.withdraws_pending.get(action.total_withdraw)
    if not withdraw_state:
        invalid_withdraw_expired_msg = f"Withdraw expired of {action.total_withdraw} did not correspond to previous withdraw request"
        return TransitionResult(channel_state, [EventInvalidReceivedWithdrawExpired(attempted_withdraw=action.total_withdraw, reason=invalid_withdraw_expired_msg)])
    is_valid_flag = is_valid_withdraw_expired(channel_state=channel_state, state_change=action, withdraw_state=withdraw_state, block_number=block_number)
    if is_valid_flag:
        del channel_state.partner_state.withdraws_pending[withdraw_state.total_withdraw]
        channel_state.partner_state.nonce = action.nonce
        coop_settle = channel_state.partner_state.initiated_coop_settle
        if coop_settle is not None:
            if coop_settle.total_withdraw_initiator == withdraw_state.total_withdraw and coop_settle.expiration == withdraw_state.expiration:
                channel_state.partner_state.initiated_coop_settle = None
        send_processed = SendProcessed(recipient=channel_state.partner_state.address,
                                       recipient_metadata=withdraw_state.recipient_metadata,
                                       message_identifier=action.message_identifier,
                                       canonical_identifier="CANONICAL_IDENTIFIER_UNORDERED_QUEUE")
        events = [send_processed]
    else:
        error_msg = is_valid_flag.as_error_message
        assert error_msg, "is_valid_withdraw_expired should return error msg if not valid"
        events = [EventInvalidReceivedWithdrawExpired(attempted_withdraw=action.total_withdraw, reason=error_msg)]
    return TransitionResult(channel_state, events)

def handle_refundtransfer(received_transfer: Any, channel_state: NettingChannelState, refund: ReceiveTransferRefund) -> Tuple[bool, List[Event], Optional[str]]:
    is_valid_flag, msg, pending_locks = is_valid_refund(refund=refund, channel_state=channel_state, sender_state=channel_state.partner_state, receiver_state=channel_state.our_state, received_transfer=received_transfer)
    events: List[Event]
    if is_valid_flag:
        assert pending_locks is not None, "is_valid_refund should return pending locks if valid"
        channel_state.partner_state.balance_proof = refund.transfer.balance_proof
        channel_state.partner_state.nonce = refund.transfer.balance_proof.nonce
        channel_state.partner_state.pending_locks = pending_locks
        lock = refund.transfer.lock
        channel_state.partner_state.secrethashes_to_lockedlocks[lock.secrethash] = lock
        recipient_address = channel_state.partner_state.address
        recipient_metadata = {}  # Placeholder for get_address_metadata(recipient_address, received_transfer.route_states)
        send_processed = SendProcessed(recipient=refund.transfer.balance_proof.sender,
                                       recipient_metadata=recipient_metadata,
                                       message_identifier=refund.transfer.message_identifier,
                                       canonical_identifier="CANONICAL_IDENTIFIER_UNORDERED_QUEUE")
        events = [send_processed]
    else:
        assert msg, "is_valid_refund should return error msg if not valid"
        events = [EventInvalidReceivedTransferRefund(payment_identifier=received_transfer.payment_identifier, reason=msg)]
    return is_valid_flag, events, msg

def handle_receive_lock_expired(channel_state: NettingChannelState,
                                state_change: ReceiveLockExpired,
                                block_number: BlockNumber,
                                recipient_metadata: Optional[AddressMetadata] = None) -> TransitionResult:
    is_valid_flag, msg, pending_locks = is_valid_lock_expired(state_change=state_change, channel_state=channel_state, sender_state=channel_state.partner_state, receiver_state=channel_state.our_state, block_number=block_number)
    events: List[Event] = []
    if is_valid_flag:
        assert pending_locks is not None, "is_valid_lock_expired should return pending locks if valid"
        channel_state.partner_state.balance_proof = state_change.balance_proof
        channel_state.partner_state.nonce = state_change.balance_proof.nonce
        channel_state.partner_state.pending_locks = pending_locks
        _del_unclaimed_lock(channel_state.partner_state, state_change.secrethash)
        send_processed = SendProcessed(recipient=state_change.balance_proof.sender,
                                       recipient_metadata=recipient_metadata,
                                       message_identifier=state_change.message_identifier,
                                       canonical_identifier="CANONICAL_IDENTIFIER_UNORDERED_QUEUE")
        events = [send_processed]
    else:
        assert msg, "is_valid_lock_expired should return error msg if not valid"
        events = [EventInvalidReceivedLockExpired(secrethash=state_change.secrethash, reason=msg)]
    return TransitionResult(channel_state, events)

def handle_receive_lockedtransfer(channel_state: NettingChannelState,
                                  mediated_transfer: Any,
                                  recipient_metadata: Optional[AddressMetadata] = None) -> Tuple[bool, List[Event], Optional[str]]:
    is_valid_flag, msg, pending_locks = is_valid_lockedtransfer(mediated_transfer, channel_state, channel_state.partner_state, channel_state.our_state)
    events: List[Event]
    if is_valid_flag:
        assert pending_locks is not None, "is_valid_lockedtransfer should return pending locks if valid"
        channel_state.partner_state.balance_proof = mediated_transfer.balance_proof
        channel_state.partner_state.nonce = mediated_transfer.balance_proof.nonce
        channel_state.partner_state.pending_locks = pending_locks
        lock = mediated_transfer.lock
        channel_state.partner_state.secrethashes_to_lockedlocks[lock.secrethash] = lock
        send_processed = SendProcessed(recipient=mediated_transfer.balance_proof.sender,
                                       recipient_metadata=recipient_metadata,
                                       message_identifier=mediated_transfer.message_identifier,
                                       canonical_identifier="CANONICAL_IDENTIFIER_UNORDERED_QUEUE")
        events = [send_processed]
    else:
        assert msg, "is_valid_lockedtransfer should return error msg if not valid"
        events = [EventInvalidReceivedLockedTransfer(payment_identifier=mediated_transfer.payment_identifier, reason=msg)]
    return is_valid_flag, events, msg

def handle_unlock(channel_state: NettingChannelState,
                  unlock: ReceiveUnlock,
                  recipient_metadata: Optional[AddressMetadata] = None) -> Tuple[bool, List[Event], Optional[str]]:
    is_valid_flag, msg, unlocked_pending_locks = is_valid_unlock(unlock, channel_state, channel_state.partner_state)
    if is_valid_flag:
        assert unlocked_pending_locks is not None, "is_valid_unlock should return pending locks if valid"
        channel_state.partner_state.balance_proof = unlock.balance_proof
        channel_state.partner_state.nonce = unlock.balance_proof.nonce
        channel_state.partner_state.pending_locks = unlocked_pending_locks
        _del_lock(channel_state.partner_state, unlock.secrethash)
        send_processed = SendProcessed(recipient=unlock.balance_proof.sender,
                                       recipient_metadata=recipient_metadata,
                                       message_identifier=unlock.message_identifier,
                                       canonical_identifier="CANONICAL_IDENTIFIER_UNORDERED_QUEUE")
        events: List[Event] = [send_processed]
    else:
        assert msg, "is_valid_unlock should return error msg if not valid"
        events = [EventInvalidReceivedUnlock(secrethash=unlock.secrethash, reason=msg)]
    return is_valid_flag, events, msg

@handle_state_transitions.register
def _handle_block(action: Block,
                  channel_state: NettingChannelState,
                  block_number: BlockNumber,
                  pseudo_random_generator: random.Random,
                  **kwargs: Any) -> TransitionResult:
    assert action.block_number == block_number, "Block number mismatch"
    events: List[Event] = []
    if get_status(channel_state) == ChannelState.STATE_OPENED:
        expired_withdraws = events_for_expired_withdraws(channel_state=channel_state,
                                                         block_number=block_number,
                                                         pseudo_random_generator=pseudo_random_generator)
        events.extend(expired_withdraws)
    if get_status(channel_state) == ChannelState.STATE_CLOSED:
        assert channel_state.close_transaction, "channel get_status is STATE_CLOSED, but close_transaction is not set"
        assert channel_state.close_transaction.finished_block_number, "channel get_status is STATE_CLOSED, but close_transaction block number is missing"
        closed_block_number = channel_state.close_transaction.finished_block_number
        settlement_end = closed_block_number + channel_state.settle_timeout
        if action.block_number > settlement_end:
            channel_state.settle_transaction = type("TxStatus", (), {"finished_block_number": action.block_number})
            event = ContractSendChannelSettle(canonical_identifier=channel_state.canonical_identifier,
                                              triggered_by_block_hash=action.block_hash)
            events.append(event)
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_channel_closed(action: Any,
                           channel_state: NettingChannelState,
                           **kwargs: Any) -> TransitionResult:
    events: List[Event] = []
    just_closed = (action.channel_identifier == channel_state.identifier and get_status(channel_state) in [ChannelState.STATE_OPENED, ChannelState.STATE_CLOSING])
    if just_closed:
        set_closed(channel_state, action.block_number)
        balance_proof = channel_state.partner_state.balance_proof
        call_update = (action.transaction_from != channel_state.our_state.address and balance_proof is not None and channel_state.update_transaction is None)
        if call_update:
            expiration = block_number + channel_state.settle_timeout  # block_number should be defined in outer context
            update = ContractSendChannelUpdateTransfer(expiration=expiration,
                                                       balance_proof=balance_proof,  # type: ignore
                                                       triggered_by_block_hash=action.block_hash)
            channel_state.update_transaction = type("TxStatus", (), {"finished_block_number": action.block_number, "result": "SUCCESS"})
            events.append(update)
    return TransitionResult(channel_state, events)

@handle_state_transitions.register
def _handle_channel_updated_transfer(action: Any,
                                     channel_state: NettingChannelState,
                                     block_number: BlockNumber,
                                     **kwargs: Any) -> TransitionResult:
    if action.channel_identifier == channel_state.identifier:
        channel_state.update_transaction = type("TxStatus", (), {"finished_block_number": block_number, "result": "SUCCESS"})
    return TransitionResult(channel_state, [])

@handle_state_transitions.register
def _handle_channel_settled(action: ContractReceiveChannelSettled,
                            channel_state: NettingChannelState,
                            **kwargs: Any) -> TransitionResult:
    events: List[Event] = []
    if action.channel_identifier == channel_state.identifier:
        set_settled(channel_state, action.block_number)  # action.block_number assumed present
        our_locksroot = action.our_onchain_locksroot
        partner_locksroot = action.partner_onchain_locksroot
        should_clear_channel = (our_locksroot == b"\x00" and partner_locksroot == b"\x00")
        is_coop_settle = False
        initiator_lock_check = (action.our_onchain_locksroot == b"\x00")
        partner_lock_check = (action.partner_onchain_locksroot == b"\x00")
        if channel_state.our_state.initiated_coop_settle:
            coop_settle = channel_state.our_state.initiated_coop_settle
            initiator_transfer_check = (coop_settle.total_withdraw_initiator == action.our_transferred_amount)
            partner_transfer_check = (coop_settle.total_withdraw_partner == action.partner_transferred_amount)
            if initiator_transfer_check and initiator_lock_check and partner_transfer_check and partner_lock_check:
                set_coop_settled(channel_state.our_state, action.block_number)  # action.block_number assumed present
                is_coop_settle = True
        if channel_state.partner_state.initiated_coop_settle:
            coop_settle = channel_state.partner_state.initiated_coop_settle
            partner_transfer_check = (coop_settle.total_withdraw_initiator == action.partner_transferred_amount)
            initiator_transfer_check = (coop_settle.total_withdraw_partner == action.our_transferred_amount)
            if initiator_transfer_check and initiator_lock_check and partner_transfer_check and partner_lock_check:
                set_coop_settled(channel_state.partner_state, action.block_number)
                is_coop_settle = True
        if is_coop_settle:
            channel_state.partner_state.onchain_total_withdraw = action.partner_transferred_amount
            channel_state.our_state.onchain_total_withdraw = action.our_transferred_amount
        if should_clear_channel:
            return TransitionResult(None, events)
        channel_state.our_state.onchain_locksroot = our_locksroot
        channel_state.partner_state.onchain_locksroot = partner_locksroot
        onchain_unlock = ContractSendChannelBatchUnlock(canonical_identifier=channel_state.canonical_identifier,
                                                         sender=channel_state.partner_state.address,
                                                         triggered_by_block_hash=action.block_hash)
        events.append(onchain_unlock)
    return TransitionResult(channel_state, events)

def update_fee_schedule_after_balance_change(channel_state: NettingChannelState, fee_config: Any) -> List[Event]:
    proportional_imbalance_fee = get_proportional_imbalance_fee(channel_state.token_address)
    imbalance_penalty = calculate_imbalance_fees(channel_capacity=get_capacity(channel_state),
                                                 proportional_imbalance_fee=proportional_imbalance_fee)
    channel_state.fee_schedule = FeeScheduleState(cap_fees=channel_state.fee_schedule.cap_fees,
                                                  flat=channel_state.fee_schedule.flat,
                                                  proportional=channel_state.fee_schedule.proportional,
                                                  imbalance_penalty=imbalance_penalty)
    return []

@handle_state_transitions.register
def _handle_channel_deposit(action: ContractReceiveChannelDeposit,
                            channel_state: NettingChannelState,
                            **kwargs: Any) -> TransitionResult:
    participant_address = action.deposit_transaction.participant_address  # type: ignore
    contract_balance = action.deposit_transaction.contract_balance  # type: ignore
    if participant_address == channel_state.our_state.address:
        update_contract_balance(channel_state.our_state, contract_balance)
    elif participant_address == channel_state.partner_state.address:
        update_contract_balance(channel_state.partner_state, contract_balance)
    update_fee_schedule_after_balance_change(channel_state, action.fee_config)  # type: ignore
    return TransitionResult(channel_state, [])

@handle_state_transitions.register
def _handle_channel_withdraw(action: ContractReceiveChannelWithdraw,
                             channel_state: NettingChannelState,
                             **kwargs: Any) -> TransitionResult:
    participants = (channel_state.our_state.address, channel_state.partner_state.address)
    if action.participant not in participants:
        return TransitionResult(channel_state, [])
    if action.participant == channel_state.our_state.address:
        end_state = channel_state.our_state
    else:
        end_state = channel_state.partner_state
    withdraw_state = end_state.withdraws_pending.get(action.total_withdraw)  # type: ignore
    if withdraw_state:
        del end_state.withdraws_pending[action.total_withdraw]  # type: ignore
    end_state.onchain_total_withdraw = action.total_withdraw  # type: ignore
    update_fee_schedule_after_balance_change(channel_state, action.fee_config)  # type: ignore
    return TransitionResult(channel_state, [])

@handle_state_transitions.register
def _handle_channel_batch_unlock(action: ContractReceiveChannelBatchUnlock,
                                 channel_state: NettingChannelState,
                                 **kwargs: Any) -> TransitionResult:
    events: List[Event] = []
    new_channel_state: Optional[NettingChannelState] = channel_state
    if get_status(channel_state) == ChannelState.STATE_SETTLED:
        our_state = channel_state.our_state
        partner_state = channel_state.partner_state
        if action.sender == our_state.address:
            our_state.onchain_locksroot = b"\x00"
        elif action.sender == partner_state.address:
            partner_state.onchain_locksroot = b"\x00"
        no_unlock_left_to_do = (our_state.onchain_locksroot == b"\x00" and partner_state.onchain_locksroot == b"\x00")
        if no_unlock_left_to_do:
            new_channel_state = None
    return TransitionResult(new_channel_state, events)

def sanity_check(channel_state: NettingChannelState) -> None:
    partner_state = channel_state.partner_state
    our_state = channel_state.our_state
    previous = 0
    coop_settle = (channel_state.our_state.initiated_coop_settle is not None or channel_state.partner_state.initiated_coop_settle is not None)
    for total_withdraw, withdraw_state in our_state.withdraws_pending.items():
        if not coop_settle:
            assert withdraw_state.total_withdraw > previous, "total_withdraw must be ordered"
        assert total_withdraw == withdraw_state.total_withdraw, "Total withdraw mismatch"
        previous = withdraw_state.total_withdraw
    our_balance = get_balance(our_state, partner_state)
    partner_balance = get_balance(partner_state, our_state)
    msg = "The balance can never be negative, that would be equivalent to a loan or a double spend."
    assert our_balance >= 0, msg
    assert partner_balance >= 0, msg
    channel_capacity = get_capacity(channel_state)
    msg = "The whole deposit of the channel has to be accounted for."
    assert our_balance + partner_balance == channel_capacity, msg
    our_locked = get_amount_locked(our_state)
    partner_locked = get_amount_locked(partner_state)
    our_bp = get_current_balanceproof(our_state)
    partner_bp = get_current_balanceproof(partner_state)
    our_bp_locked_amount = our_bp[3]
    partner_bp_locked_amount = partner_bp[3]
    msg = "The sum of the lock's amounts, and the value of the balance proof locked_amount must be equal, otherwise settle will not reserve the proper amount of tokens."
    assert partner_locked == partner_bp_locked_amount, msg
    assert our_locked == our_bp_locked_amount, msg
    our_distributable = get_distributable(our_state, partner_state)
    partner_distributable = get_distributable(partner_state, our_state)
    assert our_distributable + our_locked <= our_balance, "distributable + locked must not exceed balance (own)"
    assert partner_distributable + partner_locked <= partner_balance, "distributable + locked must not exceed balance (partner)"
    our_locksroot = compute_locksroot(our_state.pending_locks)
    partner_locksroot = compute_locksroot(partner_state.pending_locks)
    msg = "The balance proof locks root must match the existing locks, otherwise it is not possible to prove on-chain that a given lock was pending."
    assert our_locksroot == our_bp[0], msg
    assert partner_locksroot == partner_bp[0], msg
    msg = "The lock mappings and the pending locks must be synchronized, otherwise there is a bug"
    for lock in partner_state.secrethashes_to_lockedlocks.values():
        assert lock.encoded in partner_state.pending_locks.locks, msg
    for partial_unlock in partner_state.secrethashes_to_unlockedlocks.values():
        assert partial_unlock.encoded in partner_state.pending_locks.locks, msg
    for partial_unlock in partner_state.secrethashes_to_onchain_unlockedlocks.values():
        assert partial_unlock.encoded in partner_state.pending_locks.locks, msg
    for lock in our_state.secrethashes_to_lockedlocks.values():
        assert lock.encoded in our_state.pending_locks.locks, msg
    for partial_unlock in our_state.secrethashes_to_unlockedlocks.values():
        assert partial_unlock.encoded in our_state.pending_locks.locks, msg
    for partial_unlock in our_state.secrethashes_to_onchain_unlockedlocks.values():
        assert partial_unlock.encoded in our_state.pending_locks.locks, msg

def state_transition(channel_state: NettingChannelState,
                     state_change: StateChange,
                     block_number: BlockNumber,
                     block_hash: BlockHash,
                     pseudo_random_generator: random.Random) -> TransitionResult:
    iteration: TransitionResult = handle_state_transitions(state_change,
                                                           channel_state=channel_state,
                                                           block_number=block_number,
                                                           block_hash=block_hash,
                                                           pseudo_random_generator=pseudo_random_generator)
    if iteration.new_state is not None:
        sanity_check(iteration.new_state)
    return iteration
