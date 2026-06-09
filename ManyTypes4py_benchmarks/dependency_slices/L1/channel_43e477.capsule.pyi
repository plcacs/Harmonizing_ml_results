# === Third-party dependency: eth_utils ===
# Used symbols: encode_hex, keccak, to_hex

# === Internal dependency: raiden.constants ===
UINT256_MAX = 2 ** 256 - 1
LOCKSROOT_OF_NO_LOCKS = Locksroot(...)
MAXIMUM_PENDING_TRANSFERS = 160

# === Internal dependency: raiden.settings ===
DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS = BlockTimeout(...)

# === Internal dependency: raiden.transfer.architecture ===
class Event: ...
class TransitionResult(Generic[T]): ...
class SuccessOrError: ...

# === Internal dependency: raiden.transfer.events ===
class SendWithdrawRequest(SendMessageEvent):
    ...
class SendWithdrawConfirmation(SendMessageEvent):
class SendWithdrawExpired(SendMessageEvent): ...
class ContractSendChannelWithdraw(ContractSendEvent):
class ContractSendChannelCoopSettle(ContractSendEvent):
class ContractSendChannelClose(ContractSendEvent):
class ContractSendChannelSettle(ContractSendEvent):
class ContractSendChannelUpdateTransfer(ContractSendExpirableEvent):
class ContractSendChannelBatchUnlock(ContractSendEvent):
class EventInvalidReceivedTransferRefund(Event):
class EventInvalidReceivedLockExpired(Event):
class EventInvalidReceivedLockedTransfer(Event):
class EventInvalidReceivedUnlock(Event):
class EventInvalidReceivedWithdrawRequest(Event):
class EventInvalidReceivedWithdraw(Event):
class EventInvalidReceivedWithdrawExpired(Event):
class EventInvalidActionWithdraw(Event): ...
class EventInvalidActionCoopSettle(Event): ...
class EventInvalidActionSetRevealTimeout(Event): ...
class SendProcessed(SendMessageEvent):

# === Internal dependency: raiden.transfer.identifiers ===
class CanonicalIdentifier:
    ...
CANONICAL_IDENTIFIER_UNORDERED_QUEUE = CanonicalIdentifier(...)

# === Internal dependency: raiden.transfer.mediated_transfer.events ===
class SendLockExpired(SendMessageEvent):
    ...
class SendLockedTransfer(SendMessageEvent):
class SendUnlock(SendMessageEvent):

# === Internal dependency: raiden.transfer.mediated_transfer.mediation_fee ===
class FeeScheduleState(State):
    ...
def calculate_imbalance_fees(channel_capacity, proportional_imbalance_fee): ...

# === Internal dependency: raiden.transfer.mediated_transfer.state ===
class LockedTransferUnsignedState(LockedTransferState):
    ...

# === Internal dependency: raiden.transfer.state ===
class ChannelState(Enum):
    STATE_CLOSING = 'waiting_for_close'
    STATE_OPENED = 'opened'
def message_identifier_from_prng(prng): ...
class HashTimeLockState(State):
    ...
class UnlockPartialProofState(State):
class TransactionExecutionStatus(State):
class PendingLocksState(State):
class ExpiredWithdrawState: ...
class PendingWithdrawState:
class CoopSettleState:
class NettingChannelState(State): ...
def get_address_metadata(address, route_states): ...
from raiden.transfer.architecture import BalanceProofSignedState
from raiden.transfer.architecture import BalanceProofUnsignedState
CHANNEL_STATES_PRIOR_TO_CLOSED = (ChannelState.STATE_OPENED, ChannelState.STATE_CLOSING)

# === Internal dependency: raiden.transfer.utils ===
def hash_balance_data(transferred_amount, locked_amount, locksroot): ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address): ...

# === Internal dependency: raiden.utils.packing ===
def pack_balance_proof(nonce, balance_hash, additional_hash, canonical_identifier): ...
def pack_withdraw(canonical_identifier, participant, total_withdraw, expiration_block): ...

# === Internal dependency: raiden.utils.signer ===
def recover(data, signature, hasher=...): ...

# === Internal dependency: raiden.utils.typing ===
def typecheck(value, expected): ...
from typing import Tuple
from typing import Union
from web3.types import Nonce
from raiden_contracts.utils.type_aliases import BlockExpiration
from raiden_contracts.utils.type_aliases import Locksroot
from raiden_contracts.utils.type_aliases import TokenAmount
MYPY_ANNOTATION = 'This assert is used to tell mypy what is the type of the variable'
T_Balance = int
Balance = NewType(...)
T_BlockTimeout = int
BlockTimeout = NewType(...)
T_LockedAmount = int
LockedAmount = NewType(...)
T_TargetAddress = bytes
TargetAddress = NewType(...)
T_Secret = bytes
Secret = NewType(...)
T_EncodedData = bytes
EncodedData = NewType(...)
T_WithdrawAmount = int
WithdrawAmount = NewType(...)