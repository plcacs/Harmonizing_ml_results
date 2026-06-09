from typing import Any

# === Third-party dependency: eth_utils ===
# Used symbols: encode_hex, keccak, to_hex

# === Internal dependency: raiden.constants ===
UINT256_MAX: Any
LOCKSROOT_OF_NO_LOCKS: Locksroot
MAXIMUM_PENDING_TRANSFERS: int

# === Internal dependency: raiden.settings ===
DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS: BlockTimeout

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
class CanonicalIdentifier: ...
CANONICAL_IDENTIFIER_UNORDERED_QUEUE: CanonicalIdentifier

# === Internal dependency: raiden.transfer.mediated_transfer.events ===
class SendLockExpired(SendMessageEvent):
    ...
class SendLockedTransfer(SendMessageEvent):
class SendUnlock(SendMessageEvent):

# === Internal dependency: raiden.transfer.mediated_transfer.mediation_fee ===
class FeeScheduleState(State):
    ...
def calculate_imbalance_fees(channel_capacity: TokenAmount, proportional_imbalance_fee: ProportionalFeeAmount) -> Optional[List[Tuple[TokenAmount, FeeAmount]]]: ...

# === Internal dependency: raiden.transfer.mediated_transfer.state ===
class LockedTransferUnsignedState(LockedTransferState):
    ...

# === Internal dependency: raiden.transfer.state ===
class ChannelState(Enum): ...
def message_identifier_from_prng(prng: Random) -> MessageID: ...
class HashTimeLockState(State):
    ...
class UnlockPartialProofState(State):
class TransactionExecutionStatus(State):
class PendingLocksState(State):
class ExpiredWithdrawState: ...
class PendingWithdrawState:
class CoopSettleState:
class NettingChannelState(State): ...
def get_address_metadata(address: Address, route_states: List[RouteState]) -> Optional[AddressMetadata]: ...
# re-export: from raiden.transfer.architecture import BalanceProofSignedState
# re-export: from raiden.transfer.architecture import BalanceProofUnsignedState
CHANNEL_STATES_PRIOR_TO_CLOSED: Any

# === Internal dependency: raiden.transfer.utils ===
def hash_balance_data(transferred_amount: TokenAmount, locked_amount: LockedAmount, locksroot: Locksroot) -> BalanceHash: ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address: AddressTypes) -> ChecksumAddress: ...

# === Internal dependency: raiden.utils.packing ===
def pack_balance_proof(nonce: Nonce, balance_hash: BalanceHash, additional_hash: AdditionalHash, canonical_identifier: CanonicalIdentifier) -> bytes: ...
def pack_withdraw(canonical_identifier: CanonicalIdentifier, participant: Address, total_withdraw: WithdrawAmount, expiration_block: BlockExpiration) -> bytes: ...

# === Internal dependency: raiden.utils.signer ===
def recover(data: bytes, signature: Signature, hasher: Callable[[bytes], bytes] = ...) -> Address: ...

# === Internal dependency: raiden.utils.typing ===
def typecheck(value: Any, expected: Union[Type, Tuple[Type, ...]]) -> None: ...
# re-export: from typing import Tuple
# re-export: from typing import Union
# re-export: from web3.types import Nonce
# re-export: from raiden_contracts.utils.type_aliases import BlockExpiration
# re-export: from raiden_contracts.utils.type_aliases import Locksroot
# re-export: from raiden_contracts.utils.type_aliases import TokenAmount
MYPY_ANNOTATION: str
Balance: NewType
BlockTimeout: NewType
LockedAmount: NewType
TargetAddress: NewType
Secret: NewType
EncodedData: NewType
WithdrawAmount: NewType