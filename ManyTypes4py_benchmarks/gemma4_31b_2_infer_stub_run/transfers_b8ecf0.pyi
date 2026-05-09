from dataclasses import dataclass
from typing import Optional, Type, TypeVar
from raiden.messages.abstract import SignedRetrieableMessage
from raiden.messages.cmdid import CmdId
from raiden.messages.metadata import Metadata
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.mediated_transfer.events import SendLockExpired, SendSecretRequest, SendSecretReveal, SendUnlock
from raiden.utils.typing import (
    AdditionalHash,
    Address,
    BlockExpiration,
    ChainID,
    ChannelID,
    InitiatorAddress,
    LockedAmount,
    Locksroot,
    MessageID,
    Nonce,
    PaymentAmount,
    PaymentID,
    PaymentWithFeeAmount,
    Secret,
    SecretHash,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkAddress,
)

T = TypeVar("T", bound="LockedTransferBase")

def assert_envelope_values(
    nonce: Nonce,
    channel_identifier: ChannelID,
    transferred_amount: PaymentAmount,
    locked_amount: LockedAmount,
    locksroot: Locksroot,
) -> None: ...

def assert_transfer_values(
    payment_identifier: PaymentID,
    token: TokenAddress,
    recipient: Address,
) -> None: ...

@dataclass(repr=False, eq=False)
class Lock:
    amount: PaymentWithFeeAmount
    expiration: BlockExpiration
    secrethash: SecretHash

    def __post_init__(self) -> None: ...

    @property
    def as_bytes(self) -> bytes: ...

    @property
    def lockhash(self) -> bytes: ...

    @classmethod
    def from_bytes(cls, serialized: bytes) -> "Lock": ...

@dataclass(repr=False, eq=False)
class EnvelopeMessage(SignedRetrieableMessage):
    nonce: Nonce
    channel_identifier: ChannelID
    transferred_amount: PaymentAmount
    locked_amount: LockedAmount
    locksroot: Locksroot

    def __post_init__(self) -> None: ...

    @property
    def message_hash(self) -> bytes: ...

    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class SecretRequest(SignedRetrieableMessage):
    cmdid: Type[CmdId]
    message_identifier: MessageID
    payment_identifier: PaymentID
    secrethash: SecretHash
    amount: PaymentAmount
    expiration: BlockExpiration

    @classmethod
    def from_event(cls, event: SendSecretRequest) -> "SecretRequest": ...

    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class Unlock(EnvelopeMessage):
    cmdid: Type[CmdId]
    payment_identifier: PaymentID
    secret: Secret

    def __post_init__(self) -> None: ...

    @property
    def secrethash(self) -> SecretHash: ...

    @classmethod
    def from_event(cls, event: SendUnlock) -> "Unlock": ...

    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class RevealSecret(SignedRetrieableMessage):
    cmdid: Type[CmdId]
    message_identifier: MessageID
    secret: Secret

    @property
    def secrethash(self) -> SecretHash: ...

    @classmethod
    def from_event(cls, event: SendSecretReveal) -> "RevealSecret": ...

    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class LockedTransferBase(EnvelopeMessage):
    payment_identifier: PaymentID
    token: TokenAddress
    recipient: Address
    target: TargetAddress
    initiator: InitiatorAddress
    lock: Lock

    def __post_init__(self) -> None: ...

    @classmethod
    def from_event(cls: Type[T], event: Any) -> T: ...

    def _packed_data(self) -> bytes: ...

    @classmethod
    def _pack_locked_transfer_data(
        cls,
        cmdid: CmdId,
        message_identifier: MessageID,
        payment_identifier: PaymentID,
        token: TokenAddress,
        recipient: Address,
        target: TargetAddress,
        initiator: InitiatorAddress,
        lock: Lock,
    ) -> bytes: ...

@dataclass(repr=False, eq=False)
class LockedTransfer(LockedTransferBase):
    cmdid: Type[CmdId]
    metadata: Optional[Metadata] = None

    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class RefundTransfer(LockedTransferBase):
    cmdid: Type[CmdId]

    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class LockExpired(EnvelopeMessage):
    cmdid: Type[CmdId]
    message_identifier: MessageID
    recipient: Address
    secrethash: SecretHash

    @classmethod
    def from_event(cls, event: SendLockExpired) -> "LockExpired": ...

    @property
    def message_hash(self) -> bytes: ...