from dataclasses import dataclass, field
from typing import Any, ClassVar

from raiden.messages.abstract import SignedRetrieableMessage
from raiden.messages.cmdid import CmdId
from raiden.messages.metadata import Metadata
from raiden.transfer.mediated_transfer.events import (
    SendLockExpired,
    SendSecretRequest,
    SendSecretReveal,
    SendUnlock,
)
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

def assert_envelope_values(
    nonce: int,
    channel_identifier: int,
    transferred_amount: int,
    locked_amount: int,
    locksroot: bytes,
) -> None: ...

def assert_transfer_values(
    payment_identifier: int,
    token: bytes,
    recipient: bytes,
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
    def from_bytes(cls, serialized: bytes) -> Lock: ...

@dataclass(repr=False, eq=False)
class EnvelopeMessage(SignedRetrieableMessage):
    nonce: Nonce
    channel_identifier: ChannelID
    transferred_amount: TokenAmount
    locked_amount: LockedAmount
    locksroot: Locksroot
    chain_id: ChainID
    token_network_address: TokenNetworkAddress

    def __post_init__(self) -> None: ...

    @property
    def message_hash(self) -> bytes: ...

    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class SecretRequest(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    payment_identifier: PaymentID
    secrethash: SecretHash
    amount: PaymentWithFeeAmount
    expiration: BlockExpiration

    @classmethod
    def from_event(cls, event: SendSecretRequest) -> SecretRequest: ...

    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class Unlock(EnvelopeMessage):
    cmdid: ClassVar[CmdId]
    secret: Secret
    payment_identifier: PaymentID

    def __post_init__(self) -> None: ...

    @property
    def secrethash(self) -> bytes: ...

    @classmethod
    def from_event(cls, event: SendUnlock) -> Unlock: ...

    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class RevealSecret(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    secret: Secret

    @property
    def secrethash(self) -> bytes: ...

    @classmethod
    def from_event(cls, event: SendSecretReveal) -> RevealSecret: ...

    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class LockedTransferBase(EnvelopeMessage):
    payment_identifier: PaymentID
    token: TokenAddress
    recipient: Address
    lock: Lock
    target: TargetAddress
    initiator: InitiatorAddress
    metadata: Metadata | None

    def __post_init__(self) -> None: ...

    @classmethod
    def from_event(cls, event: Any) -> LockedTransferBase: ...

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
    cmdid: ClassVar[CmdId]

    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class RefundTransfer(LockedTransferBase):
    cmdid: ClassVar[CmdId]

    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class LockExpired(EnvelopeMessage):
    cmdid: ClassVar[CmdId]
    recipient: Address
    secrethash: SecretHash

    @classmethod
    def from_event(cls, event: SendLockExpired) -> LockExpired: ...

    @property
    def message_hash(self) -> bytes: ...