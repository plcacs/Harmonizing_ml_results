```pyi
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from raiden.messages.abstract import SignedRetrieableMessage
from raiden.messages.cmdid import CmdId
from raiden.messages.metadata import Metadata
from raiden.transfer.identifiers import CanonicalIdentifier
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
    nonce: Nonce,
    channel_identifier: ChannelID,
    transferred_amount: TokenAmount,
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
    def from_bytes(cls, serialized: bytes) -> Lock: ...

@dataclass(repr=False, eq=False)
class EnvelopeMessage(SignedRetrieableMessage):
    chain_id: ChainID
    message_identifier: MessageID
    nonce: Nonce
    token_network_address: TokenNetworkAddress
    channel_identifier: ChannelID
    transferred_amount: TokenAmount
    locked_amount: LockedAmount
    locksroot: Locksroot
    signature: bytes
    def __post_init__(self) -> None: ...
    @property
    def message_hash(self) -> bytes: ...
    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class SecretRequest(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    message_identifier: MessageID
    payment_identifier: PaymentID
    secrethash: SecretHash
    amount: PaymentWithFeeAmount
    expiration: BlockExpiration
    signature: bytes
    @classmethod
    def from_event(cls, event: Any) -> SecretRequest: ...
    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class Unlock(EnvelopeMessage):
    cmdid: ClassVar[CmdId]
    payment_identifier: PaymentID
    secret: Secret
    def __post_init__(self) -> None: ...
    @property
    def secrethash(self) -> bytes: ...
    @classmethod
    def from_event(cls, event: Any) -> Unlock: ...
    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class RevealSecret(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    message_identifier: MessageID
    secret: Secret
    signature: bytes
    @property
    def secrethash(self) -> bytes: ...
    @classmethod
    def from_event(cls, event: Any) -> RevealSecret: ...
    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class LockedTransferBase(EnvelopeMessage):
    payment_identifier: PaymentID
    token: TokenAddress
    recipient: Address
    target: TargetAddress
    initiator: InitiatorAddress
    lock: Lock
    metadata: Optional[Metadata]
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
    def from_event(cls, event: Any) -> LockExpired: ...
    @property
    def message_hash(self) -> bytes: ...
```