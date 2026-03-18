from typing import Any, ClassVar
from raiden.messages.abstract import SignedRetrieableMessage
from raiden.messages.cmdid import CmdId
from raiden.utils.typing import (
    Address,
    BlockExpiration,
    InitiatorAddress,
    MessageID,
    PaymentID,
    PaymentWithFeeAmount,
    Secret,
    SecretHash,
    TargetAddress,
    TokenAddress,
)


def assert_envelope_values(nonce: Any, channel_identifier: Any, transferred_amount: Any, locked_amount: Any, locksroot: Any) -> None: ...
def assert_transfer_values(payment_identifier: Any, token: Any, recipient: Any) -> None: ...


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


class EnvelopeMessage(SignedRetrieableMessage):
    def __post_init__(self) -> None: ...
    @property
    def message_hash(self) -> bytes: ...
    def _data_to_sign(self) -> bytes: ...


class SecretRequest(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]

    @classmethod
    def from_event(cls, event: Any) -> "SecretRequest": ...
    def _data_to_sign(self) -> bytes: ...


class Unlock(EnvelopeMessage):
    cmdid: ClassVar[CmdId]
    secret: Secret

    def __post_init__(self) -> None: ...
    @property
    def secrethash(self) -> SecretHash: ...
    @classmethod
    def from_event(cls, event: Any) -> "Unlock": ...
    @property
    def message_hash(self) -> bytes: ...


class RevealSecret(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    secret: Secret

    @property
    def secrethash(self) -> SecretHash: ...
    @classmethod
    def from_event(cls, event: Any) -> "RevealSecret": ...
    def _data_to_sign(self) -> bytes: ...


class LockedTransferBase(EnvelopeMessage):
    def __post_init__(self) -> None: ...
    @classmethod
    def from_event(cls, event: Any) -> "LockedTransferBase": ...
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


class LockedTransfer(LockedTransferBase):
    cmdid: ClassVar[CmdId]

    @property
    def message_hash(self) -> bytes: ...


class RefundTransfer(LockedTransferBase):
    cmdid: ClassVar[CmdId]

    @property
    def message_hash(self) -> bytes: ...


class LockExpired(EnvelopeMessage):
    cmdid: ClassVar[CmdId]

    @classmethod
    def from_event(cls, event: Any) -> "LockExpired": ...
    @property
    def message_hash(self) -> bytes: ...