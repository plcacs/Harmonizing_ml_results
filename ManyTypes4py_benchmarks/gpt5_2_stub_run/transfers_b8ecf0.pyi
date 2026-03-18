from typing import Any, ClassVar

from raiden.messages.abstract import SignedRetrieableMessage
from raiden.messages.cmdid import CmdId
from raiden.utils.typing import Address, BlockExpiration, InitiatorAddress, MessageID, PaymentID, PaymentWithFeeAmount, Secret, SecretHash, TargetAddress, TokenAddress


def assert_envelope_values(nonce: int, channel_identifier: int, transferred_amount: int, locked_amount: int, locksroot: bytes) -> None: ...
def assert_transfer_values(payment_identifier: int, token: bytes, recipient: bytes) -> None: ...


class Lock:
    amount: PaymentWithFeeAmount
    expiration: BlockExpiration
    secrethash: SecretHash
    def __init__(self, amount: PaymentWithFeeAmount, expiration: BlockExpiration, secrethash: SecretHash) -> None: ...
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
    def secrethash(self) -> bytes: ...
    @classmethod
    def from_event(cls, event: Any) -> "Unlock": ...
    @property
    def message_hash(self) -> bytes: ...


class RevealSecret(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    secret: Secret
    @property
    def secrethash(self) -> bytes: ...
    @classmethod
    def from_event(cls, event: Any) -> "RevealSecret": ...
    def _data_to_sign(self) -> bytes: ...


class LockedTransferBase(EnvelopeMessage):
    token: TokenAddress
    recipient: Address
    target: TargetAddress
    initiator: InitiatorAddress
    lock: Lock
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
    secrethash: SecretHash
    @classmethod
    def from_event(cls, event: Any) -> "LockExpired": ...
    @property
    def message_hash(self) -> bytes: ...