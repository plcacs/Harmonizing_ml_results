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
def assert_transfer_values(payment_identifier: PaymentID, token: TokenAddress, recipient: Address) -> None: ...


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
    chain_id: ChainID
    nonce: Nonce
    token_network_address: TokenNetworkAddress
    channel_identifier: ChannelID
    transferred_amount: TokenAmount
    locked_amount: LockedAmount
    locksroot: Locksroot

    def __post_init__(self) -> None: ...
    @property
    def message_hash(self) -> bytes: ...
    def _data_to_sign(self) -> bytes: ...


class SecretRequest(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    payment_identifier: PaymentID
    secrethash: SecretHash
    amount: PaymentAmount
    expiration: BlockExpiration

    @classmethod
    def from_event(cls, event: SendSecretRequest) -> "SecretRequest": ...
    def _data_to_sign(self) -> bytes: ...


class Unlock(EnvelopeMessage):
    cmdid: ClassVar[CmdId]
    payment_identifier: PaymentID
    secret: Secret

    def __post_init__(self) -> None: ...
    @property
    def secrethash(self) -> SecretHash: ...
    @classmethod
    def from_event(cls, event: SendUnlock) -> "Unlock": ...
    @property
    def message_hash(self) -> bytes: ...


class RevealSecret(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    secret: Secret

    @property
    def secrethash(self) -> SecretHash: ...
    @classmethod
    def from_event(cls, event: SendSecretReveal) -> "RevealSecret": ...
    def _data_to_sign(self) -> bytes: ...


class LockedTransferBase(EnvelopeMessage):
    payment_identifier: PaymentID
    token: TokenAddress
    recipient: Address
    target: TargetAddress
    initiator: InitiatorAddress
    lock: Lock
    metadata: Metadata

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
    recipient: Address
    secrethash: SecretHash

    @classmethod
    def from_event(cls, event: SendLockExpired) -> "LockExpired": ...
    @property
    def message_hash(self) -> bytes: ...