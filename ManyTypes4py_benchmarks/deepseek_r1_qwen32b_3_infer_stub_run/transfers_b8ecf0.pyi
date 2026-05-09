from dataclasses import dataclass
from hashlib import sha256
from typing import Any, ClassVar, Optional

from eth_hash.auto import keccak
from eth_utils import keccak as eth_hash_keccak
from raiden.constants import EMPTY_SIGNATURE, UINT64_MAX, UINT256_MAX
from raiden.messages.abstract import SignedRetrieableMessage
from raiden.messages.cmdid import CmdId
from raiden.messages.metadata import Metadata
from raiden.transfer.identifiers import CanonicalIdentifier
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
    ClassVar,
    CmdId,
    InitiatorAddress,
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

def assert_envelope_values(nonce: int, channel_identifier: int, transferred_amount: int, locked_amount: int, locksroot: bytes) -> None: ...

def assert_transfer_values(payment_identifier: int, token: bytes, recipient: bytes) -> None: ...

@dataclass(repr=False, eq=False)
class Lock:
    amount: int
    expiration: int
    secrethash: bytes
    def __post_init__(self) -> None: ...
    @property
    def as_bytes(self) -> bytes: ...
    @property
    def lockhash(self) -> bytes: ...
    @classmethod
    def from_bytes(cls, serialized: bytes) -> 'Lock': ...

@dataclass(repr=False, eq=False)
class EnvelopeMessage(SignedRetrieableMessage):
    chain_id: ChainID
    message_identifier: MessageID
    payment_identifier: PaymentID
    nonce: Nonce
    token_network_address: TokenNetworkAddress
    channel_identifier: ChannelID
    transferred_amount: TokenAmount
    locked_amount: TokenAmount
    locksroot: Locksroot
    recipient: Address
    metadata: Metadata
    signature: bytes
    def __post_init__(self) -> None: ...
    @property
    def message_hash(self) -> bytes: ...
    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class SecretRequest(SignedRetrieableMessage):
    message_identifier: MessageID
    payment_identifier: PaymentID
    secrethash: SecretHash
    amount: PaymentAmount
    expiration: BlockExpiration
    signature: bytes
    @classmethod
    def from_event(cls, event: SendSecretRequest) -> 'SecretRequest': ...
    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class Unlock(EnvelopeMessage):
    secret: Secret
    @classmethod
    def from_event(cls, event: SendUnlock) -> 'Unlock': ...
    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class RevealSecret(SignedRetrieableMessage):
    secret: Secret
    @classmethod
    def from_event(cls, event: Any) -> 'RevealSecret': ...
    def _data_to_sign(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class LockedTransferBase(EnvelopeMessage):
    lock: Lock
    target: TargetAddress
    initiator: InitiatorAddress
    @classmethod
    def from_event(cls, event: Any) -> 'LockedTransferBase': ...
    def _packed_data(self) -> bytes: ...
    @classmethod
    def _pack_locked_transfer_data(cls, cmdid: CmdId, message_identifier: MessageID, payment_identifier: PaymentID, token: TokenAddress, recipient: Address, target: TargetAddress, initiator: InitiatorAddress, lock: Lock) -> bytes: ...

@dataclass(repr=False, eq=False)
class LockedTransfer(LockedTransferBase):
    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class RefundTransfer(LockedTransferBase):
    @property
    def message_hash(self) -> bytes: ...

@dataclass(repr=False, eq=False)
class LockExpired(EnvelopeMessage):
    secrethash: SecretHash
    @classmethod
    def from_event(cls, event: SendLockExpired) -> 'LockExpired': ...
    @property
    def message_hash(self) -> bytes: ...