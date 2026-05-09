from dataclasses import dataclass
from hashlib import sha256
from typing import Any, ClassVar, Optional, Union
from eth_utils import keccak
from raiden.constants import EMPTY_SIGNATURE, UINT64_MAX, UINT256_MAX
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
    ClassVar,
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
    UINT256_MAX,
    UINT64_MAX,
)

def assert_envelope_values(nonce: Nonce, channel_identifier: ChannelID, transferred_amount: TokenAmount, locked_amount: TokenAmount, locksroot: Locksroot) -> None: ...

def assert_transfer_values(payment_identifier: PaymentID, token: TokenAddress, recipient: Address) -> None: ...

@dataclass(repr=False, eq=False)
class Lock:
    amount: PaymentWithFeeAmount
    expiration: BlockExpiration
    secrethash: SecretHash
    __post_init__: Any
    as_bytes: property
    lockhash: property
    from_bytes: ClassVar[classmethod]

@dataclass(repr=False, eq=False)
class EnvelopeMessage(SignedRetrieableMessage):
    nonce: Nonce
    channel_identifier: ChannelID
    transferred_amount: TokenAmount
    locked_amount: TokenAmount
    locksroot: Locksroot
    chain_id: ChainID
    token_network_address: TokenNetworkAddress
    message_identifier: MessageID
    signature: bytes
    metadata: Optional[Metadata]
    __post_init__: Any
    message_hash: property
    _data_to_sign: Any

@dataclass(repr=False, eq=False)
class SecretRequest(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    message_identifier: MessageID
    payment_identifier: PaymentID
    secrethash: SecretHash
    amount: PaymentAmount
    expiration: BlockExpiration
    signature: bytes
    from_event: ClassVar[classmethod]
    _data_to_sign: Any

@dataclass(repr=False, eq=False)
class Unlock(EnvelopeMessage):
    cmdid: ClassVar[CmdId]
    secret: Secret
    __post_init__: Any
    secrethash: property
    from_event: ClassVar[classmethod]
    message_hash: property

@dataclass(repr=False, eq=False)
class RevealSecret(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId]
    secret: Secret
    signature: bytes
    from_event: ClassVar[classmethod]
    _data_to_sign: Any

@dataclass(repr=False, eq=False)
class LockedTransferBase(EnvelopeMessage):
    token: TokenAddress
    recipient: Address
    target: TargetAddress
    initiator: InitiatorAddress
    lock: Lock
    __post_init__: Any
    from_event: ClassVar[classmethod]
    _packed_data: Any
    _pack_locked_transfer_data: ClassVar[classmethod]

@dataclass(repr=False, eq=False)
class LockedTransfer(LockedTransferBase):
    cmdid: ClassVar[CmdId]
    message_hash: property

@dataclass(repr=False, eq=False)
class RefundTransfer(LockedTransferBase):
    cmdid: ClassVar[CmdId]
    message_hash: property

@dataclass(repr=False, eq=False)
class LockExpired(EnvelopeMessage):
    cmdid: ClassVar[CmdId]
    recipient: Address
    secrethash: SecretHash
    from_event: ClassVar[classmethod]
    message_hash: property