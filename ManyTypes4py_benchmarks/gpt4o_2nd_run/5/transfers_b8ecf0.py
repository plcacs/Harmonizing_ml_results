from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, ClassVar, Optional
import eth_hash.auto as eth_hash
from eth_utils import keccak
from raiden.constants import EMPTY_SIGNATURE, UINT64_MAX, UINT256_MAX
from raiden.messages.abstract import SignedRetrieableMessage
from raiden.messages.cmdid import CmdId
from raiden.messages.metadata import Metadata
from raiden.transfer.identifiers import CanonicalIdentifier
from raiden.transfer.mediated_transfer.events import SendLockExpired, SendSecretRequest, SendSecretReveal, SendUnlock
from raiden.transfer.utils import hash_balance_data
from raiden.utils.packing import pack_balance_proof
from raiden.utils.predicates import ishash
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
    locksroot: Locksroot
) -> None:
    if nonce <= 0:
        raise ValueError('nonce cannot be zero or negative')
    if nonce > UINT64_MAX:
        raise ValueError('nonce is too large')
    if channel_identifier <= 0:
        raise ValueError('channel id cannot be zero or negative')
    if channel_identifier > UINT256_MAX:
        raise ValueError('channel id is too large')
    if transferred_amount < 0:
        raise ValueError('transferred_amount cannot be negative')
    if transferred_amount > UINT256_MAX:
        raise ValueError('transferred_amount is too large')
    if locked_amount < 0:
        raise ValueError('locked_amount cannot be negative')
    if locked_amount > UINT256_MAX:
        raise ValueError('locked_amount is too large')
    if len(locksroot) != 32:
        raise ValueError('locksroot must have length 32')

def assert_transfer_values(
    payment_identifier: PaymentID,
    token: TokenAddress,
    recipient: Address
) -> None:
    if payment_identifier < 0:
        raise ValueError('payment_identifier cannot be negative')
    if payment_identifier > UINT64_MAX:
        raise ValueError('payment_identifier is too large')
    if len(token) != 20:
        raise ValueError('token is an invalid address')
    if len(recipient) != 20:
        raise ValueError('recipient is an invalid address')

@dataclass(repr=False, eq=False)
class Lock:
    amount: PaymentWithFeeAmount
    expiration: BlockExpiration
    secrethash: SecretHash

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError(f'amount {self.amount} needs to be positive')
        if self.amount > UINT256_MAX:
            raise ValueError(f'amount {self.amount} is too large')
        if self.expiration < 0:
            raise ValueError(f'expiration {self.expiration} needs to be positive')
        if self.expiration > UINT256_MAX:
            raise ValueError(f'expiration {self.expiration} is too large')
        if not ishash(self.secrethash):
            raise ValueError('secrethash {self.secrethash} is not a valid hash')

    @property
    def as_bytes(self) -> bytes:
        return (
            self.expiration.to_bytes(32, byteorder='big') +
            self.amount.to_bytes(32, byteorder='big') +
            self.secrethash
        )

    @property
    def lockhash(self) -> bytes:
        return keccak(self.as_bytes)

    @classmethod
    def from_bytes(cls, serialized: bytes) -> 'Lock':
        return cls(
            expiration=BlockExpiration(int.from_bytes(serialized[:32], byteorder='big')),
            amount=PaymentWithFeeAmount(int.from_bytes(serialized[32:64], byteorder='big')),
            secrethash=SecretHash(serialized[64:])
        )

@dataclass(repr=False, eq=False)
class EnvelopeMessage(SignedRetrieableMessage):
    nonce: Nonce
    channel_identifier: ChannelID
    transferred_amount: TokenAmount
    locked_amount: LockedAmount
    locksroot: Locksroot
    chain_id: ChainID
    token_network_address: TokenNetworkAddress

    def __post_init__(self) -> None:
        assert_envelope_values(
            self.nonce,
            self.channel_identifier,
            self.transferred_amount,
            self.locked_amount,
            self.locksroot
        )

    @property
    def message_hash(self) -> bytes:
        raise NotImplementedError

    def _data_to_sign(self) -> bytes:
        balance_hash = hash_balance_data(
            self.transferred_amount,
            self.locked_amount,
            self.locksroot
        )
        balance_proof_packed = pack_balance_proof(
            nonce=self.nonce,
            balance_hash=balance_hash,
            additional_hash=AdditionalHash(self.message_hash),
            canonical_identifier=CanonicalIdentifier(
                chain_identifier=self.chain_id,
                token_network_address=self.token_network_address,
                channel_identifier=self.channel_identifier
            )
        )
        return balance_proof_packed

@dataclass(repr=False, eq=False)
class SecretRequest(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId] = CmdId.SECRETREQUEST
    message_identifier: MessageID
    payment_identifier: PaymentID
    secrethash: SecretHash
    amount: PaymentAmount
    expiration: BlockExpiration
    signature: bytes

    @classmethod
    def from_event(cls, event: SendSecretRequest) -> 'SecretRequest':
        return cls(
            message_identifier=event.message_identifier,
            payment_identifier=event.payment_identifier,
            secrethash=event.secrethash,
            amount=event.amount,
            expiration=event.expiration,
            signature=EMPTY_SIGNATURE
        )

    def _data_to_sign(self) -> bytes:
        return (
            bytes([self.cmdid.value, 0, 0, 0]) +
            self.message_identifier.to_bytes(8, byteorder='big') +
            self.payment_identifier.to_bytes(8, byteorder='big') +
            self.secrethash +
            self.amount.to_bytes(32, byteorder='big') +
            self.expiration.to_bytes(32, byteorder='big')
        )

@dataclass(repr=False, eq=False)
class Unlock(EnvelopeMessage):
    cmdid: ClassVar[CmdId] = CmdId.UNLOCK
    secret: Secret
    message_identifier: MessageID
    payment_identifier: PaymentID

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.payment_identifier < 0:
            raise ValueError('payment_identifier cannot be negative')
        if self.payment_identifier > UINT64_MAX:
            raise ValueError('payment_identifier is too large')
        if len(self.secret) != 32:
            raise ValueError('secret must have 32 bytes')

    @property
    def secrethash(self) -> bytes:
        return sha256(self.secret).digest()

    @classmethod
    def from_event(cls, event: SendUnlock) -> 'Unlock':
        balance_proof = event.balance_proof
        return cls(
            chain_id=balance_proof.chain_id,
            message_identifier=event.message_identifier,
            payment_identifier=event.payment_identifier,
            nonce=balance_proof.nonce,
            token_network_address=balance_proof.token_network_address,
            channel_identifier=balance_proof.channel_identifier,
            transferred_amount=balance_proof.transferred_amount,
            locked_amount=balance_proof.locked_amount,
            locksroot=balance_proof.locksroot,
            secret=event.secret,
            signature=EMPTY_SIGNATURE
        )

    @property
    def message_hash(self) -> bytes:
        return eth_hash.keccak(
            bytes([self.cmdid.value]) +
            self.message_identifier.to_bytes(8, byteorder='big') +
            self.payment_identifier.to_bytes(8, byteorder='big') +
            self.secret
        )

@dataclass(repr=False, eq=False)
class RevealSecret(SignedRetrieableMessage):
    cmdid: ClassVar[CmdId] = CmdId.REVEALSECRET
    message_identifier: MessageID
    secret: Secret
    signature: bytes

    @property
    def secrethash(self) -> bytes:
        return sha256(self.secret).digest()

    @classmethod
    def from_event(cls, event: SendSecretReveal) -> 'RevealSecret':
        return cls(
            message_identifier=event.message_identifier,
            secret=event.secret,
            signature=EMPTY_SIGNATURE
        )

    def _data_to_sign(self) -> bytes:
        return (
            bytes([self.cmdid.value, 0, 0, 0]) +
            self.message_identifier.to_bytes(8, byteorder='big') +
            self.secret
        )

@dataclass(repr=False, eq=False)
class LockedTransferBase(EnvelopeMessage):
    payment_identifier: PaymentID
    token: TokenAddress
    recipient: Address
    target: TargetAddress
    initiator: InitiatorAddress
    lock: Lock
    metadata: Optional[Metadata]

    def __post_init__(self) -> None:
        super().__post_init__()
        assert_transfer_values(self.payment_identifier, self.token, self.recipient)
        if len(self.target) != 20:
            raise ValueError('target is an invalid address')
        if len(self.initiator) != 20:
            raise ValueError('initiator is an invalid address')

    @classmethod
    def from_event(cls, event: SendLockExpired) -> 'LockedTransferBase':
        transfer = event.transfer
        balance_proof = transfer.balance_proof
        lock = Lock(
            amount=transfer.lock.amount,
            expiration=transfer.lock.expiration,
            secrethash=transfer.lock.secrethash
        )
        return cls(
            chain_id=balance_proof.chain_id,
            message_identifier=event.message_identifier,
            payment_identifier=transfer.payment_identifier,
            nonce=balance_proof.nonce,
            token_network_address=balance_proof.token_network_address,
            token=transfer.token,
            channel_identifier=balance_proof.channel_identifier,
            transferred_amount=balance_proof.transferred_amount,
            locked_amount=balance_proof.locked_amount,
            recipient=event.recipient,
            locksroot=balance_proof.locksroot,
            lock=lock,
            target=transfer.target,
            initiator=transfer.initiator,
            signature=EMPTY_SIGNATURE,
            metadata=Metadata.from_event(event=event)
        )

    def _packed_data(self) -> bytes:
        return LockedTransferBase._pack_locked_transfer_data(
            self.cmdid,
            self.message_identifier,
            self.payment_identifier,
            self.token,
            self.recipient,
            self.target,
            self.initiator,
            self.lock
        )

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
        lock: Lock
    ) -> bytes:
        return (
            bytes([cmdid.value]) +
            message_identifier.to_bytes(8, byteorder='big') +
            payment_identifier.to_bytes(8, byteorder='big') +
            lock.expiration.to_bytes(32, byteorder='big') +
            token +
            recipient +
            target +
            initiator +
            lock.secrethash +
            lock.amount.to_bytes(32, byteorder='big')
        )

@dataclass(repr=False, eq=False)
class LockedTransfer(LockedTransferBase):
    cmdid: ClassVar[CmdId] = CmdId.LOCKEDTRANSFER

    @property
    def message_hash(self) -> bytes:
        metadata_hash = self.metadata and self.metadata.hash or b''
        return keccak(self._packed_data() + metadata_hash)

@dataclass(repr=False, eq=False)
class RefundTransfer(LockedTransferBase):
    cmdid: ClassVar[CmdId] = CmdId.REFUNDTRANSFER

    @property
    def message_hash(self) -> bytes:
        return keccak(self._packed_data())

@dataclass(repr=False, eq=False)
class LockExpired(EnvelopeMessage):
    cmdid: ClassVar[CmdId] = CmdId.LOCKEXPIRED
    message_identifier: MessageID
    recipient: Address
    secrethash: SecretHash

    @classmethod
    def from_event(cls, event: SendLockExpired) -> 'LockExpired':
        balance_proof = event.balance_proof
        return cls(
            chain_id=balance_proof.chain_id,
            nonce=balance_proof.nonce,
            token_network_address=balance_proof.token_network_address,
            channel_identifier=balance_proof.channel_identifier,
            transferred_amount=balance_proof.transferred_amount,
            locked_amount=balance_proof.locked_amount,
            locksroot=balance_proof.locksroot,
            message_identifier=event.message_identifier,
            recipient=event.recipient,
            secrethash=event.secrethash,
            signature=EMPTY_SIGNATURE
        )

    @property
    def message_hash(self) -> bytes:
        return eth_hash.keccak(
            bytes([self.cmdid.value]) +
            self.message_identifier.to_bytes(8, byteorder='big') +
            self.recipient +
            self.secrethash
        )
