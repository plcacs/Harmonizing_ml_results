from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, ClassVar
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
from raiden.utils.typing import AdditionalHash, Address, BlockExpiration, ChainID, ChannelID, InitiatorAddress, LockedAmount, Locksroot, MessageID, Nonce, PaymentAmount, PaymentID, PaymentWithFeeAmount, Secret, SecretHash, TargetAddress, TokenAddress, TokenAmount, TokenNetworkAddress


def assert_envelope_values(
    nonce: int,
    channel_identifier: int,
    transferred_amount: int,
    locked_amount: int,
    locksroot: bytes,
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


def assert_transfer_values(payment_identifier: int, token: bytes, recipient: bytes) -> None:
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
            raise ValueError(f'secrethash {self.secrethash} is not a valid hash')

    @property
    def as_bytes(self) -> bytes:
        return (
            self.expiration.to_bytes(32, byteorder='big')
            + self.amount.to_bytes(32, byteorder='big')
            + self.secrethash
        )

    @property
    def lockhash(self) -> bytes:
        return keccak(self.as_bytes)

    @classmethod
    def from_bytes(cls, serialized: bytes) -> "Lock":
        expiration = BlockExpiration(int.from_bytes(serialized[:32], byteorder='big'))
        amount = PaymentWithFeeAmount(int.from_bytes(serialized[32:64], byteorder='big'))
        secrethash = SecretHash(serialized[64:])
        return cls(expiration=expiration, amount=amount, secrethash=secrethash)


@dataclass(repr=False, eq=False)
class EnvelopeMessage(SignedRetrieableMessage):
    chain_id: ChainID
    message_identifier: MessageID
    nonce: Nonce
    token_network_address: TokenNetworkAddress
    channel_identifier: ChannelID
    transferred_amount: PaymentAmount
    locked_amount: LockedAmount
    locksroot: Locksroot
    signature: bytes

    def __post_init__(self) -> None:
        assert_envelope_values(
            self.nonce,
            self.channel_identifier,
            self.transferred_amount,
            self.locked_amount,
            self.locksroot,
        )

    @property
    def message_hash(self) -> bytes:
        raise NotImplementedError

    def _data_to_sign(self) -> bytes:
        balance_hash: bytes = hash_balance_data(
            self.transferred_amount, self.locked_amount, self.locksroot
        )
        additional_hash: AdditionalHash = AdditionalHash(self.message_hash)
        canonical_identifier = CanonicalIdentifier(
            chain_identifier=self.chain_id,
            token_network_address=self.token_network_address,
            channel_identifier=self.channel_identifier,
        )
        balance_proof_packed: bytes = pack_balance_proof(
            nonce=self.nonce,
            balance_hash=balance_hash,
            additional_hash=additional_hash,
            canonical_identifier=canonical_identifier,
        )
        return balance_proof_packed


@dataclass(repr=False, eq=False)
class SecretRequest(SignedRetrieableMessage):
    chain_id: ChainID = None  # Assuming not used for _data_to_sign
    message_identifier: MessageID
    payment_identifier: PaymentID
    secrethash: SecretHash
    amount: TokenAmount
    expiration: BlockExpiration
    signature: bytes
    cmdid: ClassVar[CmdId] = CmdId.SECRETREQUEST

    @classmethod
    def from_event(cls, event: SendSecretRequest) -> "SecretRequest":
        return cls(
            message_identifier=event.message_identifier,
            payment_identifier=event.payment_identifier,
            secrethash=event.secrethash,
            amount=event.amount,
            expiration=event.expiration,
            signature=EMPTY_SIGNATURE,
        )

    def _data_to_sign(self) -> bytes:
        return (
            bytes([self.cmdid.value, 0, 0, 0])
            + self.message_identifier.to_bytes(8, byteorder='big')
            + self.payment_identifier.to_bytes(8, byteorder='big')
            + self.secrethash
            + self.amount.to_bytes(32, byteorder='big')
            + self.expiration.to_bytes(32, byteorder='big')
        )


@dataclass(repr=False, eq=False)
class Unlock(EnvelopeMessage):
    payment_identifier: PaymentID
    secret: bytes = field(repr=False)
    cmdid: ClassVar[CmdId] = CmdId.UNLOCK

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
    def from_event(cls, event: SendUnlock) -> "Unlock":
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
            signature=EMPTY_SIGNATURE,
        )

    @property
    def message_hash(self) -> bytes:
        return eth_hash.keccak(
            bytes([self.cmdid.value])
            + self.message_identifier.to_bytes(8, byteorder='big')
            + self.payment_identifier.to_bytes(8, byteorder='big')
            + self.secret
        )


@dataclass(repr=False, eq=False)
class RevealSecret(SignedRetrieableMessage):
    message_identifier: MessageID
    secret: bytes = field(repr=False)
    signature: bytes
    cmdid: ClassVar[CmdId] = CmdId.REVEALSECRET

    @property
    def secrethash(self) -> bytes:
        return sha256(self.secret).digest()

    @classmethod
    def from_event(cls, event: SendSecretReveal) -> "RevealSecret":
        return cls(
            message_identifier=event.message_identifier,
            secret=event.secret,
            signature=EMPTY_SIGNATURE,
        )

    def _data_to_sign(self) -> bytes:
        return (
            bytes([self.cmdid.value, 0, 0, 0])
            + self.message_identifier.to_bytes(8, byteorder='big')
            + self.secret
        )


@dataclass(repr=False, eq=False)
class LockedTransferBase(EnvelopeMessage):
    payment_identifier: PaymentID
    token: TokenAddress
    recipient: Address
    lock: Lock
    target: TargetAddress
    initiator: InitiatorAddress
    signature: bytes
    metadata: Metadata

    def __post_init__(self) -> None:
        super().__post_init__()
        assert_transfer_values(self.payment_identifier, self.token, self.recipient)
        if len(self.target) != 20:
            raise ValueError('target is an invalid address')
        if len(self.initiator) != 20:
            raise ValueError('initiator is an invalid address')

    @classmethod
    def from_event(cls, event: Any) -> "LockedTransferBase":
        transfer = event.transfer
        balance_proof = transfer.balance_proof
        lock_obj = Lock(
            amount=transfer.lock.amount,
            expiration=transfer.lock.expiration,
            secrethash=transfer.lock.secrethash,
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
            lock=lock_obj,
            target=transfer.target,
            initiator=transfer.initiator,
            signature=EMPTY_SIGNATURE,
            metadata=Metadata.from_event(event=event),
        )

    def _packed_data(self) -> bytes:
        return self._pack_locked_transfer_data(
            self.cmdid,
            self.message_identifier,
            self.payment_identifier,
            self.token,
            self.recipient,
            self.target,
            self.initiator,
            self.lock,
        )

    @classmethod
    def _pack_locked_transfer_data(
        cls,
        cmdid: CmdId,
        message_identifier: int,
        payment_identifier: int,
        token: bytes,
        recipient: bytes,
        target: bytes,
        initiator: bytes,
        lock: Lock,
    ) -> bytes:
        return (
            bytes([cmdid.value])
            + message_identifier.to_bytes(8, byteorder='big')
            + payment_identifier.to_bytes(8, byteorder='big')
            + lock.expiration.to_bytes(32, byteorder='big')
            + token
            + recipient
            + target
            + initiator
            + lock.secrethash
            + lock.amount.to_bytes(32, byteorder='big')
        )


@dataclass(repr=False, eq=False)
class LockedTransfer(LockedTransferBase):
    cmdid: ClassVar[CmdId] = CmdId.LOCKEDTRANSFER

    @property
    def message_hash(self) -> bytes:
        metadata_hash: bytes = self.metadata.hash if self.metadata and self.metadata.hash else b""
        return keccak(self._packed_data() + metadata_hash)


@dataclass(repr=False, eq=False)
class RefundTransfer(LockedTransferBase):
    cmdid: ClassVar[CmdId] = CmdId.REFUNDTRANSFER

    @property
    def message_hash(self) -> bytes:
        return keccak(self._packed_data())


@dataclass(repr=False, eq=False)
class LockExpired(EnvelopeMessage):
    message_identifier: MessageID
    recipient: Address
    secrethash: SecretHash
    signature: bytes
    cmdid: ClassVar[CmdId] = CmdId.LOCKEXPIRED

    @classmethod
    def from_event(cls, event: SendLockExpired) -> "LockExpired":
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
            signature=EMPTY_SIGNATURE,
        )

    @property
    def message_hash(self) -> bytes:
        return eth_hash.keccak(
            bytes([self.cmdid.value])
            + self.message_identifier.to_bytes(8, byteorder='big')
            + self.recipient
            + self.secrethash
        )