from abc import ABC
from functools import cached_property
from typing import Dict, Sequence, Tuple, Type
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
from eth_utils import to_bytes
import rlp
from rlp.sedes import CountableList, big_endian_int, binary
from eth._utils.transactions import create_transaction_signature, extract_transaction_sender, validate_transaction_signature
from eth.abc import ComputationAPI, ReceiptAPI, SignedTransactionAPI, TransactionDecoderAPI, UnsignedTransactionAPI
from eth.rlp.logs import Log
from eth.rlp.receipts import Receipt
from eth.rlp.sedes import address
from eth.rlp.transactions import SignedTransactionMethods
from eth.validation import validate_canonical_address, validate_is_bytes, validate_is_list_like, validate_is_transaction_access_list, validate_uint64, validate_uint256
from eth.vm.forks.berlin.constants import ACCESS_LIST_TRANSACTION_TYPE
from eth.vm.forks.berlin.transactions import AccessListPayloadDecoder, AccountAccesses, TypedTransaction, _calculate_txn_intrinsic_gas_berlin
from eth.vm.forks.london.constants import DYNAMIC_FEE_TRANSACTION_TYPE
from eth.vm.forks.shanghai.transactions import ShanghaiLegacyTransaction, ShanghaiTransactionBuilder, ShanghaiUnsignedLegacyTransaction
from ..london.transactions import DynamicFeePayloadDecoder
from .constants import BLOB_TX_TYPE
from .receipts import CancunReceiptBuilder

class CancunLegacyTransaction(ShanghaiLegacyTransaction, ABC):
    """Cancun legacy transaction class."""
    pass

class CancunUnsignedLegacyTransaction(ShanghaiUnsignedLegacyTransaction):
    """Cancun unsigned legacy transaction class."""

    def as_signed_transaction(self, private_key: PrivateKey, chain_id: int = None) -> CancunLegacyTransaction:
        """As signed transaction method.

        Args:
            private_key (PrivateKey): Private key to sign the transaction.
            chain_id (int, optional): Chain ID. Defaults to None.

        Returns:
            CancunLegacyTransaction: Signed transaction.
        """
        v, r, s = create_transaction_signature(self, private_key, chain_id=chain_id)
        return CancunLegacyTransaction(nonce=self.nonce, gas_price=self.gas_price, gas=self.gas, to=self.to, value=self.value, data=self.data, v=v, r=r, s=s)

class UnsignedBlobTransaction(rlp.Serializable, UnsignedTransactionAPI):
    """Unsigned blob transaction class."""
    _type_id: int = BLOB_TX_TYPE
    fields: Tuple[str, rlp.sedes.sedes] = [('chain_id', big_endian_int), ('nonce', big_endian_int), ('max_priority_fee_per_gas', big_endian_int), ('max_fee_per_gas', big_endian_int), ('gas', big_endian_int), ('to', address), ('value', big_endian_int), ('data', binary), ('access_list', CountableList(AccountAccesses)), ('max_fee_per_blob_gas', big_endian_int), ('blob_versioned_hashes', CountableList(binary))]

    def as_signed_transaction(self, private_key: PrivateKey, chain_id: int = None) -> CancunTypedTransaction:
        """As signed transaction method.

        Args:
            private_key (PrivateKey): Private key to sign the transaction.
            chain_id (int, optional): Chain ID. Defaults to None.

        Returns:
            CancunTypedTransaction: Signed transaction.
        """
        message = self.get_message_for_signing()
        signature = private_key.sign_msg(message)
        y_parity, r, s = signature.vrs
        signed_transaction = BlobTransaction(self.chain_id, self.nonce, self.max_priority_fee_per_gas, self.max_fee_per_gas, self.gas, self.to, self.value, self.data, self.access_list, self.max_fee_per_blob_gas, self.blob_versioned_hashes, y_parity, r, s)
        return CancunTypedTransaction(self._type_id, signed_transaction)

    def validate(self) -> None:
        """Validate transaction method."""
        validate_uint256(self.chain_id, title='Transaction.chain_id')
        validate_uint64(self.nonce, title='Transaction.nonce')
        validate_uint256(self.max_fee_per_gas, title='Transaction.max_fee_per_gas')
        validate_uint256(self.max_priority_fee_per_gas, title='Transaction.max_priority_fee_per_gas')
        validate_uint256(self.gas, title='Transaction.gas')
        validate_canonical_address(self.to, title='Transaction.to')
        validate_uint256(self.value, title='Transaction.value')
        validate_is_bytes(self.data, title='Transaction.data')
        validate_is_transaction_access_list(self.access_list)
        validate_uint256(self.max_fee_per_blob_gas, title='Transaction.max_fee_per_blob_gas')
        validate_is_list_like(self.blob_versioned_hashes, title='Transaction.blob_versioned_hashes')
        for blob_versioned_hash in self.blob_versioned_hashes:
            validate_is_bytes(blob_versioned_hash, title='Transaction.blob_versioned_hash', size=32)

    @cached_property
    def _type_byte(self) -> bytes:
        """Type byte property."""
        return to_bytes(self._type_id)

    def get_message_for_signing(self) -> bytes:
        """Get message for signing method.

        Returns:
            bytes: Message to be signed.
        """
        payload = rlp.encode(self)
        return self._type_byte + payload

    def gas_used_by(self, computation: ComputationAPI) -> int:
        """Gas used by method.

        Args:
            computation (ComputationAPI): Computation API.

        Returns:
            int: Gas used.
        """
        return self.intrinsic_gas + computation.get_gas_used()

    def get_intrinsic_gas(self) -> int:
        """Get intrinsic gas method.

        Returns:
            int: Intrinsic gas.
        """
        return _calculate_txn_intrinsic_gas_berlin(self)

    @property
    def intrinsic_gas(self) -> int:
        """Intrinsic gas property.

        Returns:
            int: Intrinsic gas.
        """
        return self.get_intrinsic_gas()

class BlobTransaction(rlp.Serializable, SignedTransactionMethods, SignedTransactionAPI):
    """Blob transaction class."""
    _type_id: int = BLOB_TX_TYPE
    fields: Tuple[str, rlp.sedes.sedes] = [('chain_id', big_endian_int), ('nonce', big_endian_int), ('max_priority_fee_per_gas', big_endian_int), ('max_fee_per_gas', big_endian_int), ('gas', big_endian_int), ('to', address), ('value', big_endian_int), ('data', binary), ('access_list', CountableList(AccountAccesses)), ('max_fee_per_blob_gas', big_endian_int), ('blob_versioned_hashes', CountableList(binary)), ('y_parity', big_endian_int), ('r', big_endian_int), ('s', big_endian_int)]

    @property
    def gas_price(self) -> None:
        """Gas price property.

        Raises:
            AttributeError: Gas price is no longer available.
        """
        raise AttributeError('Gas price is no longer available.See max_priority_fee_per_gas or max_fee_per_gas')

    def get_sender(self) -> Address:
        """Get sender method.

        Returns:
            Address: Sender address.
        """
        return extract_transaction_sender(self)

    def get_message_for_signing(self) -> bytes:
        """Get message for signing method.

        Returns:
            bytes: Message to be signed.
        """
        unsigned = UnsignedBlobTransaction(self.chain_id, self.nonce, self.max_priority_fee_per_gas, self.max_fee_per_gas, self.gas, self.to, self.value, self.data, self.access_list, self.max_fee_per_blob_gas, self.blob_versioned_hashes)
        payload = rlp.encode(unsigned)
        return self._type_byte + payload

    def check_signature_validity(self) -> None:
        """Check signature validity method."""
        validate_transaction_signature(self)

    @cached_property
    def _type_byte(self) -> bytes:
        """Type byte property.

        Returns:
            bytes: Type byte.
        """
        return to_bytes(self._type_id)

    @cached_property
    def hash(self) -> None:
        """Hash property.

        Raises:
            NotImplementedError: Hash method not implemented.
        """
        raise NotImplementedError('Call hash() on the TypedTransaction instead')

    def get_intrinsic_gas(self) -> int:
        """Get intrinsic gas method.

        Returns:
            int: Intrinsic gas.
        """
        return _calculate_txn_intrinsic_gas_berlin(self)

    def encode(self) -> bytes:
        """Encode method.

        Returns:
            bytes: Encoded transaction.
        """
        return rlp.encode(self)

    def make_receipt(self, status: Hash32, gas_used: int, log_entries: Sequence[Tuple[Address, Sequence[Hash32], bytes]]) -> Receipt:
        """Make receipt method.

        Args:
            status (Hash32): Status.
            gas_used (int): Gas used.
            log_entries (Sequence[Tuple[Address, Sequence[Hash32], bytes]]): Log entries.

        Returns:
            Receipt: Receipt.
        """
        logs = [Log(address, topics, data) for address, topics, data in log_entries]
        return Receipt(state_root=status, gas_used=gas_used, logs=logs)

class BlobPayloadDecoder(TransactionDecoderAPI):
    """Blob payload decoder class."""

    @classmethod
    def decode(cls, payload: bytes) -> BlobTransaction:
        """Decode method.

        Args:
            payload (bytes): Payload to be decoded.

        Returns:
            BlobTransaction: Decoded transaction.
        """
        return rlp.decode(payload, sedes=BlobTransaction)

class CancunTypedTransaction(TypedTransaction):
    """Cancun typed transaction class."""
    decoders: Dict[int, Type[TransactionDecoderAPI]] = {ACCESS_LIST_TRANSACTION_TYPE: AccessListPayloadDecoder, DYNAMIC_FEE_TRANSACTION_TYPE: DynamicFeePayloadDecoder, BLOB_TX_TYPE: BlobPayloadDecoder}
    receipt_builder: Type[ReceiptAPI] = CancunReceiptBuilder

    @property
    def max_fee_per_blob_gas(self) -> int:
        """Max fee per blob gas property.

        Returns:
            int: Max fee per blob gas.
        """
        return self._inner.max_fee_per_blob_gas

    @property
    def blob_versioned_hashes(self) -> Sequence[bytes]:
        """Blob versioned hashes property.

        Returns:
            Sequence[bytes]: Blob versioned hashes.
        """
        return self._inner.blob_versioned_hashes

class CancunTransactionBuilder(ShanghaiTransactionBuilder):
    """Cancun transaction builder class."""
    legacy_signed: Type[ShanghaiLegacyTransaction] = CancunLegacyTransaction
    legacy_unsigned: Type[ShanghaiUnsignedLegacyTransaction] = CancunUnsignedLegacyTransaction
    typed_transaction: Type[CancunTypedTransaction] = CancunTypedTransaction

    @classmethod
    def new_unsigned_blob_transaction(cls, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: AccountAccesses, max_fee_per_blob_gas: int, blob_versioned_hashes: Sequence[bytes]) -> UnsignedBlobTransaction:
        """New unsigned blob transaction method.

        Args:
            chain_id (int): Chain ID.
            nonce (int): Nonce.
            max_priority_fee_per_gas (int): Max priority fee per gas.
            max_fee_per_gas (int): Max fee per gas.
            gas (int): Gas.
            to (Address): To address.
            value (int): Value.
            data (bytes): Data.
            access_list (AccountAccesses): Access list.
            max_fee_per_blob_gas (int): Max fee per blob gas.
            blob_versioned_hashes (Sequence[bytes]): Blob versioned hashes.

        Returns:
            UnsignedBlobTransaction: Unsigned blob transaction.
        """
        transaction = UnsignedBlobTransaction(chain_id, nonce, max_priority_fee_per_gas, max_fee_per_gas, gas, to, value, data, access_list, max_fee_per_blob_gas, blob_versioned_hashes)
        return transaction

    @classmethod
    def new_blob_transaction(cls, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: AccountAccesses, max_fee_per_blob_gas: int, blob_versioned_hashes: Sequence[bytes], y_parity: int, r: int, s: int) -> CancunTypedTransaction:
        """New blob transaction method.

        Args:
            chain_id (int): Chain ID.
            nonce (int): Nonce.
            max_priority_fee_per_gas (int): Max priority fee per gas.
            max_fee_per_gas (int): Max fee per gas.
            gas (int): Gas.
            to (Address): To address.
            value (int): Value.
            data (bytes): Data.
            access_list (AccountAccesses): Access list.
            max_fee_per_blob_gas (int): Max fee per blob gas.
            blob_versioned_hashes (Sequence[bytes]): Blob versioned hashes.
            y_parity (int): Y parity.
            r (int): R.
            s (int): S.

        Returns:
            CancunTypedTransaction: Cancun typed transaction.
        """
        transaction = BlobTransaction(chain_id, nonce, max_priority_fee_per_gas, max_fee_per_gas, gas, to, value, data, access_list, max_fee_per_blob_gas, blob_versioned_hashes, y_parity, r, s)
        return CancunTypedTransaction(cls.typed_transaction._type_id, transaction)
