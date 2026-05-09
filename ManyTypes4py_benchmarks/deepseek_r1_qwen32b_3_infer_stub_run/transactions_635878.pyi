from typing import Any, Dict, List, Optional, Tuple, Union
from cached_property import cached_property
from eth_hash.auto import keccak
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
from eth_utils import ValidationError
import rlp
from eth.abc import (
    ComputationAPI,
    DecodedZeroOrOneLayerRLP,
    ReceiptAPI,
    SignedTransactionAPI,
    TransactionBuilderAPI,
    TransactionDecoderAPI,
    UnsignedTransactionAPI,
)
from eth.rlp.logs import Log
from eth.rlp.receipts import Receipt
from eth.rlp.sedes import address
from eth.vm.forks.muir_glacier.transactions import (
    MuirGlacierTransaction,
    MuirGlacierUnsignedTransaction,
)

class BerlinLegacyTransaction(MuirGlacierTransaction):
    ...

class BerlinUnsignedLegacyTransaction(MuirGlacierUnsignedTransaction):
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = ...) -> BerlinLegacyTransaction:
        ...

class AccountAccesses(rlp.Serializable):
    fields = [('account', address), ('storage_keys', List[BigEndianInt])]
    ...

class UnsignedAccessListTransaction(rlp.Serializable, UnsignedTransactionAPI):
    fields = [
        ('chain_id', big_endian_int),
        ('nonce', big_endian_int),
        ('gas_price', big_endian_int),
        ('gas', big_endian_int),
        ('to', address),
        ('value', big_endian_int),
        ('data', binary),
        ('access_list', List[AccountAccesses]),
    ]
    def get_message_for_signing(self) -> bytes:
        ...
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = ...) -> TypedTransaction:
        ...
    def validate(self) -> None:
        ...
    def gas_used_by(self, computation: ComputationAPI) -> int:
        ...
    def get_intrinsic_gas(self) -> int:
        ...
    @property
    def intrinsic_gas(self) -> int:
        ...
    @property
    def max_priority_fee_per_gas(self) -> int:
        ...
    @property
    def max_fee_per_gas(self) -> int:
        ...

class AccessListTransaction(rlp.Serializable, SignedTransactionMethods, SignedTransactionAPI):
    fields = [
        ('chain_id', big_endian_int),
        ('nonce', big_endian_int),
        ('gas_price', big_endian_int),
        ('gas', big_endian_int),
        ('to', address),
        ('value', big_endian_int),
        ('data', binary),
        ('access_list', List[AccountAccesses]),
        ('y_parity', big_endian_int),
        ('r', big_endian_int),
        ('s', big_endian_int),
    ]
    def get_sender(self) -> Address:
        ...
    def get_message_for_signing(self) -> bytes:
        ...
    def check_signature_validity(self) -> None:
        ...
    def make_receipt(self, status: bytes, gas_used: int, log_entries: List[Tuple[Address, List[bytes], bytes]]) -> ReceiptAPI:
        ...
    @property
    def max_priority_fee_per_gas(self) -> int:
        ...
    @property
    def max_fee_per_gas(self) -> int:
        ...
    @property
    def max_fee_per_blob_gas(self) -> int:
        ...
    @property
    def blob_versioned_hashes(self) -> List[bytes]:
        ...

class AccessListPayloadDecoder(TransactionDecoderAPI):
    @classmethod
    def decode(cls, payload: bytes) -> AccessListTransaction:
        ...

class TypedTransaction(SignedTransactionMethods, SignedTransactionAPI, TransactionDecoderAPI):
    rlp_type = Binary
    decoders: Dict[int, TransactionDecoderAPI]
    receipt_builder = BerlinReceiptBuilder
    def __init__(self, type_id: int, proxy_target: SignedTransactionAPI) -> None:
        ...
    @classmethod
    def get_payload_codec(cls, type_id: int) -> TransactionDecoderAPI:
        ...
    def encode(self) -> bytes:
        ...
    @classmethod
    def decode(cls, encoded: bytes) -> TypedTransaction:
        ...
    @classmethod
    def serialize(cls, obj: SignedTransactionAPI) -> bytes:
        ...
    @classmethod
    def deserialize(cls, encoded_unchecked: bytes) -> TypedTransaction:
        ...
    @property
    def chain_id(self) -> int:
        ...
    @property
    def nonce(self) -> int:
        ...
    @property
    def gas_price(self) -> int:
        ...
    @property
    def max_priority_fee_per_gas(self) -> int:
        ...
    @property
    def max_fee_per_gas(self) -> int:
        ...
    @property
    def max_fee_per_blob_gas(self) -> int:
        ...
    @property
    def blob_versioned_hashes(self) -> List[bytes]:
        ...
    @property
    def gas(self) -> int:
        ...
    @property
    def to(self) -> Optional[Address]:
        ...
    @property
    def value(self) -> int:
        ...
    @property
    def data(self) -> bytes:
        ...
    @property
    def y_parity(self) -> int:
        ...
    @property
    def r(self) -> int:
        ...
    @property
    def s(self) -> int:
        ...
    @property
    def access_list(self) -> List[AccountAccesses]:
        ...
    def get_sender(self) -> Address:
        ...
    def check_signature_validity(self) -> None:
        ...
    @property
    def hash(self) -> Hash32:
        ...
    def get_intrinsic_gas(self) -> int:
        ...
    def copy(self, **overrides: Any) -> TypedTransaction:
        ...
    def __eq__(self, other: Any) -> bool:
        ...
    def make_receipt(self, status: bytes, gas_used: int, log_entries: List[Tuple[Address, List[bytes], bytes]]) -> ReceiptAPI:
        ...
    def __hash__(self) -> int:
        ...

class BerlinTransactionBuilder(TransactionBuilderAPI):
    legacy_signed = BerlinLegacyTransaction
    legacy_unsigned = BerlinUnsignedLegacyTransaction
    typed_transaction = TypedTransaction
    @classmethod
    def decode(cls, encoded: bytes) -> Union[TypedTransaction, BerlinLegacyTransaction]:
        ...
    @classmethod
    def deserialize(cls, encoded: Union[bytes, SignedTransactionAPI]) -> Union[TypedTransaction, BerlinLegacyTransaction]:
        ...
    @classmethod
    def serialize(cls, obj: Union[TypedTransaction, BerlinLegacyTransaction]) -> bytes:
        ...
    @classmethod
    def create_unsigned_transaction(cls, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes) -> BerlinUnsignedLegacyTransaction:
        ...
    @classmethod
    def new_transaction(cls, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes, v: int, r: int, s: int) -> BerlinLegacyTransaction:
        ...
    @classmethod
    def new_unsigned_access_list_transaction(cls, chain_id: int, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes, access_list: List[AccountAccesses]) -> UnsignedAccessListTransaction:
        ...
    @classmethod
    def new_access_list_transaction(cls, chain_id: int, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes, access_list: List[AccountAccesses], y_parity: int, r: int, s: int) -> TypedTransaction:
        ...

def _calculate_txn_intrinsic_gas_berlin(klass: Any) -> int:
    ...