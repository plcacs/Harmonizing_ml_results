from typing import Any, Dict, Sequence, Tuple, Type, Union, Optional, cast, overload
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
import rlp
from eth.abc import ComputationAPI, SignedTransactionAPI, TransactionBuilderAPI, TransactionDecoderAPI
from eth.rlp.logs import Log
from eth.rlp.receipts import Receipt
from .receipts import BerlinReceiptBuilder

class BerlinLegacyTransaction(rlp.Serializable): ...

class BerlinUnsignedLegacyTransaction(rlp.Serializable):
    nonce: int
    gas_price: int
    gas: int
    to: Address
    value: int
    data: bytes

    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = ...) -> BerlinLegacyTransaction: ...

class AccountAccesses(rlp.Serializable):
    account: Address
    storage_keys: Sequence[bytes]

class UnsignedAccessListTransaction(rlp.Serializable, SignedTransactionAPI):
    _type_id: int
    chain_id: int
    nonce: int
    gas_price: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: Sequence[AccountAccesses]

    @property
    def _type_byte(self) -> bytes: ...

    def get_message_for_signing(self) -> bytes: ...

    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = ...) -> 'TypedTransaction': ...

    def validate(self) -> None: ...

    def gas_used_by(self, computation: ComputationAPI) -> int: ...

    def get_intrinsic_gas(self) -> int: ...

    @property
    def intrinsic_gas(self) -> int: ...

    @property
    def max_priority_fee_per_gas(self) -> int: ...

    @property
    def max_fee_per_gas(self) -> int: ...

class AccessListTransaction(rlp.Serializable, SignedTransactionAPI):
    _type_id: int
    chain_id: int
    nonce: int
    gas_price: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: Sequence[AccountAccesses]
    y_parity: int
    r: int
    s: int

    def get_sender(self) -> Address: ...

    def get_message_for_signing(self) -> bytes: ...

    def check_signature_validity(self) -> None: ...

    @property
    def _type_byte(self) -> bytes: ...

    @property
    def hash(self) -> Hash32: ...

    def get_intrinsic_gas(self) -> int: ...

    def encode(self) -> bytes: ...

    def make_receipt(self, status: int, gas_used: int, log_entries: Sequence[Tuple[Address, Sequence[bytes], bytes]]) -> Receipt: ...

    @property
    def max_priority_fee_per_gas(self) -> int: ...

    @property
    def max_fee_per_gas(self) -> int: ...

    @property
    def max_fee_per_blob_gas(self) -> Any: ...

    @property
    def blob_versioned_hashes(self) -> Any: ...

class AccessListPayloadDecoder(TransactionDecoderAPI):
    @classmethod
    def decode(cls, payload: bytes) -> AccessListTransaction: ...

class TypedTransaction(SignedTransactionAPI, TransactionDecoderAPI):
    rlp_type: Any
    decoders: Dict[int, Type[AccessListPayloadDecoder]]
    receipt_builder: Type[BerlinReceiptBuilder]

    def __init__(self, type_id: int, proxy_target: Any) -> None: ...

    @classmethod
    def get_payload_codec(cls, type_id: int) -> Type[AccessListPayloadDecoder]: ...

    def encode(self) -> bytes: ...

    @classmethod
    def decode(cls, encoded: bytes) -> 'TypedTransaction': ...

    @classmethod
    def serialize(cls, obj: 'TypedTransaction') -> bytes: ...

    @classmethod
    def deserialize(cls, encoded_unchecked: bytes) -> 'TypedTransaction': ...

    @property
    def _type_byte(self) -> bytes: ...

    @property
    def chain_id(self) -> int: ...

    @property
    def nonce(self) -> int: ...

    @property
    def gas_price(self) -> int: ...

    @property
    def max_priority_fee_per_gas(self) -> int: ...

    @property
    def max_fee_per_gas(self) -> int: ...

    @property
    def max_fee_per_blob_gas(self) -> Any: ...

    @property
    def blob_versioned_hashes(self) -> Any: ...

    @property
    def gas(self) -> int: ...

    @property
    def to(self) -> Address: ...

    @property
    def value(self) -> int: ...

    @property
    def data(self) -> bytes: ...

    @property
    def y_parity(self) -> int: ...

    @property
    def r(self) -> int: ...

    @property
    def s(self) -> int: ...

    @property
    def access_list(self) -> Sequence[AccountAccesses]: ...

    def get_sender(self) -> Address: ...

    def get_message_for_signing(self) -> bytes: ...

    def check_signature_validity(self) -> None: ...

    @property
    def hash(self) -> Hash32: ...

    def get_intrinsic_gas(self) -> int: ...

    def copy(self, **overrides: Any) -> 'TypedTransaction': ...

    def __eq__(self, other: object) -> bool: ...

    def make_receipt(self, status: int, gas_used: int, log_entries: Sequence[Tuple[Address, Sequence[bytes], bytes]]) -> Receipt: ...

    def __hash__(self) -> int: ...

class BerlinTransactionBuilder(TransactionBuilderAPI):
    legacy_signed: Type[BerlinLegacyTransaction]
    legacy_unsigned: Type[BerlinUnsignedLegacyTransaction]
    typed_transaction: Type[TypedTransaction]

    @classmethod
    def decode(cls, encoded: bytes) -> Union[TypedTransaction, BerlinLegacyTransaction]: ...

    @classmethod
    def deserialize(cls, encoded: bytes) -> Union[TypedTransaction, BerlinLegacyTransaction]: ...

    @classmethod
    def serialize(cls, obj: Union[TypedTransaction, BerlinLegacyTransaction]) -> bytes: ...

    @classmethod
    def create_unsigned_transaction(cls, *, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes) -> BerlinUnsignedLegacyTransaction: ...

    @classmethod
    def new_transaction(cls, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes, v: int, r: int, s: int) -> BerlinLegacyTransaction: ...

    @classmethod
    def new_unsigned_access_list_transaction(cls, chain_id: int, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes, access_list: Sequence[AccountAccesses]) -> UnsignedAccessListTransaction: ...

    @classmethod
    def new_access_list_transaction(cls, chain_id: int, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes, access_list: Sequence[AccountAccesses], y_parity: int, r: int, s: int) -> TypedTransaction: ...

def _calculate_txn_intrinsic_gas_berlin(klass: Union[UnsignedAccessListTransaction, AccessListTransaction]) -> int: ...