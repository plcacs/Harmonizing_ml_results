from typing import Any, Dict, Sequence, Tuple, Type, Union, cast

from cached_property import cached_property
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
from rlp.sedes import Binary, CountableList

from eth.abc import (
    ComputationAPI,
    DecodedZeroOrOneLayerRLP,
    ReceiptAPI,
    SignedTransactionAPI,
    TransactionBuilderAPI,
    TransactionDecoderAPI,
    UnsignedTransactionAPI,
)
from eth.rlp.receipts import Receipt
from eth.rlp.transactions import SignedTransactionMethods
from eth.vm.forks.muir_glacier.transactions import (
    MuirGlacierTransaction,
    MuirGlacierUnsignedTransaction,
)
from .receipts import BerlinReceiptBuilder

class BerlinLegacyTransaction(MuirGlacierTransaction): ...

class BerlinUnsignedLegacyTransaction(MuirGlacierUnsignedTransaction):
    def as_signed_transaction(
        self, private_key: PrivateKey, chain_id: int = ...
    ) -> BerlinLegacyTransaction: ...

class AccountAccesses(rlp.Serializable):
    fields: list[Tuple[str, Any]]
    account: Address
    storage_keys: Sequence[int]

class UnsignedAccessListTransaction(rlp.Serializable, UnsignedTransactionAPI):
    _type_id: int
    fields: list[Tuple[str, Any]]
    chain_id: int
    nonce: int
    gas_price: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: Sequence[AccountAccesses]

    @cached_property
    def _type_byte(self) -> bytes: ...
    def get_message_for_signing(self) -> bytes: ...
    def as_signed_transaction(
        self, private_key: PrivateKey, chain_id: int = ...
    ) -> TypedTransaction: ...
    def validate(self) -> None: ...
    def gas_used_by(self, computation: ComputationAPI) -> int: ...
    def get_intrinsic_gas(self) -> int: ...
    @property
    def intrinsic_gas(self) -> int: ...
    @property
    def max_priority_fee_per_gas(self) -> int: ...
    @property
    def max_fee_per_gas(self) -> int: ...

class AccessListTransaction(rlp.Serializable, SignedTransactionMethods, SignedTransactionAPI):
    _type_id: int
    fields: list[Tuple[str, Any]]
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
    @cached_property
    def _type_byte(self) -> bytes: ...
    @cached_property
    def hash(self) -> Hash32: ...
    def get_intrinsic_gas(self) -> int: ...
    def encode(self) -> bytes: ...
    def make_receipt(
        self,
        status: bytes,
        gas_used: int,
        log_entries: Sequence[Tuple[bytes, Sequence[int], bytes]],
    ) -> Receipt: ...
    @property
    def max_priority_fee_per_gas(self) -> int: ...
    @property
    def max_fee_per_gas(self) -> int: ...
    @property
    def max_fee_per_blob_gas(self) -> int: ...
    @property
    def blob_versioned_hashes(self) -> Sequence[Hash32]: ...

class AccessListPayloadDecoder(TransactionDecoderAPI):
    @classmethod
    def decode(cls, payload: bytes) -> AccessListTransaction: ...

class TypedTransaction(SignedTransactionMethods, SignedTransactionAPI, TransactionDecoderAPI):
    rlp_type: Binary
    decoders: Dict[int, Type[TransactionDecoderAPI]]
    receipt_builder: Type[BerlinReceiptBuilder]
    type_id: int
    _inner: SignedTransactionAPI

    def __init__(self, type_id: int, proxy_target: SignedTransactionAPI) -> None: ...
    @classmethod
    def get_payload_codec(cls, type_id: int) -> Type[TransactionDecoderAPI]: ...
    def encode(self) -> bytes: ...
    @classmethod
    def decode(cls, encoded: bytes) -> TypedTransaction: ...
    @classmethod
    def serialize(cls, obj: TypedTransaction) -> bytes: ...
    @classmethod
    def deserialize(cls, encoded_unchecked: bytes) -> TypedTransaction: ...
    @cached_property
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
    def max_fee_per_blob_gas(self) -> int: ...
    @property
    def blob_versioned_hashes(self) -> Sequence[Hash32]: ...
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
    @cached_property
    def hash(self) -> Hash32: ...
    def get_intrinsic_gas(self) -> int: ...
    def copy(self, **overrides: Any) -> TypedTransaction: ...
    def __eq__(self, other: object) -> bool: ...
    def make_receipt(
        self,
        status: bytes,
        gas_used: int,
        log_entries: Sequence[Tuple[bytes, Sequence[int], bytes]],
    ) -> ReceiptAPI: ...
    def __hash__(self) -> int: ...

class BerlinTransactionBuilder(TransactionBuilderAPI):
    legacy_signed: Type[BerlinLegacyTransaction]
    legacy_unsigned: Type[BerlinUnsignedLegacyTransaction]
    typed_transaction: Type[TypedTransaction]

    @classmethod
    def decode(cls, encoded: bytes) -> SignedTransactionAPI: ...
    @classmethod
    def deserialize(cls, encoded: DecodedZeroOrOneLayerRLP) -> SignedTransactionAPI: ...
    @classmethod
    def serialize(cls, obj: SignedTransactionAPI) -> DecodedZeroOrOneLayerRLP: ...
    @classmethod
    def create_unsigned_transaction(
        cls,
        *,
        nonce: int,
        gas_price: int,
        gas: int,
        to: Address,
        value: int,
        data: bytes,
    ) -> BerlinUnsignedLegacyTransaction: ...
    @classmethod
    def new_transaction(
        cls,
        nonce: int,
        gas_price: int,
        gas: int,
        to: Address,
        value: int,
        data: bytes,
        v: int,
        r: int,
        s: int,
    ) -> BerlinLegacyTransaction: ...
    @classmethod
    def new_unsigned_access_list_transaction(
        cls,
        chain_id: int,
        nonce: int,
        gas_price: int,
        gas: int,
        to: Address,
        value: int,
        data: bytes,
        access_list: Sequence[AccountAccesses],
    ) -> UnsignedAccessListTransaction: ...
    @classmethod
    def new_access_list_transaction(
        cls,
        chain_id: int,
        nonce: int,
        gas_price: int,
        gas: int,
        to: Address,
        value: int,
        data: bytes,
        access_list: Sequence[AccountAccesses],
        y_parity: int,
        r: int,
        s: int,
    ) -> TypedTransaction: ...

def _calculate_txn_intrinsic_gas_berlin(
    klass: Union[UnsignedAccessListTransaction, AccessListTransaction],
) -> int: ...