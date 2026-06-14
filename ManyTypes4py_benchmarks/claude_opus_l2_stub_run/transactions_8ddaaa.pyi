from typing import Any, Dict, Sequence, Tuple, Type

from cached_property import cached_property
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
import rlp
from rlp.sedes import CountableList, big_endian_int, binary

from eth.abc import (
    ComputationAPI,
    ReceiptAPI,
    SignedTransactionAPI,
    TransactionDecoderAPI,
    UnsignedTransactionAPI,
)
from eth.rlp.receipts import Receipt
from eth.rlp.sedes import address as address
from eth.rlp.transactions import SignedTransactionMethods
from eth.vm.forks.berlin.transactions import (
    AccessListPayloadDecoder,
    AccountAccesses,
    BerlinLegacyTransaction,
    BerlinTransactionBuilder,
    BerlinUnsignedLegacyTransaction,
    TypedTransaction,
)
from .constants import DYNAMIC_FEE_TRANSACTION_TYPE as DYNAMIC_FEE_TRANSACTION_TYPE
from .receipts import LondonReceiptBuilder


class LondonLegacyTransaction(BerlinLegacyTransaction): ...


class LondonUnsignedLegacyTransaction(BerlinUnsignedLegacyTransaction):
    def as_signed_transaction(
        self, private_key: PrivateKey, chain_id: int = ...
    ) -> LondonLegacyTransaction: ...


class UnsignedDynamicFeeTransaction(rlp.Serializable, UnsignedTransactionAPI):
    _type_id: int
    fields: list[Tuple[str, Any]]

    chain_id: int
    nonce: int
    max_priority_fee_per_gas: int
    max_fee_per_gas: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: Any

    @cached_property
    def _type_byte(self) -> bytes: ...
    def get_message_for_signing(self) -> bytes: ...
    def as_signed_transaction(
        self, private_key: PrivateKey, chain_id: int = ...
    ) -> LondonTypedTransaction: ...
    def validate(self) -> None: ...
    def gas_used_by(self, computation: ComputationAPI) -> int: ...
    def get_intrinsic_gas(self) -> int: ...
    @property
    def intrinsic_gas(self) -> int: ...


class DynamicFeeTransaction(rlp.Serializable, SignedTransactionMethods, SignedTransactionAPI):
    _type_id: int
    fields: list[Tuple[str, Any]]

    chain_id: int
    nonce: int
    max_priority_fee_per_gas: int
    max_fee_per_gas: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: Any
    y_parity: int
    r: int
    s: int

    @property
    def gas_price(self) -> int: ...
    @property
    def max_fee_per_blob_gas(self) -> int: ...
    @property
    def blob_versioned_hashes(self) -> Sequence[Hash32]: ...
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
        log_entries: Tuple[Tuple[bytes, Tuple[int, ...], bytes], ...],
    ) -> ReceiptAPI: ...


class DynamicFeePayloadDecoder(TransactionDecoderAPI):
    @classmethod
    def decode(cls, payload: bytes) -> SignedTransactionAPI: ...


class LondonTypedTransaction(TypedTransaction):
    decoders: Dict[int, Type[TransactionDecoderAPI]]
    receipt_builder: Type[LondonReceiptBuilder]


class LondonTransactionBuilder(BerlinTransactionBuilder):
    legacy_signed: Type[LondonLegacyTransaction]
    legacy_unsigned: Type[LondonUnsignedLegacyTransaction]
    typed_transaction: Type[LondonTypedTransaction]

    @classmethod
    def new_unsigned_dynamic_fee_transaction(
        cls,
        chain_id: int,
        nonce: int,
        max_priority_fee_per_gas: int,
        max_fee_per_gas: int,
        gas: int,
        to: Address,
        value: int,
        data: bytes,
        access_list: Sequence[Any],
    ) -> UnsignedDynamicFeeTransaction: ...

    @classmethod
    def new_dynamic_fee_transaction(
        cls,
        chain_id: int,
        nonce: int,
        max_priority_fee_per_gas: int,
        max_fee_per_gas: int,
        gas: int,
        to: Address,
        value: int,
        data: bytes,
        access_list: Sequence[Any],
        y_parity: int,
        r: int,
        s: int,
    ) -> LondonTypedTransaction: ...