from typing import Dict, Sequence, Tuple, Type, Optional
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
from eth.abc import (
    ComputationAPI,
    ReceiptAPI,
    SignedTransactionAPI,
    TransactionDecoderAPI,
    UnsignedTransactionAPI,
)
from eth.rlp.transactions import SignedTransactionMethods
from eth.vm.forks.berlin.transactions import (
    BerlinLegacyTransaction,
    BerlinUnsignedLegacyTransaction,
    TypedTransaction,
)
from .receipts import LondonReceiptBuilder

class LondonLegacyTransaction(BerlinLegacyTransaction):
    def __init__(self, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes, v: int, r: int, s: int):
        ...

class LondonUnsignedLegacyTransaction(BerlinUnsignedLegacyTransaction):
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = None) -> LondonLegacyTransaction:
        ...

class UnsignedDynamicFeeTransaction(rlp.Serializable, UnsignedTransactionAPI):
    chain_id: int
    nonce: int
    max_priority_fee_per_gas: int
    max_fee_per_gas: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: Sequence[AccountAccesses]

    def __init__(self, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: Sequence[AccountAccesses]):
        ...

    def get_message_for_signing(self) -> bytes:
        ...

    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = None) -> LondonTypedTransaction:
        ...

    def validate(self) -> None:
        ...

    def gas_used_by(self, computation: ComputationAPI) -> int:
        ...

    @property
    def intrinsic_gas(self) -> int:
        ...

class DynamicFeeTransaction(rlp.Serializable, SignedTransactionMethods, SignedTransactionAPI):
    chain_id: int
    nonce: int
    max_priority_fee_per_gas: int
    max_fee_per_gas: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: Sequence[AccountAccesses]
    y_parity: int
    r: int
    s: int

    def __init__(self, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: Sequence[AccountAccesses], y_parity: int, r: int, s: int):
        ...

    @property
    def gas_price(self) -> int:
        ...

    @property
    def max_fee_per_blob_gas(self) -> int:
        ...

    @property
    def blob_versioned_hashes(self) -> Sequence[Hash32]:
        ...

    def get_sender(self) -> Address:
        ...

    def get_message_for_signing(self) -> bytes:
        ...

    def check_signature_validity(self) -> None:
        ...

    @property
    def hash(self) -> Hash32:
        ...

    def get_intrinsic_gas(self) -> int:
        ...

    def encode(self) -> bytes:
        ...

    def make_receipt(self, status: bytes, gas_used: int, log_entries: Sequence[Tuple[Address, Sequence[Hash32], bytes]]) -> Receipt:
        ...

class DynamicFeePayloadDecoder(TransactionDecoderAPI):
    @classmethod
    def decode(cls, payload: bytes) -> DynamicFeeTransaction:
        ...

class LondonTypedTransaction(TypedTransaction):
    receipt_builder: Type[LondonReceiptBuilder]

    def __init__(self, type_id: int, transaction: DynamicFeeTransaction):
        ...

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
        access_list: Sequence[AccountAccesses],
    ) -> UnsignedDynamicFeeTransaction:
        ...

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
        access_list: Sequence[AccountAccesses],
        y_parity: int,
        r: int,
        s: int,
    ) -> LondonTypedTransaction:
        ...