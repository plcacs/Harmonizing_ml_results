from typing import Dict, List, Optional, Sequence, Tuple, Union
from cached_property import cached_property
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
from eth_utils import to_bytes
import rlp
from rlp.sedes import CountableList, big_endian_int, binary
from eth.abc import ComputationAPI, ReceiptAPI, SignedTransactionAPI, TransactionDecoderAPI, UnsignedTransactionAPI
from eth.rlp.logs import Log
from eth.rlp.receipts import Receipt
from eth.rlp.sedes import address
from eth.vm.forks.berlin.transactions import TypedTransaction

class LondonLegacyTransaction(BerlinLegacyTransaction):
    pass

class LondonUnsignedLegacyTransaction(BerlinUnsignedLegacyTransaction):
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = None) -> LondonLegacyTransaction: ...

class UnsignedDynamicFeeTransaction(rlp.Serializable, UnsignedTransactionAPI):
    chain_id: int
    nonce: int
    max_priority_fee_per_gas: int
    max_fee_per_gas: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: CountableList[Tuple[Address, List[int]]]

    def __init__(self, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: CountableList[Tuple[Address, List[int]]]) -> None: ...

    @cached_property
    def _type_byte(self) -> bytes: ...

    def get_message_for_signing(self) -> bytes: ...

    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = None) -> LondonTypedTransaction: ...

    def validate(self) -> None: ...

    def gas_used_by(self, computation: ComputationAPI) -> int: ...

    def get_intrinsic_gas(self) -> int: ...

    @property
    def intrinsic_gas(self) -> int: ...

class DynamicFeeTransaction(rlp.Serializable, SignedTransactionMethods, SignedTransactionAPI):
    chain_id: int
    nonce: int
    max_priority_fee_per_gas: int
    max_fee_per_gas: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: CountableList[Tuple[Address, List[int]]]
    y_parity: int
    r: int
    s: int

    def __init__(self, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: CountableList[Tuple[Address, List[int]]], y_parity: int, r: int, s: int) -> None: ...

    @property
    def gas_price(self) -> Any: ...

    @property
    def max_fee_per_blob_gas(self) -> Any: ...

    @property
    def blob_versioned_hashes(self) -> Any: ...

    def get_sender(self) -> Address: ...

    def get_message_for_signing(self) -> bytes: ...

    def check_signature_validity(self) -> None: ...

    @cached_property
    def _type_byte(self) -> bytes: ...

    @cached_property
    def hash(self) -> Any: ...

    def get_intrinsic_gas(self) -> int: ...

    def encode(self) -> bytes: ...

    def make_receipt(self, status: bytes, gas_used: int, log_entries: List[Tuple[Address, List[Hash32], bytes]]) -> Receipt: ...

class DynamicFeePayloadDecoder(TransactionDecoderAPI):
    @classmethod
    def decode(cls, payload: bytes) -> DynamicFeeTransaction: ...

class LondonTypedTransaction(TypedTransaction):
    decoders: Dict[int, TransactionDecoderAPI]
    receipt_builder: LondonReceiptBuilder

    def __init__(self, transaction_type: int, transaction: DynamicFeeTransaction) -> None: ...

class LondonTransactionBuilder(BerlinTransactionBuilder):
    legacy_signed: Type[LondonLegacyTransaction]
    legacy_unsigned: Type[LondonUnsignedLegacyTransaction]
    typed_transaction: Type[LondonTypedTransaction]

    @classmethod
    def new_unsigned_dynamic_fee_transaction(cls, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: CountableList[Tuple[Address, List[int]]]) -> UnsignedDynamicFeeTransaction: ...

    @classmethod
    def new_dynamic_fee_transaction(cls, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: CountableList[Tuple[Address, List[int]]], y_parity: int, r: int, s: int) -> LondonTypedTransaction: ...