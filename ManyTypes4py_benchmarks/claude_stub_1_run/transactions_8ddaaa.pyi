```pyi
from typing import Any, Dict, List, Sequence, Tuple, Type
from cached_property import cached_property
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
from eth_utils import to_bytes
import rlp
from rlp.sedes import CountableList, big_endian_int, binary
from eth._utils.transactions import create_transaction_signature, extract_transaction_sender, validate_transaction_signature
from eth.abc import ComputationAPI, ReceiptAPI, SignedTransactionAPI, TransactionDecoderAPI, UnsignedTransactionAPI
from eth.constants import CREATE_CONTRACT_ADDRESS
from eth.rlp.logs import Log
from eth.rlp.receipts import Receipt
from eth.rlp.sedes import address
from eth.rlp.transactions import SignedTransactionMethods
from eth.validation import validate_canonical_address, validate_is_bytes, validate_is_transaction_access_list, validate_uint64, validate_uint256
from eth.vm.forks.berlin.constants import ACCESS_LIST_TRANSACTION_TYPE
from eth.vm.forks.berlin.transactions import AccessListPayloadDecoder, AccountAccesses, BerlinLegacyTransaction, BerlinTransactionBuilder, BerlinUnsignedLegacyTransaction, TypedTransaction, _calculate_txn_intrinsic_gas_berlin
from .constants import DYNAMIC_FEE_TRANSACTION_TYPE
from .receipts import LondonReceiptBuilder

class LondonLegacyTransaction(BerlinLegacyTransaction): ...

class LondonUnsignedLegacyTransaction(BerlinUnsignedLegacyTransaction):
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Any = ...) -> LondonLegacyTransaction: ...

class UnsignedDynamicFeeTransaction(rlp.Serializable, UnsignedTransactionAPI):
    _type_id: int
    fields: Sequence[Tuple[str, Any]]
    chain_id: int
    nonce: int
    max_priority_fee_per_gas: int
    max_fee_per_gas: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: Sequence[AccountAccesses]
    
    @cached_property
    def _type_byte(self) -> bytes: ...
    
    def get_message_for_signing(self) -> bytes: ...
    
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Any = ...) -> LondonTypedTransaction: ...
    
    def validate(self) -> None: ...
    
    def gas_used_by(self, computation: ComputationAPI) -> int: ...
    
    def get_intrinsic_gas(self) -> int: ...
    
    @property
    def intrinsic_gas(self) -> int: ...

class DynamicFeeTransaction(rlp.Serializable, SignedTransactionMethods, SignedTransactionAPI):
    _type_id: int
    fields: Sequence[Tuple[str, Any]]
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
    def hash(self) -> Hash32: ...
    
    def get_intrinsic_gas(self) -> int: ...
    
    def encode(self) -> bytes: ...
    
    def make_receipt(self, status: Any, gas_used: int, log_entries: Sequence[Tuple[Address, Sequence[int], bytes]]) -> Receipt: ...

class DynamicFeePayloadDecoder(TransactionDecoderAPI):
    @classmethod
    def decode(cls, payload: bytes) -> DynamicFeeTransaction: ...

class LondonTypedTransaction(TypedTransaction):
    decoders: Dict[int, Type[TransactionDecoderAPI]]
    receipt_builder: Type[LondonReceiptBuilder]

class LondonTransactionBuilder(BerlinTransactionBuilder):
    legacy_signed: Type[LondonLegacyTransaction]
    legacy_unsigned: Type[LondonUnsignedLegacyTransaction]
    typed_transaction: Type[LondonTypedTransaction]
    
    @classmethod
    def new_unsigned_dynamic_fee_transaction(cls, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: Sequence[AccountAccesses]) -> UnsignedDynamicFeeTransaction: ...
    
    @classmethod
    def new_dynamic_fee_transaction(cls, chain_id: int, nonce: int, max_priority_fee_per_gas: int, max_fee_per_gas: int, gas: int, to: Address, value: int, data: bytes, access_list: Sequence[AccountAccesses], y_parity: int, r: int, s: int) -> LondonTypedTransaction: ...
```