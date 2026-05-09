from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
from cached_property import cached_property
from eth_hash.auto import keccak
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
from eth_utils import ValidationError
import rlp
from eth.abc import ComputationAPI, DecodedZeroOrOneLayerRLP, ReceiptAPI, SignedTransactionAPI, TransactionBuilderAPI, TransactionDecoderAPI, UnsignedTransactionAPI
from eth.rlp.logs import Log
from eth.rlp.receipts import Receipt
from eth.rlp.sedes import address
from eth.vm.forks.muir_glacier.transactions import MuirGlacierTransaction, MuirGlacierUnsignedTransaction
from .constants import ACCESS_LIST_ADDRESS_COST_EIP_2930, ACCESS_LIST_STORAGE_KEY_COST_EIP_2930, ACCESS_LIST_TRANSACTION_TYPE, VALID_TRANSACTION_TYPES
from .receipts import BerlinReceiptBuilder

class BerlinLegacyTransaction(MuirGlacierTransaction):
    ...

class BerlinUnsignedLegacyTransaction(MuirGlacierUnsignedTransaction):
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = None) -> BerlinLegacyTransaction:
        ...

class AccountAccesses(rlp.Serializable):
    account: Address
    storage_keys: List[int]
    ...

class UnsignedAccessListTransaction(rlp.Serializable, UnsignedTransactionAPI):
    chain_id: int
    nonce: int
    gas_price: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: List[AccountAccesses]
    ...

    def get_message_for_signing(self) -> bytes:
        ...

    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = None) -> TypedTransaction:
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
    chain_id: int
    nonce: int
    gas_price: int
    gas: int
    to: Address
    value: int
    data: bytes
    access_list: List[AccountAccesses]
    y_parity: int
    r: int
    s: int
    ...

    def get_sender(self) -> Address:
        ...

    def get_message_for_signing(self) -> bytes:
        ...

    def check_signature_validity(self) -> None:
        ...

    def encode(self) -> bytes:
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
    def __init__(self, type_id: int, proxy_target: SignedTransactionAPI) -> None:
        ...

    @classmethod
    def get_payload_codec(cls, type_id: int) -> TransactionDecoderAPI:
        ...

    def encode(self) -> bytes:
        ...

    @classmethod
    def decode(cls, encoded: bytes) -> SignedTransactionAPI:
        ...

    @classmethod
    def serialize(cls, obj: SignedTransactionAPI) -> bytes:
        ...

    @classmethod
    def deserialize(cls, encoded_unchecked: bytes) -> SignedTransactionAPI:
        ...

    @property
    def hash(self) -> Hash32:
        ...

    def get_intrinsic_gas(self) -> int:
        ...

    def copy(self, **overrides: Any) -> SignedTransactionAPI:
        ...

    def make_receipt(self, status: bytes, gas_used: int, log_entries: List[Tuple[Address, List[bytes], bytes]]) -> ReceiptAPI:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __hash__(self) -> int:
        ...

class BerlinTransactionBuilder(TransactionBuilderAPI):
    @classmethod
    def decode(cls, encoded: bytes) -> SignedTransactionAPI:
        ...

    @classmethod
    def deserialize(cls, encoded: bytes) -> SignedTransactionAPI:
        ...

    @classmethod
    def serialize(cls, obj: SignedTransactionAPI) -> bytes:
        ...

    @classmethod
    def create_unsigned_transaction(cls, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes) -> BerlinUnsignedLegacyTransaction:
        ...

    @classmethod
    def new_transaction(cls, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes, v: int, r: int, s: int) -> BerlinLegacyTransaction:
        ...

    @classmethod
    def new_unsigned_access_list_transaction(cls, chain_id: int, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes, access_list: List[AccountAccesses]) -> UnsignedAccessListTransaction:
        ...

    @classmethod
    def new_access_list_transaction(cls, chain_id: int, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes, access_list: List[AccountAccesses], y_parity: int, r: int, s: int) -> TypedTransaction:
        ...

def _calculate_txn_intrinsic_gas_berlin(klass: Type[SignedTransactionAPI]) -> int:
    ...