from typing import Dict, Sequence, Tuple, Optional, Any, Type, Union
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
from eth.abc import ComputationAPI, ReceiptAPI, SignedTransactionAPI, TransactionDecoderAPI, UnsignedTransactionAPI
from eth.rlp.logs import Log
from eth.rlp.receipts import Receipt
from eth.vm.forks.berlin.constants import ACCESS_LIST_TRANSACTION_TYPE
from eth.vm.forks.berlin.transactions import AccessListPayloadDecoder, AccountAccesses, BerlinLegacyTransaction, BerlinTransactionBuilder, BerlinUnsignedLegacyTransaction, TypedTransaction
from .constants import DYNAMIC_FEE_TRANSACTION_TYPE
from .receipts import LondonReceiptBuilder

class LondonLegacyTransaction(BerlinLegacyTransaction): ...

class LondonUnsignedLegacyTransaction(BerlinUnsignedLegacyTransaction):
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = ...) -> LondonLegacyTransaction: ...

class UnsignedDynamicFeeTransaction(UnsignedTransactionAPI):
    _type_id: int
    fields: list[Tuple[str, Any]]
    
    @property
    def _type_byte(self) -> bytes: ...
    
    def get_message_for_signing(self) -> bytes: ...
    
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = ...) -> 'LondonTypedTransaction': ...
    
    def validate(self) -> None: ...
    
    def gas_used_by(self, computation: ComputationAPI) -> int: ...
    
    def get_intrinsic_gas(self) -> int: ...
    
    @property
    def intrinsic_gas(self) -> int: ...

class DynamicFeeTransaction(SignedTransactionAPI):
    _type_id: int
    fields: list[Tuple[str, Any]]

    @property
    def gas_price(self) -> Any: ...
    
    @property
    def max_fee_per_blob_gas(self) -> Any: ...
    
    @property
    def blob_versioned_hashes(self) -> Any: ...
    
    def get_sender(self) -> Address: ...
    
    def get_message_for_signing(self) -> bytes: ...
    
    def check_signature_validity(self) -> None: ...
    
    @property
    def _type_byte(self) -> bytes: ...
    
    @property
    def hash(self) -> Any: ...
    
    def get_intrinsic_gas(self) -> int: ...
    
    def encode(self) -> bytes: ...
    
    def make_receipt(self, status: Any, gas_used: int, log_entries: Sequence[Tuple[Address, Sequence[Hash32], bytes]]) -> Receipt: ...

class DynamicFeePayloadDecoder(TransactionDecoderAPI):
    @classmethod
    def decode(cls, payload: bytes) -> DynamicFeeTransaction: ...

class LondonTypedTransaction(TypedTransaction):
    decoders: Dict[int, Type[TransactionDecoderAPI]]
    receipt_builder: Type[ReceiptAPI]

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
        access_list: Sequence[AccountAccesses]
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
        access_list: Sequence[AccountAccesses], 
        y_parity: int, 
        r: int, 
        s: int
    ) -> LondonTypedTransaction: ...