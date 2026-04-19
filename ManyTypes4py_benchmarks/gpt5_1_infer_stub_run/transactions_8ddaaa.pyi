from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Type, Union
import rlp
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, Hash32
from eth.abc import ComputationAPI, ReceiptAPI, SignedTransactionAPI, TransactionDecoderAPI, UnsignedTransactionAPI
from eth.vm.forks.berlin.transactions import (
    AccessListPayloadDecoder,
    BerlinLegacyTransaction,
    BerlinTransactionBuilder,
    BerlinUnsignedLegacyTransaction,
    TypedTransaction,
)
from eth.rlp.transactions import SignedTransactionMethods
from .constants import DYNAMIC_FEE_TRANSACTION_TYPE
from .receipts import LondonReceiptBuilder

class LondonLegacyTransaction(BerlinLegacyTransaction):
    ...

class LondonUnsignedLegacyTransaction(BerlinUnsignedLegacyTransaction):
    def as_signed_transaction(
        self, private_key: PrivateKey, chain_id: Optional[int] = ...
    ) -> LondonLegacyTransaction: ...

class UnsignedDynamicFeeTransaction(rlp.Serializable, UnsignedTransactionAPI):
    _type_id: ClassVar[int]
    fields: ClassVar[Sequence[Tuple[str, Any]]]

    @property
    def _type_byte(self) -> bytes: ...

    def get_message_for_signing(self) -> bytes: ...

    def as_signed_transaction(
        self, private_key: PrivateKey, chain_id: Optional[int] = ...
    ) -> "LondonTypedTransaction": ...

    def validate(self) -> None: ...

    def gas_used_by(self, computation: ComputationAPI) -> int: ...

    def get_intrinsic_gas(self) -> int: ...

    @property
    def intrinsic_gas(self) -> int: ...

class DynamicFeeTransaction(rlp.Serializable, SignedTransactionMethods, SignedTransactionAPI):
    _type_id: ClassVar[int]
    fields: ClassVar[Sequence[Tuple[str, Any]]]

    @property
    def gas_price(self) -> int: ...

    @property
    def max_fee_per_blob_gas(self) -> int: ...

    @property
    def blob_versioned_hashes(self) -> Sequence[Hash32]: ...

    def get_sender(self) -> Address: ...

    def get_message_for_signing(self) -> bytes: ...

    def check_signature_validity(self) -> None: ...

    @property
    def _type_byte(self) -> bytes: ...

    @property
    def hash(self) -> Hash32: ...

    def get_intrinsic_gas(self) -> int: ...

    def encode(self) -> bytes: ...

    def make_receipt(
        self,
        status: Union[int, bytes],
        gas_used: int,
        log_entries: Sequence[Tuple[Address, Sequence[Hash32], bytes]],
    ) -> ReceiptAPI: ...

class DynamicFeePayloadDecoder(TransactionDecoderAPI):
    @classmethod
    def decode(cls, payload: bytes) -> DynamicFeeTransaction: ...

class LondonTypedTransaction(TypedTransaction):
    decoders: ClassVar[Dict[int, Type[TransactionDecoderAPI]]]
    receipt_builder: ClassVar[Type[LondonReceiptBuilder]]

class LondonTransactionBuilder(BerlinTransactionBuilder):
    legacy_signed: ClassVar[Type[LondonLegacyTransaction]]
    legacy_unsigned: ClassVar[Type[LondonUnsignedLegacyTransaction]]
    typed_transaction: ClassVar[Type[LondonTypedTransaction]]

    @classmethod
    def new_unsigned_dynamic_fee_transaction(
        cls,
        chain_id: int,
        nonce: int,
        max_priority_fee_per_gas: int,
        max_fee_per_gas: int,
        gas: int,
        to: bytes,
        value: int,
        data: bytes,
        access_list: Sequence[Tuple[Address, Sequence[Hash32]]],
    ) -> UnsignedDynamicFeeTransaction: ...

    @classmethod
    def new_dynamic_fee_transaction(
        cls,
        chain_id: int,
        nonce: int,
        max_priority_fee_per_gas: int,
        max_fee_per_gas: int,
        gas: int,
        to: bytes,
        value: int,
        data: bytes,
        access_list: Sequence[Tuple[Address, Sequence[Hash32]]],
        y_parity: int,
        r: int,
        s: int,
    ) -> LondonTypedTransaction: ...