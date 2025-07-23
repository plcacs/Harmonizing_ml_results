from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, ContextManager, Dict, FrozenSet, Hashable,
    Iterable, Iterator, List, MutableMapping, NamedTuple, Optional, Sequence, Tuple,
    Type, TypeVar, Union, Generic, Set, cast
)
from eth_bloom import BloomFilter
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, BlockNumber, Hash32
from eth_utils import ExtendedDebugLogger
from eth.constants import BLANK_ROOT_HASH
from eth.exceptions import VMError
from eth.typing import AccountState, BytesOrView, ChainGaps, HeaderParams, JournalDBCheckpoint, VMConfiguration

if TYPE_CHECKING:
    from eth.vm.forks.cancun.transactions import BlobTransaction

T = TypeVar('T')
DecodedZeroOrOneLayerRLP = Union[bytes, List[bytes]]

class MiningHeaderAPI(ABC):
    @property
    @abstractmethod
    def hash(self) -> Hash32: ...
    
    @property
    @abstractmethod
    def mining_hash(self) -> Hash32: ...
    
    @property
    @abstractmethod
    def hex_hash(self) -> str: ...
    
    @property
    @abstractmethod
    def is_genesis(self) -> bool: ...
    
    @abstractmethod
    def build_changeset(self, *args: Any, **kwargs: Any) -> Any: ...
    
    @abstractmethod
    def as_dict(self) -> Dict[str, Any]: ...
    
    @property
    @abstractmethod
    def base_fee_per_gas(self) -> Optional[int]: ...
    
    @property
    @abstractmethod
    def withdrawals_root(self) -> Optional[Hash32]: ...

class BlockHeaderSedesAPI(ABC):
    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: bytes) -> 'BlockHeaderAPI': ...
    
    @classmethod
    @abstractmethod
    def serialize(cls, obj: 'BlockHeaderAPI') -> bytes: ...

class BlockHeaderAPI(MiningHeaderAPI, BlockHeaderSedesAPI):
    @abstractmethod
    def copy(self, *args: Any, **kwargs: Any) -> 'BlockHeaderAPI': ...
    
    @property
    @abstractmethod
    def parent_beacon_block_root(self) -> Optional[Hash32]: ...
    
    @property
    @abstractmethod
    def blob_gas_used(self) -> Optional[int]: ...
    
    @property
    @abstractmethod
    def excess_blob_gas(self) -> Optional[int]: ...

class LogAPI(ABC):
    @property
    @abstractmethod
    def bloomables(self) -> Sequence[Hashable]: ...

class ReceiptAPI(ABC):
    @property
    @abstractmethod
    def state_root(self) -> bytes: ...
    
    @property
    @abstractmethod
    def gas_used(self) -> int: ...
    
    @property
    @abstractmethod
    def bloom(self) -> int: ...
    
    @property
    @abstractmethod
    def logs(self) -> Sequence[LogAPI]: ...
    
    @property
    @abstractmethod
    def bloom_filter(self) -> BloomFilter: ...
    
    def copy(self, *args: Any, **kwargs: Any) -> 'ReceiptAPI': ...
    
    @abstractmethod
    def encode(self) -> bytes: ...

class ReceiptDecoderAPI(ABC):
    @classmethod
    @abstractmethod
    def decode(cls, encoded: bytes) -> ReceiptAPI: ...

class ReceiptBuilderAPI(ReceiptDecoderAPI):
    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: bytes) -> ReceiptAPI: ...
    
    @classmethod
    @abstractmethod
    def serialize(cls, obj: ReceiptAPI) -> bytes: ...

class BaseTransactionAPI(ABC):
    @abstractmethod
    def validate(self) -> None: ...
    
    @property
    @abstractmethod
    def intrinsic_gas(self) -> int: ...
    
    @abstractmethod
    def get_intrinsic_gas(self) -> int: ...
    
    @abstractmethod
    def gas_used_by(self, computation: 'ComputationAPI') -> int: ...
    
    @abstractmethod
    def copy(self, **overrides: Any) -> 'BaseTransactionAPI': ...
    
    @property
    @abstractmethod
    def access_list(self) -> Sequence[Tuple[Address, Sequence[int]]]: ...

class TransactionFieldsAPI(ABC):
    @property
    @abstractmethod
    def nonce(self) -> int: ...
    
    @property
    @abstractmethod
    def gas_price(self) -> int: ...
    
    @property
    @abstractmethod
    def max_fee_per_gas(self) -> int: ...
    
    @property
    @abstractmethod
    def max_priority_fee_per_gas(self) -> int: ...
    
    @property
    @abstractmethod
    def gas(self) -> int: ...
    
    @property
    @abstractmethod
    def to(self) -> Address: ...
    
    @property
    @abstractmethod
    def value(self) -> int: ...
    
    @property
    @abstractmethod
    def data(self) -> bytes: ...
    
    @property
    @abstractmethod
    def r(self) -> int: ...
    
    @property
    @abstractmethod
    def s(self) -> int: ...
    
    @property
    @abstractmethod
    def hash(self) -> Hash32: ...
    
    @property
    @abstractmethod
    def chain_id(self) -> Optional[int]: ...
    
    @property
    @abstractmethod
    def max_fee_per_blob_gas(self) -> Optional[int]: ...
    
    @property
    @abstractmethod
    def blob_versioned_hashes(self) -> Optional[Sequence[Hash32]]: ...

class LegacyTransactionFieldsAPI(TransactionFieldsAPI):
    @property
    @abstractmethod
    def v(self) -> int: ...

class UnsignedTransactionAPI(BaseTransactionAPI):
    @abstractmethod
    def as_signed_transaction(self, private_key: PrivateKey, chain_id: Optional[int] = None) -> 'SignedTransactionAPI': ...

class TransactionDecoderAPI(ABC):
    @classmethod
    @abstractmethod
    def decode(cls, encoded: bytes) -> 'SignedTransactionAPI': ...

class TransactionBuilderAPI(TransactionDecoderAPI):
    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: bytes) -> 'SignedTransactionAPI': ...
    
    @classmethod
    @abstractmethod
    def serialize(cls, obj: 'SignedTransactionAPI') -> bytes: ...
    
    @classmethod
    @abstractmethod
    def create_unsigned_transaction(cls, *, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes) -> UnsignedTransactionAPI: ...
    
    @classmethod
    @abstractmethod
    def new_transaction(cls, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes, v: int, r: int, s: int) -> 'SignedTransactionAPI': ...

class SignedTransactionAPI(BaseTransactionAPI, TransactionFieldsAPI):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    
    @property
    @abstractmethod
    def sender(self) -> Address: ...
    
    @property
    @abstractmethod
    def y_parity(self) -> int: ...
    
    @abstractmethod
    def validate(self) -> None: ...
    
    @property
    @abstractmethod
    def is_signature_valid(self) -> bool: ...
    
    @abstractmethod
    def check_signature_validity(self) -> None: ...
    
    @abstractmethod
    def get_sender(self) -> Address: ...
    
    @abstractmethod
    def get_message_for_signing(self) -> bytes: ...
    
    def as_dict(self) -> Dict[str, Any]: ...
    
    @abstractmethod
    def make_receipt(self, status: bytes, gas_used: int, log_entries: Sequence[LogAPI]) -> ReceiptAPI: ...
    
    @abstractmethod
    def encode(self) -> bytes: ...

class WithdrawalAPI(ABC):
    @property
    @abstractmethod
    def index(self) -> int: ...
    
    @property
    @abstractmethod
    def validator_index(self) -> int: ...
    
    @property
    @abstractmethod
    def address(self) -> Address: ...
    
    @property
    @abstractmethod
    def amount(self) -> int: ...
    
    @property
    @abstractmethod
    def hash(self) -> Hash32: ...
    
    @abstractmethod
    def validate(self) -> None: ...
    
    @abstractmethod
    def encode(self) -> bytes: ...

class BlockAPI(ABC):
    transaction_builder: ClassVar[Optional[Type[TransactionBuilderAPI]] = None
    receipt_builder: ClassVar[Optional[Type[ReceiptBuilderAPI]]] = None
    
    @abstractmethod
    def __init__(self, header: BlockHeaderAPI, transactions: Sequence[SignedTransactionAPI], uncles: Sequence[BlockHeaderAPI], withdrawals: Optional[Sequence[WithdrawalAPI]] = None) -> None: ...
    
    @classmethod
    @abstractmethod
    def get_transaction_builder(cls) -> Type[TransactionBuilderAPI]: ...
    
    @classmethod
    @abstractmethod
    def get_receipt_builder(cls) -> Type[ReceiptBuilderAPI]: ...
    
    @classmethod
    @abstractmethod
    def from_header(cls, header: BlockHeaderAPI, chaindb: 'ChainDatabaseAPI') -> 'BlockAPI': ...
    
    @abstractmethod
    def get_receipts(self, chaindb: 'ChainDatabaseAPI') -> Sequence[ReceiptAPI]: ...
    
    @property
    @abstractmethod
    def hash(self) -> Hash32: ...
    
    @property
    @abstractmethod
    def number(self) -> BlockNumber: ...
    
    @property
    @abstractmethod
    def is_genesis(self) -> bool: ...
    
    def copy(self, *args: Any, **kwargs: Any) -> 'BlockAPI': ...

class MetaWitnessAPI(ABC):
    @property
    @abstractmethod
    def hashes(self) -> Sequence[Hash32]: ...
    
    @property
    @abstractmethod
    def accounts_queried(self) -> Sequence[Address]: ...
    
    @property
    @abstractmethod
    def account_bytecodes_queried(self) -> Sequence[Address]: ...
    
    @abstractmethod
    def get_slots_queried(self, address: Address) -> Sequence[int]: ...
    
    @property
    @abstractmethod
    def total_slots_queried(self) -> int: ...

class BlockAndMetaWitness(NamedTuple):
    block: BlockAPI
    meta_witness: MetaWitnessAPI

class BlockPersistResult(NamedTuple):
    block: BlockAPI
    receipts: Sequence[ReceiptAPI]
    computations: Sequence['ComputationAPI']

class BlockImportResult(NamedTuple):
    imported_block: BlockAPI
    new_canonical_blocks: Sequence[BlockAPI]
    old_canonical_blocks: Sequence[BlockAPI]

class SchemaAPI(ABC):
    @staticmethod
    @abstractmethod
    def make_header_chain_gaps_lookup_key() -> bytes: ...
    
    @staticmethod
    @abstractmethod
    def make_canonical_head_hash_lookup_key() -> bytes: ...
    
    @staticmethod
    @abstractmethod
    def make_block_number_to_hash_lookup_key(block_number: BlockNumber) -> bytes: ...
    
    @staticmethod
    @abstractmethod
    def make_block_hash_to_score_lookup_key(block_hash: Hash32) -> bytes: ...
    
    @staticmethod
    @abstractmethod
    def make_transaction_hash_to_block_lookup_key(transaction_hash: Hash32) -> bytes: ...
    
    @staticmethod
    @abstractmethod
    def make_withdrawal_hash_to_block_lookup_key(withdrawal_hash: Hash32) -> bytes: ...

class DatabaseAPI(MutableMapping[bytes, bytes], ABC):
    @abstractmethod
    def set(self, key: bytes, value: bytes) -> None: ...
    
    @abstractmethod
    def exists(self, key: bytes) -> bool: ...
    
    @abstractmethod
    def delete(self, key: bytes) -> None: ...

class AtomicWriteBatchAPI(DatabaseAPI): ...

class AtomicDatabaseAPI(DatabaseAPI):
    @abstractmethod
    def atomic_batch(self) -> ContextManager[AtomicWriteBatchAPI]: ...

class HeaderDatabaseAPI(ABC):
    @abstractmethod
    def __init__(self, db: AtomicDatabaseAPI) -> None: ...
    
    @abstractmethod
    def get_header_chain_gaps(self) -> Tuple[Sequence[Tuple[BlockNumber, BlockNumber]], BlockNumber]: ...
    
    @abstractmethod
    def get_canonical_block_hash(self, block_number: BlockNumber) -> Hash32: ...
    
    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number: BlockNumber) -> BlockHeaderAPI: ...
    
    @abstractmethod
    def get_canonical_head(self) -> BlockHeaderAPI: ...
    
    @abstractmethod
    def get_block_header_by_hash(self, block_hash: Hash32) -> BlockHeaderAPI: ...
    
    @abstractmethod
    def get_score(self, block_hash: Hash32) -> int: ...
    
    @abstractmethod
    def header_exists(self, block_hash: Hash32) -> bool: ...
    
    @abstractmethod
    def persist_checkpoint_header(self, header: BlockHeaderAPI, score: int) -> None: ...
    
    @abstractmethod
    def persist_header(self, header: BlockHeaderAPI) -> Tuple[Sequence[BlockHeaderAPI], Sequence[BlockHeaderAPI]]: ...
    
    @abstractmethod
    def persist_header_chain(self, headers: Sequence[BlockHeaderAPI], genesis_parent_hash: Optional[Hash32] = None) -> Tuple[Sequence[BlockHeaderAPI], Sequence[BlockHeaderAPI]]: ...

class ChainDatabaseAPI(HeaderDatabaseAPI):
    @abstractmethod
    def get_block_uncles(self, uncles_hash: Hash32) -> Sequence[BlockHeaderAPI]: ...
    
    @abstractmethod
    def persist_block(self, block: BlockAPI, genesis_parent_hash: Optional[Hash32] = None) -> None: ...
    
    @abstractmethod
    def persist_unexecuted_block(self, block: BlockAPI, receipts: Sequence[ReceiptAPI], genesis_parent_hash: Optional[Hash32] = None) -> None: ...
    
    @abstractmethod
    def persist_uncles(self, uncles: Sequence[BlockHeaderAPI]) -> Hash32: ...
    
    @abstractmethod
    def add_receipt(self, block_header: BlockHeaderAPI, index_key: bytes, receipt: ReceiptAPI) -> Hash32: ...
    
    @abstractmethod
    def add_transaction(self, block_header: BlockHeaderAPI, index_key: bytes, transaction: SignedTransactionAPI) -> Hash32: ...
    
    @abstractmethod
    def get_block_transactions(self, block_header: BlockHeaderAPI, transaction_decoder: TransactionDecoderAPI) -> Sequence[SignedTransactionAPI]: ...
    
    @abstractmethod
    def get_block_transaction_hashes(self, block_header: BlockHeaderAPI) -> Sequence[Hash32]: ...
    
    @abstractmethod
    def get_receipt_by_index(self, block_number: BlockNumber, receipt_index: int, receipt_decoder: ReceiptDecoderAPI) -> ReceiptAPI: ...
    
    @abstractmethod
    def get_receipts(self, header: BlockHeaderAPI, receipt_decoder: ReceiptDecoderAPI) -> Sequence[ReceiptAPI]: ...
    
    @abstractmethod
    def get_transaction_by_index(self, block_number: BlockNumber, transaction_index: int, transaction_decoder: TransactionDecoderAPI) -> SignedTransactionAPI: ...
    
    @abstractmethod
    def get_transaction_index(self, transaction_hash: Hash32) -> Tuple[BlockNumber, int]: ...
    
    @abstractmethod
    def get_block_withdrawals(self, block_header: BlockHeaderAPI) -> Sequence[WithdrawalAPI]: ...
    
    @abstractmethod
    def exists(self, key: bytes) -> bool: ...
    
    @abstractmethod
    def get(self, key: bytes) -> bytes: ...
    
    @abstractmethod
    def persist_trie_data_dict(self, trie_data_dict: Dict[bytes, bytes]) -> None: ...

class GasMeterAPI(ABC):
    @abstractmethod
    def consume_gas(self, amount: int, reason: str) -> None: ...
    
    @abstractmethod
    def return_gas(self, amount: int) -> None: ...
    
    @abstractmethod
    def refund_gas(self, amount: int) -> None: ...

class MessageAPI(ABC):
    __slots__ = ['code', '_code_address', 'create_address', 'data', 'depth', 'gas', 'is_static', 'sender', 'should_transfer_value', '_storage_addressto', 'value']
    
    @property
    @abstractmethod
    def code_address(self) -> Address: ...
    
    @property
    @abstractmethod
    def storage_address(self) -> Address: ...
    
    @property
    @abstractmethod
    def is_create(self) -> bool: ...
    
    @property
    @abstractmethod
    def data_as_bytes(self) -> bytes: ...

class OpcodeAPI(ABC):
    @abstractmethod
    def __call__(self, computation: 'ComputationAPI') -> None: ...
    
    @classmethod
    @abstractmethod
    def as_opcode(cls, logic_fn: Callable[['ComputationAPI'], None], mnemonic: str, gas_cost: int) -> Type['OpcodeAPI']: ...

class ChainContextAPI(ABC):
    @abstractmethod
    def __init__(self, chain_id: int) -> None: ...
    
    @property
    @abstractmethod
    def chain_id(self) -> int: ...

class TransactionContextAPI(ABC):
    @abstractmethod
    def __init__(self, gas_price: int, origin: Address) -> None: ...
    
    @abstractmethod
    def get_next_log_counter(self) -> int: ...
    
    @property
    @abstractmethod
    def gas_price(self) -> int: ...
    
    @property
    @abstractmethod
    def origin(self) -> Address: ...
    
    @property
    @abstractmethod
    def blob_versioned_hashes(self) -> Optional[Sequence[Hash32