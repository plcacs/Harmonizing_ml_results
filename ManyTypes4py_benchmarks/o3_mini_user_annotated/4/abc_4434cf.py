from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, ContextManager, Dict, FrozenSet, Hashable, Iterable, Iterator, List, MutableMapping, NamedTuple, Optional, Sequence, Tuple, Type, TypeVar, Union

T = TypeVar("T")


DecodedZeroOrOneLayerRLP = Union[bytes, List[bytes]]


class MiningHeaderAPI(ABC):
    parent_hash: bytes
    uncles_hash: bytes
    coinbase: bytes
    state_root: bytes
    transaction_root: bytes
    receipt_root: bytes
    bloom: int
    difficulty: int
    block_number: int
    gas_limit: int
    gas_used: int
    timestamp: int
    extra_data: bytes

    @property
    @abstractmethod
    def hash(self) -> bytes:
        ...

    @property
    @abstractmethod
    def mining_hash(self) -> bytes:
        ...

    @property
    @abstractmethod
    def hex_hash(self) -> str:
        ...

    @property
    @abstractmethod
    def is_genesis(self) -> bool:
        ...

    @abstractmethod
    def build_changeset(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def as_dict(self) -> Dict[Hashable, Any]:
        ...

    @property
    @abstractmethod
    def base_fee_per_gas(self) -> Optional[int]:
        ...

    @property
    @abstractmethod
    def withdrawals_root(self) -> Optional[bytes]:
        ...


class BlockHeaderSedesAPI(ABC):
    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: List[bytes]) -> "BlockHeaderAPI":
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj: "BlockHeaderAPI") -> List[bytes]:
        ...


class BlockHeaderAPI(MiningHeaderAPI, BlockHeaderSedesAPI):
    mix_hash: bytes
    nonce: bytes

    @abstractmethod
    def copy(self, *args: Any, **kwargs: Any) -> "BlockHeaderAPI":
        ...

    @property
    @abstractmethod
    def parent_beacon_block_root(self) -> Optional[bytes]:
        ...

    @property
    @abstractmethod
    def blob_gas_used(self) -> int:
        ...

    @property
    @abstractmethod
    def excess_blob_gas(self) -> int:
        ...


class LogAPI(ABC):
    address: bytes
    topics: Sequence[int]
    data: bytes

    @property
    @abstractmethod
    def bloomables(self) -> Tuple[bytes, ...]:
        ...


class ReceiptAPI(ABC):
    @property
    @abstractmethod
    def state_root(self) -> bytes:
        ...

    @property
    @abstractmethod
    def gas_used(self) -> int:
        ...

    @property
    @abstractmethod
    def bloom(self) -> int:
        ...

    @property
    @abstractmethod
    def logs(self) -> Sequence[LogAPI]:
        ...

    @property
    @abstractmethod
    def bloom_filter(self) -> Any:
        ...

    def copy(self, *args: Any, **kwargs: Any) -> "ReceiptAPI":
        ...

    @abstractmethod
    def encode(self) -> bytes:
        ...


class ReceiptDecoderAPI(ABC):
    @classmethod
    @abstractmethod
    def decode(cls, encoded: bytes) -> ReceiptAPI:
        ...


class ReceiptBuilderAPI(ReceiptDecoderAPI):
    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: DecodedZeroOrOneLayerRLP) -> "ReceiptAPI":
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj: "ReceiptAPI") -> DecodedZeroOrOneLayerRLP:
        ...


class BaseTransactionAPI(ABC):
    @abstractmethod
    def validate(self) -> None:
        ...

    @property
    @abstractmethod
    def intrinsic_gas(self) -> int:
        ...

    @abstractmethod
    def get_intrinsic_gas(self) -> int:
        ...

    @abstractmethod
    def gas_used_by(self, computation: "ComputationAPI") -> int:
        ...

    @abstractmethod
    def copy(self: T, **overrides: Any) -> T:
        ...

    @property
    @abstractmethod
    def access_list(self) -> Sequence[Tuple[bytes, Sequence[int]]]:
        ...


class TransactionFieldsAPI(ABC):
    @property
    @abstractmethod
    def nonce(self) -> int:
        ...

    @property
    @abstractmethod
    def gas_price(self) -> int:
        ...

    @property
    @abstractmethod
    def max_fee_per_gas(self) -> int:
        ...

    @property
    @abstractmethod
    def max_priority_fee_per_gas(self) -> int:
        ...

    @property
    @abstractmethod
    def gas(self) -> int:
        ...

    @property
    @abstractmethod
    def to(self) -> bytes:
        ...

    @property
    @abstractmethod
    def value(self) -> int:
        ...

    @property
    @abstractmethod
    def data(self) -> bytes:
        ...

    @property
    @abstractmethod
    def r(self) -> int:
        ...

    @property
    @abstractmethod
    def s(self) -> int:
        ...

    @property
    @abstractmethod
    def hash(self) -> bytes:
        ...

    @property
    @abstractmethod
    def chain_id(self) -> Optional[int]:
        ...

    @property
    @abstractmethod
    def max_fee_per_blob_gas(self) -> int:
        ...

    @property
    @abstractmethod
    def blob_versioned_hashes(self) -> Sequence[bytes]:
        ...


class LegacyTransactionFieldsAPI(TransactionFieldsAPI):
    @property
    @abstractmethod
    def v(self) -> int:
        ...


class UnsignedTransactionAPI(BaseTransactionAPI):
    nonce: int
    gas_price: int
    gas: int
    to: bytes
    value: int
    data: bytes

    @abstractmethod
    def as_signed_transaction(self, private_key: bytes, chain_id: Optional[int] = None) -> "SignedTransactionAPI":
        ...


class TransactionDecoderAPI(ABC):
    @classmethod
    @abstractmethod
    def decode(cls, encoded: bytes) -> "SignedTransactionAPI":
        ...


class TransactionBuilderAPI(TransactionDecoderAPI):
    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: DecodedZeroOrOneLayerRLP) -> "SignedTransactionAPI":
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj: "SignedTransactionAPI") -> DecodedZeroOrOneLayerRLP:
        ...

    @classmethod
    @abstractmethod
    def create_unsigned_transaction(
        cls,
        *,
        nonce: int,
        gas_price: int,
        gas: int,
        to: bytes,
        value: int,
        data: bytes,
    ) -> UnsignedTransactionAPI:
        ...

    @classmethod
    @abstractmethod
    def new_transaction(
        cls,
        nonce: int,
        gas_price: int,
        gas: int,
        to: bytes,
        value: int,
        data: bytes,
        v: int,
        r: int,
        s: int,
    ) -> "SignedTransactionAPI":
        ...


class SignedTransactionAPI(BaseTransactionAPI, TransactionFieldsAPI):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @property
    @abstractmethod
    def sender(self) -> bytes:
        ...

    @property
    @abstractmethod
    def y_parity(self) -> int:
        ...

    type_id: Optional[int]

    @abstractmethod
    def validate(self) -> None:
        ...

    @property
    @abstractmethod
    def is_signature_valid(self) -> bool:
        ...

    @abstractmethod
    def check_signature_validity(self) -> None:
        ...

    @abstractmethod
    def get_sender(self) -> bytes:
        ...

    @abstractmethod
    def get_message_for_signing(self) -> bytes:
        ...

    def as_dict(self) -> Dict[Hashable, Any]:
        ...

    @abstractmethod
    def make_receipt(
        self,
        status: bytes,
        gas_used: int,
        log_entries: Tuple[Tuple[bytes, Tuple[int, ...], bytes], ...],
    ) -> ReceiptAPI:
        ...

    @abstractmethod
    def encode(self) -> bytes:
        ...


class WithdrawalAPI(ABC):
    @property
    @abstractmethod
    def index(self) -> int:
        ...

    @property
    @abstractmethod
    def validator_index(self) -> int:
        ...

    @property
    @abstractmethod
    def address(self) -> bytes:
        ...

    @property
    @abstractmethod
    def amount(self) -> int:
        ...

    @property
    @abstractmethod
    def hash(self) -> bytes:
        ...

    @abstractmethod
    def validate(self) -> None:
        ...

    @abstractmethod
    def encode(self) -> bytes:
        ...


class BlockAPI(ABC):
    header: BlockHeaderAPI
    transactions: Tuple[SignedTransactionAPI, ...]
    uncles: Tuple[BlockHeaderAPI, ...]
    withdrawals: Tuple[WithdrawalAPI, ...]

    transaction_builder: Type[TransactionBuilderAPI] = None
    receipt_builder: Type[ReceiptBuilderAPI] = None

    @abstractmethod
    def __init__(
        self,
        header: BlockHeaderAPI,
        transactions: Sequence[SignedTransactionAPI],
        uncles: Sequence[BlockHeaderAPI],
        withdrawals: Optional[Sequence[WithdrawalAPI]] = None,
    ) -> None:
        ...

    @classmethod
    @abstractmethod
    def get_transaction_builder(cls) -> Type[TransactionBuilderAPI]:
        ...

    @classmethod
    @abstractmethod
    def get_receipt_builder(cls) -> Type[ReceiptBuilderAPI]:
        ...

    @classmethod
    @abstractmethod
    def from_header(cls, header: BlockHeaderAPI, chaindb: "ChainDatabaseAPI") -> "BlockAPI":
        ...

    @abstractmethod
    def get_receipts(self, chaindb: "ChainDatabaseAPI") -> Tuple[ReceiptAPI, ...]:
        ...

    @property
    @abstractmethod
    def hash(self) -> bytes:
        ...

    @property
    @abstractmethod
    def number(self) -> int:
        ...

    @property
    @abstractmethod
    def is_genesis(self) -> bool:
        ...

    def copy(self, *args: Any, **kwargs: Any) -> "BlockAPI":
        ...


class MetaWitnessAPI(ABC):
    @property
    @abstractmethod
    def hashes(self) -> FrozenSet[bytes]:
        ...

    @property
    @abstractmethod
    def accounts_queried(self) -> FrozenSet[bytes]:
        ...

    @property
    @abstractmethod
    def account_bytecodes_queried(self) -> FrozenSet[bytes]:
        ...

    @abstractmethod
    def get_slots_queried(self, address: bytes) -> FrozenSet[int]:
        ...

    @property
    @abstractmethod
    def total_slots_queried(self) -> int:
        ...


class BlockAndMetaWitness(NamedTuple):
    block: BlockAPI
    meta_witness: MetaWitnessAPI


class BlockPersistResult(NamedTuple):
    imported_block: BlockAPI
    new_canonical_blocks: Tuple[BlockAPI, ...]
    old_canonical_blocks: Tuple[BlockAPI, ...]


class BlockImportResult(NamedTuple):
    imported_block: BlockAPI
    new_canonical_blocks: Tuple[BlockAPI, ...]
    old_canonical_blocks: Tuple[BlockAPI, ...]
    meta_witness: MetaWitnessAPI


class SchemaAPI(ABC):
    @staticmethod
    @abstractmethod
    def make_header_chain_gaps_lookup_key() -> bytes:
        ...

    @staticmethod
    @abstractmethod
    def make_canonical_head_hash_lookup_key() -> bytes:
        ...

    @staticmethod
    @abstractmethod
    def make_block_number_to_hash_lookup_key(block_number: int) -> bytes:
        ...

    @staticmethod
    @abstractmethod
    def make_block_hash_to_score_lookup_key(block_hash: bytes) -> bytes:
        ...

    @staticmethod
    @abstractmethod
    def make_transaction_hash_to_block_lookup_key(transaction_hash: bytes) -> bytes:
        ...

    @staticmethod
    @abstractmethod
    def make_withdrawal_hash_to_block_lookup_key(withdrawal_hash: bytes) -> bytes:
        ...


class DatabaseAPI(MutableMapping[bytes, bytes], ABC):
    @abstractmethod
    def set(self, key: bytes, value: bytes) -> None:
        ...

    @abstractmethod
    def exists(self, key: bytes) -> bool:
        ...

    @abstractmethod
    def delete(self, key: bytes) -> None:
        ...


class AtomicWriteBatchAPI(DatabaseAPI):
    pass


class AtomicDatabaseAPI(DatabaseAPI):
    @abstractmethod
    def atomic_batch(self) -> ContextManager[AtomicWriteBatchAPI]:
        ...


class HeaderDatabaseAPI(ABC):
    db: AtomicDatabaseAPI

    @abstractmethod
    def __init__(self, db: AtomicDatabaseAPI) -> None:
        ...

    @abstractmethod
    def get_header_chain_gaps(self) -> Any:
        ...

    @abstractmethod
    def get_canonical_block_hash(self, block_number: int) -> bytes:
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number: int) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_canonical_head(self) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash: bytes) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_score(self, block_hash: bytes) -> int:
        ...

    @abstractmethod
    def header_exists(self, block_hash: bytes) -> bool:
        ...

    @abstractmethod
    def persist_checkpoint_header(self, header: BlockHeaderAPI, score: int) -> None:
        ...

    @abstractmethod
    def persist_header(self, header: BlockHeaderAPI) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]]:
        ...

    @abstractmethod
    def persist_header_chain(
        self, headers: Sequence[BlockHeaderAPI], genesis_parent_hash: Optional[bytes] = None
    ) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]]:
        ...


class ChainDatabaseAPI(HeaderDatabaseAPI):
    @abstractmethod
    def get_block_uncles(self, uncles_hash: bytes) -> Tuple[BlockHeaderAPI, ...]:
        ...

    @abstractmethod
    def persist_block(
        self, block: BlockAPI, genesis_parent_hash: Optional[bytes] = None
    ) -> Tuple[Tuple[bytes, ...], Tuple[bytes, ...]]:
        ...

    @abstractmethod
    def persist_unexecuted_block(
        self, block: BlockAPI, receipts: Tuple[ReceiptAPI, ...], genesis_parent_hash: Optional[bytes] = None
    ) -> Tuple[Tuple[bytes, ...], Tuple[bytes, ...]]:
        ...

    @abstractmethod
    def persist_uncles(self, uncles: Tuple[BlockHeaderAPI, ...]) -> bytes:
        ...

    @abstractmethod
    def add_receipt(
        self, block_header: BlockHeaderAPI, index_key: int, receipt: ReceiptAPI
    ) -> bytes:
        ...

    @abstractmethod
    def add_transaction(
        self, block_header: BlockHeaderAPI, index_key: int, transaction: SignedTransactionAPI
    ) -> bytes:
        ...

    @abstractmethod
    def get_block_transactions(
        self, block_header: BlockHeaderAPI, transaction_decoder: Type[TransactionDecoderAPI]
    ) -> Tuple[SignedTransactionAPI, ...]:
        ...

    @abstractmethod
    def get_block_transaction_hashes(self, block_header: BlockHeaderAPI) -> Tuple[bytes, ...]:
        ...

    @abstractmethod
    def get_receipt_by_index(
        self,
        block_number: int,
        receipt_index: int,
        receipt_decoder: Type[ReceiptDecoderAPI],
    ) -> ReceiptAPI:
        ...

    @abstractmethod
    def get_receipts(
        self, header: BlockHeaderAPI, receipt_decoder: Type[ReceiptDecoderAPI]
    ) -> Tuple[ReceiptAPI, ...]:
        ...

    @abstractmethod
    def get_transaction_by_index(
        self,
        block_number: int,
        transaction_index: int,
        transaction_decoder: Type[TransactionDecoderAPI],
    ) -> SignedTransactionAPI:
        ...

    @abstractmethod
    def get_transaction_index(self, transaction_hash: bytes) -> Tuple[int, int]:
        ...

    @abstractmethod
    def get_block_withdrawals(self, block_header: BlockHeaderAPI) -> Tuple[WithdrawalAPI, ...]:
        ...

    @abstractmethod
    def exists(self, key: bytes) -> bool:
        ...

    @abstractmethod
    def get(self, key: bytes) -> bytes:
        ...

    @abstractmethod
    def persist_trie_data_dict(self, trie_data_dict: Dict[bytes, bytes]) -> None:
        ...


class GasMeterAPI(ABC):
    start_gas: int
    gas_refunded: int
    gas_remaining: int

    @abstractmethod
    def consume_gas(self, amount: int, reason: str) -> None:
        ...

    @abstractmethod
    def return_gas(self, amount: int) -> None:
        ...

    @abstractmethod
    def refund_gas(self, amount: int) -> None:
        ...


class MessageAPI(ABC):
    code: bytes
    _code_address: bytes
    create_address: bytes
    data: Union[bytes, memoryview]
    depth: int
    gas: int
    is_static: bool
    sender: bytes
    should_transfer_value: bool
    _storage_address: bytes
    to: bytes
    value: int

    __slots__ = [
        "code",
        "_code_address",
        "create_address",
        "data",
        "depth",
        "gas",
        "is_static",
        "sender",
        "should_transfer_value",
        "_storage_address",
        "to",
        "value",
    ]

    @property
    @abstractmethod
    def code_address(self) -> bytes:
        ...

    @property
    @abstractmethod
    def storage_address(self) -> bytes:
        ...

    @property
    @abstractmethod
    def is_create(self) -> bool:
        ...

    @property
    @abstractmethod
    def data_as_bytes(self) -> bytes:
        ...


class OpcodeAPI(ABC):
    mnemonic: str

    @abstractmethod
    def __call__(self, computation: "ComputationAPI") -> None:
        ...

    @classmethod
    @abstractmethod
    def as_opcode(
        cls: Type[T],
        logic_fn: Callable[["ComputationAPI"], None],
        mnemonic: str,
        gas_cost: int,
    ) -> "OpcodeAPI":
        ...


class ChainContextAPI(ABC):
    @abstractmethod
    def __init__(self, chain_id: Optional[int]) -> None:
        ...

    @property
    @abstractmethod
    def chain_id(self) -> int:
        ...


class TransactionContextAPI(ABC):
    @abstractmethod
    def __init__(self, gas_price: int, origin: bytes) -> None:
        ...

    @abstractmethod
    def get_next_log_counter(self) -> int:
        ...

    @property
    @abstractmethod
    def gas_price(self) -> int:
        ...

    @property
    @abstractmethod
    def origin(self) -> bytes:
        ...

    @property
    @abstractmethod
    def blob_versioned_hashes(self) -> Sequence[bytes]:
        ...


class MemoryAPI(ABC):
    @abstractmethod
    def extend(self, start_position: int, size: int) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def write(self, start_position: int, size: int, value: bytes) -> None:
        ...

    @abstractmethod
    def read(self, start_position: int, size: int) -> memoryview:
        ...

    @abstractmethod
    def read_bytes(self, start_position: int, size: int) -> bytes:
        ...

    @abstractmethod
    def copy(self, destination: int, source: int, length: int) -> bytes:
        ...


class StackAPI(ABC):
    @abstractmethod
    def push_int(self, value: int) -> None:
        ...

    @abstractmethod
    def push_bytes(self, value: bytes) -> None:
        ...

    @abstractmethod
    def pop1_bytes(self) -> bytes:
        ...

    @abstractmethod
    def pop1_int(self) -> int:
        ...

    @abstractmethod
    def pop1_any(self) -> Union[int, bytes]:
        ...

    @abstractmethod
    def pop_any(self, num_items: int) -> Tuple[Union[int, bytes], ...]:
        ...

    @abstractmethod
    def pop_ints(self, num_items: int) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def pop_bytes(self, num_items: int) -> Tuple[bytes, ...]:
        ...

    @abstractmethod
    def swap(self, position: int) -> None:
        ...

    @abstractmethod
    def dup(self, position: int) -> None:
        ...


class CodeStreamAPI(ABC):
    program_counter: int

    @abstractmethod
    def read(self, size: int) -> bytes:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        ...

    @abstractmethod
    def peek(self) -> int:
        ...

    @abstractmethod
    def seek(self, program_counter: int) -> ContextManager["CodeStreamAPI"]:
        ...

    @abstractmethod
    def is_valid_opcode(self, position: int) -> bool:
        ...


class StackManipulationAPI(ABC):
    @abstractmethod
    def stack_pop_ints(self, num_items: int) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def stack_pop_bytes(self, num_items: int) -> Tuple[bytes, ...]:
        ...

    @abstractmethod
    def stack_pop_any(self, num_items: int) -> Tuple[Union[int, bytes], ...]:
        ...

    @abstractmethod
    def stack_pop1_int(self) -> int:
        ...

    @abstractmethod
    def stack_pop1_bytes(self) -> bytes:
        ...

    @abstractmethod
    def stack_pop1_any(self) -> Union[int, bytes]:
        ...

    @abstractmethod
    def stack_push_int(self, value: int) -> None:
        ...

    @abstractmethod
    def stack_push_bytes(self, value: bytes) -> None:
        ...


class ExecutionContextAPI(ABC):
    @property
    @abstractmethod
    def coinbase(self) -> bytes:
        ...

    @property
    @abstractmethod
    def timestamp(self) -> int:
        ...

    @property
    @abstractmethod
    def block_number(self) -> int:
        ...

    @property
    @abstractmethod
    def difficulty(self) -> int:
        ...

    @property
    @abstractmethod
    def mix_hash(self) -> bytes:
        ...

    @property
    @abstractmethod
    def gas_limit(self) -> int:
        ...

    @property
    @abstractmethod
    def prev_hashes(self) -> Iterable[bytes]:
        ...

    @property
    @abstractmethod
    def chain_id(self) -> int:
        ...

    @property
    @abstractmethod
    def base_fee_per_gas(self) -> Optional[int]:
        ...

    @property
    @abstractmethod
    def excess_blob_gas(self) -> Optional[int]:
        ...


class ComputationAPI(ContextManager["ComputationAPI"], StackManipulationAPI):
    logger: Any

    state: "StateAPI"
    msg: MessageAPI
    transaction_context: TransactionContextAPI
    code: CodeStreamAPI
    children: List["ComputationAPI"]
    return_data: bytes
    accounts_to_delete: List[bytes]
    beneficiaries: List[bytes]
    contracts_created: List[bytes]

    _memory: MemoryAPI
    _stack: StackAPI
    _gas_meter: GasMeterAPI
    _error: Exception
    _output: bytes
    _log_entries: List[Tuple[int, bytes, Tuple[int, ...], bytes]]

    opcodes: Dict[int, OpcodeAPI]
    _precompiles: Dict[bytes, Callable[["ComputationAPI"], "ComputationAPI"]]

    @abstractmethod
    def __init__(
        self,
        state: "StateAPI",
        message: MessageAPI,
        transaction_context: TransactionContextAPI,
    ) -> None:
        ...

    @abstractmethod
    def _configure_gas_meter(self) -> GasMeterAPI:
        ...

    @property
    @abstractmethod
    def is_origin_computation(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_success(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_error(self) -> bool:
        ...

    @property
    @abstractmethod
    def error(self) -> Exception:
        ...

    @error.setter
    def error(self, value: Exception) -> None:
        raise NotImplementedError

    @abstractmethod
    def raise_if_error(self) -> None:
        ...

    @property
    @abstractmethod
    def should_burn_gas(self) -> bool:
        ...

    @property
    @abstractmethod
    def should_return_gas(self) -> bool:
        ...

    @property
    @abstractmethod
    def should_erase_return_data(self) -> bool:
        ...

    @abstractmethod
    def extend_memory(self, start_position: int, size: int) -> None:
        ...

    @abstractmethod
    def memory_write(self, start_position: int, size: int, value: bytes) -> None:
        ...

    @abstractmethod
    def memory_read_bytes(self, start_position: int, size: int) -> bytes:
        ...

    @abstractmethod
    def memory_copy(self, destination: int, source: int, length: int) -> bytes:
        ...

    @abstractmethod
    def get_gas_meter(self) -> GasMeterAPI:
        ...

    @abstractmethod
    def consume_gas(self, amount: int, reason: str) -> None:
        ...

    @abstractmethod
    def return_gas(self, amount: int) -> None:
        ...

    @abstractmethod
    def refund_gas(self, amount: int) -> None:
        ...

    @abstractmethod
    def get_gas_used(self) -> int:
        ...

    @abstractmethod
    def get_gas_remaining(self) -> int:
        ...

    @abstractmethod
    def stack_swap(self, position: int) -> None:
        ...

    @abstractmethod
    def stack_dup(self, position: int) -> None:
        ...

    @property
    @abstractmethod
    def output(self) -> bytes:
        ...

    @output.setter
    def output(self, value: bytes) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def precompiles(self) -> Dict[bytes, Callable[["ComputationAPI"], None]]:
        ...

    @classmethod
    @abstractmethod
    def get_precompiles(cls) -> Dict[bytes, Callable[["ComputationAPI"], None]]:
        ...

    @abstractmethod
    def get_opcode_fn(self, opcode: int) -> OpcodeAPI:
        ...

    @abstractmethod
    def prepare_child_message(
        self,
        gas: int,
        to: bytes,
        value: int,
        data: Union[bytes, memoryview],
        code: bytes,
        **kwargs: Any,
    ) -> MessageAPI:
        ...

    @abstractmethod
    def apply_child_computation(self, child_msg: MessageAPI) -> "ComputationAPI":
        ...

    @abstractmethod
    def generate_child_computation(self, child_msg: MessageAPI) -> "ComputationAPI":
        ...

    @abstractmethod
    def add_child_computation(self, child_computation: "ComputationAPI") -> None:
        ...

    @abstractmethod
    def get_gas_refund(self) -> int:
        ...

    @abstractmethod
    def register_account_for_deletion(self, beneficiary: bytes) -> None:
        ...

    @abstractmethod
    def get_accounts_for_deletion(self) -> List[bytes]:
        ...

    @abstractmethod
    def get_self_destruct_beneficiaries(self) -> List[bytes]:
        ...

    @abstractmethod
    def add_log_entry(self, account: bytes, topics: Tuple[int, ...], data: bytes) -> None:
        ...

    @abstractmethod
    def get_raw_log_entries(self) -> Tuple[Tuple[int, bytes, Tuple[int, ...], bytes], ...]:
        ...

    @abstractmethod
    def get_log_entries(self) -> Tuple[Tuple[bytes, Tuple[int, ...], bytes], ...]:
        ...

    @classmethod
    @abstractmethod
    def apply_message(
        cls,
        state: "StateAPI",
        message: MessageAPI,
        transaction_context: TransactionContextAPI,
        parent_computation: Optional["ComputationAPI"] = None,
    ) -> "ComputationAPI":
        ...

    @classmethod
    @abstractmethod
    def apply_create_message(
        cls,
        state: "StateAPI",
        message: MessageAPI,
        transaction_context: TransactionContextAPI,
        parent_computation: Optional["ComputationAPI"] = None,
    ) -> "ComputationAPI":
        ...

    @classmethod
    @abstractmethod
    def apply_computation(
        cls,
        state: "StateAPI",
        message: MessageAPI,
        transaction_context: TransactionContextAPI,
    ) -> "ComputationAPI":
        ...


class AccountStorageDatabaseAPI(ABC):
    @abstractmethod
    def get(self, slot: int, from_journal: bool = True) -> int:
        ...

    @abstractmethod
    def set(self, slot: int, value: int) -> None:
        ...

    @abstractmethod
    def delete(self) -> None:
        ...

    @abstractmethod
    def record(self, checkpoint: Any) -> None:
        ...

    @abstractmethod
    def discard(self, checkpoint: Any) -> None:
        ...

    @abstractmethod
    def commit(self, checkpoint: Any) -> None:
        ...

    @abstractmethod
    def lock_changes(self) -> None:
        ...

    @abstractmethod
    def make_storage_root(self) -> None:
        ...

    @property
    @abstractmethod
    def has_changed_root(self) -> bool:
        ...

    @abstractmethod
    def get_changed_root(self) -> bytes:
        ...

    @abstractmethod
    def persist(self, db: DatabaseAPI) -> None:
        ...

    @abstractmethod
    def get_accessed_slots(self) -> FrozenSet[int]:
        ...


class AccountAPI(ABC):
    nonce: int
    balance: int
    storage_root: bytes
    code_hash: bytes


class TransientStorageAPI(ABC):
    @abstractmethod
    def record(self, checkpoint: Any) -> None:
        ...

    @abstractmethod
    def commit(self, snapshot: Any) -> None:
        ...

    @abstractmethod
    def discard(self, snapshot: Any) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def get_transient_storage(self, address: bytes, slot: int) -> bytes:
        ...

    @abstractmethod
    def set_transient_storage(self, address: bytes, slot: int, value: bytes) -> None:
        ...


class AccountDatabaseAPI(ABC):
    @abstractmethod
    def __init__(self, db: AtomicDatabaseAPI, state_root: bytes) -> None:
        ...

    @property
    @abstractmethod
    def state_root(self) -> bytes:
        ...

    @state_root.setter
    def state_root(self, value: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def has_root(self, state_root: bytes) -> bool:
        ...

    @abstractmethod
    def get_storage(self, address: bytes, slot: int, from_journal: bool = True) -> int:
        ...

    @abstractmethod
    def set_storage(self, address: bytes, slot: int, value: int) -> None:
        ...

    @abstractmethod
    def delete_storage(self, address: bytes) -> None:
        ...

    @abstractmethod
    def is_storage_warm(self, address: bytes, slot: int) -> bool:
        ...

    @abstractmethod
    def mark_storage_warm(self, address: bytes, slot: int) -> None:
        ...

    @abstractmethod
    def get_balance(self, address: bytes) -> int:
        ...

    @abstractmethod
    def set_balance(self, address: bytes, balance: int) -> None:
        ...

    @abstractmethod
    def get_nonce(self, address: bytes) -> int:
        ...

    @abstractmethod
    def set_nonce(self, address: bytes, nonce: int) -> None:
        ...

    @abstractmethod
    def increment_nonce(self, address: bytes) -> None:
        ...

    @abstractmethod
    def set_code(self, address: bytes, code: bytes) -> None:
        ...

    @abstractmethod
    def get_code(self, address: bytes) -> bytes:
        ...

    @abstractmethod
    def get_code_hash(self, address: bytes) -> bytes:
        ...

    @abstractmethod
    def delete_code(self, address: bytes) -> None:
        ...

    @abstractmethod
    def account_has_code_or_nonce(self, address: bytes) -> bool:
        ...

    @abstractmethod
    def delete_account(self, address: bytes) -> None:
        ...

    @abstractmethod
    def account_exists(self, address: bytes) -> bool:
        ...

    @abstractmethod
    def touch_account(self, address: bytes) -> None:
        ...

    @abstractmethod
    def account_is_empty(self, address: bytes) -> bool:
        ...

    @abstractmethod
    def is_address_warm(self, address: bytes) -> bool:
        ...

    @abstractmethod
    def mark_address_warm(self, address: bytes) -> None:
        ...

    @abstractmethod
    def record(self) -> Any:
        ...

    @abstractmethod
    def discard(self, checkpoint: Any) -> None:
        ...

    @abstractmethod
    def commit(self, checkpoint: Any) -> None:
        ...

    @abstractmethod
    def lock_changes(self) -> None:
        ...

    @abstractmethod
    def make_state_root(self) -> bytes:
        ...

    @abstractmethod
    def persist(self) -> MetaWitnessAPI:
        ...


class TransactionExecutorAPI(ABC):
    @abstractmethod
    def __init__(self, vm_state: "StateAPI") -> None:
        ...

    @abstractmethod
    def __call__(self, transaction: SignedTransactionAPI) -> "ComputationAPI":
        ...

    @abstractmethod
    def validate_transaction(self, transaction: SignedTransactionAPI) -> None:
        ...

    @abstractmethod
    def build_evm_message(self, transaction: SignedTransactionAPI) -> MessageAPI:
        ...

    @abstractmethod
    def build_computation(self, message: MessageAPI, transaction: SignedTransactionAPI) -> "ComputationAPI":
        ...

    @abstractmethod
    def finalize_computation(self, transaction: SignedTransactionAPI, computation: "ComputationAPI") -> "ComputationAPI":
        ...

    @abstractmethod
    def calc_data_fee(self, transaction: Any) -> int:
        ...


class ConfigurableAPI(ABC):
    @classmethod
    @abstractmethod
    def configure(cls: Type[T], __name__: Optional[str] = None, **overrides: Any) -> Type[T]:
        ...


class StateAPI(ConfigurableAPI):
    execution_context: ExecutionContextAPI

    computation_class: Type[ComputationAPI]
    transaction_context_class: Type[TransactionContextAPI]
    account_db_class: Type[AccountDatabaseAPI]
    transaction_executor_class: Optional[Type[TransactionExecutorAPI]] = None

    @abstractmethod
    def __init__(self, db: AtomicDatabaseAPI, execution_context: ExecutionContextAPI, state_root: bytes) -> None:
        ...

    @property
    @abstractmethod
    def logger(self) -> Any:
        ...

    @property
    @abstractmethod
    def coinbase(self) -> bytes:
        ...

    @property
    @abstractmethod
    def timestamp(self) -> int:
        ...

    @property
    @abstractmethod
    def block_number(self) -> int:
        ...

    @property
    @abstractmethod
    def difficulty(self) -> int:
        ...

    @property
    @abstractmethod
    def mix_hash(self) -> bytes:
        ...

    @property
    @abstractmethod
    def gas_limit(self) -> int:
        ...

    @property
    @abstractmethod
    def base_fee(self) -> int:
        ...

    @abstractmethod
    def get_gas_price(self, transaction: SignedTransactionAPI) -> int:
        ...

    @abstractmethod
    def get_tip(self, transaction: SignedTransactionAPI) -> int:
        ...

    @property
    @abstractmethod
    def blob_base_fee(self) -> int:
        ...

    @classmethod
    @abstractmethod
    def get_account_db_class(cls) -> Type[AccountDatabaseAPI]:
        ...

    @property
    @abstractmethod
    def state_root(self) -> bytes:
        ...

    @abstractmethod
    def make_state_root(self) -> bytes:
        ...

    @abstractmethod
    def get_storage(self, address: bytes, slot: int, from_journal: bool = True) -> int:
        ...

    @abstractmethod
    def set_storage(self, address: bytes, slot: int, value: int) -> None:
        ...

    @abstractmethod
    def delete_storage(self, address: bytes) -> None:
        ...

    @abstractmethod
    def delete_account(self, address: bytes) -> None:
        ...

    @abstractmethod
    def get_balance(self, address: bytes) -> int:
        ...

    @abstractmethod
    def set_balance(self, address: bytes, balance: int) -> None:
        ...

    @abstractmethod
    def delta_balance(self, address: bytes, delta: int) -> None:
        ...

    @abstractmethod
    def get_nonce(self, address: bytes) -> int:
        ...

    @abstractmethod
    def set_nonce(self, address: bytes, nonce: int) -> None:
        ...

    @abstractmethod
    def increment_nonce(self, address: bytes) -> None:
        ...

    @abstractmethod
    def get_code(self, address: bytes) -> bytes:
        ...

    @abstractmethod
    def set_code(self, address: bytes, code: bytes) -> None:
        ...

    @abstractmethod
    def get_code_hash(self, address: bytes) -> bytes:
        ...

    @abstractmethod
    def delete_code(self, address: bytes) -> None:
        ...

    @abstractmethod
    def has_code_or_nonce(self, address: bytes) -> bool:
        ...

    @abstractmethod
    def account_exists(self, address: bytes) -> bool:
        ...

    @abstractmethod
    def touch_account(self, address: bytes) -> None:
        ...

    @abstractmethod
    def account_is_empty(self, address: bytes) -> bool:
        ...

    @abstractmethod
    def is_storage_warm(self, address: bytes, slot: int) -> bool:
        ...

    @abstractmethod
    def mark_storage_warm(self, address: bytes, slot: int) -> None:
        ...

    @abstractmethod
    def is_address_warm(self, address: bytes) -> bool:
        ...

    @abstractmethod
    def mark_address_warm(self, address: bytes) -> None:
        ...

    @abstractmethod
    def get_transient_storage(self, address: bytes, slot: int) -> bytes:
        ...

    @abstractmethod
    def set_transient_storage(self, address: bytes, slot: int, value: bytes) -> None:
        ...

    @abstractmethod
    def clear_transient_storage(self) -> None:
        ...

    @abstractmethod
    def snapshot(self) -> Tuple[bytes, Any]:
        ...

    @abstractmethod
    def revert(self, snapshot: Tuple[bytes, Any]) -> None:
        ...

    @abstractmethod
    def commit(self, snapshot: Tuple[bytes, Any]) -> None:
        ...

    @abstractmethod
    def lock_changes(self) -> None:
        ...

    @abstractmethod
    def persist(self) -> MetaWitnessAPI:
        ...

    @abstractmethod
    def get_ancestor_hash(self, block_number: int) -> bytes:
        ...

    @abstractmethod
    def get_computation(self, message: MessageAPI, transaction_context: TransactionContextAPI) -> ComputationAPI:
        ...

    @classmethod
    @abstractmethod
    def get_transaction_context_class(cls) -> Type[TransactionContextAPI]:
        ...

    @abstractmethod
    def apply_transaction(self, transaction: SignedTransactionAPI) -> ComputationAPI:
        ...

    @abstractmethod
    def get_transaction_executor(self) -> TransactionExecutorAPI:
        ...

    @abstractmethod
    def costless_execute_transaction(self, transaction: SignedTransactionAPI) -> ComputationAPI:
        ...

    @abstractmethod
    def override_transaction_context(self, gas_price: int) -> ContextManager[None]:
        ...

    @abstractmethod
    def validate_transaction(self, transaction: SignedTransactionAPI) -> None:
        ...

    @abstractmethod
    def get_transaction_context(self, transaction: SignedTransactionAPI) -> TransactionContextAPI:
        ...

    def apply_withdrawal(self, withdrawal: WithdrawalAPI) -> None:
        ...

    def apply_all_withdrawals(self, withdrawals: Sequence[WithdrawalAPI]) -> None:
        ...


class ConsensusContextAPI(ABC):
    @abstractmethod
    def __init__(self, db: AtomicDatabaseAPI) -> None:
        ...


class ConsensusAPI(ABC):
    @abstractmethod
    def __init__(self, context: ConsensusContextAPI) -> None:
        ...

    @abstractmethod
    def validate_seal(self, header: BlockHeaderAPI) -> None:
        ...

    @abstractmethod
    def validate_seal_extension(self, header: BlockHeaderAPI, parents: Iterable[BlockHeaderAPI]) -> None:
        ...

    @classmethod
    @abstractmethod
    def get_fee_recipient(cls, header: BlockHeaderAPI) -> bytes:
        ...


class VirtualMachineAPI(ConfigurableAPI):
    fork: str
    chaindb: ChainDatabaseAPI
    extra_data_max_bytes: ClassVar[int]
    consensus_class: Type[ConsensusAPI]
    consensus_context: ConsensusContextAPI

    @abstractmethod
    def __init__(
        self,
        header: BlockHeaderAPI,
        chaindb: ChainDatabaseAPI,
        chain_context: "ChainContextAPI",
        consensus_context: ConsensusContextAPI,
    ) -> None:
        ...

    @property
    @abstractmethod
    def state(self) -> StateAPI:
        ...

    @classmethod
    @abstractmethod
    def build_state(
        cls,
        db: AtomicDatabaseAPI,
        header: BlockHeaderAPI,
        chain_context: "ChainContextAPI",
        previous_hashes: Iterable[bytes] = (),
    ) -> StateAPI:
        ...

    @abstractmethod
    def get_header(self) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_block(self) -> BlockAPI:
        ...

    def transaction_applied_hook(
        self,
        transaction_index: int,
        transactions: Sequence[SignedTransactionAPI],
        base_header: BlockHeaderAPI,
        partial_header: BlockHeaderAPI,
        computation: ComputationAPI,
        receipt: ReceiptAPI,
    ) -> None:
        pass

    @abstractmethod
    def apply_transaction(self, header: BlockHeaderAPI, transaction: SignedTransactionAPI) -> Tuple[ReceiptAPI, ComputationAPI]:
        ...

    @staticmethod
    @abstractmethod
    def create_execution_context(header: BlockHeaderAPI, prev_hashes: Iterable[bytes], chain_context: "ChainContextAPI") -> ExecutionContextAPI:
        ...

    @abstractmethod
    def execute_bytecode(
        self,
        origin: bytes,
        gas_price: int,
        gas: int,
        to: bytes,
        sender: bytes,
        value: int,
        data: bytes,
        code: bytes,
        code_address: Optional[bytes] = None,
    ) -> ComputationAPI:
        ...

    @abstractmethod
    def apply_all_transactions(
        self,
        transactions: Sequence[SignedTransactionAPI],
        base_header: BlockHeaderAPI,
    ) -> Tuple[BlockHeaderAPI, Tuple[ReceiptAPI, ...], Tuple[ComputationAPI, ...]]:
        ...

    def apply_all_withdrawals(self, withdrawals: Sequence[WithdrawalAPI]) -> None:
        ...

    @abstractmethod
    def make_receipt(
        self,
        base_header: BlockHeaderAPI,
        transaction: SignedTransactionAPI,
        computation: ComputationAPI,
        state: StateAPI,
    ) -> ReceiptAPI:
        ...

    @abstractmethod
    def import_block(self, block: BlockAPI) -> BlockAndMetaWitness:
        ...

    @abstractmethod
    def mine_block(self, block: BlockAPI, *args: Any, **kwargs: Any) -> BlockAndMetaWitness:
        ...

    @abstractmethod
    def set_block_transactions_and_withdrawals(
        self,
        base_block: BlockAPI,
        new_header: BlockHeaderAPI,
        transactions: Sequence[SignedTransactionAPI],
        receipts: Sequence[ReceiptAPI],
        withdrawals: Optional[Sequence[WithdrawalAPI]] = None,
    ) -> BlockAPI:
        ...

    @abstractmethod
    def finalize_block(self, block: BlockAPI) -> BlockAndMetaWitness:
        ...

    @abstractmethod
    def pack_block(self, block: BlockAPI, *args: Any, **kwargs: Any) -> BlockAPI:
        ...

    @abstractmethod
    def add_receipt_to_header(self, old_header: BlockHeaderAPI, receipt: ReceiptAPI) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def increment_blob_gas_used(self, old_header: BlockHeaderAPI, transaction: TransactionFieldsAPI) -> BlockHeaderAPI:
        ...

    @classmethod
    @abstractmethod
    def compute_difficulty(cls, parent_header: BlockHeaderAPI, timestamp: int) -> int:
        ...

    @abstractmethod
    def configure_header(self, **header_params: Any) -> BlockHeaderAPI:
        ...

    @classmethod
    @abstractmethod
    def create_header_from_parent(cls, parent_header: BlockHeaderAPI, **header_params: Any) -> BlockHeaderAPI:
        ...

    @classmethod
    @abstractmethod
    def generate_block_from_parent_header_and_coinbase(cls, parent_header: BlockHeaderAPI, coinbase: bytes) -> BlockAPI:
        ...

    @classmethod
    @abstractmethod
    def create_genesis_header(cls, **genesis_params: Any) -> BlockHeaderAPI:
        ...

    @classmethod
    @abstractmethod
    def get_block_class(cls) -> Type[BlockAPI]:
        ...

    @staticmethod
    @abstractmethod
    def get_block_reward() -> int:
        ...

    @classmethod
    @abstractmethod
    def get_nephew_reward(cls) -> int:
        ...

    @classmethod
    @abstractmethod
    def get_prev_hashes(cls, last_block_hash: bytes, chaindb: ChainDatabaseAPI) -> Optional[Iterable[bytes]]:
        ...

    @property
    @abstractmethod
    def previous_hashes(self) -> Optional[Iterable[bytes]]:
        ...

    @staticmethod
    @abstractmethod
    def get_uncle_reward(block_number: int, uncle: BlockHeaderAPI) -> int:
        ...

    @abstractmethod
    def create_transaction(self, *args: Any, **kwargs: Any) -> SignedTransactionAPI:
        ...

    @classmethod
    @abstractmethod
    def create_unsigned_transaction(
        cls,
        *,
        nonce: int,
        gas_price: int,
        gas: int,
        to: bytes,
        value: int,
        data: bytes,
    ) -> UnsignedTransactionAPI:
        ...

    @classmethod
    @abstractmethod
    def get_transaction_builder(cls) -> Type[TransactionBuilderAPI]:
        ...

    @classmethod
    @abstractmethod
    def get_receipt_builder(cls) -> Type[ReceiptBuilderAPI]:
        ...

    @classmethod
    @abstractmethod
    def validate_receipt(cls, receipt: ReceiptAPI) -> None:
        ...

    @abstractmethod
    def validate_block(self, block: BlockAPI) -> None:
        ...

    @classmethod
    @abstractmethod
    def validate_header(cls, header: BlockHeaderAPI, parent_header: BlockHeaderAPI) -> None:
        ...

    @abstractmethod
    def validate_transaction_against_header(self, base_header: BlockHeaderAPI, transaction: SignedTransactionAPI) -> None:
        ...

    @abstractmethod
    def validate_seal(self, header: BlockHeaderAPI) -> None:
        ...

    @abstractmethod
    def validate_seal_extension(self, header: BlockHeaderAPI, parents: Iterable[BlockHeaderAPI]) -> None:
        ...

    @classmethod
    @abstractmethod
    def validate_uncle(cls, block: BlockAPI, uncle: BlockHeaderAPI, uncle_parent: BlockHeaderAPI) -> None:
        ...

    @classmethod
    @abstractmethod
    def get_state_class(cls) -> Type[StateAPI]:
        ...

    @abstractmethod
    def in_costless_state(self) -> ContextManager[StateAPI]:
        ...


class VirtualMachineModifierAPI(ABC):
    @abstractmethod
    def amend_vm_configuration(self, vm_config: Any) -> Any:
        ...


class HeaderChainAPI(ABC):
    header: BlockHeaderAPI
    chain_id: int
    vm_configuration: Tuple[Tuple[int, Type[VirtualMachineAPI]], ...]

    @abstractmethod
    def __init__(self, base_db: AtomicDatabaseAPI, header: Optional[BlockHeaderAPI] = None) -> None:
        ...

    @classmethod
    @abstractmethod
    def from_genesis_header(cls, base_db: AtomicDatabaseAPI, genesis_header: BlockHeaderAPI) -> "HeaderChainAPI":
        ...

    @classmethod
    @abstractmethod
    def get_headerdb_class(cls) -> Type[HeaderDatabaseAPI]:
        ...

    def get_canonical_block_hash(self, block_number: int) -> bytes:
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number: int) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_canonical_head(self) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash: bytes) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def header_exists(self, block_hash: bytes) -> bool:
        ...

    @abstractmethod
    def import_header(self, header: BlockHeaderAPI) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]]:
        ...


class ChainAPI(ConfigurableAPI):
    vm_configuration: Tuple[Tuple[int, Type[VirtualMachineAPI]], ...]
    chain_id: int
    chaindb: ChainDatabaseAPI
    consensus_context_class: Type[ConsensusContextAPI]

    @classmethod
    @abstractmethod
    def get_chaindb_class(cls) -> Type[ChainDatabaseAPI]:
        ...

    @classmethod
    @abstractmethod
    def from_genesis(
        cls,
        base_db: AtomicDatabaseAPI,
        genesis_params: Dict[str, Any],
        genesis_state: Optional[Dict[Any, Any]] = None,
    ) -> "ChainAPI":
        ...

    @classmethod
    @abstractmethod
    def from_genesis_header(cls, base_db: AtomicDatabaseAPI, genesis_header: BlockHeaderAPI) -> "ChainAPI":
        ...

    @classmethod
    @abstractmethod
    def get_vm_class(cls, header: BlockHeaderAPI) -> Type[VirtualMachineAPI]:
        ...

    @abstractmethod
    def get_vm(self, header: Optional[BlockHeaderAPI] = None) -> VirtualMachineAPI:
        ...

    @classmethod
    def get_vm_class_for_block_number(cls, block_number: int) -> Type[VirtualMachineAPI]:
        ...

    @abstractmethod
    def create_header_from_parent(self, parent_header: BlockHeaderAPI, **header_params: Dict[str, Any]) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash: bytes) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number: int) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_canonical_head(self) -> BlockHeaderAPI:
        ...

    @abstractmethod
    def get_score(self, block_hash: bytes) -> int:
        ...

    @abstractmethod
    def get_ancestors(self, limit: int, header: BlockHeaderAPI) -> Tuple[BlockAPI, ...]:
        ...

    @abstractmethod
    def get_block(self) -> BlockAPI:
        ...

    @abstractmethod
    def get_block_by_hash(self, block_hash: bytes) -> BlockAPI:
        ...

    @abstractmethod
    def get_block_by_header(self, block_header: BlockHeaderAPI) -> BlockAPI:
        ...

    @abstractmethod
    def get_canonical_block_by_number(self, block_number: int) -> BlockAPI:
        ...

    @abstractmethod
    def get_canonical_block_hash(self, block_number: int) -> bytes:
        ...

    @abstractmethod
    def build_block_with_transactions_and_withdrawals(
        self,
        transactions: Tuple[SignedTransactionAPI, ...],
        parent_header: Optional[BlockHeaderAPI] = None,
        withdrawals: Optional[Tuple[WithdrawalAPI, ...]] = None,
    ) -> Tuple[BlockAPI, Tuple[ReceiptAPI, ...], Tuple[ComputationAPI, ...]]:
        ...

    @abstractmethod
    def create_transaction(self, *args: Any, **kwargs: Any) -> SignedTransactionAPI:
        ...

    @abstractmethod
    def create_unsigned_transaction(
        self,
        *,
        nonce: int,
        gas_price: int,
        gas: int,
        to: bytes,
        value: int,
        data: bytes,
    ) -> UnsignedTransactionAPI:
        ...

    @abstractmethod
    def get_canonical_transaction_index(self, transaction_hash: bytes) -> Tuple[int, int]:
        ...

    @abstractmethod
    def get_canonical_transaction(self, transaction_hash: bytes) -> SignedTransactionAPI:
        ...

    @abstractmethod
    def get_canonical_transaction_by_index(self, block_number: int, index: int) -> SignedTransactionAPI:
        ...

    @abstractmethod
    def get_transaction_receipt(self, transaction_hash: bytes) -> ReceiptAPI:
        ...

    @abstractmethod
    def get_transaction_receipt_by_index(self, block_number: int, index: int) -> ReceiptAPI:
        ...

    @abstractmethod
    def get_transaction_result(self, transaction: SignedTransactionAPI, at_header: BlockHeaderAPI) -> bytes:
        ...

    @abstractmethod
    def estimate_gas(self, transaction: SignedTransactionAPI, at_header: Optional[BlockHeaderAPI] = None) -> int:
        ...

    @abstractmethod
    def import_block(self, block: BlockAPI, perform_validation: bool = True) -> Any:
        ...

    @abstractmethod
    def validate_receipt(self, receipt: ReceiptAPI, at_header: BlockHeaderAPI) -> None:
        ...

    @abstractmethod
    def validate_block(self, block: BlockAPI) -> None:
        ...

    @abstractmethod
    def validate_seal(self, header: BlockHeaderAPI) -> None:
        ...

    @abstractmethod
    def validate_uncles(self, block: BlockAPI) -> None:
        ...

    @abstractmethod
    def validate_chain(self, root: BlockHeaderAPI, descendants: Tuple[BlockHeaderAPI, ...], seal_check_random_sample_rate: int = 1) -> None:
        ...

    @abstractmethod
    def validate_chain_extension(self, headers: Tuple[BlockHeaderAPI, ...]) -> None:
        ...


class MiningChainAPI(ChainAPI):
    header: BlockHeaderAPI

    @abstractmethod
    def __init__(self, base_db: AtomicDatabaseAPI, header: Optional[BlockHeaderAPI] = None) -> None:
        ...

    @abstractmethod
    def set_header_timestamp(self, timestamp: int) -> None:
        ...

    @abstractmethod
    def mine_all(
        self,
        transactions: Sequence[SignedTransactionAPI],
        *args: Any,
        parent_header: Optional[BlockHeaderAPI] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Tuple[ReceiptAPI, ...], Tuple[ComputationAPI, ...]]:
        ...

    @abstractmethod
    def apply_transaction(self, transaction: SignedTransactionAPI) -> Tuple[BlockAPI, ReceiptAPI, ComputationAPI]:
        ...

    @abstractmethod
    def mine_block(self, *args: Any, **kwargs: Any) -> BlockAPI:
        ...

    @abstractmethod
    def mine_block_extended(self, *args: Any, **kwargs: Any) -> BlockAndMetaWitness:
        ...