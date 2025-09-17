from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, MutableMapping, ContextManager, Callable
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from eth_bloom import BloomFilter
from eth_keys.datatypes import PrivateKey
from eth_typing import Address, BlockNumber, Hash32
from eth_utils import ExtendedDebugLogger
from eth.constants import BLANK_ROOT_HASH
from eth.exceptions import VMError
from eth.typing import AccountState, BytesOrView, ChainGaps, HeaderParams, JournalDBCheckpoint, VMConfiguration

T = TypeVar("T")
DecodedZeroOrOneLayerRLP = Union[bytes, List[bytes]]


class MiningHeaderAPI(ABC):
    """
    A class to define a block header without ``mix_hash`` and ``nonce`` which can act as
    a temporary representation during mining before the block header is sealed.
    """

    @property
    @abstractmethod
    def hash(self) -> Hash32:
        """
        Return the hash of the block header.
        """
        ...

    @property
    @abstractmethod
    def mining_hash(self) -> Hash32:
        """
        Return the mining hash of the block header.
        """
        ...

    @property
    @abstractmethod
    def hex_hash(self) -> str:
        """
        Return the hash as a hex string.
        """
        ...

    @property
    @abstractmethod
    def is_genesis(self) -> bool:
        """
        Return ``True`` if this header represents the genesis block of the chain,
        otherwise ``False``.
        """
        ...

    @abstractmethod
    def build_changeset(self, *args: Any, **kwargs: Any) -> Any:
        """
        Open a changeset to modify the header.
        """
        ...

    @abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the header.
        """
        ...

    @property
    @abstractmethod
    def base_fee_per_gas(self) -> Optional[int]:
        """
        Return the base fee per gas of the block.

        Set to None in pre-EIP-1559 (London) header.
        """
        ...

    @property
    @abstractmethod
    def withdrawals_root(self) -> Optional[Hash32]:
        """
        Return the withdrawals root of the block.

        Set to None in pre-Shanghai header.
        """
        ...


class BlockHeaderSedesAPI(ABC):
    """
    Serialize and deserialize RLP for a header.
    """

    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: Any) -> Any:
        """
        Extract a header from an encoded RLP object.
        """
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj: Any) -> bytes:
        """
        Encode a header to a series of bytes used by RLP.
        """
        ...


class BlockHeaderAPI(MiningHeaderAPI, BlockHeaderSedesAPI):
    """
    A class derived from :class:`~eth.abc.MiningHeaderAPI` to define a block header
    after it is sealed.
    """

    @abstractmethod
    def copy(self, *args: Any, **kwargs: Any) -> BlockHeaderAPI:
        """
        Return a copy of the header, optionally overwriting any of its properties.
        """
        ...

    @property
    @abstractmethod
    def parent_beacon_block_root(self) -> Hash32:
        """
        Return the hash of the parent beacon block.
        """
        ...

    @property
    @abstractmethod
    def blob_gas_used(self) -> int:
        """
        Return blob gas used.
        """
        ...

    @property
    @abstractmethod
    def excess_blob_gas(self) -> int:
        """
        Return excess blob gas.
        """
        ...


class LogAPI(ABC):
    """
    A class to define a written log.
    """

    @property
    @abstractmethod
    def bloomables(self) -> Iterable[Any]:
        ...
        

class ReceiptAPI(ABC):
    """
    A class to define a receipt to capture the outcome of a transaction.
    """

    @property
    @abstractmethod
    def state_root(self) -> Hash32:
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
    def logs(self) -> Iterable[Any]:
        ...

    @property
    @abstractmethod
    def bloom_filter(self) -> BloomFilter:
        ...

    def copy(self, *args: Any, **kwargs: Any) -> ReceiptAPI:
        """
        Return a copy of the receipt, optionally overwriting any of its properties.
        """
        ...

    @abstractmethod
    def encode(self) -> bytes:
        """
        This encodes a receipt, no matter if it's: a legacy receipt, a
        typed receipt, or the payload of a typed receipt.
        """
        ...


class ReceiptDecoderAPI(ABC):
    """
    Responsible for decoding receipts from bytestrings.
    """

    @classmethod
    @abstractmethod
    def decode(cls, encoded: Any) -> ReceiptAPI:
        """
        Decodes a receipt from an encoded bytestring.
        """
        ...


class ReceiptBuilderAPI(ReceiptDecoderAPI):
    """
    Responsible for encoding and decoding receipts.
    """

    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: Any) -> ReceiptAPI:
        """
        Extract a receipt from an encoded RLP object.
        """
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj: ReceiptAPI) -> bytes:
        """
        Encode a receipt to a series of bytes used by RLP.
        """
        ...


class BaseTransactionAPI(ABC):
    """
    A class to define all common methods of a transaction.
    """

    @abstractmethod
    def validate(self) -> None:
        """
        Hook called during instantiation to ensure that all transaction
        parameters pass validation rules.
        """
        ...

    @property
    @abstractmethod
    def intrinsic_gas(self) -> int:
        """
        Convenience property for the return value of `get_intrinsic_gas`
        """
        ...

    @abstractmethod
    def get_intrinsic_gas(self) -> int:
        """
        Return the intrinsic gas for the transaction.
        """
        ...

    @abstractmethod
    def gas_used_by(self, computation: Any) -> int:
        """
        Return the gas used by the given computation.
        """
        ...

    @abstractmethod
    def copy(self, **overrides: Any) -> BaseTransactionAPI:
        """
        Return a copy of the transaction.
        """
        ...

    @property
    @abstractmethod
    def access_list(self) -> Any:
        """
        Get addresses to be accessed by a transaction, and their storage slots.
        """
        ...


class TransactionFieldsAPI(ABC):
    """
    A class to define all common transaction fields.
    """

    @property
    @abstractmethod
    def nonce(self) -> int:
        ...

    @property
    @abstractmethod
    def gas_price(self) -> int:
        """
        Will raise :class:`AttributeError` if get or set on a 1559 transaction.
        """
        ...

    @property
    @abstractmethod
    def max_fee_per_gas(self) -> int:
        """
        Will default to gas_price if this is a pre-1559 transaction.
        """
        ...

    @property
    @abstractmethod
    def max_priority_fee_per_gas(self) -> int:
        """
        Will default to gas_price if this is a pre-1559 transaction.
        """
        ...

    @property
    @abstractmethod
    def gas(self) -> int:
        ...

    @property
    @abstractmethod
    def to(self) -> Optional[Address]:
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
    def hash(self) -> Hash32:
        """
        Return the hash of the transaction.
        """
        ...

    @property
    @abstractmethod
    def chain_id(self) -> int:
        ...

    @property
    @abstractmethod
    def max_fee_per_blob_gas(self) -> int:
        ...

    @property
    @abstractmethod
    def blob_versioned_hashes(self) -> Sequence[Hash32]:
        ...


class LegacyTransactionFieldsAPI(TransactionFieldsAPI):
    @property
    @abstractmethod
    def v(self) -> int:
        """
        In old transactions, this v field combines the y_parity bit and the
        chain ID.
        """
        ...


class UnsignedTransactionAPI(BaseTransactionAPI):
    """
    A class representing a transaction before it is signed.
    """

    @abstractmethod
    def as_signed_transaction(
        self, private_key: PrivateKey, chain_id: Optional[int] = None
    ) -> SignedTransactionAPI:
        """
        Return a signed version of this transaction.
        """
        ...


class TransactionDecoderAPI(ABC):
    """
    Responsible for decoding transactions from bytestrings.
    """

    @classmethod
    @abstractmethod
    def decode(cls, encoded: Any) -> BaseTransactionAPI:
        """
        Decode a transaction from an encoded bytestring.
        """
        ...


class TransactionBuilderAPI(TransactionDecoderAPI):
    """
    Responsible for creating and encoding transactions.
    """

    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: Any) -> BaseTransactionAPI:
        """
        Extract a transaction from an encoded RLP object.
        """
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj: BaseTransactionAPI) -> bytes:
        """
        Encode a transaction to a series of bytes used by RLP.
        """
        ...

    @classmethod
    @abstractmethod
    def create_unsigned_transaction(
        cls, *, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes
    ) -> UnsignedTransactionAPI:
        """
        Create an unsigned transaction.
        """
        ...

    @classmethod
    @abstractmethod
    def new_transaction(
        cls, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes, v: int, r: int, s: int
    ) -> SignedTransactionAPI:
        """
        Create a signed transaction.
        """
        ...


class SignedTransactionAPI(BaseTransactionAPI, TransactionFieldsAPI):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    """
    A class representing a transaction that was signed with a private key.
    """

    @property
    @abstractmethod
    def sender(self) -> Address:
        """
        Convenience and performance property for the sender address.
        """
        ...

    @property
    @abstractmethod
    def y_parity(self) -> int:
        """
        The bit used to disambiguate elliptic curve signatures.
        """
        ...

    @abstractmethod
    def validate(self) -> None:
        """
        Validate transaction parameters.
        """
        ...

    @property
    @abstractmethod
    def is_signature_valid(self) -> bool:
        """
        Return ``True`` if the signature is valid.
        """
        ...

    @abstractmethod
    def check_signature_validity(self) -> None:
        """
        Check signature validity; raise error if invalid.
        """
        ...

    @abstractmethod
    def get_sender(self) -> Address:
        """
        Get the 20-byte sender address.
        """
        ...

    @abstractmethod
    def get_message_for_signing(self) -> bytes:
        """
        Return the bytestring for signing.
        """
        ...

    def as_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of the transaction.
        """
        ...

    @abstractmethod
    def make_receipt(self, status: Any, gas_used: int, log_entries: Iterable[Any]) -> ReceiptAPI:
        """
        Build a receipt for this transaction.
        """
        ...

    @abstractmethod
    def encode(self) -> bytes:
        """
        Encode the transaction.
        """
        ...


class WithdrawalAPI(ABC):
    """
    A class to define a withdrawal.
    """

    @property
    @abstractmethod
    def index(self) -> int:
        """
        A monotonically increasing index for each withdrawal.
        """
        ...

    @property
    @abstractmethod
    def validator_index(self) -> int:
        """
        The index for the validator corresponding to the withdrawal.
        """
        ...

    @property
    @abstractmethod
    def address(self) -> Address:
        """
        The recipient address for the withdrawn ether.
        """
        ...

    @property
    @abstractmethod
    def amount(self) -> int:
        """
        The withdrawal amount in gwei.
        """
        ...

    @property
    @abstractmethod
    def hash(self) -> Hash32:
        """
        Return the hash of the withdrawal.
        """
        ...

    @abstractmethod
    def validate(self) -> None:
        """
        Validate withdrawal fields.
        """
        ...

    @abstractmethod
    def encode(self) -> bytes:
        """
        Return the encoded withdrawal.
        """
        ...


class BlockAPI(ABC):
    """
    A class to define a block.
    """
    transaction_builder: ClassVar[Optional[Type[TransactionBuilderAPI]]] = None
    receipt_builder: ClassVar[Optional[Type[ReceiptBuilderAPI]]] = None

    @abstractmethod
    def __init__(
        self,
        header: Any,
        transactions: Iterable[Any],
        uncles: Iterable[Any],
        withdrawals: Optional[Iterable[Any]] = None,
    ) -> None:
        ...

    @classmethod
    @abstractmethod
    def get_transaction_builder(cls) -> Type[TransactionBuilderAPI]:
        """
        Return the transaction builder for the block.
        """
        ...

    @classmethod
    @abstractmethod
    def get_receipt_builder(cls) -> Type[ReceiptBuilderAPI]:
        """
        Return the receipt builder for the block.
        """
        ...

    @classmethod
    @abstractmethod
    def from_header(cls, header: Any, chaindb: Any) -> BlockAPI:
        """
        Instantiate a block from the given header and chaindb.
        """
        ...

    @abstractmethod
    def get_receipts(self, chaindb: Any) -> Iterable[ReceiptAPI]:
        """
        Fetch receipts for this block.
        """
        ...

    @property
    @abstractmethod
    def hash(self) -> Hash32:
        """
        Return the block hash.
        """
        ...

    @property
    @abstractmethod
    def number(self) -> int:
        """
        Return the block number.
        """
        ...

    @property
    @abstractmethod
    def is_genesis(self) -> bool:
        """
        True if this is the genesis block.
        """
        ...

    def copy(self, *args: Any, **kwargs: Any) -> BlockAPI:
        """
        Return a copy of the block.
        """
        ...


class MetaWitnessAPI(ABC):
    @property
    @abstractmethod
    def hashes(self) -> Iterable[Hash32]:
        ...

    @property
    @abstractmethod
    def accounts_queried(self) -> FrozenSet[Address]:
        ...

    @property
    @abstractmethod
    def account_bytecodes_queried(self) -> FrozenSet[bytes]:
        ...

    @abstractmethod
    def get_slots_queried(self, address: Address) -> Iterable[Any]:
        ...

    @property
    @abstractmethod
    def total_slots_queried(self) -> int:
        """
        Total storage slots queried.
        """
        ...


class BlockAndMetaWitness(NamedTuple):
    # Fields can be defined as needed.
    ...


class BlockPersistResult(NamedTuple):
    # Fields can be defined as needed.
    ...


class BlockImportResult(NamedTuple):
    # Fields can be defined as needed.
    ...


class SchemaAPI(ABC):
    """
    A class representing a database schema.
    """

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
    def make_block_number_to_hash_lookup_key(block_number: BlockNumber) -> bytes:
        ...
    
    @staticmethod
    @abstractmethod
    def make_block_hash_to_score_lookup_key(block_hash: Hash32) -> bytes:
        ...
    
    @staticmethod
    @abstractmethod
    def make_transaction_hash_to_block_lookup_key(transaction_hash: Hash32) -> bytes:
        ...
    
    @staticmethod
    @abstractmethod
    def make_withdrawal_hash_to_block_lookup_key(withdrawal_hash: Hash32) -> bytes:
        ...


class DatabaseAPI(MutableMapping[bytes, bytes], ABC):
    """
    A class representing a database.
    """

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
    """
    The object returned by an atomic database during batching.
    """
    pass


class AtomicDatabaseAPI(DatabaseAPI):
    """
    Immediately write out changes if they are not in an atomic batch context.
    """

    @abstractmethod
    def atomic_batch(self) -> ContextManager[AtomicWriteBatchAPI]:
        ...


class HeaderDatabaseAPI(ABC):
    """
    A database for block headers.
    """

    @abstractmethod
    def __init__(self, db: AtomicDatabaseAPI) -> None:
        ...

    @abstractmethod
    def get_header_chain_gaps(self) -> ChainGaps:
        ...

    @abstractmethod
    def get_canonical_block_hash(self, block_number: BlockNumber) -> Hash32:
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number: BlockNumber) -> MiningHeaderAPI:
        ...

    @abstractmethod
    def get_canonical_head(self) -> MiningHeaderAPI:
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash: Hash32) -> MiningHeaderAPI:
        ...

    @abstractmethod
    def get_score(self, block_hash: Hash32) -> int:
        ...

    @abstractmethod
    def header_exists(self, block_hash: Hash32) -> bool:
        ...

    @abstractmethod
    def persist_checkpoint_header(self, header: Any, score: int) -> None:
        ...

    @abstractmethod
    def persist_header(self, header: Any) -> Tuple[Iterable[Any], Iterable[Any]]:
        ...

    @abstractmethod
    def persist_header_chain(
        self, headers: Iterable[Any], genesis_parent_hash: Optional[Hash32] = None
    ) -> Tuple[Iterable[Any], Iterable[Any]]:
        ...


class ChainDatabaseAPI(HeaderDatabaseAPI):
    """
    A database for chain data.
    """

    @abstractmethod
    def get_block_uncles(self, uncles_hash: Hash32) -> Iterable[Any]:
        ...

    @abstractmethod
    def persist_block(self, block: BlockAPI, genesis_parent_hash: Optional[Hash32] = None) -> None:
        ...

    @abstractmethod
    def persist_unexecuted_block(
        self, block: BlockAPI, receipts: Iterable[ReceiptAPI], genesis_parent_hash: Optional[Hash32] = None
    ) -> None:
        ...

    @abstractmethod
    def persist_uncles(self, uncles: Iterable[Any]) -> Hash32:
        ...

    @abstractmethod
    def add_receipt(self, block_header: MiningHeaderAPI, index_key: Any, receipt: ReceiptAPI) -> Hash32:
        ...

    @abstractmethod
    def add_transaction(self, block_header: MiningHeaderAPI, index_key: Any, transaction: BaseTransactionAPI) -> Hash32:
        ...

    @abstractmethod
    def get_block_transactions(self, block_header: MiningHeaderAPI, transaction_decoder: TransactionDecoderAPI) -> Iterable[Any]:
        ...

    @abstractmethod
    def get_block_transaction_hashes(self, block_header: MiningHeaderAPI) -> Tuple[Hash32, ...]:
        ...

    @abstractmethod
    def get_receipt_by_index(self, block_number: BlockNumber, receipt_index: int, receipt_decoder: ReceiptDecoderAPI) -> ReceiptAPI:
        ...

    @abstractmethod
    def get_receipts(self, header: MiningHeaderAPI, receipt_decoder: ReceiptDecoderAPI) -> Tuple[ReceiptAPI, ...]:
        ...

    @abstractmethod
    def get_transaction_by_index(self, block_number: BlockNumber, transaction_index: int, transaction_decoder: TransactionDecoderAPI) -> BaseTransactionAPI:
        ...

    @abstractmethod
    def get_transaction_index(self, transaction_hash: Hash32) -> Tuple[int, int]:
        ...


class GasMeterAPI(ABC):
    """
    A class to define a gas meter.
    """

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
    """
    A message for VM computation.
    """
    __slots__ = ['code', '_code_address', 'create_address', 'data', 'depth', 'gas', 'is_static', 'sender', 'should_transfer_value', '_storage_addressto', 'value']

    code: bytes
    _code_address: Optional[Address]
    create_address: Optional[Address]
    data: bytes
    depth: int
    gas: int
    is_static: bool
    sender: Address
    should_transfer_value: bool
    _storage_addressto: Any
    value: int

    @property
    @abstractmethod
    def code_address(self) -> Address:
        ...
    
    @property
    @abstractmethod
    def storage_address(self) -> Address:
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
    """
    A class representing an opcode.
    """

    @abstractmethod
    def __call__(self, computation: Any) -> Any:
        ...

    @classmethod
    @abstractmethod
    def as_opcode(cls, logic_fn: Callable[..., Any], mnemonic: str, gas_cost: int) -> OpcodeAPI:
        ...


class ChainContextAPI(ABC):
    """
    Immutable chain context information.
    """

    @abstractmethod
    def __init__(self, chain_id: int) -> None:
        ...

    @property
    @abstractmethod
    def chain_id(self) -> int:
        ...


class TransactionContextAPI(ABC):
    """
    Immutable transaction context information.
    """

    @abstractmethod
    def __init__(self, gas_price: int, origin: Address) -> None:
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
    def origin(self) -> Address:
        ...

    @property
    @abstractmethod
    def blob_versioned_hashes(self) -> Sequence[Hash32]:
        ...


class MemoryAPI(ABC):
    """
    A class representing the memory of the VM.
    """

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
    def copy(self, destination: int, source: int, length: int) -> None:
        ...


class StackAPI(ABC):
    """
    A class representing the stack of the VM.
    """

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
    def pop1_any(self) -> Any:
        ...

    @abstractmethod
    def pop_any(self, num_items: int) -> Tuple[Any, ...]:
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
    """
    A class representing a stream of EVM code.
    """

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
    def seek(self, program_counter: int) -> ContextManager[int]:
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
    def stack_pop_any(self, num_items: int) -> Tuple[Any, ...]:
        ...

    @abstractmethod
    def stack_pop1_int(self) -> int:
        ...

    @abstractmethod
    def stack_pop1_bytes(self) -> bytes:
        ...

    @abstractmethod
    def stack_pop1_any(self) -> Any:
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
    def coinbase(self) -> Address:
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
    def mix_hash(self) -> Hash32:
        ...

    @property
    @abstractmethod
    def gas_limit(self) -> int:
        ...

    @property
    @abstractmethod
    def prev_hashes(self) -> Iterable[Hash32]:
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
    def excess_blob_gas(self) -> int:
        ...


class ComputationAPI(ContextManager[ComputationAPI], StackManipulationAPI, ABC):
    return_data: bytes = b""
    contracts_created: List[Any] = []
    _output: bytes = b""

    @abstractmethod
    def __init__(self, state: StateAPI, message: MessageAPI, transaction_context: TransactionContextAPI) -> None:
        ...

    @abstractmethod
    def _configure_gas_meter(self) -> None:
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
    def error(self) -> VMError:
        ...

    @error.setter
    def error(self, value: VMError) -> None:
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
    def memory_copy(self, destination: int, source: int, length: int) -> None:
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
    def precompiles(self) -> Dict[Address, Callable[..., Any]]:
        ...

    @classmethod
    @abstractmethod
    def get_precompiles(cls) -> Dict[Address, Callable[..., Any]]:
        ...

    @abstractmethod
    def get_opcode_fn(self, opcode: int) -> Callable[..., Any]:
        ...

    @abstractmethod
    def prepare_child_message(
        self, gas: int, to: Optional[Address], value: int, data: bytes, code: bytes, **kwargs: Any
    ) -> MessageAPI:
        ...

    @abstractmethod
    def apply_child_computation(self, child_msg: MessageAPI) -> Any:
        ...

    @abstractmethod
    def generate_child_computation(self, child_msg: MessageAPI) -> ComputationAPI:
        ...

    @abstractmethod
    def add_child_computation(self, child_computation: ComputationAPI) -> None:
        ...

    @abstractmethod
    def get_gas_refund(self) -> int:
        ...

    @abstractmethod
    def register_account_for_deletion(self, beneficiary: Address) -> None:
        ...

    @abstractmethod
    def get_accounts_for_deletion(self) -> Tuple[Address, ...]:
        ...

    @abstractmethod
    def get_self_destruct_beneficiaries(self) -> List[Address]:
        ...

    @abstractmethod
    def add_log_entry(self, account: Address, topics: Iterable[int], data: bytes) -> None:
        ...

    @abstractmethod
    def get_raw_log_entries(self) -> Tuple[Any, ...]:
        ...

    @abstractmethod
    def get_log_entries(self) -> Tuple[Any, ...]:
        ...

    @classmethod
    @abstractmethod
    def apply_message(
        cls,
        state: StateAPI,
        message: MessageAPI,
        transaction_context: TransactionContextAPI,
        parent_computation: Optional[ComputationAPI] = None,
    ) -> ComputationAPI:
        ...

    @classmethod
    @abstractmethod
    def apply_create_message(
        cls,
        state: StateAPI,
        message: MessageAPI,
        transaction_context: TransactionContextAPI,
        parent_computation: Optional[ComputationAPI] = None,
    ) -> ComputationAPI:
        ...

    @classmethod
    @abstractmethod
    def apply_computation(
        cls, state: StateAPI, message: MessageAPI, transaction_context: TransactionContextAPI
    ) -> ComputationAPI:
        ...


class AccountStorageDatabaseAPI(ABC):
    """
    Storage cache and write batch for a single account.
    """

    @abstractmethod
    def get(self, slot: Hashable, from_journal: bool = True) -> Any:
        ...

    @abstractmethod
    def set(self, slot: Hashable, value: Any) -> None:
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
    def make_storage_root(self) -> Any:
        ...

    @property
    @abstractmethod
    def has_changed_root(self) -> bool:
        ...

    @abstractmethod
    def get_changed_root(self) -> Any:
        ...

    @abstractmethod
    def persist(self, db: DatabaseAPI) -> None:
        ...

    @abstractmethod
    def get_accessed_slots(self) -> List[Any]:
        ...


class AccountAPI(ABC):
    """
    A class representing an Ethereum account.
    """
    pass


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
    def get_transient_storage(self, address: Address, slot: Any) -> Any:
        ...

    @abstractmethod
    def set_transient_storage(self, address: Address, slot: Any, value: Any) -> None:
        ...


class AccountDatabaseAPI(ABC):
    @abstractmethod
    def __init__(self, db: DatabaseAPI, state_root: Hash32 = BLANK_ROOT_HASH) -> None:
        ...

    @property
    @abstractmethod
    def state_root(self) -> Hash32:
        ...

    @state_root.setter
    def state_root(self, value: Hash32) -> None:
        raise NotImplementedError

    @abstractmethod
    def has_root(self, state_root: Hash32) -> bool:
        ...

    @abstractmethod
    def get_storage(self, address: Address, slot: Any, from_journal: bool = True) -> Any:
        ...

    @abstractmethod
    def set_storage(self, address: Address, slot: Any, value: Any) -> None:
        ...

    @abstractmethod
    def delete_storage(self, address: Address) -> None:
        ...

    @abstractmethod
    def is_storage_warm(self, address: Address, slot: Any) -> bool:
        ...

    @abstractmethod
    def mark_storage_warm(self, address: Address, slot: Any) -> None:
        ...

    @abstractmethod
    def get_balance(self, address: Address) -> int:
        ...

    @abstractmethod
    def set_balance(self, address: Address, balance: int) -> None:
        ...

    @abstractmethod
    def get_nonce(self, address: Address) -> int:
        ...

    @abstractmethod
    def set_nonce(self, address: Address, nonce: int) -> None:
        ...

    @abstractmethod
    def increment_nonce(self, address: Address) -> None:
        ...

    @abstractmethod
    def set_code(self, address: Address, code: bytes) -> None:
        ...

    @abstractmethod
    def get_code(self, address: Address) -> bytes:
        ...

    @abstractmethod
    def get_code_hash(self, address: Address) -> Hash32:
        ...

    @abstractmethod
    def delete_code(self, address: Address) -> None:
        ...

    @abstractmethod
    def account_has_code_or_nonce(self, address: Address) -> bool:
        ...

    @abstractmethod
    def delete_account(self, address: Address) -> None:
        ...

    @abstractmethod
    def account_exists(self, address: Address) -> bool:
        ...

    @abstractmethod
    def touch_account(self, address: Address) -> None:
        ...

    @abstractmethod
    def account_is_empty(self, address: Address) -> bool:
        ...

    @abstractmethod
    def is_address_warm(self, address: Address) -> bool:
        ...

    @abstractmethod
    def mark_address_warm(self, address: Address) -> None:
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
    def make_state_root(self) -> Hash32:
        ...

    @abstractmethod
    def persist(self) -> None:
        ...


class TransactionExecutorAPI(ABC):
    @abstractmethod
    def __init__(self, vm_state: StateAPI) -> None:
        ...

    @abstractmethod
    def __call__(self, transaction: BaseTransactionAPI) -> ComputationAPI:
        ...

    @abstractmethod
    def validate_transaction(self, transaction: BaseTransactionAPI) -> None:
        ...

    @abstractmethod
    def build_evm_message(self, transaction: BaseTransactionAPI) -> MessageAPI:
        ...

    @abstractmethod
    def build_computation(self, message: MessageAPI, transaction: BaseTransactionAPI) -> ComputationAPI:
        ...

    @abstractmethod
    def finalize_computation(self, transaction: BaseTransactionAPI, computation: ComputationAPI) -> None:
        ...

    @abstractmethod
    def calc_data_fee(self, transaction: BaseTransactionAPI) -> int:
        ...


class ConfigurableAPI(ABC):
    @classmethod
    @abstractmethod
    def configure(cls, __name__: Optional[str] = None, **overrides: Any) -> Type[Any]:
        ...


class StateAPI(ConfigurableAPI, ABC):
    transaction_executor_class: ClassVar[Optional[Type[TransactionExecutorAPI]]] = None

    @abstractmethod
    def __init__(self, db: DatabaseAPI, execution_context: ExecutionContextAPI, state_root: Hash32) -> None:
        ...

    @property
    @abstractmethod
    def logger(self) -> ExtendedDebugLogger:
        ...

    @property
    @abstractmethod
    def coinbase(self) -> Address:
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
    def mix_hash(self) -> Hash32:
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
    def get_gas_price(self, transaction: BaseTransactionAPI) -> int:
        ...

    @abstractmethod
    def get_tip(self, transaction: BaseTransactionAPI) -> int:
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
    def state_root(self) -> Hash32:
        ...

    @abstractmethod
    def make_state_root(self) -> Hash32:
        ...

    @abstractmethod
    def get_storage(self, address: Address, slot: Any, from_journal: bool = True) -> Any:
        ...

    @abstractmethod
    def set_storage(self, address: Address, slot: Any, value: Any) -> None:
        ...

    @abstractmethod
    def delete_storage(self, address: Address) -> None:
        ...

    @abstractmethod
    def delete_account(self, address: Address) -> None:
        ...

    @abstractmethod
    def get_balance(self, address: Address) -> int:
        ...

    @abstractmethod
    def set_balance(self, address: Address, balance: int) -> None:
        ...

    @abstractmethod
    def delta_balance(self, address: Address, delta: int) -> None:
        ...

    @abstractmethod
    def get_nonce(self, address: Address) -> int:
        ...

    @abstractmethod
    def set_nonce(self, address: Address, nonce: int) -> None:
        ...

    @abstractmethod
    def increment_nonce(self, address: Address) -> None:
        ...

    @abstractmethod
    def get_code(self, address: Address) -> bytes:
        ...

    @abstractmethod
    def set_code(self, address: Address, code: bytes) -> None:
        ...

    @abstractmethod
    def get_code_hash(self, address: Address) -> Hash32:
        ...

    @abstractmethod
    def delete_code(self, address: Address) -> None:
        ...

    @abstractmethod
    def has_code_or_nonce(self, address: Address) -> bool:
        ...

    @abstractmethod
    def account_exists(self, address: Address) -> bool:
        ...

    @abstractmethod
    def touch_account(self, address: Address) -> None:
        ...

    @abstractmethod
    def account_is_empty(self, address: Address) -> bool:
        ...

    @abstractmethod
    def is_storage_warm(self, address: Address, slot: Any) -> bool:
        ...

    @abstractmethod
    def mark_storage_warm(self, address: Address, slot: Any) -> None:
        ...

    @abstractmethod
    def is_address_warm(self, address: Address) -> bool:
        ...

    @abstractmethod
    def mark_address_warm(self, address: Address) -> None:
        ...

    @abstractmethod
    def get_transient_storage(self, address: Address, slot: Any) -> Any:
        ...

    @abstractmethod
    def set_transient_storage(self, address: Address, slot: Any, value: Any) -> None:
        ...

    @abstractmethod
    def clear_transient_storage(self) -> None:
        ...

    @abstractmethod
    def snapshot(self) -> Any:
        ...

    @abstractmethod
    def revert(self, snapshot: Any) -> None:
        ...

    @abstractmethod
    def commit(self, snapshot: Any) -> None:
        ...

    @abstractmethod
    def lock_changes(self) -> None:
        ...

    @abstractmethod
    def persist(self) -> None:
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
    def apply_transaction(self, transaction: BaseTransactionAPI) -> ComputationAPI:
        ...

    @abstractmethod
    def get_transaction_executor(self) -> TransactionExecutorAPI:
        ...

    @abstractmethod
    def costless_execute_transaction(self, transaction: BaseTransactionAPI) -> ComputationAPI:
        ...

    @abstractmethod
    def override_transaction_context(self, gas_price: int) -> ContextManager[None]:
        ...

    @abstractmethod
    def validate_transaction(self, transaction: BaseTransactionAPI) -> None:
        ...

    @abstractmethod
    def get_transaction_context(self, transaction: BaseTransactionAPI) -> TransactionContextAPI:
        ...

    def apply_withdrawal(self, withdrawal: WithdrawalAPI) -> None:
        ...

    def apply_all_withdrawals(self, withdrawals: Iterable[WithdrawalAPI]) -> None:
        ...


class ConsensusContextAPI(ABC):
    @abstractmethod
    def __init__(self, db: DatabaseAPI) -> None:
        ...


class ConsensusAPI(ABC):
    @abstractmethod
    def __init__(self, context: ConsensusContextAPI) -> None:
        ...

    @abstractmethod
    def validate_seal(self, header: Any) -> None:
        ...

    @abstractmethod
    def validate_seal_extension(self, header: Any, parents: Iterable[Any]) -> None:
        ...

    @classmethod
    @abstractmethod
    def get_fee_recipient(cls, header: Any) -> Address:
        ...


class VirtualMachineAPI(ConfigurableAPI, ABC):
    @abstractmethod
    def __init__(self, header: Any, chaindb: DatabaseAPI, chain_context: Any, consensus_context: ConsensusContextAPI) -> None:
        ...

    @property
    @abstractmethod
    def state(self) -> StateAPI:
        ...

    @classmethod
    @abstractmethod
    def build_state(
        cls, db: DatabaseAPI, header: Any, chain_context: Any, previous_hashes: Iterable[Hash32] = ()
    ) -> StateAPI:
        ...

    @abstractmethod
    def get_header(self) -> Any:
        ...

    @abstractmethod
    def get_block(self) -> BlockAPI:
        ...

    def transaction_applied_hook(
        self,
        transaction_index: int,
        transactions: Iterable[BaseTransactionAPI],
        base_header: Any,
        partial_header: Any,
        computation: ComputationAPI,
        receipt: ReceiptAPI,
    ) -> None:
        ...

    @abstractmethod
    def apply_transaction(self, header: Any, transaction: BaseTransactionAPI) -> ComputationAPI:
        ...

    @staticmethod
    @abstractmethod
    def create_execution_context(header: Any, prev_hashes: Iterable[Hash32], chain_context: Any) -> ExecutionContextAPI:
        ...

    @abstractmethod
    def execute_bytecode(
        self,
        origin: Address,
        gas_price: int,
        gas: int,
        to: Optional[Address],
        sender: Address,
        value: int,
        data: bytes,
        code: bytes,
        code_address: Optional[Address] = None,
    ) -> ComputationAPI:
        ...

    @abstractmethod
    def apply_all_transactions(
        self, transactions: Iterable[BaseTransactionAPI], base_header: Any
    ) -> Tuple[Any, Tuple[ReceiptAPI, ...], Iterable[ComputationAPI]]:
        ...

    def apply_all_withdrawals(self, withdrawals: Iterable[WithdrawalAPI]) -> None:
        ...

    @abstractmethod
    def make_receipt(self, base_header: Any, transaction: BaseTransactionAPI, computation: ComputationAPI, state: StateAPI) -> ReceiptAPI:
        ...

    @abstractmethod
    def import_block(self, block: BlockAPI) -> BlockImportResult:
        ...

    @abstractmethod
    def mine_block(self, block: BlockAPI, *args: Any, **kwargs: Any) -> BlockAPI:
        ...

    @abstractmethod
    def set_block_transactions_and_withdrawals(
        self,
        base_block: BlockAPI,
        new_header: Any,
        transactions: Iterable[BaseTransactionAPI],
        receipts: Iterable[ReceiptAPI],
        withdrawals: Optional[Iterable[WithdrawalAPI]] = None,
    ) -> BlockAPI:
        ...

    @abstractmethod
    def finalize_block(self, block: BlockAPI) -> None:
        ...

    @abstractmethod
    def pack_block(self, block: BlockAPI, *args: Any, **kwargs: Any) -> BlockAPI:
        ...

    @abstractmethod
    def add_receipt_to_header(self, old_header: Any, receipt: ReceiptAPI) -> Any:
        ...

    @abstractmethod
    def increment_blob_gas_used(self, old_header: Any, transaction: BaseTransactionAPI) -> Any:
        ...

    @classmethod
    @abstractmethod
    def compute_difficulty(cls, parent_header: Any, timestamp: int) -> int:
        ...

    @abstractmethod
    def configure_header(self, **header_params: Any) -> Any:
        ...

    @classmethod
    @abstractmethod
    def create_header_from_parent(cls, parent_header: Any, **header_params: Any) -> Any:
        ...

    @classmethod
    @abstractmethod
    def generate_block_from_parent_header_and_coinbase(cls, parent_header: Any, coinbase: Address) -> BlockAPI:
        ...

    @classmethod
    @abstractmethod
    def create_genesis_header(cls, **genesis_params: Any) -> Any:
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
    def get_prev_hashes(cls, last_block_hash: Hash32, chaindb: DatabaseAPI) -> Iterable[Hash32]:
        ...

    @property
    @abstractmethod
    def previous_hashes(self) -> Tuple[Hash32, ...]:
        ...

    @staticmethod
    @abstractmethod
    def get_uncle_reward(block_number: int, uncle: Any) -> int:
        ...

    @abstractmethod
    def create_transaction(self, *args: Any, **kwargs: Any) -> SignedTransactionAPI:
        ...

    @classmethod
    @abstractmethod
    def create_unsigned_transaction(
        cls, *, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes
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
    def validate_header(cls, header: Any, parent_header: Any) -> None:
        ...

    @abstractmethod
    def validate_transaction_against_header(self, base_header: Any, transaction: BaseTransactionAPI) -> None:
        ...

    @abstractmethod
    def validate_seal(self, header: Any) -> None:
        ...

    @abstractmethod
    def validate_seal_extension(self, header: Any, parents: Iterable[Any]) -> None:
        ...

    @classmethod
    @abstractmethod
    def validate_uncle(cls, block: BlockAPI, uncle: Any, uncle_parent: Any) -> None:
        ...

    @classmethod
    @abstractmethod
    def get_state_class(cls) -> Type[StateAPI]:
        ...

    @abstractmethod
    def in_costless_state(self) -> ContextManager[None]:
        ...


class VirtualMachineModifierAPI(ABC):
    @abstractmethod
    def amend_vm_configuration(self, vm_config: Any) -> Any:
        ...


class HeaderChainAPI(ABC):
    @abstractmethod
    def __init__(self, base_db: DatabaseAPI, header: Optional[Any] = None) -> None:
        ...

    @classmethod
    @abstractmethod
    def from_genesis_header(cls, base_db: DatabaseAPI, genesis_header: Any) -> HeaderChainAPI:
        ...

    @classmethod
    @abstractmethod
    def get_headerdb_class(cls) -> Type[HeaderDatabaseAPI]:
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number: int) -> Any:
        ...

    @abstractmethod
    def get_canonical_head(self) -> Any:
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash: Hash32) -> Any:
        ...

    @abstractmethod
    def header_exists(self, block_hash: Hash32) -> bool:
        ...

    @abstractmethod
    def import_header(self, header: Any) -> Iterable[Any]:
        ...


class ChainAPI(ConfigurableAPI, ABC):
    @classmethod
    @abstractmethod
    def get_chaindb_class(cls) -> Type[ChainDatabaseAPI]:
        ...

    @classmethod
    @abstractmethod
    def from_genesis(cls, base_db: DatabaseAPI, genesis_params: Dict[str, Any], genesis_state: Optional[Any] = None) -> ChainAPI:
        ...

    @classmethod
    @abstractmethod
    def from_genesis_header(cls, base_db: DatabaseAPI, genesis_header: Any) -> ChainAPI:
        ...

    @classmethod
    @abstractmethod
    def get_vm_class(cls, header: Any) -> Type[VirtualMachineAPI]:
        ...

    @abstractmethod
    def get_vm(self, header: Optional[Any] = None) -> VirtualMachineAPI:
        ...

    @classmethod
    def get_vm_class_for_block_number(cls, block_number: int) -> Type[VirtualMachineAPI]:
        ...

    @abstractmethod
    def create_header_from_parent(self, parent_header: Any, **header_params: Any) -> Any:
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash: Hash32) -> Any:
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number: int) -> Any:
        ...

    @abstractmethod
    def get_canonical_head(self) -> Any:
        ...

    @abstractmethod
    def get_score(self, block_hash: Hash32) -> int:
        ...

    @abstractmethod
    def get_ancestors(self, limit: int, header: Any) -> Iterable[Any]:
        ...

    @abstractmethod
    def get_block(self) -> BlockAPI:
        ...

    @abstractmethod
    def get_block_by_hash(self, block_hash: Hash32) -> BlockAPI:
        ...

    @abstractmethod
    def get_block_by_header(self, block_header: Any) -> BlockAPI:
        ...

    @abstractmethod
    def get_canonical_block_by_number(self, block_number: int) -> BlockAPI:
        ...

    @abstractmethod
    def get_canonical_block_hash(self, block_number: int) -> Hash32:
        ...

    @abstractmethod
    def build_block_with_transactions_and_withdrawals(
        self,
        transactions: Iterable[BaseTransactionAPI],
        parent_header: Optional[Any] = None,
        withdrawals: Optional[Iterable[WithdrawalAPI]] = None,
    ) -> Tuple[BlockAPI, Tuple[ReceiptAPI, ...], Iterable[ComputationAPI]]:
        ...

    @abstractmethod
    def create_transaction(self, *args: Any, **kwargs: Any) -> SignedTransactionAPI:
        ...

    @abstractmethod
    def create_unsigned_transaction(cls, *, nonce: int, gas_price: int, gas: int, to: Optional[Address], value: int, data: bytes) -> UnsignedTransactionAPI:
        ...

    @abstractmethod
    def get_canonical_transaction_index(self, transaction_hash: Hash32) -> Tuple[int, int]:
        ...

    @abstractmethod
    def get_canonical_transaction(self, transaction_hash: Hash32) -> BaseTransactionAPI:
        ...

    @abstractmethod
    def get_canonical_transaction_by_index(self, block_number: int, index: int) -> BaseTransactionAPI:
        ...

    @abstractmethod
    def get_transaction_receipt(self, transaction_hash: Hash32) -> ReceiptAPI:
        ...

    @abstractmethod
    def get_transaction_receipt_by_index(self, block_number: int, index: int) -> ReceiptAPI:
        ...

    @abstractmethod
    def get_transaction_result(self, transaction: BaseTransactionAPI, at_header: Any) -> Any:
        ...

    @abstractmethod
    def estimate_gas(self, transaction: BaseTransactionAPI, at_header: Optional[Any] = None) -> int:
        ...

    @abstractmethod
    def import_block(self, block: BlockAPI, perform_validation: bool = True) -> Tuple[
        BlockAPI, Tuple[BlockAPI, ...], Tuple[BlockAPI, ...]
    ]:
        ...

    @abstractmethod
    def validate_receipt(self, receipt: ReceiptAPI, at_header: Any) -> None:
        ...

    @abstractmethod
    def validate_block(self, block: BlockAPI) -> None:
        ...

    @abstractmethod
    def validate_seal(self, header: Any) -> None:
        ...

    @abstractmethod
    def validate_uncles(self, block: BlockAPI) -> None:
        ...

    @abstractmethod
    def validate_chain(self, root: Any, descendants: Iterable[Any], seal_check_random_sample_rate: int = 1) -> None:
        ...

    @abstractmethod
    def validate_chain_extension(self, headers: Iterable[Any]) -> None:
        ...


class MiningChainAPI(ChainAPI, ABC):
    @abstractmethod
    def __init__(self, base_db: DatabaseAPI, header: Optional[Any] = None) -> None:
        ...

    @abstractmethod
    def set_header_timestamp(self, timestamp: int) -> None:
        ...

    @abstractmethod
    def mine_all(self, transactions: Iterable[BaseTransactionAPI], *args: Any, parent_header: Optional[Any] = None, **kwargs: Any) -> BlockAPI:
        ...

    @abstractmethod
    def apply_transaction(self, transaction: BaseTransactionAPI) -> None:
        ...

    @abstractmethod
    def mine_block(self, *args: Any, **kwargs: Any) -> BlockAPI:
        ...

    @abstractmethod
    def mine_block_extended(self, *args: Any, **kwargs: Any) -> Tuple[BlockAPI, Any]:
        ...