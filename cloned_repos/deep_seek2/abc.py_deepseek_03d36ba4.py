from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    List,
    MutableMapping,
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
from eth.typing import (
    AccountState,
    BytesOrView,
    ChainGaps,
    HeaderParams,
    JournalDBCheckpoint,
    VMConfiguration,
)

if TYPE_CHECKING:
    from eth.vm.forks.cancun.transactions import BlobTransaction

T = TypeVar("T")

# A decoded RLP object of unknown interpretation, with a maximum "depth" of 1.
DecodedZeroOrOneLayerRLP = Union[bytes, List[bytes]]


class MiningHeaderAPI(ABC):
    """
    A class to define a block header without ``mix_hash`` and ``nonce`` which can act as
    a temporary representation during mining before the block header is sealed.
    """

    parent_hash: Hash32
    uncles_hash: Hash32
    coinbase: Address
    state_root: Hash32
    transaction_root: Hash32
    receipt_root: Hash32
    bloom: int
    difficulty: int
    block_number: BlockNumber
    gas_limit: int
    gas_used: int
    timestamp: int
    extra_data: bytes

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

    # We can remove this API and inherit from rlp.Serializable when it becomes typesafe
    @abstractmethod
    def build_changeset(self, *args: Any, **kwargs: Any) -> Any:
        """
        Open a changeset to modify the header.
        """
        ...

    # We can remove this API and inherit from rlp.Serializable when it becomes typesafe
    @abstractmethod
    def as_dict(self) -> Dict[Hashable, Any]:
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

    The header may be one of several definitions, like a London (EIP-1559) or
    pre-London header.
    """

    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: List[bytes]) -> "BlockHeaderAPI":
        """
        Extract a header from an encoded RLP object.

        This method is used by rlp.decode(..., sedes=TransactionBuilderAPI).
        """
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj: "BlockHeaderAPI") -> List[bytes]:
        """
        Encode a header to a series of bytes used by RLP.

        This method is used by rlp.encode(obj).
        """
        ...


class BlockHeaderAPI(MiningHeaderAPI, BlockHeaderSedesAPI):
    """
    A class derived from :class:`~eth.abc.MiningHeaderAPI` to define a block header
    after it is sealed.
    """

    mix_hash: Hash32
    nonce: bytes

    # We can remove this API and inherit from rlp.Serializable when it becomes typesafe
    @abstractmethod
    def copy(self, *args: Any, **kwargs: Any) -> "BlockHeaderAPI":
        """
        Return a copy of the header, optionally overwriting any of its properties.
        """
        ...

    @property
    @abstractmethod
    def parent_beacon_block_root(self) -> Optional[Hash32]:
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

    address: Address
    topics: Sequence[int]
    data: bytes

    @property
    @abstractmethod
    def bloomables(self) -> Tuple[bytes, ...]:
        ...


class ReceiptAPI(ABC):
    """
    A class to define a receipt to capture the outcome of a transaction.
    """

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
    def bloom_filter(self) -> BloomFilter:
        ...

    # We can remove this API and inherit from rlp.Serializable when it becomes typesafe
    def copy(self, *args: Any, **kwargs: Any) -> "ReceiptAPI":  # noqa: B027
        """
        Return a copy of the receipt, optionally overwriting any of its properties.
        """
        # This method isn't marked abstract because derived classes implement it by
        # deriving from rlp.Serializable but mypy won't recognize it as implemented.
        ...

    @abstractmethod
    def encode(self) -> bytes:
        """
        This encodes a receipt, no matter if it's: a legacy receipt, a
        typed receipt, or the payload of a typed receipt. See more
        context in decode.
        """
        ...


class ReceiptDecoderAPI(ABC):
    """
    Responsible for decoding receipts from bytestrings.
    """

    @classmethod
    @abstractmethod
    def decode(cls, encoded: bytes) -> ReceiptAPI:
        """
        This decodes a receipt that is encoded to either a typed
        receipt, a legacy receipt, or the body of a typed receipt. It assumes
        that typed receipts are *not* rlp-encoded first.

        If dealing with an object that is always rlp encoded first, then use this instead:

            rlp.decode(encoded, sedes=ReceiptBuilderAPI)

        For example, you may receive a list of receipts via a devp2p request.
        Each receipt is either a (legacy) rlp list, or a (new-style)
        bytestring. Even if the receipt is a bytestring, it's wrapped in an rlp
        bytestring, in that context. New-style receipts will *not* be wrapped
        in an RLP bytestring in other contexts. They will just be an EIP-2718
        type-byte plus payload of concatenated bytes, which cannot be decoded
        as RLP. This happens for example, when calculating the receipt root
        hash.
        """
        ...


class ReceiptBuilderAPI(ReceiptDecoderAPI):
    """
    Responsible for encoding and decoding receipts.

    Most simply, the builder is responsible for some pieces of the encoding for
    RLP. In legacy transactions, this happens using rlp.Serializeable. It is
    also responsible for initializing the transactions. The two transaction
    initializers assume legacy transactions, for now.

    Some VMs support multiple distinct transaction types. In that case, the
    builder is responsible for dispatching on the different types.
    """

    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: DecodedZeroOrOneLayerRLP) -> "ReceiptAPI":
        """
        Extract a receipt from an encoded RLP object.

        This method is used by rlp.decode(..., sedes=ReceiptBuilderAPI).
        """
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj: "ReceiptAPI") -> DecodedZeroOrOneLayerRLP:
        """
        Encode a receipt to a series of bytes used by RLP.

        In the case of legacy receipt, it will actually be a list of
        bytes. That doesn't show up here, because pyrlp doesn't export type
        annotations.

        This method is used by rlp.encode(obj).
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
        Return the intrinsic gas for the transaction which is defined as the amount of
        gas that is needed before any code runs.
        """
        ...

    @abstractmethod
    def gas_used_by(self, computation: "ComputationAPI") -> int:
        """
        Return the gas used by the given computation. In Frontier,
        for example, this is sum of the intrinsic cost and the gas used
        during computation.
        """
        ...

    # We can remove this API and inherit from rlp.Serializable when it becomes typesafe
    @abstractmethod
    def copy(self: T, **overrides: Any) -> T:
        """
        Return a copy of the transaction.
        """
        ...

    @property
    @abstractmethod
    def access_list(self) -> Sequence[Tuple[Address, Sequence[int]]]:
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
    def to(self) -> Address:
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
    def chain_id(self) -> Optional[int]:
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
        chain ID. All new usages should prefer accessing those fields directly.
        But if you must access the original v, then you can cast to this API
        first (after checking that type_id is None).
        """
        ...


class UnsignedTransactionAPI(BaseTransactionAPI):
    """
    A class representing a transaction before it is signed.
    """

    nonce: int
    gas_price: int
    gas: int
    to: Address
    value: int
    data: bytes

    #
    # API that must be implemented by all Transaction subclasses.
    #
    @abstractmethod
    def as_signed_transaction(
        self, private_key: PrivateKey, chain_id: int = None
    ) -> "SignedTransactionAPI":
        """
        Return a version of this transaction which has been signed using the
        provided `private_key`
        """
        ...


class TransactionDecoderAPI(ABC):
    """
    Responsible for decoding transactions from bytestrings.

    Some VMs support multiple distinct transaction types. In that case, the
    decoder is responsible for dispatching on the different types.
    """

    @classmethod
    @abstractmethod
    def decode(cls, encoded: bytes) -> "SignedTransactionAPI":
        """
        This decodes a transaction that is encoded to either a typed
        transaction or a legacy transaction, or even the payload of one of the
        transaction types. It assumes that typed transactions are *not*
        rlp-encoded first.

        If dealing with an object that is rlp encoded first, then use this instead:

            rlp.decode(encoded, sedes=TransactionBuilderAPI)

        For example, you may receive a list of transactions via a devp2p
        request.  Each transaction is either a (legacy) rlp list, or a
        (new-style) bytestring. Even if the transaction is a bytestring, it's
        wrapped in an rlp bytestring, in that context. New-style transactions
        will *not* be wrapped in an RLP bytestring in other contexts. They will
        just be an EIP-2718 type-byte plus payload of concatenated bytes, which
        cannot be decoded as RLP. An example context for this is calculating
        the transaction root hash.
        """
        ...


class TransactionBuilderAPI(TransactionDecoderAPI):
    """
    Responsible for creating and encoding transactions.

    Most simply, the builder is responsible for some pieces of the encoding for
    RLP. In legacy transactions, this happens using rlp.Serializeable. It is
    also responsible for initializing the transactions. The two transaction
    initializers assume legacy transactions, for now.

    Some VMs support multiple distinct transaction types. In that case, the
    builder is responsible for dispatching on the different types.
    """

    @classmethod
    @abstractmethod
    def deserialize(cls, encoded: DecodedZeroOrOneLayerRLP) -> "SignedTransactionAPI":
        """
        Extract a transaction from an encoded RLP object.

        This method is used by rlp.decode(..., sedes=TransactionBuilderAPI).
        """
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj: "SignedTransactionAPI") -> DecodedZeroOrOneLayerRLP:
        """
        Encode a transaction to a series of bytes used by RLP.

        In the case of legacy transactions, it will actually be a list of
        bytes. That doesn't show up here, because pyrlp doesn't export type
        annotations.

        This method is used by rlp.encode(obj).
        """
        ...

    @classmethod
    @abstractmethod
    def create_unsigned_transaction(
        cls,
        *,
        nonce: int,
        gas_price: int,
        gas: int,
        to: Address,
        value: int,
        data: bytes,
    ) -> UnsignedTransactionAPI:
        """
        Create an unsigned transaction.
        """
        ...

    @classmethod
    @abstractmethod
    def new_transaction(
        cls,
        nonce: int,
        gas_price: int,
        gas: int,
        to: Address,
        value: int,
        data: bytes,
        v: int,
        r: int,
        s: int,
    ) -> "SignedTransactionAPI":
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
        Convenience and performance property for the return value of `get_sender`
        """
        ...

    @property
    @abstractmethod
    def y_parity(self) -> int:
        """
        The bit used to disambiguate elliptic curve signatures.

        The only values this method will return are 0 or 1.
        """
        ...

    type_id: Optional[int]
    """
    The type of EIP-2718 transaction

    Each EIP-2718 transaction includes a type id (which is the leading
    byte, as encoded).

    If this transaction is a legacy transaction, that it has no type. Then,
    type_id will be None.
    """

    # +-------------------------------------------------------------+
    # | API that must be implemented by all Transaction subclasses. |
    # +-------------------------------------------------------------+

    #
    # Validation
    #
    @abstractmethod
    def validate(self) -> None:
        """
        Hook called during instantiation to ensure that all transaction
        parameters pass validation rules.
        """
        ...

    #
    # Signature and Sender
    #
    @property
    @abstractmethod
    def is_signature_valid(self) -> bool:
        """
        Return ``True`` if the signature is valid, otherwise ``False``.
        """
        ...

    @abstractmethod
    def check_signature_validity(self) -> None:
        """
        Check if the signature is valid. Raise a ``ValidationError`` if the signature
        is invalid.
        """
        ...

    @abstractmethod
    def get_sender(self) -> Address:
        """
        Get the 20-byte address which sent this transaction.

        This can be a slow operation. ``transaction.sender`` is always preferred.
        """
        ...

    #
    # Conversion to and creation of unsigned transactions.
    #
    @abstractmethod
    def get_message_for_signing(self) -> bytes:
        """
        Return the bytestring that should be signed in order to create a signed
        transaction.
        """
        ...

    # We can remove this API and inherit from rlp.Serializable when it becomes typesafe
    def as_dict(self) -> Dict[Hashable, Any]:
        """
        Return a dictionary representation of the transaction.
        """
        ...

    @abstractmethod
    def make_receipt(
        self,
        status: bytes,
        gas_used: int,
        log_entries: Tuple[Tuple[bytes, Tuple[int, ...], bytes],
    ) -> ReceiptAPI:
        """
        Build a receipt for this transaction.

        Transactions have this responsibility because there are different types
        of transactions, which have different types of receipts. (See
        access-list transactions, which change the receipt encoding)

        :param status: success or failure (used to be the state root after execution)
        :param gas_used: cumulative usage of this transaction and the previous
            ones in the header
        :param log_entries: logs generated during execution
        """
        ...

    @abstractmethod
    def encode(self) -> bytes:
        """
        This encodes a transaction, no matter if it's: a legacy transaction, a
        typed transaction, or the payload of a typed transaction. See more
        context in decode.
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
        A monotonically increasing index, starting from 0, that increments by 1 per
        withdrawal to uniquely identify each withdrawal.
        """
        ...

    @property
    @abstractmethod
    def validator_index(self) -> int:
        """
        The index for the validator on the consensus layer the withdrawal corresponds
        to.
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
        The nonzero amount of ether to withdraw, given in gwei (10**9 wei).
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
        withdrawals: Optional[
            Sequence[WithdrawalAPI]
        ] = None,  # only present post-Shanghai
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
    def from_header(
        cls, header: BlockHeaderAPI, chaindb: "ChainDatabaseAPI"
    ) -> "BlockAPI":
        """
        Instantiate a block from the given ``header`` and the ``chaindb``.
        """
        ...

    @abstractmethod
    def get_receipts(self, chaindb: "ChainDatabaseAPI") -> Tuple[ReceiptAPI, ...]:
        """
        Fetch the receipts for this block from the given ``chaindb``.
        """
        ...

    @property
    @abstractmethod
    def hash(self) -> Hash32:
        """
        Return the hash of the block.
        """
        ...

    @property
    @abstractmethod
    def number(self) -> BlockNumber:
        """
        Return the number of the block.
        """
        ...

    @property
    @abstractmethod
    def is_genesis(self) -> bool:
        """
        Return ``True`` if this block represents the genesis block of the chain,
        otherwise ``False``.
        """
        ...

    # We can remove this API and inherit from rlp.Serializable when it becomes typesafe
    def copy(self, *args: Any, **kwargs: Any) -> "BlockAPI":  # noqa: B027
        """
        Return a copy of the block, optionally overwriting any of its properties.
        """
        # This method isn't marked abstract because derived classes implement it by
        # deriving from rlp.Serializable but mypy won't recognize it as implemented.
        ...


class MetaWitnessAPI(ABC):
    @property
    @abstractmethod
    def hashes(self) -> FrozenSet[Hash32]:
        ...

    @property
    @abstractmethod
    def accounts_queried(self) -> FrozenSet[Address]:
        ...

    @property
    @abstractmethod
    def account_bytecodes_queried(self) -> FrozenSet[Address]:
        ...

    @abstractmethod
    def get_slots_queried(self, address: Address) -> FrozenSet[int]:
        ...

    @property
    @abstractmethod
    def total_slots_queried(self) -> int:
        """
        Summed across all accounts, how many storage slots were queried?
        """
        ...


class BlockAndMetaWitness(NamedTuple):
    """
    After evaluating a block using the VirtualMachine, this information
    becomes available.
    """

    block: BlockAPI
    meta_witness: MetaWitnessAPI


class BlockPersistResult(NamedTuple):
    """
    After persisting a block into the active chain, this information
    becomes available.
    """

    imported_block: BlockAPI
    new_canonical_blocks: Tuple[BlockAPI, ...]
    old_canonical_blocks: Tuple[BlockAPI, ...]


class BlockImportResult(NamedTuple):
    """
    After importing and persisting a block into the active chain, this information
    becomes available.
    """

    imported_block: BlockAPI
    new_canonical_blocks: Tuple[BlockAPI, ...]
    old_canonical_blocks: Tuple[BlockAPI, ...]
    meta_witness: MetaWitnessAPI


class SchemaAPI(ABC):
    """
    A class representing a database schema that maps values to lookup keys.
    """

    @staticmethod
    @abstractmethod
    def make_header_chain_gaps_lookup_key() -> bytes:
        """
        Return the lookup key to retrieve the header chain integrity info from the
        database.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_canonical_head_hash_lookup_key() -> bytes:
        """
        Return the lookup key to retrieve the canonical head from the database.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_block_number_to_hash_lookup_key(block_number: BlockNumber) -> bytes:
        """
        Return the lookup key to retrieve a block hash from a block number.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_block_hash_to_score_lookup_key(block_hash: Hash32) -> bytes:
        """
        Return the lookup key to retrieve the score from a block hash.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_transaction_hash_to_block_lookup_key(transaction_hash: Hash32) -> bytes:
        """
        Return the lookup key to retrieve a transaction key from a transaction hash.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_withdrawal_hash_to_block_lookup_key(withdrawal_hash: Hash32) -> bytes:
        """
        Return the lookup key to retrieve a withdrawal key from a withdrawal hash.
        """
        ...


class DatabaseAPI(MutableMapping[bytes, bytes], ABC):
    """
    A class representing a database.
    """

    @abstractmethod
    def set(self, key: bytes, value: bytes) -> None:
        """
        Assign the ``value`` to the ``key``.
        """
        ...

    @abstractmethod
    def exists(self, key: bytes) -> bool:
        """
        Return ``True`` if the ``key`` exists in the database, otherwise ``False``.
        """
        ...

    @abstractmethod
    def delete(self, key: bytes) -> None:
        """
        Delete the given ``key`` from the database.
        """
        ...


class AtomicWriteBatchAPI(DatabaseAPI):
    """
    The readable/writeable object returned by an atomic database when we start building
    a batch of writes to commit.

    Reads to this database will observe writes written during batching,
    but the writes will not actually persist until this object is committed.
    """


class AtomicDatabaseAPI(DatabaseAPI):
    """
    Like ``BatchDB``, but immediately write out changes if they are
    not in an ``atomic_batch()`` context.
    """

    @abstractmethod
    def atomic_batch(self) -> ContextManager[AtomicWriteBatchAPI]:
        """
        Return a :class:`~typing.ContextManager` to write an atomic batch to the
        database.
        """
        ...


class HeaderDatabaseAPI(ABC):
    """
    A class representing a database for block headers.
    """

    db: AtomicDatabaseAPI

    @abstractmethod
    def __init__(self, db: AtomicDatabaseAPI) -> None:
        """
        Instantiate the database from an :class:`~eth.abc.AtomicDatabaseAPI`.
        """
        ...

    @abstractmethod
    def get_header_chain_gaps(self) -> ChainGaps:
        """
        Return information about gaps in the chain of headers. This consists of an
        ordered sequence of block ranges describing the integrity of the chain. Each
        block range describes a missing segment in the chain and each range is defined
        with inclusive boundaries, meaning the first value describes the first missing
        block of that segment and the second value describes the last missing block
        of the segment.

        In addition to the sequences of block ranges a block number is included that
        indicates the number of the first header that is known to be missing at the
        very tip of the chain.
        """

    #
    # Canonical Chain API
    #
    @abstractmethod
    def get_canonical_block_hash(self, block_number: BlockNumber) -> Hash32:
        """
        Return the block hash for the canonical block at the given number.

        Raise ``BlockNotFound`` if there's no block header with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(
        self, block_number: BlockNumber
    ) -> BlockHeaderAPI:
        """
        Return the block header with the given number in the canonical chain.

        Raise ``HeaderNotFound`` if there's no block header with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_head(self) -> BlockHeaderAPI:
        """
        Return the current block header at the head of the chain.
        """
        ...

    #
    # Header API
    #
    @abstractmethod
    def get_block_header_by_hash(self, block_hash: Hash32) -> BlockHeaderAPI:
        """
        Return the block header for the given ``block_hash``.
        Raise ``HeaderNotFound`` if no header with the given ``block_hash`` exists
        in the database.
        """
        ...

    @abstractmethod
    def get_score(self, block_hash: Hash32) -> int:
        """
        Return the score for the given ``block_hash``.
        """
        ...

    @abstractmethod
    def header_exists(self, block_hash: Hash32) -> bool:
        """
        Return ``True`` if the ``block_hash`` exists in the database,
        otherwise ``False``.
        """
        ...

    @abstractmethod
    def persist_checkpoint_header(self, header: BlockHeaderAPI, score: int) -> None:
        """
        Persist a checkpoint header with a trusted score. Persisting the checkpoint
        header automatically sets it as the new canonical head.
        """
        ...

    @abstractmethod
    def persist_header(
        self, header: BlockHeaderAPI
    ) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]]:
        """
        Persist the ``header`` in the database.
        Return two iterable of headers, the first containing the new canonical header,
        the second containing the old canonical headers
        """
        ...

    @abstractmethod
    def persist_header_chain(
        self,
        headers: Sequence[BlockHeaderAPI],
        genesis_parent_hash: Hash32 = None,
    ) -> Tuple[Tuple[BlockHeaderAPI, ...], Tuple[BlockHeaderAPI, ...]]:
        """
        Persist a chain of headers in the database.
        Return two iterable of headers, the first containing the new canonical headers,
        the second containing the old canonical headers

        :param genesis_parent_hash: *optional* parent hash of the block that is treated
            as genesis. Providing a ``genesis_parent_hash`` allows storage of headers
            that aren't (yet) connected back to the true genesis header.

        """
        ...


class ChainDatabaseAPI(HeaderDatabaseAPI):
    """
    A class representing a database for chain data. This class is derived from
    :class:`~eth.abc.HeaderDatabaseAPI`.
    """

    #
    # Header API
    #
    @abstractmethod
    def get_block_uncles(self, uncles_hash: Hash32) -> Tuple[BlockHeaderAPI, ...]:
        """
        Return an iterable of uncle headers specified by the given ``uncles_hash``
        """
        ...

    #
    # Block API
    #
    @abstractmethod
    def persist_block(
        self,
        block: BlockAPI,
        genesis_parent_hash: Hash32 = None,
    ) -> Tuple[Tuple[Hash32, ...], Tuple[Hash32, ...]]:
        """
        Persist the given block's header and uncles.

        :param block: the block that gets persisted
        :param genesis_parent_hash: *optional* parent hash of the header that is treated
            as genesis. Providing a ``genesis_parent_hash`` allows storage of blocks
            that aren't (yet) connected back to the true genesis header.

        .. warning::
            This API assumes all block transactions have been persisted already. Use
            :meth:`eth.abc.ChainDatabaseAPI.persist_unexecuted_block` to persist blocks
            that were not executed.
        """
        ...

    @abstractmethod
    def persist_unexecuted_block(
        self,
        block: BlockAPI,
        receipts: Tuple[ReceiptAPI, ...],
        genesis_parent_hash: Hash32 = None,
    ) -> Tuple[Tuple[Hash32, ...], Tuple[Hash32, ...]]:
        """
        Persist the given block's header, uncles, transactions, and receipts. Does
        **not** validate if state transitions are valid.

        :param block: the block that gets persisted
        :param receipts: the receipts for the given block
        :param genesis_parent_hash: *optional* parent hash of the header that is treated
            as genesis. Providing a ``genesis_parent_hash`` allows storage of blocks
            that aren't (yet) connected back to the true genesis header.

        This API should be used to persist blocks that the EVM does not execute but
        which it stores to make them available. It ensures to persist receipts and
        transactions which :meth:`eth.abc.ChainDatabaseAPI.persist_block` in contrast
        assumes to be persisted separately.
        """

    @abstractmethod
    def persist_uncles(self, uncles: Tuple[BlockHeaderAPI]) -> Hash32:
        """
        Persist the list of uncles to the database.

        Return the uncles hash.
        """
        ...

    #
    # Transaction API
    #
    @abstractmethod
    def add_receipt(
        self, block_header: BlockHeaderAPI, index_key: int, receipt: ReceiptAPI
    ) -> Hash32:
        """
        Add the given receipt to the provided block header.

        Return the updated `receipts_root` for updated block header.
        """
        ...

    @abstractmethod
    def add_transaction(
        self,
        block_header: BlockHeaderAPI,
        index_key: int,
        transaction: SignedTransactionAPI,
    ) -> Hash32:
        """
        Add the given transaction to the provided block header.

        Return the updated `transactions_root` for updated block header.
        """
        ...

    @abstractmethod
    def get_block_transactions(
        self,
        block_header: BlockHeaderAPI,
        transaction_decoder: Type[TransactionDecoderAPI],
    ) -> Tuple[SignedTransactionAPI, ...]:
        """
        Return an iterable of transactions for the block speficied by the
        given block header.
        """
        ...

    @abstractmethod
    def get_block_transaction_hashes(
        self, block_header: BlockHeaderAPI
    ) -> Tuple[Hash32, ...]:
       