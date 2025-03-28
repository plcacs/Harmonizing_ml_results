from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, ContextManager, Dict, FrozenSet, Hashable, Iterable, Iterator, List, MutableMapping, NamedTuple, Optional, Sequence, Tuple, Type, TypeVar, Union
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
    def hash(self):
        """
        Return the hash of the block header.
        """
        ...

    @property
    @abstractmethod
    def mining_hash(self):
        """
        Return the mining hash of the block header.
        """
        ...

    @property
    @abstractmethod
    def hex_hash(self):
        """
        Return the hash as a hex string.
        """
        ...

    @property
    @abstractmethod
    def is_genesis(self):
        """
        Return ``True`` if this header represents the genesis block of the chain,
        otherwise ``False``.
        """
        ...

    @abstractmethod
    def build_changeset(self, *args: Any, **kwargs: Any):
        """
        Open a changeset to modify the header.
        """
        ...

    @abstractmethod
    def as_dict(self):
        """
        Return a dictionary representation of the header.
        """
        ...

    @property
    @abstractmethod
    def base_fee_per_gas(self):
        """
        Return the base fee per gas of the block.

        Set to None in pre-EIP-1559 (London) header.
        """
        ...

    @property
    @abstractmethod
    def withdrawals_root(self):
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
    def deserialize(cls, encoded):
        """
        Extract a header from an encoded RLP object.

        This method is used by rlp.decode(..., sedes=TransactionBuilderAPI).
        """
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj):
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

    @abstractmethod
    def copy(self, *args: Any, **kwargs: Any):
        """
        Return a copy of the header, optionally overwriting any of its properties.
        """
        ...

    @property
    @abstractmethod
    def parent_beacon_block_root(self):
        """
        Return the hash of the parent beacon block.
        """
        ...

    @property
    @abstractmethod
    def blob_gas_used(self):
        """
        Return blob gas used.
        """
        ...

    @property
    @abstractmethod
    def excess_blob_gas(self):
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
    def bloomables(self):
        ...

class ReceiptAPI(ABC):
    """
    A class to define a receipt to capture the outcome of a transaction.
    """

    @property
    @abstractmethod
    def state_root(self):
        ...

    @property
    @abstractmethod
    def gas_used(self):
        ...

    @property
    @abstractmethod
    def bloom(self):
        ...

    @property
    @abstractmethod
    def logs(self):
        ...

    @property
    @abstractmethod
    def bloom_filter(self):
        ...

    def copy(self, *args: Any, **kwargs: Any):
        """
        Return a copy of the receipt, optionally overwriting any of its properties.
        """
        ...

    @abstractmethod
    def encode(self):
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
    def decode(cls, encoded):
        """
        This decodes a receipt that is encoded to either a typed
        receipt, a legacy receipt, or the body of a typed receipt. It assumes
        that typed receipts are *not* rlp-encoded first.

        If dealing with an object that is always rlp encoded, then use this instead:

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
    RLP. In legacy transactions, this happens using rlp.Serializeable.

    Some VMs support multiple distinct transaction types. In that case, the
    builder is responsible for dispatching on the different types.
    """

    @classmethod
    @abstractmethod
    def deserialize(cls, encoded):
        """
        Extract a receipt from an encoded RLP object.

        This method is used by rlp.decode(..., sedes=ReceiptBuilderAPI).
        """
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj):
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
    def validate(self):
        """
        Hook called during instantiation to ensure that all transaction
        parameters pass validation rules.
        """
        ...

    @property
    @abstractmethod
    def intrinsic_gas(self):
        """
        Convenience property for the return value of `get_intrinsic_gas`
        """
        ...

    @abstractmethod
    def get_intrinsic_gas(self):
        """
        Return the intrinsic gas for the transaction which is defined as the amount of
        gas that is needed before any code runs.
        """
        ...

    @abstractmethod
    def gas_used_by(self, computation):
        """
        Return the gas used by the given computation. In Frontier,
        for example, this is sum of the intrinsic cost and the gas used
        during computation.
        """
        ...

    @abstractmethod
    def copy(self, **overrides: Any):
        """
        Return a copy of the transaction.
        """
        ...

    @property
    @abstractmethod
    def access_list(self):
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
    def nonce(self):
        ...

    @property
    @abstractmethod
    def gas_price(self):
        """
        Will raise :class:`AttributeError` if get or set on a 1559 transaction.
        """
        ...

    @property
    @abstractmethod
    def max_fee_per_gas(self):
        """
        Will default to gas_price if this is a pre-1559 transaction.
        """
        ...

    @property
    @abstractmethod
    def max_priority_fee_per_gas(self):
        """
        Will default to gas_price if this is a pre-1559 transaction.
        """
        ...

    @property
    @abstractmethod
    def gas(self):
        ...

    @property
    @abstractmethod
    def to(self):
        ...

    @property
    @abstractmethod
    def value(self):
        ...

    @property
    @abstractmethod
    def data(self):
        ...

    @property
    @abstractmethod
    def r(self):
        ...

    @property
    @abstractmethod
    def s(self):
        ...

    @property
    @abstractmethod
    def hash(self):
        """
        Return the hash of the transaction.
        """
        ...

    @property
    @abstractmethod
    def chain_id(self):
        ...

    @property
    @abstractmethod
    def max_fee_per_blob_gas(self):
        ...

    @property
    @abstractmethod
    def blob_versioned_hashes(self):
        ...

class LegacyTransactionFieldsAPI(TransactionFieldsAPI):

    @property
    @abstractmethod
    def v(self):
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

    @abstractmethod
    def as_signed_transaction(self, private_key, chain_id=None):
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
    def decode(cls, encoded):
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
    def deserialize(cls, encoded):
        """
        Extract a transaction from an encoded RLP object.

        This method is used by rlp.decode(..., sedes=TransactionBuilderAPI).
        """
        ...

    @classmethod
    @abstractmethod
    def serialize(cls, obj):
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
    def create_unsigned_transaction(cls, *, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes):
        """
        Create an unsigned transaction.
        """
        ...

    @classmethod
    @abstractmethod
    def new_transaction(cls, nonce, gas_price, gas, to, value, data, v, r, s):
        """
        Create a signed transaction.
        """
        ...

class SignedTransactionAPI(BaseTransactionAPI, TransactionFieldsAPI):

    def __init__(self, *args: Any, **kwargs: Any):
        ...
    '\n    A class representing a transaction that was signed with a private key.\n    '

    @property
    @abstractmethod
    def sender(self):
        """
        Convenience and performance property for the return value of `get_sender`
        """
        ...

    @property
    @abstractmethod
    def y_parity(self):
        """
        The bit used to disambiguate elliptic curve signatures.

        The only values this method will return are 0 or 1.
        """
        ...
    type_id: Optional[int]
    '\n    The type of EIP-2718 transaction\n\n    Each EIP-2718 transaction includes a type id (which is the leading\n    byte, as encoded).\n\n    If this transaction is a legacy transaction, that it has no type. Then,\n    type_id will be None.\n    '

    @abstractmethod
    def validate(self):
        """
        Hook called during instantiation to ensure that all transaction
        parameters pass validation rules.
        """
        ...

    @property
    @abstractmethod
    def is_signature_valid(self):
        """
        Return ``True`` if the signature is valid, otherwise ``False``.
        """
        ...

    @abstractmethod
    def check_signature_validity(self):
        """
        Check if the signature is valid. Raise a ``ValidationError`` if the signature
        is invalid.
        """
        ...

    @abstractmethod
    def get_sender(self):
        """
        Get the 20-byte address which sent this transaction.

        This can be a slow operation. ``transaction.sender`` is always preferred.
        """
        ...

    @abstractmethod
    def get_message_for_signing(self):
        """
        Return the bytestring that should be signed in order to create a signed
        transaction.
        """
        ...

    def as_dict(self):
        """
        Return a dictionary representation of the transaction.
        """
        ...

    @abstractmethod
    def make_receipt(self, status, gas_used, log_entries):
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
    def encode(self):
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
    def index(self):
        """
        A monotonically increasing index, starting from 0, that increments by 1 per
        withdrawal to uniquely identify each withdrawal.
        """
        ...

    @property
    @abstractmethod
    def validator_index(self):
        """
        The index for the validator on the consensus layer the withdrawal corresponds
        to.
        """
        ...

    @property
    @abstractmethod
    def address(self):
        """
        The recipient address for the withdrawn ether.
        """
        ...

    @property
    @abstractmethod
    def amount(self):
        """
        The nonzero amount of ether to withdraw, given in gwei (10**9 wei).
        """
        ...

    @property
    @abstractmethod
    def hash(self):
        """
        Return the hash of the withdrawal.
        """
        ...

    @abstractmethod
    def validate(self):
        """
        Validate withdrawal fields.
        """
        ...

    @abstractmethod
    def encode(self):
        """
        Return the encoded withdrawal.
        """
        ...

class BlockAPI(ABC):
    """
    A class to define a block.
    """
    header: BlockHeaderAPI
    transactions: Tuple['SignedTransactionAPI', ...]
    uncles: Tuple[BlockHeaderAPI, ...]
    withdrawals: Tuple[WithdrawalAPI, ...]
    transaction_builder: Type['TransactionBuilderAPI'] = None
    receipt_builder: Type['ReceiptBuilderAPI'] = None

    @abstractmethod
    def __init__(self, header, transactions, uncles, withdrawals=None):
        ...

    @classmethod
    @abstractmethod
    def get_transaction_builder(cls):
        """
        Return the transaction builder for the block.
        """
        ...

    @classmethod
    @abstractmethod
    def get_receipt_builder(cls):
        """
        Return the receipt builder for the block.
        """
        ...

    @classmethod
    @abstractmethod
    def from_header(cls, header, chaindb):
        """
        Instantiate a block from the given ``header`` and the ``chaindb``.
        """
        ...

    @abstractmethod
    def get_receipts(self, chaindb):
        """
        Fetch the receipts for this block from the given ``chaindb``.
        """
        ...

    @property
    @abstractmethod
    def hash(self):
        """
        Return the hash of the block.
        """
        ...

    @property
    @abstractmethod
    def number(self):
        """
        Return the number of the block.
        """
        ...

    @property
    @abstractmethod
    def is_genesis(self):
        """
        Return ``True`` if this block represents the genesis block of the chain,
        otherwise ``False``.
        """
        ...

    def copy(self, *args: Any, **kwargs: Any):
        """
        Return a copy of the block, optionally overwriting any of its properties.
        """
        ...

class MetaWitnessAPI(ABC):

    @property
    @abstractmethod
    def hashes(self):
        ...

    @property
    @abstractmethod
    def accounts_queried(self):
        ...

    @property
    @abstractmethod
    def account_bytecodes_queried(self):
        ...

    @abstractmethod
    def get_slots_queried(self, address):
        ...

    @property
    @abstractmethod
    def total_slots_queried(self):
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
    def make_header_chain_gaps_lookup_key():
        """
        Return the lookup key to retrieve the header chain integrity info from the
        database.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_canonical_head_hash_lookup_key():
        """
        Return the lookup key to retrieve the canonical head from the database.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_block_number_to_hash_lookup_key(block_number):
        """
        Return the lookup key to retrieve a block hash from a block number.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_block_hash_to_score_lookup_key(block_hash):
        """
        Return the lookup key to retrieve the score from a block hash.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_transaction_hash_to_block_lookup_key(transaction_hash):
        """
        Return the lookup key to retrieve a transaction key from a transaction hash.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_withdrawal_hash_to_block_lookup_key(withdrawal_hash):
        """
        Return the lookup key to retrieve a withdrawal key from a withdrawal hash.
        """
        ...

class DatabaseAPI(MutableMapping[bytes, bytes], ABC):
    """
    A class representing a database.
    """

    @abstractmethod
    def set(self, key, value):
        """
        Assign the ``value`` to the ``key``.
        """
        ...

    @abstractmethod
    def exists(self, key):
        """
        Return ``True`` if the ``key`` exists in the database, otherwise ``False``.
        """
        ...

    @abstractmethod
    def delete(self, key):
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
    def atomic_batch(self):
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
    def __init__(self, db):
        """
        Instantiate the database from an :class:`~eth.abc.AtomicDatabaseAPI`.
        """
        ...

    @abstractmethod
    def get_header_chain_gaps(self):
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
        ...

    @abstractmethod
    def get_canonical_block_hash(self, block_number):
        """
        Return the block hash for the canonical block at the given number.

        Raise ``BlockNotFound`` if there's no block header with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number):
        """
        Return the block header with the given number in the canonical chain.

        Raise ``HeaderNotFound`` if there's no block header with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_head(self):
        """
        Return the current block header at the head of the chain.
        """
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash):
        """
        Return the block header for the given ``block_hash``.
        Raise ``HeaderNotFound`` if no header with the given ``block_hash`` exists
        in the database.
        """
        ...

    @abstractmethod
    def get_score(self, block_hash):
        """
        Return the score for the given ``block_hash``.
        """
        ...

    @abstractmethod
    def header_exists(self, block_hash):
        """
        Return ``True`` if the ``block_hash`` exists in the database,
        otherwise ``False``.
        """
        ...

    @abstractmethod
    def persist_checkpoint_header(self, header, score):
        """
        Persist a checkpoint header with a trusted score. Persisting the checkpoint
        header automatically sets it as the new canonical head.
        """
        ...

    @abstractmethod
    def persist_header(self, header):
        """
        Persist the ``header`` in the database.
        Return two iterable of headers, the first containing the new canonical header,
        the second containing the old canonical headers
        """
        ...

    @abstractmethod
    def persist_header_chain(self, headers, genesis_parent_hash=None):
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

    @abstractmethod
    def get_block_uncles(self, uncles_hash):
        """
        Return an iterable of uncle headers specified by the given ``uncles_hash``
        """
        ...

    @abstractmethod
    def persist_block(self, block, genesis_parent_hash=None):
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
    def persist_unexecuted_block(self, block, receipts, genesis_parent_hash=None):
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
    def persist_uncles(self, uncles):
        """
        Persist the list of uncles to the database.

        Return the uncles hash.
        """
        ...

    @abstractmethod
    def add_receipt(self, block_header, index_key, receipt):
        """
        Add the given receipt to the provided block header.

        Return the updated `receipts_root` for updated block header.
        """
        ...

    @abstractmethod
    def add_transaction(self, block_header, index_key, transaction):
        """
        Add the given transaction to the provided block header.

        Return the updated `transactions_root` for updated block header.
        """
        ...

    @abstractmethod
    def get_block_transactions(self, block_header, transaction_decoder):
        """
        Return an iterable of transactions for the block speficied by the
        given block header.
        """
        ...

    @abstractmethod
    def get_block_transaction_hashes(self, block_header):
        """
        Return a tuple cointaining the hashes of the transactions of the
        given ``block_header``.
        """
        ...

    @abstractmethod
    def get_receipt_by_index(self, block_number, receipt_index, receipt_decoder):
        """
        Return the receipt of the transaction at specified index
        for the block header obtained by the specified block number
        """
        ...

    @abstractmethod
    def get_receipts(self, header, receipt_decoder):
        """
        Return a tuple of receipts for the block specified by the given
        block header.
        """
        ...

    @abstractmethod
    def get_transaction_by_index(self, block_number, transaction_index, transaction_decoder):
        """
        Return the transaction at the specified `transaction_index` from the
        block specified by `block_number` from the canonical chain.

        Raise ``TransactionNotFound`` if no block with that ``block_number`` exists.
        """
        ...

    @abstractmethod
    def get_transaction_index(self, transaction_hash):
        """
        Return a 2-tuple of (block_number, transaction_index) indicating which
        block the given transaction can be found in and at what index in the
        block transactions.

        Raise ``TransactionNotFound`` if the transaction_hash is not found in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_block_withdrawals(self, block_header):
        """
        Return an iterable of withdrawals for the block specified by the
        given block header.
        """
        ...

    @abstractmethod
    def exists(self, key):
        """
        Return ``True`` if the given key exists in the database.
        """
        ...

    @abstractmethod
    def get(self, key):
        """
        Return the value for the given key or a KeyError if it doesn't exist in the
        database.
        """
        ...

    @abstractmethod
    def persist_trie_data_dict(self, trie_data_dict):
        """
        Store raw trie data to db from a dict
        """
        ...

class GasMeterAPI(ABC):
    """
    A class to define a gas meter.
    """
    start_gas: int
    gas_refunded: int
    gas_remaining: int

    @abstractmethod
    def consume_gas(self, amount, reason):
        """
        Consume ``amount`` of gas for a defined ``reason``.
        """
        ...

    @abstractmethod
    def return_gas(self, amount):
        """
        Return ``amount`` of gas.
        """
        ...

    @abstractmethod
    def refund_gas(self, amount):
        """
        Refund ``amount`` of gas.
        """
        ...

class MessageAPI(ABC):
    """
    A message for VM computation.
    """
    code: bytes
    _code_address: Address
    create_address: Address
    data: BytesOrView
    depth: int
    gas: int
    is_static: bool
    sender: Address
    should_transfer_value: bool
    _storage_address: Address
    to: Address
    value: int
    __slots__ = ['code', '_code_address', 'create_address', 'data', 'depth', 'gas', 'is_static', 'sender', 'should_transfer_value', '_storage_address', 'to', 'value']

    @property
    @abstractmethod
    def code_address(self):
        ...

    @property
    @abstractmethod
    def storage_address(self):
        ...

    @property
    @abstractmethod
    def is_create(self):
        ...

    @property
    @abstractmethod
    def data_as_bytes(self):
        ...

class OpcodeAPI(ABC):
    """
    A class representing an opcode.
    """
    mnemonic: str

    @abstractmethod
    def __call__(self, computation):
        """
        Execute the logic of the opcode.
        """
        ...

    @classmethod
    @abstractmethod
    def as_opcode(cls, logic_fn, mnemonic, gas_cost):
        """
        Class factory method for turning vanilla functions into Opcodes.
        """
        ...

class ChainContextAPI(ABC):
    """
    Immutable chain context information that remains constant over the VM execution.
    """

    @abstractmethod
    def __init__(self, chain_id=None):
        """
        Initialize the chain context with the given ``chain_id``.
        """
        ...

    @property
    @abstractmethod
    def chain_id(self):
        """
        Return the chain id of the chain context.
        """
        ...

class TransactionContextAPI(ABC):
    """
    Immutable transaction context information that remains constant over the
    VM execution.
    """

    @abstractmethod
    def __init__(self, gas_price, origin):
        """
        Initialize the transaction context from the given ``gas_price`` and
        ``origin`` address.
        """
        ...

    @abstractmethod
    def get_next_log_counter(self):
        """
        Increment and return the log counter.
        """
        ...

    @property
    @abstractmethod
    def gas_price(self):
        """
        Return the gas price of the transaction context.
        """
        ...

    @property
    @abstractmethod
    def origin(self):
        """
        Return the origin of the transaction context.
        """
        ...

    @property
    @abstractmethod
    def blob_versioned_hashes(self):
        """
        Return the blob versioned hashes of the transaction context.
        """
        ...

class MemoryAPI(ABC):
    """
    A class representing the memory of the :class:`~eth.abc.VirtualMachineAPI`.
    """

    @abstractmethod
    def extend(self, start_position, size):
        """
        Extend the memory from the given ``start_position`` to the provided ``size``.
        """
        ...

    @abstractmethod
    def __len__(self):
        """
        Return the length of the memory.
        """
        ...

    @abstractmethod
    def write(self, start_position, size, value):
        """
        Write `value` into memory.
        """
        ...

    @abstractmethod
    def read(self, start_position, size):
        """
        Return a view into the memory
        """
        ...

    @abstractmethod
    def read_bytes(self, start_position, size):
        """
        Read a value from memory and return a fresh bytes instance
        """
        ...

    @abstractmethod
    def copy(self, destination, source, length):
        """
        Copy bytes of memory with size ``length`` from ``source`` to ``destination``
        """
        ...

class StackAPI(ABC):
    """
    A class representing the stack of the :class:`~eth.abc.VirtualMachineAPI`.
    """

    @abstractmethod
    def push_int(self, value):
        """
        Push an integer item onto the stack.
        """
        ...

    @abstractmethod
    def push_bytes(self, value):
        """
        Push a bytes item onto the stack.
        """
        ...

    @abstractmethod
    def pop1_bytes(self):
        """
        Pop and return a bytes element from the stack.

        Raise `eth.exceptions.InsufficientStack` if the stack was empty.
        """
        ...

    @abstractmethod
    def pop1_int(self):
        """
        Pop and return an integer from the stack.

        Raise `eth.exceptions.InsufficientStack` if the stack was empty.
        """
        ...

    @abstractmethod
    def pop1_any(self):
        """
        Pop and return an element from the stack.
        The type of each element will be int or bytes, depending on whether it was
        pushed with push_bytes or push_int.

        Raise `eth.exceptions.InsufficientStack` if the stack was empty.
        """
        ...

    @abstractmethod
    def pop_any(self, num_items):
        """
        Pop and return a tuple of items of length ``num_items`` from the stack.
        The type of each element will be int or bytes, depending on whether it was
        pushed with stack_push_bytes or stack_push_int.

        Raise `eth.exceptions.InsufficientStack` if there are not enough items on
        the stack.

        Items are ordered with the top of the stack as the first item in the tuple.
        """
        ...

    @abstractmethod
    def pop_ints(self, num_items):
        """
        Pop and return a tuple of integers of length ``num_items`` from the stack.

        Raise `eth.exceptions.InsufficientStack` if there are not enough items on
        the stack.

        Items are ordered with the top of the stack as the first item in the tuple.
        """
        ...

    @abstractmethod
    def pop_bytes(self, num_items):
        """
        Pop and return a tuple of bytes of length ``num_items`` from the stack.

        Raise `eth.exceptions.InsufficientStack` if there are not enough items on
        the stack.

        Items are ordered with the top of the stack as the first item in the tuple.
        """
        ...

    @abstractmethod
    def swap(self, position):
        """
        Perform a SWAP operation on the stack.
        """
        ...

    @abstractmethod
    def dup(self, position):
        """
        Perform a DUP operation on the stack.
        """
        ...

class CodeStreamAPI(ABC):
    """
    A class representing a stream of EVM code.
    """
    program_counter: int

    @abstractmethod
    def read(self, size):
        """
        Read and return the code from the current position of the cursor up to ``size``.
        """
        ...

    @abstractmethod
    def __len__(self):
        """
        Return the length of the code stream.
        """
        ...

    @abstractmethod
    def __getitem__(self, index):
        """
        Return the ordinal value of the byte at the given ``index``.
        """
        ...

    @abstractmethod
    def __iter__(self):
        """
        Iterate over all ordinal values of the bytes of the code stream.
        """
        ...

    @abstractmethod
    def peek(self):
        """
        Return the ordinal value of the byte at the current program counter.
        """
        ...

    @abstractmethod
    def seek(self, program_counter):
        """
        Return a :class:`~typing.ContextManager` with the program counter
        set to ``program_counter``.
        """
        ...

    @abstractmethod
    def is_valid_opcode(self, position):
        """
        Return ``True`` if a valid opcode exists at ``position``.
        """
        ...

class StackManipulationAPI(ABC):

    @abstractmethod
    def stack_pop_ints(self, num_items):
        """
        Pop the last ``num_items`` from the stack,
        returning a tuple of their ordinal values.
        """
        ...

    @abstractmethod
    def stack_pop_bytes(self, num_items):
        """
        Pop the last ``num_items`` from the stack, returning a tuple of bytes.
        """
        ...

    @abstractmethod
    def stack_pop_any(self, num_items):
        """
        Pop the last ``num_items`` from the stack, returning a tuple with potentially
        mixed values of bytes or ordinal values of bytes.
        """
        ...

    @abstractmethod
    def stack_pop1_int(self):
        """
        Pop one item from the stack and return the ordinal value
        of the represented bytes.
        """
        ...

    @abstractmethod
    def stack_pop1_bytes(self):
        """
        Pop one item from the stack and return the value as ``bytes``.
        """
        ...

    @abstractmethod
    def stack_pop1_any(self):
        """
        Pop one item from the stack and return the value either as byte or the ordinal
        value of a byte.
        """
        ...

    @abstractmethod
    def stack_push_int(self, value):
        """
        Push ``value`` on the stack which must be a 256 bit integer.
        """
        ...

    @abstractmethod
    def stack_push_bytes(self, value):
        """
        Push ``value`` on the stack which must be a 32 byte string.
        """
        ...

class ExecutionContextAPI(ABC):
    """
    A class representing context information that remains constant over the
    execution of a block.
    """

    @property
    @abstractmethod
    def coinbase(self):
        """
        Return the coinbase address of the block.
        """
        ...

    @property
    @abstractmethod
    def timestamp(self):
        """
        Return the timestamp of the block.
        """
        ...

    @property
    @abstractmethod
    def block_number(self):
        """
        Return the number of the block.
        """
        ...

    @property
    @abstractmethod
    def difficulty(self):
        """
        Return the difficulty of the block.
        """
        ...

    @property
    @abstractmethod
    def mix_hash(self):
        """
        Return the mix hash of the block
        """
        ...

    @property
    @abstractmethod
    def gas_limit(self):
        """
        Return the gas limit of the block.
        """
        ...

    @property
    @abstractmethod
    def prev_hashes(self):
        """
        Return an iterable of block hashes that precede the block.
        """
        ...

    @property
    @abstractmethod
    def chain_id(self):
        """
        Return the id of the chain.
        """
        ...

    @property
    @abstractmethod
    def base_fee_per_gas(self):
        """
        Return the base fee per gas of the block
        """
        ...

    @property
    @abstractmethod
    def excess_blob_gas(self):
        """
        Return the excess blob gas of the block
        """
        ...

class ComputationAPI(ContextManager['ComputationAPI'], StackManipulationAPI):
    """
    The base abstract class for all execution computations.
    """
    logger: ExtendedDebugLogger
    state: 'StateAPI'
    msg: MessageAPI
    transaction_context: TransactionContextAPI
    code: CodeStreamAPI
    children: List['ComputationAPI']
    return_data: bytes = b''
    accounts_to_delete: List[Address]
    beneficiaries: List[Address]
    contracts_created: List[Address] = []
    _memory: MemoryAPI
    _stack: StackAPI
    _gas_meter: GasMeterAPI
    _error: VMError
    _output: bytes = b''
    _log_entries: List[Tuple[int, Address, Tuple[int, ...], bytes]]
    opcodes: Dict[int, OpcodeAPI]
    _precompiles: Dict[Address, Callable[['ComputationAPI'], 'ComputationAPI']]

    @abstractmethod
    def __init__(self, state, message, transaction_context):
        """
        Instantiate the computation.
        """
        ...

    @abstractmethod
    def _configure_gas_meter(self):
        """
        Configure the gas meter for the computation at class initialization.
        """
        ...

    @property
    @abstractmethod
    def is_origin_computation(self):
        """
        Return ``True`` if this computation is the outermost computation at
        ``depth == 0``.
        """
        ...

    @property
    @abstractmethod
    def is_success(self):
        """
        Return ``True`` if the computation did not result in an error.
        """
        ...

    @property
    @abstractmethod
    def is_error(self):
        """
        Return ``True`` if the computation resulted in an error.
        """
        ...

    @property
    @abstractmethod
    def error(self):
        """
        Return the :class:`~eth.exceptions.VMError` of the computation.
        Raise ``AttributeError`` if no error exists.
        """
        ...

    @error.setter
    def error(self, value):
        """
        Set an :class:`~eth.exceptions.VMError` for the computation.
        """
        raise NotImplementedError

    @abstractmethod
    def raise_if_error(self):
        """
        If there was an error during computation, raise it as an exception immediately.

        :raise VMError:
        """
        ...

    @property
    @abstractmethod
    def should_burn_gas(self):
        """
        Return ``True`` if the remaining gas should be burned.
        """
        ...

    @property
    @abstractmethod
    def should_return_gas(self):
        """
        Return ``True`` if the remaining gas should be returned.
        """
        ...

    @property
    @abstractmethod
    def should_erase_return_data(self):
        """
        Return ``True`` if the return data should be zerod out due to an error.
        """
        ...

    @abstractmethod
    def extend_memory(self, start_position, size):
        """
        Extend the size of the memory to be at minimum ``start_position + size``
        bytes in length.  Raise `eth.exceptions.OutOfGas` if there is not enough
        gas to pay for extending the memory.
        """
        ...

    @abstractmethod
    def memory_write(self, start_position, size, value):
        """
        Write ``value`` to memory at ``start_position``. Require that
        ``len(value) == size``.
        """
        ...

    @abstractmethod
    def memory_read_bytes(self, start_position, size):
        """
        Read and return ``size`` bytes from memory starting at ``start_position``.
        """
        ...

    @abstractmethod
    def memory_copy(self, destination, source, length):
        """
        Copy bytes of memory with size ``length`` from ``source`` to ``destination``
        """
        ...

    @abstractmethod
    def get_gas_meter(self):
        """
        Return the gas meter for the computation.
        """
        ...

    @abstractmethod
    def consume_gas(self, amount, reason):
        """
        Consume ``amount`` of gas from the remaining gas.
        Raise `eth.exceptions.OutOfGas` if there is not enough gas remaining.
        """
        ...

    @abstractmethod
    def return_gas(self, amount):
        """
        Return ``amount`` of gas to the available gas pool.
        """
        ...

    @abstractmethod
    def refund_gas(self, amount):
        """
        Add ``amount`` of gas to the pool of gas marked to be refunded.
        """
        ...

    @abstractmethod
    def get_gas_used(self):
        """
        Return the number of used gas.
        """
        ...

    @abstractmethod
    def get_gas_remaining(self):
        """
        Return the number of remaining gas.
        """
        ...

    @abstractmethod
    def stack_swap(self, position):
        """
        Swap the item on the top of the stack with the item at ``position``.
        """
        ...

    @abstractmethod
    def stack_dup(self, position):
        """
        Duplicate the stack item at ``position`` and pushes it onto the stack.
        """
        ...

    @property
    @abstractmethod
    def output(self):
        """
        Get the return value of the computation.
        """
        ...

    @output.setter
    def output(self, value):
        """
        Set the return value of the computation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def precompiles(self):
        """
        Return a dictionary where the keys are the addresses of precompiles and the
        values are the precompile functions.
        """
        ...

    @classmethod
    @abstractmethod
    def get_precompiles(cls):
        """
        Return a dictionary where the keys are the addresses of precompiles and the
        values are the precompile functions.
        """
        ...

    @abstractmethod
    def get_opcode_fn(self, opcode):
        """
        Return the function for the given ``opcode``.
        """
        ...

    @abstractmethod
    def prepare_child_message(self, gas, to, value, data, code, **kwargs: Any):
        """
        Helper method for creating a child computation.
        """
        ...

    @abstractmethod
    def apply_child_computation(self, child_msg):
        """
        Apply the vm message ``child_msg`` as a child computation.
        """
        ...

    @abstractmethod
    def generate_child_computation(self, child_msg):
        """
        Generate a child computation from the given ``child_msg``.
        """
        ...

    @abstractmethod
    def add_child_computation(self, child_computation):
        """
        Add the given ``child_computation``.
        """
        ...

    @abstractmethod
    def get_gas_refund(self):
        """
        Return the number of refunded gas.
        """
        ...

    @abstractmethod
    def register_account_for_deletion(self, beneficiary):
        """
        Register the address of ``beneficiary`` for deletion.
        """
        ...

    @abstractmethod
    def get_accounts_for_deletion(self):
        """
        Return a tuple of addresses that are registered for deletion.
        """
        ...

    @abstractmethod
    def get_self_destruct_beneficiaries(self):
        """
        Return a list of addresses that were beneficiaries of the self-destruct
        opcode - whether or not the contract was self-destructed, post-Cancun.
        """
        ...

    @abstractmethod
    def add_log_entry(self, account, topics, data):
        """
        Add a log entry.
        """
        ...

    @abstractmethod
    def get_raw_log_entries(self):
        """
        Return a tuple of raw log entries.
        """
        ...

    @abstractmethod
    def get_log_entries(self):
        """
        Return the log entries for this computation and its children.

        They are sorted in the same order they were emitted during the transaction
        processing, and include the sequential counter as the first element of the
        tuple representing every entry.
        """
        ...

    @classmethod
    @abstractmethod
    def apply_message(cls, state, message, transaction_context, parent_computation=None):
        """
        Execute a VM message. This is where the VM-specific call logic exists.
        """
        ...

    @classmethod
    @abstractmethod
    def apply_create_message(cls, state, message, transaction_context, parent_computation=None):
        """
        Execute a VM message to create a new contract. This is where the VM-specific
        create logic exists.
        """
        ...

    @classmethod
    @abstractmethod
    def apply_computation(cls, state, message, transaction_context):
        """
        Execute the logic within the message: Either run the precompile, or
        step through each opcode.  Generally, the only VM-specific logic is for
        each opcode as it executes.

        This should rarely be called directly, because it will skip over other
        important VM-specific logic that happens before or after the execution.

        Instead, prefer :meth:`~apply_message` or :meth:`~apply_create_message`.
        """
        ...

class AccountStorageDatabaseAPI(ABC):
    """
    Storage cache and write batch for a single account. Changes are not
    merklized until :meth:`make_storage_root` is called.
    """

    @abstractmethod
    def get(self, slot, from_journal=True):
        """
        Return the value at ``slot``. Lookups take the journal into consideration
        unless ``from_journal`` is explicitly set to ``False``.
        """
        ...

    @abstractmethod
    def set(self, slot, value):
        """
        Write ``value`` into ``slot``.
        """
        ...

    @abstractmethod
    def delete(self):
        """
        Delete the entire storage at the account.
        """
        ...

    @abstractmethod
    def record(self, checkpoint):
        """
        Record changes into the given ``checkpoint``.
        """
        ...

    @abstractmethod
    def discard(self, checkpoint):
        """
        Discard the given ``checkpoint``.
        """
        ...

    @abstractmethod
    def commit(self, checkpoint):
        """
        Collapse changes into the given ``checkpoint``.
        """
        ...

    @abstractmethod
    def lock_changes(self):
        """
        Locks in changes to storage, typically just as a transaction starts.

        This is used, for example, to look up the storage value from the start
        of the transaction, when calculating gas costs in EIP-2200: net gas metering.
        """
        ...

    @abstractmethod
    def make_storage_root(self):
        """
        Force calculation of the storage root for this account
        """
        ...

    @property
    @abstractmethod
    def has_changed_root(self):
        """
        Return ``True`` if the storage root has changed.
        """
        ...

    @abstractmethod
    def get_changed_root(self):
        """
        Return the changed root hash.
        Raise ``ValidationError`` if the root has not changed.
        """
        ...

    @abstractmethod
    def persist(self, db):
        """
        Persist all changes to the database.
        """
        ...

    @abstractmethod
    def get_accessed_slots(self):
        """
        List all the slots that had been accessed since object creation.
        """
        ...

class AccountAPI(ABC):
    """
    A class representing an Ethereum account.
    """
    nonce: int
    balance: int
    storage_root: Hash32
    code_hash: Hash32

class TransientStorageAPI(ABC):

    @abstractmethod
    def record(self, checkpoint):
        """
        Record changes into the given ``checkpoint``.
        """
        ...

    @abstractmethod
    def commit(self, snapshot):
        """
        Commit the given ``checkpoint``.
        """
        ...

    @abstractmethod
    def discard(self, snapshot):
        """
        Discard the given ``checkpoint``.
        """
        ...

    @abstractmethod
    def clear(self):
        """
        Clear the transient storage.
        """
        ...

    @abstractmethod
    def get_transient_storage(self, address, slot):
        """
        Return the transient storage for ``address`` at slot ``slot``.
        """
        ...

    @abstractmethod
    def set_transient_storage(self, address, slot, value):
        """
        Set the transient storage for ``address`` at slot ``slot``.
        """
        ...

class AccountDatabaseAPI(ABC):
    """
    A class representing a database for accounts.
    """

    @abstractmethod
    def __init__(self, db, state_root=BLANK_ROOT_HASH):
        """
        Initialize the account database.
        """
        ...

    @property
    @abstractmethod
    def state_root(self):
        """
        Return the state root hash.
        """
        ...

    @state_root.setter
    def state_root(self, value):
        """
        Force-set the state root hash.
        """
        raise NotImplementedError

    @abstractmethod
    def has_root(self, state_root):
        """
        Return ``True`` if the `state_root` exists, otherwise ``False``.
        """
        ...

    @abstractmethod
    def get_storage(self, address, slot, from_journal=True):
        """
        Return the value stored at ``slot`` for the given ``address``. Take the journal
        into consideration unless ``from_journal`` is set to ``False``.
        """
        ...

    @abstractmethod
    def set_storage(self, address, slot, value):
        """
        Write ``value`` into ``slot`` for the given ``address``.
        """
        ...

    @abstractmethod
    def delete_storage(self, address):
        """
        Delete the storage at ``address``.
        """
        ...

    @abstractmethod
    def is_storage_warm(self, address, slot):
        """
        Was the storage slot accessed during this transaction?

        See EIP-2929
        """
        ...

    @abstractmethod
    def mark_storage_warm(self, address, slot):
        """
        Mark the storage slot as accessed during this transaction.

        See EIP-2929
        """
        ...

    @abstractmethod
    def get_balance(self, address):
        """
        Return the balance at ``address``.
        """
        ...

    @abstractmethod
    def set_balance(self, address, balance):
        """
        Set ``balance`` as the new balance for ``address``.
        """
        ...

    @abstractmethod
    def get_nonce(self, address):
        """
        Return the nonce for ``address``.
        """
        ...

    @abstractmethod
    def set_nonce(self, address, nonce):
        """
        Set ``nonce`` as the new nonce for ``address``.
        """
        ...

    @abstractmethod
    def increment_nonce(self, address):
        """
        Increment the nonce for ``address``.
        """
        ...

    @abstractmethod
    def set_code(self, address, code):
        """
        Set ``code`` as the new code at ``address``.
        """
        ...

    @abstractmethod
    def get_code(self, address):
        """
        Return the code at the given ``address``.
        """
        ...

    @abstractmethod
    def get_code_hash(self, address):
        """
        Return the hash of the code at ``address``.
        """
        ...

    @abstractmethod
    def delete_code(self, address):
        """
        Delete the code at ``address``.
        """
        ...

    @abstractmethod
    def account_has_code_or_nonce(self, address):
        """
        Return ``True`` if either code or a nonce exists at ``address``.
        """
        ...

    @abstractmethod
    def delete_account(self, address):
        """
        Delete the account at ``address``.
        """
        ...

    @abstractmethod
    def account_exists(self, address):
        """
        Return ``True`` if an account exists at ``address``, otherwise ``False``.
        """
        ...

    @abstractmethod
    def touch_account(self, address):
        """
        Touch the account at ``address``.
        """
        ...

    @abstractmethod
    def account_is_empty(self, address):
        """
        Return ``True`` if an account exists at ``address``.
        """
        ...

    @abstractmethod
    def is_address_warm(self, address):
        """
        Was the account accessed during this transaction?

        See EIP-2929
        """
        ...

    @abstractmethod
    def mark_address_warm(self, address):
        """
        Mark the account as accessed during this transaction.

        See EIP-2929
        """
        ...

    @abstractmethod
    def record(self):
        """
        Create and return a new checkpoint.
        """
        ...

    @abstractmethod
    def discard(self, checkpoint):
        """
        Discard the given ``checkpoint``.
        """
        ...

    @abstractmethod
    def commit(self, checkpoint):
        """
        Collapse changes into ``checkpoint``.
        """
        ...

    @abstractmethod
    def lock_changes(self):
        """
        Locks in changes across all accounts' storage databases.

        This is typically used at the end of a transaction, to make sure that
        a revert doesn't roll back through the previous transaction, and to
        be able to look up the "original" value of any account storage, where
        "original" is the beginning of a transaction (instead of the beginning
        of a block).

        See :meth:`eth.abc.AccountStorageDatabaseAPI.lock_changes` for
        what is called on each account's storage database.
        """
        ...

    @abstractmethod
    def make_state_root(self):
        """
        Generate the state root with all the current changes in AccountDB

        Current changes include every pending change to storage, as well as all account
        changes. After generating all the required tries, the final account state root
        is returned.

        This is an expensive operation, so should be called as little as possible.
        For example, pre-Byzantium, this is called after every transaction, because we
        need the state root in each receipt. Byzantium+, we only need state roots at
        the end of the block, so we *only* call it right before persistence.

        :return: the new state root
        """
        ...

    @abstractmethod
    def persist(self):
        """
        Send changes to underlying database, including the trie state
        so that it will forever be possible to read the trie from this checkpoint.

        :meth:`make_state_root` must be explicitly called before this method.
        Otherwise persist will raise a ValidationError.
        """
        ...

class TransactionExecutorAPI(ABC):
    """
    A class providing APIs to execute transactions on VM state.
    """

    @abstractmethod
    def __init__(self, vm_state):
        """
        Initialize the executor from the given ``vm_state``.
        """
        ...

    @abstractmethod
    def __call__(self, transaction):
        """
        Execute the ``transaction`` and return a :class:`eth.abc.ComputationAPI`.
        """
        ...

    @abstractmethod
    def validate_transaction(self, transaction):
        """
        Validate the given ``transaction``.
        Raise a ``ValidationError`` if the transaction is invalid.
        """
        ...

    @abstractmethod
    def build_evm_message(self, transaction):
        """
        Build and return a :class:`~eth.abc.MessageAPI` from the given ``transaction``.
        """
        ...

    @abstractmethod
    def build_computation(self, message, transaction):
        """
        Apply the ``message`` to the VM and use the given ``transaction`` to
        retrieve the context from.
        """
        ...

    @abstractmethod
    def finalize_computation(self, transaction, computation):
        """
        Finalize the ``transaction``.
        """
        ...

    @abstractmethod
    def calc_data_fee(self, transaction):
        """
        For Cancun and later, calculate the data fee for a transaction.
        """
        ...

class ConfigurableAPI(ABC):
    """
    A class providing inline subclassing.
    """

    @classmethod
    @abstractmethod
    def configure(cls, __name__=None, **overrides: Any):
        ...

class StateAPI(ConfigurableAPI):
    """
    The base class that encapsulates all of the various moving parts related to
    the state of the VM during execution.
    Each :class:`~eth.abc.VirtualMachineAPI` must be configured with a subclass of the
    :class:`~eth.abc.StateAPI`.

      .. note::

        Each :class:`~eth.abc.StateAPI` class must be configured with:

        - ``computation_class``: The :class:`~eth.abc.ComputationAPI` class for
          vm execution.
        - ``transaction_context_class``: The :class:`~eth.abc.TransactionContextAPI`
          class for vm execution.
    """
    execution_context: ExecutionContextAPI
    computation_class: Type['ComputationAPI']
    transaction_context_class: Type['TransactionContextAPI']
    account_db_class: Type['AccountDatabaseAPI']
    transaction_executor_class: Optional[Type['TransactionExecutorAPI']] = None

    @abstractmethod
    def __init__(self, db, execution_context, state_root):
        """
        Initialize the state.
        """
        ...

    @property
    @abstractmethod
    def logger(self):
        """
        Return the logger.
        """
        ...

    @property
    @abstractmethod
    def coinbase(self):
        """
        Return the current ``coinbase`` from the current :attr:`~execution_context`
        """
        ...

    @property
    @abstractmethod
    def timestamp(self):
        """
        Return the current ``timestamp`` from the current :attr:`~execution_context`
        """
        ...

    @property
    @abstractmethod
    def block_number(self):
        """
        Return the current ``block_number`` from the current :attr:`~execution_context`
        """
        ...

    @property
    @abstractmethod
    def difficulty(self):
        """
        Return the current ``difficulty`` from the current :attr:`~execution_context`
        """
        ...

    @property
    @abstractmethod
    def mix_hash(self):
        """
        Return the current ``mix_hash`` from the current :attr:`~execution_context`
        """
        ...

    @property
    @abstractmethod
    def gas_limit(self):
        """
        Return the current ``gas_limit`` from the current :attr:`~transaction_context`
        """
        ...

    @property
    @abstractmethod
    def base_fee(self):
        """
        Return the current ``base_fee`` from the current :attr:`~execution_context`

        Raises a ``NotImplementedError`` if called in an execution context
        prior to the London hard fork.
        """
        ...

    @abstractmethod
    def get_gas_price(self, transaction):
        """
        Return the gas price of the given transaction.

        Factor in the current block's base gas price, if appropriate. (See EIP-1559)
        """
        ...

    @abstractmethod
    def get_tip(self, transaction):
        """
        Return the gas price that gets allocated to the miner/validator.

        Pre-EIP-1559 that would be the full transaction gas price. After, it
        would be the tip price (potentially reduced, if the base fee is so high
        that it surpasses the transaction's maximum gas price after adding the
        tip).
        """
        ...

    @property
    @abstractmethod
    def blob_base_fee(self):
        """
        Return the current ``blob_base_fee`` from the current :attr:`~execution_context`

        Raises a ``NotImplementedError`` if called in an execution context
        prior to the Cancun hard fork.
        """
        ...

    @classmethod
    @abstractmethod
    def get_account_db_class(cls):
        """
        Return the :class:`~eth.abc.AccountDatabaseAPI` class that the
        state class uses.
        """
        ...

    @property
    @abstractmethod
    def state_root(self):
        """
        Return the current ``state_root`` from the underlying database
        """
        ...

    @abstractmethod
    def make_state_root(self):
        """
        Create and return the state root.
        """
        ...

    @abstractmethod
    def get_storage(self, address, slot, from_journal=True):
        """
        Return the storage at ``slot`` for ``address``.
        """
        ...

    @abstractmethod
    def set_storage(self, address, slot, value):
        """
        Write ``value`` to the given ``slot`` at ``address``.
        """
        ...

    @abstractmethod
    def delete_storage(self, address):
        """
        Delete the storage at the given ``address``.
        """
        ...

    @abstractmethod
    def delete_account(self, address):
        """
        Delete the account at the given ``address``.
        """
        ...

    @abstractmethod
    def get_balance(self, address):
        """
        Return the balance for the account at ``address``.
        """
        ...

    @abstractmethod
    def set_balance(self, address, balance):
        """
        Set ``balance`` to the balance at ``address``.
        """
        ...

    @abstractmethod
    def delta_balance(self, address, delta):
        """
        Apply ``delta`` to the balance at ``address``.
        """
        ...

    @abstractmethod
    def get_nonce(self, address):
        """
        Return the nonce at ``address``.
        """
        ...

    @abstractmethod
    def set_nonce(self, address, nonce):
        """
        Set ``nonce`` as the new nonce at ``address``.
        """
        ...

    @abstractmethod
    def increment_nonce(self, address):
        """
        Increment the nonce at ``address``.
        """
        ...

    @abstractmethod
    def get_code(self, address):
        """
        Return the code at ``address``.
        """
        ...

    @abstractmethod
    def set_code(self, address, code):
        """
        Set ``code`` as the new code at ``address``.
        """
        ...

    @abstractmethod
    def get_code_hash(self, address):
        """
        Return the hash of the code at ``address``.
        """
        ...

    @abstractmethod
    def delete_code(self, address):
        """
        Delete the code at ``address``.
        """
        ...

    @abstractmethod
    def has_code_or_nonce(self, address):
        """
        Return ``True`` if either a nonce or code exists at the given ``address``.
        """
        ...

    @abstractmethod
    def account_exists(self, address):
        """
        Return ``True`` if an account exists at ``address``.
        """
        ...

    @abstractmethod
    def touch_account(self, address):
        """
        Touch the account at the given ``address``.
        """
        ...

    @abstractmethod
    def account_is_empty(self, address):
        """
        Return ``True`` if the account at ``address`` is empty, otherwise ``False``.
        """
        ...

    @abstractmethod
    def is_storage_warm(self, address, slot):
        """
        Was the storage slot accessed during this transaction?

        See EIP-2929
        """
        ...

    @abstractmethod
    def mark_storage_warm(self, address, slot):
        """
        Mark the storage slot as accessed during this transaction.

        See EIP-2929
        """
        ...

    @abstractmethod
    def is_address_warm(self, address):
        """
        Was the account accessed during this transaction?

        See EIP-2929
        """
        ...

    @abstractmethod
    def mark_address_warm(self, address):
        """
        Mark the account as accessed during this transaction.

        See EIP-2929
        """
        ...

    @abstractmethod
    def get_transient_storage(self, address, slot):
        """
        Return the transient storage for ``address`` at slot ``slot``.
        """
        ...

    @abstractmethod
    def set_transient_storage(self, address, slot, value):
        """
        Set the transient storage for ``address`` at slot ``slot``.
        """
        ...

    @abstractmethod
    def clear_transient_storage(self):
        """
        Clear the transient storage. Should be done at the start of every transaction
        """
        ...

    @abstractmethod
    def snapshot(self):
        """
        Perform a full snapshot of the current state.

        Snapshots are a combination of the :attr:`~state_root` at the time of the
        snapshot and the checkpoint from the journaled DB.
        """
        ...

    @abstractmethod
    def revert(self, snapshot):
        """
        Revert the VM to the state at the snapshot
        """
        ...

    @abstractmethod
    def commit(self, snapshot):
        """
        Commit the journal to the point where the snapshot was taken.  This
        merges in any changes that were recorded since the snapshot.
        """
        ...

    @abstractmethod
    def lock_changes(self):
        """
        Locks in all changes to state, typically just as a transaction starts.

        This is used, for example, to look up the storage value from the start
        of the transaction, when calculating gas costs in EIP-2200: net gas metering.
        """
        ...

    @abstractmethod
    def persist(self):
        """
        Persist the current state to the database.
        """
        ...

    @abstractmethod
    def get_ancestor_hash(self, block_number):
        """
        Return the hash for the ancestor block with number ``block_number``.
        Return the empty bytestring ``b''`` if the block number is outside of the
        range of available block numbers (typically the last 255 blocks).
        """
        ...

    @abstractmethod
    def get_computation(self, message, transaction_context):
        """
        Return a computation instance for the given `message` and `transaction_context`
        """
        ...

    @classmethod
    @abstractmethod
    def get_transaction_context_class(cls):
        """
        Return the :class:`~eth.vm.transaction_context.BaseTransactionContext` class
        that the state class uses.
        """
        ...

    @abstractmethod
    def apply_transaction(self, transaction):
        """
        Apply transaction to the vm state

        :param transaction: the transaction to apply
        :return: the computation
        """
        ...

    @abstractmethod
    def get_transaction_executor(self):
        """
        Return the transaction executor.
        """
        ...

    @abstractmethod
    def costless_execute_transaction(self, transaction):
        """
        Execute the given ``transaction`` with a gas price of ``0``.
        """
        ...

    @abstractmethod
    def override_transaction_context(self, gas_price):
        """
        Return a :class:`~typing.ContextManager` that overwrites the current
        transaction context, applying the given ``gas_price``.
        """
        ...

    @abstractmethod
    def validate_transaction(self, transaction):
        """
        Validate the given ``transaction``.
        """
        ...

    @abstractmethod
    def get_transaction_context(self, transaction):
        """
        Return the :class:`~eth.abc.TransactionContextAPI` for the given ``transaction``
        """
        ...

    @abstractmethod
    def apply_withdrawal(self, withdrawal):
        """
        Apply a single withdrawal to the state.
        """
        ...

    @abstractmethod
    def apply_all_withdrawals(self, withdrawals):
        """
        Updates the state by applying all withdrawals.
        This does *not* update the current block or header of the VM.

        :param withdrawals: an iterable of all withdrawals to apply
        """
        ...

class ConsensusContextAPI(ABC):
    """
    A class representing a data context for the :class:`~eth.abc.ConsensusAPI` which is
    instantiated once per chain instance and stays in memory across VM runs.
    """

    @abstractmethod
    def __init__(self, db):
        """
        Initialize the context with a database.
        """
        ...

class ConsensusAPI(ABC):
    """
    A class encapsulating the consensus scheme to allow chains to run
    under different kind of EVM-compatible consensus mechanisms such
    as the Clique Proof of Authority scheme.
    """

    @abstractmethod
    def __init__(self, context):
        """
        Initialize the consensus api.
        """
        ...

    @abstractmethod
    def validate_seal(self, header):
        """
        Validate the seal on the given header, even if its parent is missing.
        """
        ...

    @abstractmethod
    def validate_seal_extension(self, header, parents):
        """
        Validate the seal on the given header when all parents must be present.
        Parent headers that are not yet in the database must be passed as ``parents``.
        """
        ...

    @classmethod
    @abstractmethod
    def get_fee_recipient(cls, header):
        """
        Return the address that should receive rewards for creating the block.
        """
        ...

class VirtualMachineAPI(ConfigurableAPI):
    """
    The :class:`~eth.abc.VirtualMachineAPI` class represents the Chain rules for a
    specific protocol definition such as the Frontier or Homestead network.

      .. note::

        Each :class:`~eth.abc.VirtualMachineAPI` class must be configured with:

        - ``block_class``: The :class:`~eth.abc.BlockAPI` class for blocks in this
            VM ruleset.
        - ``_state_class``: The :class:`~eth.abc.StateAPI` class used by this
            VM for execution.
    """
    fork: str
    chaindb: 'ChainDatabaseAPI'
    extra_data_max_bytes: ClassVar[int]
    consensus_class: Type[ConsensusAPI]
    consensus_context: ConsensusContextAPI

    @abstractmethod
    def __init__(self, header, chaindb, chain_context, consensus_context):
        """
        Initialize the virtual machine.
        """
        ...

    @property
    @abstractmethod
    def state(self):
        """
        Return the current state.
        """
        ...

    @classmethod
    @abstractmethod
    def build_state(cls, db, header, chain_context, previous_hashes=()):
        """
        You probably want `VM().state` instead of this.

        Occasionally, you want to build custom state against a particular
        header and DB, even if you don't have the VM initialized.
        This is a convenience method to do that.
        """
        ...

    @abstractmethod
    def get_header(self):
        """
        Return the current header.
        """
        ...

    @abstractmethod
    def get_block(self):
        """
        Return the current block.
        """
        ...

    def transaction_applied_hook(self, transaction_index, transactions, base_header, partial_header, computation, receipt):
        """
        A hook for a subclass to use as a way to note that a transaction was applied.
        This only gets triggered as part of `apply_all_transactions`, which is called
        by `block_import`.
        """

    @abstractmethod
    def apply_transaction(self, header, transaction):
        """
        Apply the transaction to the current block. This is a wrapper around
        :func:`~eth.vm.state.State.apply_transaction` with some extra
        orchestration logic.

        :param header: header of the block before application
        :param transaction: to apply
        """
        ...

    @staticmethod
    @abstractmethod
    def create_execution_context(header, prev_hashes, chain_context):
        """
        Create and return the :class:`~eth.abc.ExecutionContextAPI`` for the given
        ``header``, iterable of block hashes that precede the block and
        the ``chain_context``.
        """
        ...

    @abstractmethod
    def execute_bytecode(self, origin, gas_price, gas, to, sender, value, data, code, code_address=None):
        """
        Execute raw bytecode in the context of the current state of
        the virtual machine. Note that this skips over some of the logic
        that would normally happen during a call. Watch out for:

            - value (ether) is *not* transferred
            - state is *not* rolled back in case of an error
            - The target account is *not* necessarily created
            - others...

        For other potential surprises, check the implementation differences
        between :meth:`ComputationAPI.apply_computation` and
        :meth:`ComputationAPI.apply_message`. (depending on the VM fork)
        """
        ...

    @abstractmethod
    def apply_all_transactions(self, transactions, base_header):
        """
        Determine the results of applying all transactions to the base header.
        This does *not* update the current block or header of the VM.

        :param transactions: an iterable of all transactions to apply
        :param base_header: the starting header to apply transactions to
        :return: the final header, the receipts of each transaction, and the
            computations

        """
        ...

    def apply_all_withdrawals(self, withdrawals):
        """
        Updates the state by applying all withdrawals.
        This does *not* update the current block or header of the VM.

        :param withdrawals: an iterable of all withdrawals to apply
        """
        ...

    @abstractmethod
    def make_receipt(self, base_header, transaction, computation, state):
        """
        Generate the receipt resulting from applying the transaction.

        :param base_header: the header of the block before the transaction was applied.
        :param transaction: the transaction used to generate the receipt
        :param computation: the result of running the transaction computation
        :param state: the resulting state, after executing the computation

        :return: receipt
        """
        ...

    @abstractmethod
    def import_block(self, block):
        """
        Import the given block to the chain.
        """
        ...

    @abstractmethod
    def mine_block(self, block, *args: Any, **kwargs: Any):
        """
        Mine the given block. Proxies to self.pack_block method.
        """
        ...

    @abstractmethod
    def set_block_transactions_and_withdrawals(self, base_block, new_header, transactions, receipts, withdrawals=None):
        """
        Create a new block with the given ``transactions`` and/or ``withdrawals``.
        """
        ...

    @abstractmethod
    def finalize_block(self, block):
        """
        Perform any finalization steps like awarding the block mining reward,
        and persisting the final state root.
        """
        ...

    @abstractmethod
    def pack_block(self, block, *args: Any, **kwargs: Any):
        """
        Pack block for mining.

        :param bytes coinbase: 20-byte public address to receive block reward
        :param bytes uncles_hash: 32 bytes
        :param bytes state_root: 32 bytes
        :param bytes transaction_root: 32 bytes
        :param bytes receipt_root: 32 bytes
        :param int bloom:
        :param int gas_used:
        :param bytes extra_data: 32 bytes
        :param bytes mix_hash: 32 bytes
        :param bytes nonce: 8 bytes
        """
        ...

    @abstractmethod
    def add_receipt_to_header(self, old_header, receipt):
        """
        Apply the receipt to the old header, and return the resulting header.
        This may have storage-related side-effects. For example, pre-Byzantium,
        the state root hash is included in the receipt, and so must be stored
        into the database.
        """
        ...

    @abstractmethod
    def increment_blob_gas_used(self, old_header, transaction):
        """
        Update the header by incrementing the blob_gas_used for the transaction.
        """
        ...

    @classmethod
    @abstractmethod
    def compute_difficulty(cls, parent_header, timestamp):
        """
        Compute the difficulty for a block header.

        :param parent_header: the parent header
        :param timestamp: the timestamp of the child header
        """
        ...

    @abstractmethod
    def configure_header(self, **header_params: Any):
        """
        Setup the current header with the provided parameters.  This can be
        used to set fields like the gas limit or timestamp to value different
        than their computed defaults.
        """
        ...

    @classmethod
    @abstractmethod
    def create_header_from_parent(cls, parent_header, **header_params: Any):
        """
        Creates and initializes a new block header from the provided
        `parent_header`.
        """
        ...

    @classmethod
    @abstractmethod
    def generate_block_from_parent_header_and_coinbase(cls, parent_header, coinbase):
        """
        Generate block from parent header and coinbase.
        """
        ...

    @classmethod
    @abstractmethod
    def create_genesis_header(cls, **genesis_params: Any):
        """
        Create a genesis header using this VM's rules.

        This is equivalent to calling :meth:`create_header_from_parent`
        with ``parent_header`` set to None.
        """
        ...

    @classmethod
    @abstractmethod
    def get_block_class(cls):
        """
        Return the :class:`~eth.rlp.blocks.Block` class that this VM uses for blocks.
        """
        ...

    @staticmethod
    @abstractmethod
    def get_block_reward():
        """
        Return the amount in **wei** that should be given to a miner as a reward
        for this block.

          .. note::
            This is an abstract method that must be implemented in subclasses
        """
        ...

    @classmethod
    @abstractmethod
    def get_nephew_reward(cls):
        """
        Return the reward which should be given to the miner of the given `nephew`.

          .. note::
            This is an abstract method that must be implemented in subclasses
        """
        ...

    @classmethod
    @abstractmethod
    def get_prev_hashes(cls, last_block_hash, chaindb):
        """
        Return an iterable of block hashes that precede the block with the given
        ``last_block_hash``.
        """
        ...

    @property
    @abstractmethod
    def previous_hashes(self):
        """
        Convenience API for accessing the previous 255 block hashes.
        """
        ...

    @staticmethod
    @abstractmethod
    def get_uncle_reward(block_number, uncle):
        """
        Return the reward which should be given to the miner of the given `uncle`.

          .. note::
            This is an abstract method that must be implemented in subclasses
        """
        ...

    @abstractmethod
    def create_transaction(self, *args: Any, **kwargs: Any):
        """
        Proxy for instantiating a signed transaction for this VM.
        """
        ...

    @classmethod
    @abstractmethod
    def create_unsigned_transaction(cls, *, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes):
        """
        Proxy for instantiating an unsigned transaction for this VM.
        """
        ...

    @classmethod
    @abstractmethod
    def get_transaction_builder(cls):
        """
        Return the class that this VM uses to build and encode transactions.
        """
        ...

    @classmethod
    @abstractmethod
    def get_receipt_builder(cls):
        """
        Return the class that this VM uses to encode and decode receipts.
        """
        ...

    @classmethod
    @abstractmethod
    def validate_receipt(cls, receipt):
        """
        Validate the given ``receipt``.
        """
        ...

    @abstractmethod
    def validate_block(self, block):
        """
        Validate a block that is either being mined or imported.

        Since block validation (specifically the uncle validation) must have
        access to the ancestor blocks, this validation must occur at the Chain
        level.

        Cannot be used to validate genesis block.
        """
        ...

    @classmethod
    @abstractmethod
    def validate_header(cls, header, parent_header):
        """
        :raise eth.exceptions.ValidationError: if the header is not valid
        """
        ...

    @abstractmethod
    def validate_transaction_against_header(self, base_header, transaction):
        """
        Validate that the given transaction is valid to apply to the given header.

        :param base_header: header before applying the transaction
        :param transaction: the transaction to validate

        :raises: ValidationError if the transaction is not valid to apply
        """
        ...

    @abstractmethod
    def validate_seal(self, header):
        """
        Validate the seal on the given ``header``.
        """
        ...

    @abstractmethod
    def validate_seal_extension(self, header, parents):
        """
        Validate the seal on the given header when all parents must be present. Parent
        headers that are not yet in the database must exist in ``parents``.
        """
        ...

    @classmethod
    @abstractmethod
    def validate_uncle(cls, block, uncle, uncle_parent):
        """
        Validate the given uncle in the context of the given block.
        """
        ...

class HeaderChainAPI(ABC):
    """
    Like :class:`eth.abc.ChainAPI` but does only support headers, not entire blocks.
    """
    header: BlockHeaderAPI
    chain_id: int
    vm_configuration: Tuple[Tuple[BlockNumber, Type['VirtualMachineAPI']], ...]

    @abstractmethod
    def __init__(self, base_db, header=None):
        """
        Initialize the header chain.
        """
        ...

    @classmethod
    @abstractmethod
    def from_genesis_header(cls, base_db, genesis_header):
        """
        Initialize the chain from the genesis header.
        """
        ...

    @classmethod
    @abstractmethod
    def get_headerdb_class(cls):
        """
        Return the class which should be used for the `headerdb`
        """
        ...

    def get_canonical_block_hash(self, block_number):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def get_canonical_head(self):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def header_exists(self, block_hash):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def import_header(self, header):
        """
        Direct passthrough to `headerdb`

        Also updates the local `header` property to be the latest canonical head.

        Returns an iterable of headers representing the headers that are newly
        part of the canonical chain.

        - If the imported header is not part of the canonical chain then an
          empty tuple will be returned.
        - If the imported header simply extends the canonical chain then a
          length-1 tuple with the imported header will be returned.
        - If the header is part of a non-canonical chain which overtakes the
          current canonical chain then the returned tuple will contain the
          headers which are newly part of the canonical chain.
        """
        ...

class ChainAPI(ConfigurableAPI):
    """
    A Chain is a combination of one or more VM classes. Each VM is associated
    with a range of blocks. The Chain class acts as a wrapper around these other
    VM classes, delegating operations to the appropriate VM depending on the
    current block number.
    """
    vm_configuration: Tuple[Tuple[BlockNumber, Type['VirtualMachineAPI']], ...]
    chain_id: int
    chaindb: 'ChainDatabaseAPI'
    consensus_context_class: Type['ConsensusContextAPI']

    @classmethod
    @abstractmethod
    def get_chaindb_class(cls):
        """
        Return the class for the used :class:`~eth.abc.ChainDatabaseAPI`.
        """
        ...

    @classmethod
    @abstractmethod
    def from_genesis(cls, base_db, genesis_params, genesis_state=None):
        """
        Initialize the Chain from a genesis state.
        """
        ...

    @classmethod
    @abstractmethod
    def from_genesis_header(cls, base_db, genesis_header):
        """
        Initialize the chain from the genesis header.
        """
        ...

    @classmethod
    @abstractmethod
    def get_vm_class(cls, header):
        """
        Return the VM class for the given ``header``
        """
        ...

    @abstractmethod
    def get_vm(self, header=None):
        """
        Return the VM instance for the given ``header``.
        """
        ...

    @classmethod
    @abstractmethod
    def get_vm_class_for_block_number(cls, block_number):
        """
        Return the VM class for the given ``block_number``
        """
        ...

    @abstractmethod
    def create_header_from_parent(self, parent_header, **header_params: HeaderParams):
        """
        Passthrough helper to the VM class of the block descending from the
        given header.
        """
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash):
        """
        Return the requested block header as specified by ``block_hash``.
        Raise ``BlockNotFound`` if no block header with the given hash exists in the db.
        """
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number):
        """
        Return the block header with the given number in the canonical chain.

        Raise ``HeaderNotFound`` if there's no block header with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_head(self):
        """
        Return the block header at the canonical chain head.

        Raise ``CanonicalHeadNotFound`` if there's no head defined for the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_score(self, block_hash):
        """
        Return the difficulty score of the block with the given ``block_hash``.

        Raise ``HeaderNotFound`` if there is no matching block hash.
        """
        ...

    @abstractmethod
    def get_ancestors(self, limit, header):
        """
        Return `limit` number of ancestor blocks from the current canonical head.
        """
        ...

    @abstractmethod
    def get_block(self):
        """
        Return the current block at the tip of the chain.
        """
        ...

    @abstractmethod
    def get_block_by_hash(self, block_hash):
        """
        Return the requested block as specified by ``block_hash``.

        :raise eth.exceptions.HeaderNotFound: if the header is missing
        :raise eth.exceptions.BlockNotFound: if any part of the block body is missing
        """
        ...

    @abstractmethod
    def get_block_by_header(self, block_header):
        """
        Return the requested block as specified by the ``block_header``.

        :raise eth.exceptions.BlockNotFound: if any part of the block body is missing
        """
        ...

    @abstractmethod
    def get_canonical_block_by_number(self, block_number):
        """
        Return the block with the given ``block_number`` in the canonical chain.

        Raise ``BlockNotFound`` if no block with the given ``block_number`` exists
        in the canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_block_hash(self, block_number):
        """
        Return the block hash with the given ``block_number`` in the canonical chain.

        Raise ``BlockNotFound`` if there's no block with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def build_block_with_transactions_and_withdrawals(self, transactions, parent_header=None, withdrawals=None):
        """
        Generate a block with the provided transactions. This does *not* import
        that block into your chain. If you want this new block in your chain,
        run :meth:`~import_block` with the result block from this method.

        :param transactions: an iterable of transactions to insert into the block
        :param parent_header: parent of the new block -- or canonical head if ``None``
        :param withdrawals: an iterable of withdrawals to insert into the block
        :return: (new block, receipts, computations)
        """
        ...

    @abstractmethod
    def create_transaction(self, *args: Any, **kwargs: Any):
        """
        Passthrough helper to the current VM class.
        """
        ...

    @abstractmethod
    def create_unsigned_transaction(self, *, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes):
        """
        Passthrough helper to the current VM class.
        """
        ...

    @abstractmethod
    def get_canonical_transaction_index(self, transaction_hash):
        """
        Return a 2-tuple of (block_number, transaction_index) indicating which
        block the given transaction can be found in and at what index in the
        block transactions.

        Raise ``TransactionNotFound`` if the transaction does not exist in the canonical
        chain.
        """
        ...

    @abstractmethod
    def get_canonical_transaction(self, transaction_hash):
        """
        Return the requested transaction as specified by the ``transaction_hash``
        from the canonical chain.

        Raise ``TransactionNotFound`` if no transaction with the specified hash is
        found in the canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_transaction_by_index(self, block_number, index):
        """
        Return the requested transaction as specified by the ``block_number``
        and ``index`` from the canonical chain.

        Raise ``TransactionNotFound`` if no transaction exists at ``index`` at
        ``block_number`` in the canonical chain.
        """
        ...

    @abstractmethod
    def get_transaction_receipt(self, transaction_hash):
        """
        Return the requested receipt for the transaction as specified
        by the ``transaction_hash``.

        Raise ``ReceiptNotFound`` if no receipt for the specified
        ``transaction_hash`` is found in the canonical chain.
        """
        ...

    @abstractmethod
    def get_transaction_receipt_by_index(self, block_number, index):
        """
        Return the requested receipt for the transaction as specified by the
        ``block_number`` and ``index``.

        Raise ``ReceiptNotFound`` if no receipt for the specified ``block_number``
        and ``index`` is found in the canonical chain.
        """
        ...

    @abstractmethod
    def get_transaction_result(self, transaction, at_header):
        """
        Return the result of running the given transaction.
        This is referred to as a `call()` in web3.
        """
        ...

    @abstractmethod
    def estimate_gas(self, transaction, at_header=None):
        """
        Return an estimation of the amount of gas the given ``transaction`` will
        use if executed on top of the block specified by ``at_header``.
        """
        ...

    @abstractmethod
    def import_block(self, block, perform_validation=True):
        """
        Import the given ``block`` and return a 3-tuple

        - the imported block
        - a tuple of blocks which are now part of the canonical chain.
        - a tuple of blocks which were canonical and now are no longer canonical.
        """
        ...

    @abstractmethod
    def validate_receipt(self, receipt, at_header):
        """
        Validate the given ``receipt`` at the given header.
        """
        ...

    @abstractmethod
    def validate_block(self, block):
        """
        Validate a block that is either being mined or imported.

        Since block validation (specifically the uncle validation) must have
        access to the ancestor blocks, this validation must occur at the Chain
        level.
        """
        ...

    @classmethod
    @abstractmethod
    def validate_seal(cls, header):
        """
        Validate the seal on the given ``header``.
        """
        ...

class MiningChainAPI(ChainAPI):
    """
    Like :class:`~eth.abc.ChainAPI` but with APIs to create blocks incrementally.
    """
    header: BlockHeaderAPI

    @abstractmethod
    def __init__(self, base_db, header=None):
        """
        Initialize the chain.
        """
        ...

    @abstractmethod
    def set_header_timestamp(self, timestamp):
        """
        Set the timestamp of the pending header to mine.

        This is mostly useful for testing, as the timestamp will be chosen
        automatically if this method is not called.
        """
        ...

    @abstractmethod
    def mine_all(self, transactions, *args: Any, parent_header: Optional[BlockHeaderAPI]=None, **kwargs: Any):
        """
        Build a block with the given transactions, and mine it.

        Optionally, supply the parent block header to mine on top of.

        This is much faster than individually running :meth:`apply_transaction`
        and then :meth:`mine_block`.
        """
        ...

    @abstractmethod
    def apply_transaction(self, transaction):
        """
        Apply the transaction to the current tip block.

        WARNING: ReceiptAPI and Transaction trie generation is computationally
        heavy and incurs significant performance overhead.
        """
        ...

    @abstractmethod
    def mine_block(self, *args: Any, **kwargs: Any):
        """
        Mines the current block. Proxies to the current Virtual Machine.
        See VM. :meth:`~eth.vm.base.VM.mine_block`
        """
        ...

    @abstractmethod
    def mine_block_extended(self, *args: Any, **kwargs: Any):
        """
        Just like :meth:`~mine_block`, but includes extra returned info. Currently,
        the only extra info returned is the :class:`MetaWitness`.
        """
        ...

class SchemaAPI(ABC):
    """
    A class representing a database schema that maps values to lookup keys.
    """

    @staticmethod
    @abstractmethod
    def make_header_chain_gaps_lookup_key():
        """
        Return the lookup key to retrieve the header chain integrity info from the
        database.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_canonical_head_hash_lookup_key():
        """
        Return the lookup key to retrieve the canonical head from the database.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_block_number_to_hash_lookup_key(block_number):
        """
        Return the lookup key to retrieve a block hash from a block number.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_block_hash_to_score_lookup_key(block_hash):
        """
        Return the lookup key to retrieve the score from a block hash.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_transaction_hash_to_block_lookup_key(transaction_hash):
        """
        Return the lookup key to retrieve a transaction key from a transaction hash.
        """
        ...

    @staticmethod
    @abstractmethod
    def make_withdrawal_hash_to_block_lookup_key(withdrawal_hash):
        """
        Return the lookup key to retrieve a withdrawal key from a withdrawal hash.
        """
        ...

class DatabaseAPI(MutableMapping[bytes, bytes], ABC):
    """
    A class representing a database.
    """

    @abstractmethod
    def set(self, key, value):
        """
        Assign the ``value`` to the ``key``.
        """
        ...

    @abstractmethod
    def exists(self, key):
        """
        Return ``True`` if the ``key`` exists in the database, otherwise ``False``.
        """
        ...

    @abstractmethod
    def delete(self, key):
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
    def atomic_batch(self):
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
    def __init__(self, db):
        """
        Instantiate the database from an :class:`~eth.abc.AtomicDatabaseAPI`.
        """
        ...

    @abstractmethod
    def get_header_chain_gaps(self):
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
        ...

    @abstractmethod
    def get_canonical_block_hash(self, block_number):
        """
        Return the block hash for the canonical block at the given number.

        Raise ``BlockNotFound`` if there's no block header with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number):
        """
        Return the block header with the given number in the canonical chain.

        Raise ``HeaderNotFound`` if there's no block header with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_head(self):
        """
        Return the current block header at the head of the chain.
        """
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash):
        """
        Return the block header for the given ``block_hash``.
        Raise ``HeaderNotFound`` if no header with the given ``block_hash`` exists
        in the database.
        """
        ...

    @abstractmethod
    def get_score(self, block_hash):
        """
        Return the score for the given ``block_hash``.
        """
        ...

    @abstractmethod
    def header_exists(self, block_hash):
        """
        Return ``True`` if the ``block_hash`` exists in the database,
        otherwise ``False``.
        """
        ...

    @abstractmethod
    def persist_checkpoint_header(self, header, score):
        """
        Persist a checkpoint header with a trusted score. Persisting the checkpoint
        header automatically sets it as the new canonical head.
        """
        ...

    @abstractmethod
    def persist_header(self, header):
        """
        Persist the ``header`` in the database.
        Return two iterable of headers, the first containing the new canonical header,
        the second containing the old canonical headers
        """
        ...

    @abstractmethod
    def persist_header_chain(self, headers, genesis_parent_hash=None):
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

    @abstractmethod
    def get_block_uncles(self, uncles_hash):
        """
        Return an iterable of uncle headers specified by the given ``uncles_hash``
        """
        ...

    @abstractmethod
    def persist_block(self, block, genesis_parent_hash=None):
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
    def persist_unexecuted_block(self, block, receipts, genesis_parent_hash=None):
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
    def persist_uncles(self, uncles):
        """
        Persist the list of uncles to the database.

        Return the uncles hash.
        """
        ...

    @abstractmethod
    def add_receipt(self, block_header, index_key, receipt):
        """
        Add the given receipt to the provided block header.

        Return the updated `receipts_root` for updated block header.
        """
        ...

    @abstractmethod
    def add_transaction(self, block_header, index_key, transaction):
        """
        Add the given transaction to the provided block header.

        Return the updated `transactions_root` for updated block header.
        """
        ...

    @abstractmethod
    def get_block_transactions(self, block_header, transaction_decoder):
        """
        Return an iterable of transactions for the block speficied by the
        given block header.
        """
        ...

    @abstractmethod
    def get_block_transaction_hashes(self, block_header):
        """
        Return a tuple cointaining the hashes of the transactions of the
        given ``block_header``.
        """
        ...

    @abstractmethod
    def get_receipt_by_index(self, block_number, receipt_index, receipt_decoder):
        """
        Return the receipt of the transaction at specified index
        for the block header obtained by the specified block number
        """
        ...

    @abstractmethod
    def get_receipts(self, header, receipt_decoder):
        """
        Return a tuple of receipts for the block specified by the given
        block header.
        """
        ...

    @abstractmethod
    def get_transaction_by_index(self, block_number, transaction_index, transaction_decoder):
        """
        Return the transaction at the specified `transaction_index` from the
        block specified by `block_number` from the canonical chain.

        Raise ``TransactionNotFound`` if no block with that ``block_number`` exists.
        """
        ...

    @abstractmethod
    def get_transaction_index(self, transaction_hash):
        """
        Return a 2-tuple of (block_number, transaction_index) indicating which
        block the given transaction can be found in and at what index in the
        block transactions.

        Raise ``TransactionNotFound`` if the transaction_hash is not found in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_block_withdrawals(self, block_header):
        """
        Return an iterable of withdrawals for the block specified by the
        given block header.
        """
        ...

    @abstractmethod
    def get_transaction_result(self, transaction, at_header):
        """
        Return the result of running the given transaction.
        This is referred to as a `call()` in web3.
        """
        ...

    @abstractmethod
    def estimate_gas(self, transaction, at_header=None):
        """
        Return an estimation of the amount of gas the given ``transaction`` will
        use if executed on top of the block specified by ``at_header``.
        """
        ...

    @abstractmethod
    def import_block(self, block, perform_validation=True):
        """
        Import the given ``block`` and return a 3-tuple

        - the imported block
        - a tuple of blocks which are now part of the canonical chain.
        - a tuple of blocks which were canonical and now are no longer canonical.
        """
        ...

    @abstractmethod
    def validate_receipt(self, receipt, at_header):
        """
        Validate the given ``receipt`` at the given header.
        """
        ...

    @abstractmethod
    def validate_block(self, block):
        """
        Validate a block that is either being mined or imported.

        Since block validation (specifically the uncle validation) must have
        access to the ancestor blocks, this validation must occur at the Chain
        level.
        """
        ...

    @classmethod
    @abstractmethod
    def validate_seal(cls, header):
        """
        Validate the seal on the given ``header``.
        """
        ...

class VirtualMachineModifierAPI(ABC):
    """
    Amend a set of VMs for a chain. This allows modifying a chain for different
    consensus schemes.
    """

    @abstractmethod
    def amend_vm_configuration(self, vm_config):
        """
        Amend the ``vm_config`` by configuring the VM classes, and hence returning
        a modified set of VM classes.
        """
        ...

class HeaderChainAPI(ABC):
    """
    Like :class:`eth.abc.ChainAPI` but does only support headers, not entire blocks.
    """
    header: BlockHeaderAPI
    chain_id: int
    vm_configuration: Tuple[Tuple[BlockNumber, Type['VirtualMachineAPI']], ...]

    @abstractmethod
    def __init__(self, base_db, header=None):
        """
        Initialize the header chain.
        """
        ...

    @classmethod
    @abstractmethod
    def from_genesis_header(cls, base_db, genesis_header):
        """
        Initialize the chain from the genesis header.
        """
        ...

    @classmethod
    @abstractmethod
    def get_headerdb_class(cls):
        """
        Return the class which should be used for the `headerdb`
        """
        ...

    def get_canonical_block_hash(self, block_number):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def get_canonical_head(self):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def header_exists(self, block_hash):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def import_header(self, header):
        """
        Direct passthrough to `headerdb`

        Also updates the local `header` property to be the latest canonical head.

        Returns an iterable of headers representing the headers that are newly
        part of the canonical chain.

        - If the imported header is not part of the canonical chain then an
          empty tuple will be returned.
        - If the imported header simply extends the canonical chain then a
          length-1 tuple with the imported header will be returned.
        - If the header is part of a non-canonical chain which overtakes the
          current canonical chain then the returned tuple will contain the
          headers which are newly part of the canonical chain.
        """
        ...

class ChainAPI(ConfigurableAPI):
    """
    A Chain is a combination of one or more VM classes. Each VM is associated
    with a range of blocks. The Chain class acts as a wrapper around these other
    VM classes, delegating operations to the appropriate VM depending on the
    current block number.
    """
    vm_configuration: Tuple[Tuple[BlockNumber, Type['VirtualMachineAPI']], ...]
    chain_id: int
    chaindb: 'ChainDatabaseAPI'
    consensus_context_class: Type['ConsensusContextAPI']

    @classmethod
    @abstractmethod
    def get_chaindb_class(cls):
        """
        Return the class for the used :class:`~eth.abc.ChainDatabaseAPI`.
        """
        ...

    @classmethod
    @abstractmethod
    def from_genesis(cls, base_db, genesis_params, genesis_state=None):
        """
        Initialize the Chain from a genesis state.
        """
        ...

    @classmethod
    @abstractmethod
    def from_genesis_header(cls, base_db, genesis_header):
        """
        Initialize the chain from the genesis header.
        """
        ...

    @classmethod
    @abstractmethod
    def get_vm_class(cls, header):
        """
        Return the VM class for the given ``header``
        """
        ...

    @abstractmethod
    def get_vm(self, header=None):
        """
        Return the VM instance for the given ``header``.
        """
        ...

    @classmethod
    @abstractmethod
    def get_vm_class_for_block_number(cls, block_number):
        """
        Return the VM class for the given ``block_number``
        """
        ...

    @abstractmethod
    def create_header_from_parent(self, parent_header, **header_params: HeaderParams):
        """
        Passthrough helper to the VM class of the block descending from the
        given header.
        """
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash):
        """
        Return the requested block header as specified by ``block_hash``.
        Raise ``BlockNotFound`` if no block header with the given hash exists in the db.
        """
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number):
        """
        Return the block header with the given number in the canonical chain.

        Raise ``HeaderNotFound`` if there's no block header with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_head(self):
        """
        Return the block header at the canonical chain head.

        Raise ``CanonicalHeadNotFound`` if there's no head defined for the
        canonical chain.
        """
        ...

    @abstractmethod
    def get_score(self, block_hash):
        """
        Return the difficulty score of the block with the given ``block_hash``.

        Raise ``HeaderNotFound`` if there is no matching block hash.
        """
        ...

    @abstractmethod
    def get_ancestors(self, limit, header):
        """
        Return `limit` number of ancestor blocks from the current canonical head.
        """
        ...

    @abstractmethod
    def get_block(self):
        """
        Return the current block at the tip of the chain.
        """
        ...

    @abstractmethod
    def get_block_by_hash(self, block_hash):
        """
        Return the requested block as specified by ``block_hash``.

        :raise eth.exceptions.HeaderNotFound: if the header is missing
        :raise eth.exceptions.BlockNotFound: if any part of the block body is missing
        """
        ...

    @abstractmethod
    def get_block_by_header(self, block_header):
        """
        Return the requested block as specified by the ``block_header``.

        :raise eth.exceptions.BlockNotFound: if any part of the block body is missing
        """
        ...

    @abstractmethod
    def get_canonical_block_by_number(self, block_number):
        """
        Return the block with the given ``block_number`` in the canonical chain.

        Raise ``BlockNotFound`` if no block with the given ``block_number`` exists
        in the canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_block_hash(self, block_number):
        """
        Return the block hash with the given ``block_number`` in the canonical chain.

        Raise ``BlockNotFound`` if there's no block with the given number in the
        canonical chain.
        """
        ...

    @abstractmethod
    def build_block_with_transactions_and_withdrawals(self, transactions, parent_header=None, withdrawals=None):
        """
        Generate a block with the provided transactions. This does *not* import
        that block into your chain. If you want this new block in your chain,
        run :meth:`~import_block` with the result block from this method.
        """
        ...

    @abstractmethod
    def create_transaction(self, *args: Any, **kwargs: Any):
        """
        Passthrough helper to the current VM class.
        """
        ...

    @abstractmethod
    def create_unsigned_transaction(self, *, nonce: int, gas_price: int, gas: int, to: Address, value: int, data: bytes):
        """
        Passthrough helper to the current VM class.
        """
        ...

    @abstractmethod
    def get_canonical_transaction_index(self, transaction_hash):
        """
        Return a 2-tuple of (block_number, transaction_index) indicating which
        block the given transaction can be found in and at what index in the
        block transactions.

        Raise ``TransactionNotFound`` if the transaction does not exist in the canonical
        chain.
        """
        ...

    @abstractmethod
    def get_canonical_transaction(self, transaction_hash):
        """
        Return the requested transaction as specified by the ``transaction_hash``
        from the canonical chain.

        Raise ``TransactionNotFound`` if no transaction with the specified hash is
        found in the canonical chain.
        """
        ...

    @abstractmethod
    def get_canonical_transaction_by_index(self, block_number, index):
        """
        Return the requested transaction as specified by the ``block_number``
        and ``index`` from the canonical chain.

        Raise ``TransactionNotFound`` if no transaction exists at ``index`` at
        ``block_number`` in the canonical chain.
        """
        ...

    @abstractmethod
    def get_transaction_receipt(self, transaction_hash):
        """
        Return the requested receipt for the transaction as specified
        by the ``transaction_hash``.

        Raise ``ReceiptNotFound`` if no receipt for the specified
        ``transaction_hash`` is found in the canonical chain.
        """
        ...

    @abstractmethod
    def get_transaction_receipt_by_index(self, block_number, index):
        """
        Return the requested receipt for the transaction as specified by the
        ``block_number`` and ``index``.

        Raise ``ReceiptNotFound`` if no receipt for the specified ``block_number``
        and ``index`` is found in the canonical chain.
        """
        ...

    @abstractmethod
    def get_transaction_result(self, transaction, at_header):
        """
        Return the result of running the given transaction.
        This is referred to as a `call()` in web3.
        """
        ...

    @abstractmethod
    def estimate_gas(self, transaction, at_header=None):
        """
        Return an estimation of the amount of gas the given ``transaction`` will
        use if executed on top of the block specified by ``at_header``.
        """
        ...

    @abstractmethod
    def import_block(self, block, perform_validation=True):
        """
        Import the given ``block`` and return a 3-tuple

        - the imported block
        - a tuple of blocks which are now part of the canonical chain.
        - a tuple of blocks which were canonical and now are no longer canonical.
        """
        ...

    @abstractmethod
    def validate_receipt(self, receipt, at_header):
        """
        Validate the given ``receipt`` at the given header.
        """
        ...

    @abstractmethod
    def validate_block(self, block):
        """
        Validate a block that is either being mined or imported.

        Since block validation (specifically the uncle validation) must have
        access to the ancestor blocks, this validation must occur at the Chain
        level.
        """
        ...

    @classmethod
    @abstractmethod
    def validate_seal(cls, header):
        """
        Validate the seal on the given ``header``.
        """
        ...

class MiningChainAPI(ChainAPI):
    """
    Like :class:`~eth.abc.ChainAPI` but with APIs to create blocks incrementally.
    """
    header: BlockHeaderAPI

    @abstractmethod
    def __init__(self, base_db, header=None):
        """
        Initialize the chain.
        """
        ...

    @abstractmethod
    def set_header_timestamp(self, timestamp):
        """
        Set the timestamp of the pending header to mine.

        This is mostly useful for testing, as the timestamp will be chosen
        automatically if this method is not called.
        """
        ...

    @abstractmethod
    def mine_all(self, transactions, *args: Any, parent_header: Optional[BlockHeaderAPI]=None, **kwargs: Any):
        """
        Build a block with the given transactions, and mine it.

        Optionally, supply the parent block header to mine on top of.

        This is much faster than individually running :meth:`apply_transaction`
        and then :meth:`mine_block`.
        """
        ...

    @abstractmethod
    def apply_transaction(self, transaction):
        """
        Apply the transaction to the current tip block.

        WARNING: ReceiptAPI and Transaction trie generation is computationally
        heavy and incurs significant performance overhead.
        """
        ...

    @abstractmethod
    def mine_block(self, *args: Any, **kwargs: Any):
        """
        Mines the current block. Proxies to the current Virtual Machine.
        See VM. :meth:`~eth.vm.base.VM.mine_block`
        """
        ...

    @abstractmethod
    def mine_block_extended(self, *args: Any, **kwargs: Any):
        """
        Just like :meth:`~mine_block`, but includes extra returned info. Currently,
        the only extra info returned is the :class:`MetaWitness`.
        """
        ...

class HeaderChainAPI(ABC):
    """
    Like :class:`eth.abc.ChainAPI` but does only support headers, not entire blocks.
    """
    header: BlockHeaderAPI
    chain_id: int
    vm_configuration: Tuple[Tuple[BlockNumber, Type['VirtualMachineAPI']], ...]

    @abstractmethod
    def __init__(self, base_db, header=None):
        """
        Initialize the header chain.
        """
        ...

    @classmethod
    @abstractmethod
    def from_genesis_header(cls, base_db, genesis_header):
        """
        Initialize the chain from the genesis header.
        """
        ...

    @classmethod
    @abstractmethod
    def get_headerdb_class(cls):
        """
        Return the class which should be used for the `headerdb`
        """
        ...

    def get_canonical_block_hash(self, block_number):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def get_canonical_block_header_by_number(self, block_number):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def get_canonical_head(self):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def get_block_header_by_hash(self, block_hash):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def header_exists(self, block_hash):
        """
        Direct passthrough to `headerdb`
        """
        ...

    @abstractmethod
    def import_header(self, header):
        """
        Direct passthrough to `headerdb`

        Also updates the local `header` property to be the latest canonical head.

        Returns an iterable of headers representing the headers that are newly
        part of the canonical chain.

        - If the imported header is not part of the canonical chain then an
          empty tuple will be returned.
        - If the imported header simply extends the canonical chain then a
          length-1 tuple with the imported header will be returned.
        - If the header is part of a non-canonical chain which overtakes the
          current canonical chain then the returned tuple will contain the
          headers which are newly part of the canonical chain.
        """
        ...

class StateAPI(ConfigurableAPI):
    """
    The base class that encapsulates all of the various moving parts related to
    the state of the VM during execution.
    Each :class:`~eth.abc.VirtualMachineAPI` must be configured with a subclass of the
    :class:`~eth.abc.StateAPI`.
    Each :class:`~eth.abc.StateAPI` class must be configured with:

    - ``computation_class``: The :class:`~eth.abc.ComputationAPI` class for
      vm execution.
    - ``transaction_context_class``: The :class:`~eth.abc.TransactionContextAPI`
      class for vm execution.
    """