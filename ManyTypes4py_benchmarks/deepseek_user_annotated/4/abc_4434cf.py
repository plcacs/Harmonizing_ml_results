from abc import (
    ABC,
    abstractmethod,
)
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

from eth_bloom import (
    BloomFilter,
)
from eth_keys.datatypes import (
    PrivateKey,
)
from eth_typing import (
    Address,
    BlockNumber,
    Hash32,
)
from eth_utils import (
    ExtendedDebugLogger,
)

from eth.constants import (
    BLANK_ROOT_HASH,
)
from eth.exceptions import (
    VMError,
)
from eth.typing import (
    AccountState,
    BytesOrView,
    ChainGaps,
    HeaderParams,
    JournalDBCheckpoint,
    VMConfiguration,
)

if TYPE_CHECKING:
    from eth.vm.forks.cancun.transactions import (
        BlobTransaction,
    )

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
    def is_signature_valid