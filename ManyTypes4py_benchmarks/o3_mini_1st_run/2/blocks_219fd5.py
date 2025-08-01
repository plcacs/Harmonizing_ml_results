from abc import ABC
from typing import Any, List, Optional, Sequence, Tuple, Type, cast, Union
from eth_bloom import BloomFilter
from eth_typing import Address, BlockNumber, Hash32
from eth_utils import encode_hex, keccak
import rlp
from rlp.sedes import Binary, CountableList, big_endian_int, binary
from trie.exceptions import MissingTrieNode
from eth._utils.headers import new_timestamp_from_parent
from eth.abc import (
    BlockHeaderAPI,
    BlockHeaderSedesAPI,
    ChainDatabaseAPI,
    ReceiptAPI,
    ReceiptBuilderAPI,
    SignedTransactionAPI,
    TransactionBuilderAPI,
    WithdrawalAPI,
)
from eth.constants import BLANK_ROOT_HASH, EMPTY_UNCLE_HASH, GENESIS_NONCE, GENESIS_PARENT_HASH, ZERO_ADDRESS, ZERO_HASH32
from eth.exceptions import BlockNotFound, HeaderNotFound
from eth.rlp.blocks import BaseBlock
from eth.rlp.headers import BlockHeader
from eth.rlp.sedes import address, hash32, trie_root, uint256
from ..london.blocks import LondonBlockHeader
from ..london.receipts import LondonReceiptBuilder
from .transactions import ShanghaiTransactionBuilder
from .withdrawals import Withdrawal


class ShanghaiBlockHeader(rlp.Serializable, BlockHeaderAPI, ABC):
    fields: List[Tuple[str, Any]] = [
        ('parent_hash', hash32),
        ('uncles_hash', hash32),
        ('coinbase', address),
        ('state_root', trie_root),
        ('transaction_root', trie_root),
        ('receipt_root', trie_root),
        ('bloom', uint256),
        ('difficulty', big_endian_int),
        ('block_number', big_endian_int),
        ('gas_limit', big_endian_int),
        ('gas_used', big_endian_int),
        ('timestamp', big_endian_int),
        ('extra_data', binary),
        ('mix_hash', binary),
        ('nonce', Binary(8, allow_empty=True)),
        ('base_fee_per_gas', big_endian_int),
        ('withdrawals_root', trie_root)
    ]

    _hash: Optional[Hash32] = None

    def __init__(
        self,
        difficulty: int,
        block_number: int,
        gas_limit: int,
        timestamp: Optional[int] = None,
        coinbase: Address = ZERO_ADDRESS,
        parent_hash: Hash32 = ZERO_HASH32,
        uncles_hash: Hash32 = EMPTY_UNCLE_HASH,
        state_root: Hash32 = BLANK_ROOT_HASH,
        transaction_root: Hash32 = BLANK_ROOT_HASH,
        receipt_root: Hash32 = BLANK_ROOT_HASH,
        bloom: int = 0,
        gas_used: int = 0,
        extra_data: bytes = b'',
        mix_hash: bytes = ZERO_HASH32,
        nonce: bytes = GENESIS_NONCE,
        base_fee_per_gas: int = 0,
        withdrawals_root: Hash32 = BLANK_ROOT_HASH
    ) -> None:
        if timestamp is None:
            if parent_hash == ZERO_HASH32:
                timestamp = new_timestamp_from_parent(None)
            else:
                raise ValueError('Must set timestamp explicitly if this is not a genesis header')
        super().__init__(
            parent_hash=parent_hash,
            uncles_hash=uncles_hash,
            coinbase=coinbase,
            state_root=state_root,
            transaction_root=transaction_root,
            receipt_root=receipt_root,
            bloom=bloom,
            difficulty=difficulty,
            block_number=block_number,
            gas_limit=gas_limit,
            gas_used=gas_used,
            timestamp=timestamp,
            extra_data=extra_data,
            mix_hash=mix_hash,
            nonce=nonce,
            base_fee_per_gas=base_fee_per_gas,
            withdrawals_root=withdrawals_root
        )

    def __str__(self) -> str:
        return f'<ShanghaiBlockHeader #{self.block_number} {encode_hex(self.hash)[2:10]}>'

    @property
    def hash(self) -> Hash32:
        if self._hash is None:
            self._hash = keccak(rlp.encode(self))
        return cast(Hash32, self._hash)

    @property
    def mining_hash(self) -> Any:
        raise ValueError('Mining hash is not available post merge.')

    @property
    def hex_hash(self) -> str:
        return encode_hex(self.hash)

    @property
    def is_genesis(self) -> bool:
        return self.parent_hash == GENESIS_PARENT_HASH and self.block_number == 0

    @property
    def blob_gas_used(self) -> Any:
        raise AttributeError('Blob gas used is not available until Cancun.')

    @property
    def excess_blob_gas(self) -> Any:
        raise AttributeError('Excess blob gas is not available until Cancun.')

    @property
    def parent_beacon_block_root(self) -> Any:
        raise AttributeError('Parent beacon block root is not available until Cancun.')


class ShanghaiBackwardsHeader(BlockHeaderSedesAPI):
    """
    An rlp sedes class for block headers.
    It can serialize and deserialize Shanghai, London, and pre-London headers.
    """

    @classmethod
    def serialize(cls, obj: BlockHeaderAPI) -> bytes:
        return obj.serialize(obj)

    @classmethod
    def deserialize(cls, encoded: Sequence[Any]) -> BlockHeaderAPI:
        num_fields: int = len(encoded)
        if num_fields == 17:
            return ShanghaiBlockHeader.deserialize(encoded)  # type: ignore
        if num_fields == 16:
            return LondonBlockHeader.deserialize(encoded)  # type: ignore
        elif num_fields == 15:
            return BlockHeader.deserialize(encoded)  # type: ignore
        else:
            raise ValueError(f'Unexpected number of fields in block header. Got {num_fields} in {encoded!r}')


class ShanghaiBlock(BaseBlock):
    transaction_builder: Type[ShanghaiTransactionBuilder] = ShanghaiTransactionBuilder
    receipt_builder: Type[LondonReceiptBuilder] = LondonReceiptBuilder
    fields: List[Tuple[str, Any]] = [
        ('header', ShanghaiBlockHeader),
        ('transactions', CountableList(transaction_builder)),
        ('uncles', CountableList(ShanghaiBackwardsHeader, max_length=0)),
        ('withdrawals', CountableList(Withdrawal))
    ]
    bloom_filter: Optional[BloomFilter]

    def __init__(
        self,
        header: ShanghaiBlockHeader,
        transactions: Optional[Sequence[SignedTransactionAPI]] = None,
        uncles: Optional[Sequence[ShanghaiBackwardsHeader]] = None,
        withdrawals: Optional[Sequence[Withdrawal]] = None
    ) -> None:
        if transactions is None:
            transactions = []
        if uncles is None:
            uncles = []
        if withdrawals is None:
            withdrawals = []
        self.bloom_filter = BloomFilter(header.bloom)
        super().__init__(header=header, transactions=transactions, uncles=uncles, withdrawals=withdrawals)

    @property
    def number(self) -> int:
        return self.header.block_number

    @property
    def hash(self) -> Hash32:
        return self.header.hash

    @classmethod
    def get_transaction_builder(cls) -> Type[ShanghaiTransactionBuilder]:
        return cls.transaction_builder

    @classmethod
    def get_receipt_builder(cls) -> Type[LondonReceiptBuilder]:
        return cls.receipt_builder

    def get_receipts(self, chaindb: ChainDatabaseAPI) -> Sequence[ReceiptAPI]:
        return chaindb.get_receipts(self.header, self.get_receipt_builder())

    @classmethod
    def from_header(cls, header: BlockHeaderAPI, chaindb: ChainDatabaseAPI) -> "ShanghaiBlock":
        """
        Returns the block denoted by the given block header.

        :raise eth.exceptions.BlockNotFound: if transactions, uncle headers,
               or withdrawals are missing
        """
        if header.uncles_hash == EMPTY_UNCLE_HASH:
            uncles: Union[Tuple[Any, ...], List[Any]] = ()
        else:
            try:
                uncles = chaindb.get_block_uncles(header.uncles_hash)
            except HeaderNotFound as exc:
                raise BlockNotFound(f'Uncles not found in database for {header}: {exc}') from exc
        try:
            transactions = chaindb.get_block_transactions(header, cls.get_transaction_builder())
        except MissingTrieNode as exc:
            raise BlockNotFound(f'Transactions not found in database for {header}: {exc}') from exc
        try:
            withdrawals = chaindb.get_block_withdrawals(header)
        except MissingTrieNode as exc:
            raise BlockNotFound(f'Withdrawals not found in database for {header}: {exc}') from exc
        return cls(header=header, transactions=transactions, uncles=uncles, withdrawals=withdrawals)