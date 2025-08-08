from typing import List, Optional, Type, cast
from eth_hash.auto import keccak
from eth_typing import BlockNumber
from eth_typing.evm import Address, Hash32
from eth_utils import encode_hex
import rlp
from rlp.sedes import Binary, CountableList, big_endian_int, binary
from eth._utils.headers import new_timestamp_from_parent
from eth.abc import BlockHeaderAPI, BlockHeaderSedesAPI, MiningHeaderAPI, ReceiptBuilderAPI, TransactionBuilderAPI
from eth.constants import BLANK_ROOT_HASH, EMPTY_UNCLE_HASH, GENESIS_NONCE, GENESIS_PARENT_HASH, ZERO_ADDRESS, ZERO_HASH32
from eth.rlp.headers import BlockHeader
from eth.rlp.sedes import address, hash32, trie_root, uint256
from eth.vm.forks.berlin.blocks import BerlinBlock
from .receipts import LondonReceiptBuilder
from .transactions import LondonTransactionBuilder

UNMINED_LONDON_HEADER_FIELDS: List[Tuple[str, Type]] = [('parent_hash', hash32), ('uncles_hash', hash32), ('coinbase', address), ('state_root', trie_root), ('transaction_root', trie_root), ('receipt_root', trie_root), ('bloom', uint256), ('difficulty', big_endian_int), ('block_number', big_endian_int), ('gas_limit', big_endian_int), ('gas_used', big_endian_int), ('timestamp', big_endian_int), ('extra_data', binary), ('base_fee_per_gas', big_endian_int)]

class LondonMiningHeader(rlp.Serializable, MiningHeaderAPI):
    fields: List[Tuple[str, Type]] = UNMINED_LONDON_HEADER_FIELDS

class LondonBlockHeader(rlp.Serializable, BlockHeaderAPI):
    fields: List[Tuple[str, Type]] = UNMINED_LONDON_HEADER_FIELDS[:-1] + [('mix_hash', binary), ('nonce', Binary(8, allow_empty=True))] + UNMINED_LONDON_HEADER_FIELDS[-1:]

    def __init__(self, difficulty: int, block_number: int, gas_limit: int, timestamp: Optional[int] = None, coinbase: Address = ZERO_ADDRESS, parent_hash: Hash32 = ZERO_HASH32, uncles_hash: Hash32 = EMPTY_UNCLE_HASH, state_root: Hash32 = BLANK_ROOT_HASH, transaction_root: Hash32 = BLANK_ROOT_HASH, receipt_root: Hash32 = BLANK_ROOT_HASH, bloom: int = 0, gas_used: int = 0, extra_data: bytes = b'', mix_hash: Hash32 = ZERO_HASH32, nonce: int = GENESIS_NONCE, base_fee_per_gas: int = 0) -> None:
        ...

    def __str__(self) -> str:
        ...

    @property
    def hash(self) -> Hash32:
        ...

    @property
    def mining_hash(self) -> Hash32:
        ...

    @property
    def hex_hash(self) -> str:
        ...

    @property
    def is_genesis(self) -> bool:
        ...

    @property
    def withdrawals_root(self):
        ...

    @property
    def blob_gas_used(self):
        ...

    @property
    def excess_blob_gas(self):
        ...

    @property
    def parent_beacon_block_root(self):
        ...

class LondonBackwardsHeader(BlockHeaderSedesAPI):
    @classmethod
    def serialize(cls, obj) -> bytes:
        ...

    @classmethod
    def deserialize(cls, encoded: bytes):
        ...

class LondonBlock(BerlinBlock):
    transaction_builder: Type[TransactionBuilderAPI] = LondonTransactionBuilder
    receipt_builder: Type[ReceiptBuilderAPI] = LondonReceiptBuilder
    fields: List[Tuple[str, Type]] = [('header', LondonBlockHeader), ('transactions', CountableList[TransactionBuilderAPI]), ('uncles', CountableList[BlockHeaderSedesAPI])]
