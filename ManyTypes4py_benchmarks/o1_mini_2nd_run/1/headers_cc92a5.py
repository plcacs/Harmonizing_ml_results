from typing import Optional, cast, overload, Any
from eth_hash.auto import keccak
from eth_typing import Address, BlockNumber, Hash32
from eth_utils import encode_hex
import rlp
from rlp.sedes import Binary, big_endian_int, binary
from eth._utils.headers import new_timestamp_from_parent
from eth.abc import BlockHeaderAPI, MiningHeaderAPI
from eth.constants import (
    BLANK_ROOT_HASH,
    EMPTY_UNCLE_HASH,
    GENESIS_NONCE,
    GENESIS_PARENT_HASH,
    ZERO_ADDRESS,
    ZERO_HASH32
)
from eth.typing import HeaderParams
from .sedes import address, hash32, trie_root, uint256


class MiningHeader(rlp.Serializable, MiningHeaderAPI):
    fields: list[tuple[str, Any]] = [
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
        ('extra_data', binary)
    ]


class BlockHeader(rlp.Serializable, BlockHeaderAPI):
    fields: list[tuple[str, Any]] = [
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
        ('nonce', Binary(8, allow_empty=True))
    ]

    @overload
    def __init__(self, **kwargs: Any) -> None:
        ...

    @overload
    def __init__(
        self,
        difficulty: int,
        block_number: BlockNumber,
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
        nonce: bytes = GENESIS_NONCE
    ) -> None:
        ...

    def __init__(
        self,
        difficulty: int,
        block_number: BlockNumber,
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
        nonce: bytes = GENESIS_NONCE
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
            nonce=nonce
        )

    def __str__(self) -> str:
        return f'<BlockHeader #{self.block_number} {encode_hex(self.hash)[2:10]}>'

    _hash: Optional[Hash32] = None

    @property
    def hash(self) -> Hash32:
        if self._hash is None:
            self._hash = keccak(rlp.encode(self))
        return cast(Hash32, self._hash)

    @property
    def mining_hash(self) -> Hash32:
        result: Hash32 = keccak(rlp.encode(self[:-2], MiningHeader))
        return cast(Hash32, result)

    @property
    def hex_hash(self) -> str:
        return encode_hex(self.hash)

    @property
    def is_genesis(self) -> bool:
        return self.parent_hash == GENESIS_PARENT_HASH and self.block_number == 0

    @property
    def base_fee_per_gas(self) -> int:
        raise AttributeError('Base fee per gas not available until London fork')

    @property
    def withdrawals_root(self) -> Hash32:
        raise AttributeError('Withdrawals root not available until Shanghai fork')

    @property
    def blob_gas_used(self) -> int:
        raise AttributeError('Blob gas used not available until Cancun fork')

    @property
    def excess_blob_gas(self) -> int:
        raise AttributeError('Excess blob gas not available until Cancun fork')

    @property
    def parent_beacon_block_root(self) -> Hash32:
        raise AttributeError('Parent beacon block root not available until Cancun fork')
