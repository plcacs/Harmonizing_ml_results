from typing import Iterable, Optional
from eth_typing import Address, BlockNumber, Hash32
from eth._utils.generator import CachedIterable
from eth.abc import ExecutionContextAPI

class ExecutionContext(ExecutionContextAPI):
    _coinbase: Optional[Address]
    _timestamp: Optional[int]
    _block_number: Optional[BlockNumber]
    _difficulty: Optional[int]
    _mix_hash: Optional[Hash32]
    _gas_limit: Optional[int]
    _prev_hashes: Optional[CachedIterable[Hash32]]
    _chain_id: Optional[int]
    _base_fee_per_gas: Optional[int]
    _excess_blob_gas: Optional[int]

    def __init__(self, coinbase, timestamp, block_number, difficulty, mix_hash, gas_limit, prev_hashes, chain_id, base_fee_per_gas=None, excess_blob_gas=None):
        self._coinbase = coinbase
        self._timestamp = timestamp
        self._block_number = block_number
        self._difficulty = difficulty
        self._mix_hash = mix_hash
        self._gas_limit = gas_limit
        self._prev_hashes = CachedIterable(prev_hashes)
        self._chain_id = chain_id
        self._base_fee_per_gas = base_fee_per_gas
        self._excess_blob_gas = excess_blob_gas

    @property
    def coinbase(self):
        return self._coinbase

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def block_number(self):
        return self._block_number

    @property
    def difficulty(self):
        return self._difficulty

    @property
    def mix_hash(self):
        return self._mix_hash

    @property
    def gas_limit(self):
        return self._gas_limit

    @property
    def prev_hashes(self):
        return self._prev_hashes

    @property
    def chain_id(self):
        return self._chain_id

    @property
    def base_fee_per_gas(self):
        if self._base_fee_per_gas is None:
            raise AttributeError(f'This header at Block #{self.block_number} does not have a base gas fee')
        else:
            return self._base_fee_per_gas

    @property
    def excess_blob_gas(self):
        if self._excess_blob_gas is None:
            raise AttributeError(f'This header at Block #{self.block_number} does not have a excess blob gas')
        else:
            return self._excess_blob_gas