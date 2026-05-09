import pytest
from typing import Any, Generator, Tuple, List, Type, Union, Optional, Callable
from eth_hash.auto import keccak
import rlp
from eth.chains.base import MiningChain
from eth.db.atomic import AtomicDB
from eth.db.chain import ChainDB
from eth.db.schema import SchemaV1
from eth.rlp.headers import BlockHeader
from eth.vm.forks import BerlinVM, LondonVM
from eth.vm.forks.frontier.blocks import FrontierBlock
from eth.vm.forks.homestead.blocks import HomesteadBlock

A_ADDRESS: bytes = ...
B_ADDRESS: bytes = ...

def set_empty_root(chaindb: ChainDB, header: BlockHeader) -> BlockHeader: ...

@pytest.fixture
def chaindb(base_db: Any) -> ChainDB: ...

@pytest.fixture
def header(request: Any) -> BlockHeader: ...

@pytest.fixture
def block(request: Any, header: BlockHeader) -> Union[FrontierBlock, HomesteadBlock]: ...

@pytest.fixture
def chain(chain_without_block_validation: Any) -> MiningChain: ...

def test_chaindb_add_block_number_to_hash_lookup(chaindb: ChainDB, block: Any) -> None: ...

def test_block_gap_tracking(
    chain: MiningChain, 
    funded_address: bytes, 
    funded_address_private_key: bytes, 
    has_uncle: bool, 
    has_transaction: bool, 
    can_fetch_block: bool
) -> None: ...

def test_chaindb_persist_header(chaindb: ChainDB, header: BlockHeader) -> None: ...

def test_chaindb_persist_header_unknown_parent(chaindb: ChainDB, header: BlockHeader, seed: bytes) -> None: ...

def test_chaindb_persist_block(chaindb: ChainDB, block: Any) -> None: ...

def test_chaindb_get_score(chaindb: ChainDB) -> None: ...

def test_chaindb_get_block_header_by_hash(chaindb: ChainDB, block: Any, header: BlockHeader) -> None: ...

def test_chaindb_get_canonical_block_hash(chaindb: ChainDB, block: Any) -> None: ...

def mine_blocks_with_receipts(
    chain: MiningChain, 
    num_blocks: int, 
    num_tx_per_block: int, 
    funded_address: bytes, 
    funded_address_private_key: bytes
) -> Generator[Tuple[Any, List[Any]], None, None]: ...

def test_chaindb_get_receipt_and_tx_by_index(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes) -> None: ...

def mine_blocks_with_access_list_receipts(
    chain: MiningChain, 
    num_blocks: int, 
    num_tx_per_block: int, 
    funded_address: bytes, 
    funded_address_private_key: bytes
) -> Generator[Tuple[Any, List[Any]], None, None]: ...

def test_chaindb_get_access_list_receipt_and_tx_by_index(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes) -> None: ...

def test_chaindb_persist_unexecuted_block(
    chain: MiningChain, 
    chain_without_block_validation_factory: Callable[[AtomicDB], MiningChain], 
    funded_address: bytes, 
    funded_address_private_key: bytes, 
    use_persist_unexecuted_block: bool
) -> None: ...

def test_chaindb_raises_blocknotfound_on_missing_uncles(VM: Any, chaindb: ChainDB, header: BlockHeader) -> None: ...

def test_chaindb_raises_blocknotfound_on_missing_transactions(VM: Any, chaindb: ChainDB, header: BlockHeader) -> None: ...