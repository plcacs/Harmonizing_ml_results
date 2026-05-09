import pytest
from eth._utils.address import force_bytes_to_address
from eth.chains.base import MiningChain
from eth.db.chain import ChainDB
from eth.rlp.headers import BlockHeader
from eth.vm.forks import BerlinVM, LondonVM
from eth.vm.forks.frontier.blocks import FrontierBlock
from eth.vm.forks.homestead.blocks import HomesteadBlock
from hypothesis import HealthCheck, given, settings
import rlp
from typing import Any, Generator, List, Optional, Tuple, Union

@pytest_fixture
def chaindb(base_db) -> ChainDB:
    ...

@pytest_fixture
def header(request) -> BlockHeader:
    ...

@pytest_fixture
def block(request, header) -> Union[FrontierBlock, HomesteadBlock]:
    ...

@pytest_fixture
def chain(chain_without_block_validation) -> MiningChain:
    ...

def test_chaindb_add_block_number_to_hash_lookup(chaindb: ChainDB, block: Union[FrontierBlock, HomesteadBlock]) -> None:
    ...

def test_block_gap_tracking(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes, has_uncle: bool, has_transaction: bool, can_fetch_block: bool) -> None:
    ...

def test_chaindb_persist_header(chaindb: ChainDB, header: BlockHeader) -> None:
    ...

@given(seed=bytes)
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_chaindb_persist_header_unknown_parent(chaindb: ChainDB, header: BlockHeader, seed: bytes) -> None:
    ...

def test_chaindb_persist_block(chaindb: ChainDB, block: Union[FrontierBlock, HomesteadBlock]) -> None:
    ...

def test_chaindb_get_score(chaindb: ChainDB) -> None:
    ...

def test_chaindb_get_block_header_by_hash(chaindb: ChainDB, block: Union[FrontierBlock, HomesteadBlock], header: BlockHeader) -> None:
    ...

def test_chaindb_get_canonical_block_hash(chaindb: ChainDB, block: Union[FrontierBlock, HomesteadBlock]) -> bytes:
    ...

def mine_blocks_with_receipts(chain: MiningChain, num_blocks: int, num_tx_per_block: int, funded_address: bytes, funded_address_private_key: bytes) -> Generator[Tuple[Any, List[dict]], None, None]:
    ...

def test_chaindb_get_receipt_and_tx_by_index(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes) -> None:
    ...

def mine_blocks_with_access_list_receipts(chain: MiningChain, num_blocks: int, num_tx_per_block: int, funded_address: bytes, funded_address_private_key: bytes) -> Generator[Tuple[Any, List[dict]], None, None]:
    ...

def test_chaindb_get_access_list_receipt_and_tx_by_index(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes) -> None:
    ...

@pytest.mark.parametrize('use_persist_unexecuted_block', (True, pytest.param(False, marks=pytest.mark.xfail)))
def test_chaindb_persist_unexecuted_block(chain: MiningChain, chain_without_block_validation_factory: Any, funded_address: bytes, funded_address_private_key: bytes, use_persist_unexecuted_block: bool) -> None:
    ...

def test_chaindb_raises_blocknotfound_on_missing_uncles(VM: Any, chaindb: ChainDB, header: BlockHeader) -> None:
    ...

def test_chaindb_raises_blocknotfound_on_missing_transactions(VM: Any, chaindb: ChainDB, header: BlockHeader) -> None:
    ...