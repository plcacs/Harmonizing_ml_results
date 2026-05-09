import pytest
from eth_hash.auto import keccak
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import binary, integers
from rlp import DecodingError, encode, decode
from eth._utils.address import force_bytes_to_address
from eth.chains.base import MiningChain
from eth.constants import BLANK_ROOT_HASH, ZERO_ADDRESS
from eth.db.atomic import AtomicDB
from eth.db.chain import ChainDB
from eth.db.schema import SchemaV1
from eth.exceptions import (
    BlockNotFound,
    CheckpointsMustBeCanonical,
    HeaderNotFound,
    ParentNotFound,
    ReceiptNotFound,
)
from eth.rlp.headers import BlockHeader
from eth.vm.forks import BerlinVM, LondonVM
from eth.vm.forks.frontier.blocks import FrontierBlock
from eth.vm.forks.homestead.blocks import HomesteadBlock
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

A_ADDRESS: bytes = ...
B_ADDRESS: bytes = ...

def set_empty_root(chaindb: ChainDB, header: BlockHeader) -> BlockHeader: ...

@pytest.fixture
def chaindb(base_db: AtomicDB) -> ChainDB: ...

@pytest.fixture
def header(request: pytest.FixtureRequest) -> BlockHeader: ...

@pytest.fixture
def block(request: pytest.FixtureRequest, header: BlockHeader) -> Union[FrontierBlock, HomesteadBlock]: ...

@pytest.fixture
def chain(chain_without_block_validation: MiningChain) -> MiningChain: ...

def test_chaindb_add_block_number_to_hash_lookup(chaindb: ChainDB, block: BlockHeader) -> None: ...

@pytest.mark.parametrize('has_uncle, has_transaction, can_fetch_block', ((True, False, False), (False, True, False), (True, True, False), (False, False, True)))
def test_block_gap_tracking(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes, has_uncle: bool, has_transaction: bool, can_fetch_block: bool) -> None: ...

def test_chaindb_persist_header(chaindb: ChainDB, header: BlockHeader) -> None: ...

@given(seed=binary(min_size=32, max_size=32))
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_chaindb_persist_header_unknown_parent(chaindb: ChainDB, header: BlockHeader, seed: bytes) -> None: ...

def test_chaindb_persist_block(chaindb: ChainDB, block: BlockHeader) -> None: ...

def test_chaindb_get_score(chaindb: ChainDB) -> None: ...

def test_chaindb_get_block_header_by_hash(chaindb: ChainDB, block: BlockHeader, header: BlockHeader) -> None: ...

def test_chaindb_get_canonical_block_hash(chaindb: ChainDB, block: BlockHeader) -> None: ...

def mine_blocks_with_receipts(chain: MiningChain, num_blocks: int, num_tx_per_block: int, funded_address: bytes, funded_address_private_key: bytes) -> Generator[Tuple[BlockHeader, List[Any]], None, None]: ...

def test_chaindb_get_receipt_and_tx_by_index(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes) -> None: ...

def mine_blocks_with_access_list_receipts(chain: MiningChain, num_blocks: int, num_tx_per_block: int, funded_address: bytes, funded_address_private_key: bytes) -> Generator[Tuple[BlockHeader, List[Any]], None, None]: ...

def test_chaindb_get_access_list_receipt_and_tx_by_index(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes) -> None: ...

@pytest.mark.parametrize('use_persist_unexecuted_block', (True, pytest.param(False, marks=pytest.mark.xfail)))
def test_chaindb_persist_unexecuted_block(chain: MiningChain, chain_without_block_validation_factory: Callable[[AtomicDB], MiningChain], funded_address: bytes, funded_address_private_key: bytes, use_persist_unexecuted_block: bool) -> None: ...

def test_chaindb_raises_blocknotfound_on_missing_uncles(VM: Union[Type[LondonVM], Type[BerlinVM], Type[HomesteadVM], Type[FrontierBlock]], chaindb: ChainDB, header: BlockHeader) -> None: ...

def test_chaindb_raises_blocknotfound_on_missing_transactions(VM: Union[Type[LondonVM], Type[BerlinVM], Type[HomesteadVM], Type[FrontierBlock]], chaindb: ChainDB, header: BlockHeader) -> None: ...