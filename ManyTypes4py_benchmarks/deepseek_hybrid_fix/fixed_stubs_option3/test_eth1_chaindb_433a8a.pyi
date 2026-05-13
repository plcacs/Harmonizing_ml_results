import pytest
from eth_hash.auto import keccak
from hypothesis import HealthCheck, given, settings, strategies as st
import rlp
from _pytest.fixtures import FixtureRequest
from eth._utils.address import force_bytes_to_address
from eth.chains.base import MiningChain
from eth.constants import BLANK_ROOT_HASH, ZERO_ADDRESS
from eth.db.atomic import AtomicDB
from eth.db.chain import ChainDB
from eth.db.chain_gaps import GENESIS_CHAIN_GAPS
from eth.db.schema import SchemaV1
from eth.exceptions import BlockNotFound, CheckpointsMustBeCanonical, HeaderNotFound, ParentNotFound, ReceiptNotFound
from eth.rlp.headers import BlockHeader
from eth.tools.builder.chain import api
from eth.tools.factories.transaction import new_access_list_transaction, new_transaction
from eth.tools.rlp import assert_headers_eq
from eth.vm.forks import BerlinVM, LondonVM
from eth.vm.forks.frontier.blocks import FrontierBlock
from eth.vm.forks.homestead.blocks import HomesteadBlock
from typing import Any, Generator, Iterator, List, Tuple, Type, Union
from hypothesis.strategies import SearchStrategy
from eth.vm.base import BaseVM
from eth.typing import Address, PrivateKey
from eth.rlp.blocks import BaseBlock
from eth.rlp.receipts import Receipt
from eth.rlp.transactions import BaseTransaction

A_ADDRESS: bytes = b'\xaa' * 20
B_ADDRESS: bytes = b'\xbb' * 20

def set_empty_root(chaindb: ChainDB, header: BlockHeader) -> BlockHeader: ...

@pytest.fixture
def chaindb(base_db: AtomicDB) -> ChainDB: ...

@pytest.fixture(params=[0, 10, 999])
def header(request: Any) -> BlockHeader: ...

@pytest.fixture(params=[FrontierBlock, HomesteadBlock])
def block(request: Any, header: BlockHeader) -> BaseBlock: ...

@pytest.fixture
def chain(chain_without_block_validation: MiningChain) -> MiningChain: ...

def test_chaindb_add_block_number_to_hash_lookup(chaindb: ChainDB, block: BaseBlock) -> None: ...

@pytest.mark.parametrize('has_uncle, has_transaction, can_fetch_block', ((True, False, False), (False, True, False), (True, True, False), (False, False, True)))
def test_block_gap_tracking(chain: MiningChain, funded_address: Address, funded_address_private_key: PrivateKey, has_uncle: bool, has_transaction: bool, can_fetch_block: bool) -> None: ...

def test_chaindb_persist_header(chaindb: ChainDB, header: BlockHeader) -> None: ...

@given(seed=st.binary(min_size=32, max_size=32))
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_chaindb_persist_header_unknown_parent(chaindb: ChainDB, header: BlockHeader, seed: bytes) -> None: ...

def test_chaindb_persist_block(chaindb: ChainDB, block: BaseBlock) -> None: ...

def test_chaindb_get_score(chaindb: ChainDB) -> None: ...

def test_chaindb_get_block_header_by_hash(chaindb: ChainDB, block: BaseBlock, header: BlockHeader) -> None: ...

def test_chaindb_get_canonical_block_hash(chaindb: ChainDB, block: BaseBlock) -> None: ...

def mine_blocks_with_receipts(chain: MiningChain, num_blocks: int, num_tx_per_block: int, funded_address: Address, funded_address_private_key: PrivateKey) -> Generator[Tuple[BaseBlock, List[Receipt]], None, None]: ...

def test_chaindb_get_receipt_and_tx_by_index(chain: MiningChain, funded_address: Address, funded_address_private_key: PrivateKey) -> None: ...

def mine_blocks_with_access_list_receipts(chain: MiningChain, num_blocks: int, num_tx_per_block: int, funded_address: Address, funded_address_private_key: PrivateKey) -> Generator[Tuple[BaseBlock, List[Receipt]], None, None]: ...

def test_chaindb_get_access_list_receipt_and_tx_by_index(chain: MiningChain, funded_address: Address, funded_address_private_key: PrivateKey) -> None: ...

@pytest.mark.parametrize('use_persist_unexecuted_block', (True, pytest.param(False, marks=pytest.mark.xfail(reason='The `persist_block` API relies on block execution to persisttransactions and receipts. It is expected to fail this test.'))))
def test_chaindb_persist_unexecuted_block(chain: MiningChain, chain_without_block_validation_factory: Any, funded_address: Address, funded_address_private_key: PrivateKey, use_persist_unexecuted_block: bool) -> None: ...

def test_chaindb_raises_blocknotfound_on_missing_uncles(VM: Type[BaseVM], chaindb: ChainDB, header: BlockHeader) -> None: ...

def test_chaindb_raises_blocknotfound_on_missing_transactions(VM: Type[BaseVM], chaindb: ChainDB, header: BlockHeader) -> None: ...