import pytest
from eth_hash.auto import keccak
from hypothesis import HealthCheck, given, settings, strategies as st
import rlp
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

A_ADDRESS: bytes = b'\xaa' * 20
B_ADDRESS: bytes = b'\xbb' * 20

def set_empty_root(chaindb: ChainDB, header: BlockHeader) -> BlockHeader:
    return header.copy(transaction_root=BLANK_ROOT_HASH, receipt_root=BLANK_ROOT_HASH, state_root=BLANK_ROOT_HASH)

@pytest.fixture
def chaindb(base_db: AtomicDB) -> ChainDB:
    return ChainDB(base_db)

@pytest.fixture(params=[0, 10, 999])
def header(request, chaindb: ChainDB) -> BlockHeader:
    block_number: int = request.param
    difficulty: int = 1
    gas_limit: int = 1
    return BlockHeader(difficulty, block_number, gas_limit)

@pytest.fixture(params=[FrontierBlock, HomesteadBlock])
def block(request, header: BlockHeader) -> Block:
    return request.param(header)

@pytest.fixture
def chain(chain_without_block_validation: MiningChain) -> MiningChain:
    if not isinstance(chain_without_block_validation, MiningChain):
        pytest.skip('these tests require a mining chain implementation')
    else:
        return chain_without_block_validation

# ... rest of the code ...
