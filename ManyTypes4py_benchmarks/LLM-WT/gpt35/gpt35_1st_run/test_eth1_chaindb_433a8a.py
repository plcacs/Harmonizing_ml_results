from eth.chains.base import MiningChain
from eth.db.chain import ChainDB
from eth.rlp.headers import BlockHeader
from eth.tools.builder.chain import api
from eth.tools.factories.transaction import new_access_list_transaction, new_transaction
from eth.vm.forks import BerlinVM, LondonVM
from eth.vm.forks.frontier.blocks import FrontierBlock
from eth.vm.forks.homestead.blocks import HomesteadBlock
from eth._utils.address import force_bytes_to_address
from eth.constants import BLANK_ROOT_HASH, ZERO_ADDRESS
from eth.db.atomic import AtomicDB
from eth.db.schema import SchemaV1
from eth.exceptions import BlockNotFound, CheckpointsMustBeCanonical, HeaderNotFound, ParentNotFound, ReceiptNotFound
from eth.tools.rlp import assert_headers_eq
from hypothesis import HealthCheck, given, settings, strategies as st
import pytest
import rlp

A_ADDRESS: bytes = b'\xaa' * 20
B_ADDRESS: bytes = b'\xbb' * 20

def set_empty_root(chaindb: ChainDB, header: BlockHeader) -> BlockHeader:
    return header.copy(transaction_root=BLANK_ROOT_HASH, receipt_root=BLANK_ROOT_HASH, state_root=BLANK_ROOT_HASH)

def test_chaindb_add_block_number_to_hash_lookup(chaindb: ChainDB, block: FrontierBlock) -> None:
    ...

def test_block_gap_tracking(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes, has_uncle: bool, has_transaction: bool, can_fetch_block: bool) -> None:
    ...

def test_chaindb_persist_header(chaindb: ChainDB, header: BlockHeader) -> None:
    ...

def test_chaindb_persist_header_unknown_parent(chaindb: ChainDB, header: BlockHeader, seed: bytes) -> None:
    ...

def test_chaindb_persist_block(chaindb: ChainDB, block: FrontierBlock) -> None:
    ...

def test_chaindb_get_score(chaindb: ChainDB) -> None:
    ...

def test_chaindb_get_block_header_by_hash(chaindb: ChainDB, block: FrontierBlock, header: BlockHeader) -> None:
    ...

def test_chaindb_get_canonical_block_hash(chaindb: ChainDB, block: FrontierBlock) -> None:
    ...

def mine_blocks_with_receipts(chain: MiningChain, num_blocks: int, num_tx_per_block: int, funded_address: bytes, funded_address_private_key: bytes) -> None:
    ...

def test_chaindb_get_receipt_and_tx_by_index(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes) -> None:
    ...

def mine_blocks_with_access_list_receipts(chain: MiningChain, num_blocks: int, num_tx_per_block: int, funded_address: bytes, funded_address_private_key: bytes) -> None:
    ...

def test_chaindb_get_access_list_receipt_and_tx_by_index(chain: MiningChain, funded_address: bytes, funded_address_private_key: bytes) -> None:
    ...

def test_chaindb_persist_unexecuted_block(chain: MiningChain, chain_without_block_validation_factory, funded_address: bytes, funded_address_private_key: bytes, use_persist_unexecuted_block: bool) -> None:
    ...

def test_chaindb_raises_blocknotfound_on_missing_uncles(VM, chaindb: ChainDB, header: BlockHeader) -> None:
    ...

def test_chaindb_raises_blocknotfound_on_missing_transactions(VM, chaindb: ChainDB, header: BlockHeader) -> None:
    ...
