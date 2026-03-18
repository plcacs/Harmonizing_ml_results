```python
import pytest
from eth.db.atomic import AtomicDB
from eth.db.chain import ChainDB
from eth.exceptions import BlockNotFound, CheckpointsMustBeCanonical, HeaderNotFound, ParentNotFound, ReceiptNotFound
from eth.rlp.headers import BlockHeader
from eth.tools.builder.chain import api
from eth.vm.forks.frontier.blocks import FrontierBlock
from eth.vm.forks.homestead.blocks import HomesteadBlock
from hypothesis import strategies as st
from typing import Any, Generator, Iterator, Tuple, Type, Union

A_ADDRESS: bytes = ...
B_ADDRESS: bytes = ...

def set_empty_root(chaindb: Any, header: Any) -> Any: ...

@pytest.fixture
def chaindb(base_db: Any) -> ChainDB: ...

@pytest.fixture
def header(request: Any) -> BlockHeader: ...

@pytest.fixture
def block(request: Any, header: Any) -> Union[FrontierBlock, HomesteadBlock]: ...

@pytest.fixture
def chain(chain_without_block_validation: Any) -> Any: ...

def test_chaindb_add_block_number_to_hash_lookup(chaindb: ChainDB, block: Any) -> None: ...

def test_block_gap_tracking(
    chain: Any,
    funded_address: Any,
    funded_address_private_key: Any,
    has_uncle: bool,
    has_transaction: bool,
    can_fetch_block: bool
) -> None: ...

def test_chaindb_persist_header(chaindb: ChainDB, header: BlockHeader) -> None: ...

def test_chaindb_persist_header_unknown_parent(
    chaindb: ChainDB,
    header: BlockHeader,
    seed: bytes
) -> None: ...

def test_chaindb_persist_block(chaindb: ChainDB, block: Any) -> None: ...

def test_chaindb_get_score(chaindb: ChainDB) -> None: ...

def test_chaindb_get_block_header_by_hash(
    chaindb: ChainDB,
    block: Any,
    header: BlockHeader
) -> None: ...

def test_chaindb_get_canonical_block_hash(chaindb: ChainDB, block: Any) -> None: ...

def mine_blocks_with_receipts(
    chain: Any,
    num_blocks: int,
    num_tx_per_block: int,
    funded_address: Any,
    funded_address_private_key: Any
) -> Generator[Tuple[Any, Any], None, None]: ...

def test_chaindb_get_receipt_and_tx_by_index(
    chain: Any,
    funded_address: Any,
    funded_address_private_key: Any
) -> None: ...

def mine_blocks_with_access_list_receipts(
    chain: Any,
    num_blocks: int,
    num_tx_per_block: int,
    funded_address: Any,
    funded_address_private_key: Any
) -> Generator[Tuple[Any, Any], None, None]: ...

def test_chaindb_get_access_list_receipt_and_tx_by_index(
    chain: Any,
    funded_address: Any,
    funded_address_private_key: Any
) -> None: ...

def test_chaindb_persist_unexecuted_block(
    chain: Any,
    chain_without_block_validation_factory: Any,
    funded_address: Any,
    funded_address_private_key: Any,
    use_persist_unexecuted_block: bool
) -> None: ...

def test_chaindb_raises_blocknotfound_on_missing_uncles(
    VM: Any,
    chaindb: ChainDB,
    header: BlockHeader
) -> None: ...

def test_chaindb_raises_blocknotfound_on_missing_transactions(
    VM: Any,
    chaindb: ChainDB,
    header: BlockHeader
) -> None: ...
```