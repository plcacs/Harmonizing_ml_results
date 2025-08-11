import pytest
from eth_hash.auto import keccak
from eth_utils import int_to_big_endian
from eth.vm.interrupt import MissingAccountTrieNode, MissingBytecode, MissingStorageTrieNode

@pytest.fixture
def address_with_balance() -> bytes:
    return b'1' * 20

@pytest.fixture
def address_with_balance_hash(address_with_balance: Union[str, typing.Callable, int]) -> Union[bytes, typing.TextIO, str]:
    return keccak(address_with_balance)

@pytest.fixture
def balance() -> int:
    return 10 ** 18

@pytest.fixture
def address_with_bytecode() -> bytes:
    return b'2' * 20

@pytest.fixture
def bytecode() -> bytes:
    return b'aoeu'

@pytest.fixture
def bytecode_hash(bytecode: Union[str, bytes]) -> Union[bytes, str]:
    return keccak(bytecode)

@pytest.fixture
def address_with_storage() -> bytes:
    return b'3' * 20

@pytest.fixture
def address_with_storage_hash(address_with_storage: Union[bool, bytes]) -> Union[bytes, str, list[int]]:
    return keccak(address_with_storage)

@pytest.fixture
def genesis_state(address_with_balance: nucypher.blockchain.eth.registry.BaseContractRegistry, balance: nucypher.blockchain.eth.registry.BaseContractRegistry, address_with_bytecode: nucypher.blockchain.eth.registry.BaseContractRegistry, bytecode: nucypher.blockchain.eth.registry.BaseContractRegistry, address_with_storage: nucypher.blockchain.eth.registry.BaseContractRegistry) -> dict[nucypher.blockchain.eth.registry.BaseContractRegistry, dict[typing.Text, typing.Union[nucypher.blockchain.eth.registry.BaseContractRegistry,bytes,int,dict]]]:
    return {address_with_balance: {'balance': balance, 'code': b'', 'nonce': 0, 'storage': {}}, address_with_bytecode: {'balance': 0, 'code': bytecode, 'nonce': 0, 'storage': {}}, address_with_storage: {'balance': 0, 'code': b'', 'nonce': 0, 'storage': {i: i for i in range(100)}}}

@pytest.fixture
def chain(chain_without_block_validation: Union[int, tuple[eth.rlp.headers.BlockHeader]]) -> Union[int, tuple[eth.rlp.headers.BlockHeader]]:
    return chain_without_block_validation

def test_bytecode_missing_interrupt(chain: Union[typing.Callable, eth.vm.computation.BaseComputation, bytes, bytearray], bytecode: Union[eth.vm.computation.BaseComputation, bytes], bytecode_hash: typing.Iterable, address_with_bytecode: Union[typing.Callable, bytes, bytearray]) -> None:
    retrieved_bytecode = chain.get_vm().state.get_code(address_with_bytecode)
    assert retrieved_bytecode == bytecode
    assert bytecode == chain.chaindb.db[bytecode_hash]
    del chain.chaindb.db[bytecode_hash]
    with pytest.raises(MissingBytecode) as excinfo:
        chain.get_vm().state.get_code(address_with_bytecode)
    raised_exception = excinfo.value
    assert raised_exception.missing_code_hash == bytecode_hash

def test_account_missing_interrupt(chain: Any, balance: Union[str, int], address_with_balance: Any, address_with_balance_hash: Union[typing.Callable, int, str]) -> None:
    retrieved_balance = chain.get_vm().state.get_balance(address_with_balance)
    assert retrieved_balance == balance
    expected_state_root = chain.get_vm().state.state_root
    node_hash = b"\n\x01TS\x99\x15\xc0\\\xf1\x1f\xfe\x91\xe59\xe9\xaev.\xac#'\xaf\x07)0\x16Y\xda\xdd\x81\xa8\xb3"
    del chain.chaindb.db[node_hash]
    with pytest.raises(MissingAccountTrieNode) as excinfo:
        chain.get_vm().state.get_balance(address_with_balance)
    raised_exception = excinfo.value
    assert raised_exception.missing_node_hash == node_hash
    assert raised_exception.state_root_hash == expected_state_root
    assert raised_exception.address_hash == address_with_balance_hash

def test_storage_missing_interrupt(chain: Union[typing.Callable, str, None], address_with_storage: Union[typing.Callable, str, None], address_with_storage_hash: Union[bool, typing.Callable, str]) -> None:
    test_slot = 42
    retrieved_storage_value = chain.get_vm().state.get_storage(address_with_storage, test_slot)
    assert retrieved_storage_value == test_slot
    expected_storage_root = chain.get_vm().state._account_db._get_storage_root(address_with_storage)
    expected_slot_hash = keccak(int_to_big_endian(test_slot).rjust(32, b'\x00'))
    node_hash = b'bG\\-\x92\xa3\xe4\xd4\xd1\xd5\xe4\xc0r\xbc\xae\x9f\x01\xe7\xdc\xcf\xe3\x96\x9c??+\xb2o\xd5J4\xed'
    del chain.chaindb.db[node_hash]
    with pytest.raises(MissingStorageTrieNode) as excinfo:
        chain.get_vm().state.get_storage(address_with_storage, test_slot)
    raised_exception = excinfo.value
    assert raised_exception.missing_node_hash == node_hash
    assert raised_exception.storage_root_hash == expected_storage_root
    assert raised_exception.account_address == address_with_storage
    assert raised_exception.requested_key == expected_slot_hash