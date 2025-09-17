from typing import Optional, Any
from eth_typing import Address, Hash32
from eth_utils import encode_hex
from trie.exceptions import MissingTrieNode
from trie.typing import Nibbles
from eth.exceptions import PyEVMError


class EVMMissingData(PyEVMError):
    pass


class MissingAccountTrieNode(EVMMissingData, MissingTrieNode):
    """
    Raised when a main state trie node is missing from the DB, to get an account RLP
    """

    @property
    def state_root_hash(self) -> bytes:
        return self.root_hash  # type: ignore

    @property
    def address_hash(self) -> bytes:
        return self.requested_key  # type: ignore

    def __repr__(self) -> str:
        return f'MissingAccountTrieNode: {self}'

    def __str__(self) -> str:
        superclass_str: str = EVMMissingData.__str__(self)
        return (
            f'State trie database is missing node for hash {encode_hex(self.missing_node_hash)}'
            f', which is needed to look up account with address hash {encode_hex(self.address_hash)}'
            f' at root hash {encode_hex(self.state_root_hash)} -- {superclass_str}'
        )


class MissingStorageTrieNode(EVMMissingData, MissingTrieNode):
    """
    Raised when a storage trie node is missing from the DB
    """

    def __init__(
        self,
        missing_node_hash: bytes,
        storage_root_hash: bytes,
        requested_key: bytes,
        prefix: Nibbles,
        account_address: bytes,
        *args: Any
    ) -> None:
        if not isinstance(account_address, bytes):
            raise TypeError(f'Account address must be bytes, was: {account_address!r}')
        super().__init__(missing_node_hash, storage_root_hash, requested_key, prefix, account_address, *args)

    @property
    def storage_root_hash(self) -> bytes:
        return self.root_hash  # type: ignore

    @property
    def account_address(self) -> bytes:
        return self.args[4]

    def __repr__(self) -> str:
        return f'MissingStorageTrieNode: {self}'

    def __str__(self) -> str:
        superclass_str: str = EVMMissingData.__str__(self)
        return (
            f'Storage trie database is missing hash {encode_hex(self.missing_node_hash)}'
            f' needed to look up key {encode_hex(self.requested_key)} at root hash {encode_hex(self.root_hash)}'
            f' in account address {encode_hex(self.account_address)} -- {superclass_str}'
        )


class MissingBytecode(EVMMissingData):
    """
    Raised when the bytecode is missing from the database for a known bytecode hash.
    """

    def __init__(self, missing_code_hash: bytes) -> None:
        if not isinstance(missing_code_hash, bytes):
            raise TypeError(f'Missing code hash must be bytes, was: {missing_code_hash!r}')
        super().__init__(missing_code_hash)

    @property
    def missing_code_hash(self) -> bytes:
        return self.args[0]

    def __repr__(self) -> str:
        return f'MissingBytecode: {self}'

    def __str__(self) -> str:
        superclass_str: str = EVMMissingData.__str__(self)
        return (
            f'Database is missing bytecode for code hash {encode_hex(self.missing_code_hash)}'
            f' -- {superclass_str}'
        )