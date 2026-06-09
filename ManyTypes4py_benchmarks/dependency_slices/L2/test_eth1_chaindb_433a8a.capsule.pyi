from typing import Any

# === Internal dependency: eth._utils.address ===
def force_bytes_to_address(value: bytes) -> Address: ...

# === Internal dependency: eth.chains.base ===
class MiningChain(Chain, MiningChainAPI): ...

# === Internal dependency: eth.constants ===
ZERO_ADDRESS: Address
BLANK_ROOT_HASH: Hash32

# === Internal dependency: eth.db.atomic ===
class AtomicDB(BaseAtomicDB): ...

# === Internal dependency: eth.db.chain ===
class ChainDB(HeaderDB, ChainDatabaseAPI): ...

# === Internal dependency: eth.db.chain_gaps ===
GENESIS_CHAIN_GAPS: Any

# === Internal dependency: eth.db.schema ===
class SchemaV1(SchemaAPI):
    ...

# === Internal dependency: eth.exceptions ===
class HeaderNotFound(PyEVMError): ...
class BlockNotFound(PyEVMError): ...
class ReceiptNotFound(PyEVMError): ...
class ParentNotFound(HeaderNotFound): ...
class CheckpointsMustBeCanonical(PyEVMError): ...

# === Internal dependency: eth.rlp.headers ===
class BlockHeader(rlp.Serializable, BlockHeaderAPI):
    def __init__(self, **kwargs: HeaderParams) -> None: ...
    def __init__(self, difficulty: int, block_number: BlockNumber, gas_limit: int, timestamp: int = ..., coinbase: Address = ..., parent_hash: Hash32 = ..., uncles_hash: Hash32 = ..., state_root: Hash32 = ..., transaction_root: Hash32 = ..., receipt_root: Hash32 = ..., bloom: int = ..., gas_used: int = ..., extra_data: bytes = ..., mix_hash: Hash32 = ..., nonce: bytes = ...) -> None: ...
    def hash(self) -> Hash32: ...

# === Internal dependency: eth.tools.builder.chain ===
api: API

# === Internal dependency: eth.tools.rlp ===
assert_headers_eq: Any

# === Internal dependency: eth.vm.forks ===
# re-export: from .berlin import BerlinVM
# re-export: from .london import LondonVM

# === Internal dependency: eth.vm.forks.frontier.blocks ===
class FrontierBlock(BaseBlock): ...

# === Internal dependency: eth.vm.forks.homestead.blocks ===
class HomesteadBlock(FrontierBlock): ...

# === Third-party dependency: eth_hash.auto ===
keccak: Keccak256

# === Third-party dependency: eth_hash.main ===
class Keccak256: ...

# === Third-party dependency: eth_typing ===
# Used symbols: Address, Hash32

# === Third-party dependency: hypothesis ===
# Used symbols: HealthCheck, given, settings, strategies

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, param, raises, skip

# === Third-party dependency: rlp ===
# Used symbols: decode, sedes

# === Internal dependency: tests.tools.factories.transaction ===
def new_transaction(vm: VM, from_: Address, to: Address, amount: int = ..., private_key: PrivateKey = ..., gas_price: int = ..., gas: int = ..., data: bytes = ..., nonce: int = ..., chain_id: int = ...) -> Union[SignedTransactionAPI, SpoofTransaction]: ...
def new_access_list_transaction(vm: VM, from_: Address, to: Address, private_key: PrivateKey, amount: int = ..., gas_price: int = ..., gas: int = ..., data: bytes = ..., nonce: int = ..., chain_id: int = ..., access_list: Sequence[Tuple[Address, Sequence[int]]] = ...) -> AccessListTransaction: ...