# === Internal dependency: eth._utils.address ===
def force_bytes_to_address(value): ...

# === Internal dependency: eth.chains.base ===
class MiningChain(Chain, MiningChainAPI): ...

# === Internal dependency: eth.constants ===
ZERO_ADDRESS = Address(...)
BLANK_ROOT_HASH = Hash32(...)

# === Internal dependency: eth.db.atomic ===
class AtomicDB(BaseAtomicDB): ...

# === Internal dependency: eth.db.chain ===
class ChainDB(HeaderDB, ChainDatabaseAPI): ...

# === Internal dependency: eth.db.chain_gaps ===
GENESIS_CHAIN_GAPS = ((), BlockNumber(...))

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
    def __init__(self, **kwargs): ...
    def __init__(self, difficulty, block_number, gas_limit, timestamp=..., coinbase=..., parent_hash=..., uncles_hash=..., state_root=..., transaction_root=..., receipt_root=..., bloom=..., gas_used=..., extra_data=..., mix_hash=..., nonce=...): ...
    def hash(self): ...

# === Internal dependency: eth.tools.builder.chain ===
class API:
    ...
api = API(...)

# === Internal dependency: eth.tools.rlp ===
assert_headers_eq = replace_exceptions(...)(...)

# === Internal dependency: eth.vm.forks ===
from .berlin import BerlinVM
from .london import LondonVM

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
def new_transaction(vm, from_, to, amount=..., private_key=..., gas_price=..., gas=..., data=..., nonce=..., chain_id=...): ...
def new_access_list_transaction(vm, from_, to, private_key, amount=..., gas_price=..., gas=..., data=..., nonce=..., chain_id=..., access_list=...): ...