# === Internal dependency: eth.constants ===
CREATE_CONTRACT_ADDRESS = Address(...)

# === Internal dependency: eth.rlp.blocks ===
class BaseBlock(Configurable, rlp.Serializable, BlockAPI): ...

# === Third-party dependency: eth_keys ===
# Used symbols: keys

# === Third-party dependency: eth_typing ===
# Used symbols: Address

# === Third-party dependency: eth_utils ===
# Used symbols: decode_hex, encode_hex

# === Internal dependency: scripts.benchmark._utils.chain_plumbing ===
FUNDED_ADDRESS_PRIVATE_KEY = keys.PrivateKey(...)
FUNDED_ADDRESS = Address(...)
SECOND_ADDRESS_PRIVATE_KEY = keys.PrivateKey(...)
SECOND_ADDRESS = Address(...)

# === Internal dependency: scripts.benchmark._utils.compile ===
def get_compiled_contract(contract_path, contract_name): ...

# === Internal dependency: scripts.benchmark._utils.reporting ===
class DefaultStat(NamedTuple):
    def cumulate(self, stat, increment_by_counter=...): ...

# === Internal dependency: scripts.benchmark.checks.base_benchmark ===
class BaseBenchmark(ABC):
    def execute(self): ...
    def name(self): ...

# === Internal dependency: tests.tools.factories.transaction ===
def new_transaction(vm, from_, to, amount=..., private_key=..., gas_price=..., gas=..., data=..., nonce=..., chain_id=...): ...

# === Third-party dependency: web3 ===
# Used symbols: Web3