# === Internal dependency: eth.constants ===
CREATE_CONTRACT_ADDRESS: Address

# === Internal dependency: eth.rlp.blocks ===
class BaseBlock(Configurable, rlp.Serializable, BlockAPI): ...

# === Third-party dependency: eth_keys ===
# Used symbols: keys

# === Third-party dependency: eth_typing ===
# Used symbols: Address

# === Third-party dependency: eth_utils ===
# Used symbols: decode_hex, encode_hex

# === Internal dependency: scripts.benchmark._utils.chain_plumbing ===
FUNDED_ADDRESS_PRIVATE_KEY: keys
FUNDED_ADDRESS: Address
SECOND_ADDRESS_PRIVATE_KEY: keys
SECOND_ADDRESS: Address

# === Internal dependency: scripts.benchmark._utils.compile ===
def get_compiled_contract(contract_path: pathlib.Path, contract_name: str) -> Dict[str, str]: ...

# === Internal dependency: scripts.benchmark._utils.reporting ===
class DefaultStat(NamedTuple):
    def cumulate(self, stat: 'DefaultStat', increment_by_counter: bool = ...) -> 'DefaultStat': ...

# === Internal dependency: scripts.benchmark.checks.base_benchmark ===
class BaseBenchmark(ABC):
    def execute(self) -> DefaultStat: ...
    def name(self) -> DefaultStat: ...

# === Internal dependency: tests.tools.factories.transaction ===
def new_transaction(vm: VM, from_: Address, to: Address, amount: int = ..., private_key: PrivateKey = ..., gas_price: int = ..., gas: int = ..., data: bytes = ..., nonce: int = ..., chain_id: int = ...) -> Union[SignedTransactionAPI, SpoofTransaction]: ...

# === Third-party dependency: web3 ===
# Used symbols: Web3