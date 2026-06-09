from typing import Any

# === Internal dependency: eth._utils.address ===
def force_bytes_to_address(value: bytes) -> Address: ...

# === Internal dependency: eth._utils.padding ===
pad32: bytes

# === Internal dependency: eth.chains.mainnet ===
POS_MAINNET_VMS: Any
MAINNET_VMS: Any

# === Internal dependency: eth.consensus ===
# re-export: from .context import ConsensusContext

# === Internal dependency: eth.constants ===
GENESIS_DIFFICULTY: int

# === Internal dependency: eth.db.atomic ===
class AtomicDB(BaseAtomicDB):
    def __init__(self, wrapped_db: DatabaseAPI = ...) -> None: ...

# === Internal dependency: eth.db.chain ===
class ChainDB(HeaderDB, ChainDatabaseAPI): ...

# === Internal dependency: eth.exceptions ===
class VMError(PyEVMError): ...
class InvalidInstruction(VMError): ...

# === Internal dependency: eth.vm.chain_context ===
class ChainContext(ChainContextAPI):
    def __init__(self, chain_id: Optional[int]) -> None: ...

# === Internal dependency: eth.vm.forks ===
# re-export: from .tangerine_whistle import TangerineWhistleVM
# re-export: from .frontier import FrontierVM
# re-export: from .homestead import HomesteadVM
# re-export: from .spurious_dragon import SpuriousDragonVM
# re-export: from .byzantium import ByzantiumVM
# re-export: from .constantinople import ConstantinopleVM
# re-export: from .petersburg import PetersburgVM
# re-export: from .istanbul import IstanbulVM
# re-export: from .muir_glacier import MuirGlacierVM
# re-export: from .berlin import BerlinVM
# re-export: from .london import LondonVM

# === Internal dependency: eth.vm.forks.cancun.computation ===
class CancunComputation(ShanghaiComputation): ...

# === Internal dependency: eth.vm.message ===
class Message(MessageAPI):
    def __init__(self, gas: int, to: Address, sender: Address, value: int, data: BytesOrView, code: bytes, depth: int = ..., create_address: Address = ..., code_address: Address = ..., should_transfer_value: bool = ..., is_static: bool = ..., is_delegation: bool = ..., refund: int = ...) -> None: ...

# === Internal dependency: eth.vm.opcode_values ===
ADD: int
MUL: int
EXP: int
SHL: int
SHR: int
SAR: int
BALANCE: int
EXTCODEHASH: int
CHAINID: int
SELFBALANCE: int
BASEFEE: int
BLOCKHASH: int
COINBASE: int
NUMBER: int
DIFFICULTY: int
GASLIMIT: int
SLOAD: int
PUSH1: int
PUSH20: int
SELFDESTRUCT: int

# === Internal dependency: eth.vm.spoof ===
class SpoofTransaction(SpoofAttributes):
    def __init__(self, transaction: Union[SignedTransactionAPI, UnsignedTransactionAPI], **overrides: Any) -> None: ...

# === Third-party dependency: eth_utils ===
# Used symbols: ValidationError, decode_hex, encode_hex, hexstr_if_str, int_to_big_endian, to_bytes, to_canonical_address

# === Third-party dependency: pytest ===
# Used symbols: mark, param, raises, warns