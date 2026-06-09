# === Internal dependency: eth._utils.address ===
def force_bytes_to_address(value): ...

# === Internal dependency: eth._utils.padding ===
def zpad_left(value, to_size): ...
pad32 = zpad_left(...)

# === Internal dependency: eth.chains.mainnet ===
class MainnetHomesteadVM(MainnetDAOValidatorVM): ...
MINING_MAINNET_VMS = (FrontierVM, MainnetHomesteadVM, TangerineWhistleVM, SpuriousDragonVM, ByzantiumVM, PetersburgVM, IstanbulVM, MuirGlacierVM, ...)
POS_MAINNET_VMS = (ParisVM, ShanghaiVM, CancunVM, PragueVM)
MAINNET_VMS = MINING_MAINNET_VMS + POS_MAINNET_VMS

# === Internal dependency: eth.consensus ===
from .context import ConsensusContext

# === Internal dependency: eth.constants ===
GENESIS_DIFFICULTY = 17179869184

# === Internal dependency: eth.db.atomic ===
class AtomicDB(BaseAtomicDB):
    def __init__(self, wrapped_db=...): ...

# === Internal dependency: eth.db.chain ===
class ChainDB(HeaderDB, ChainDatabaseAPI): ...

# === Internal dependency: eth.exceptions ===
class VMError(PyEVMError): ...
class InvalidInstruction(VMError): ...

# === Internal dependency: eth.vm.chain_context ===
class ChainContext(ChainContextAPI):
    def __init__(self, chain_id): ...

# === Internal dependency: eth.vm.forks ===
from .tangerine_whistle import TangerineWhistleVM
from .frontier import FrontierVM
from .homestead import HomesteadVM
from .spurious_dragon import SpuriousDragonVM
from .byzantium import ByzantiumVM
from .constantinople import ConstantinopleVM
from .petersburg import PetersburgVM
from .istanbul import IstanbulVM
from .muir_glacier import MuirGlacierVM
from .berlin import BerlinVM
from .london import LondonVM

# === Internal dependency: eth.vm.forks.cancun.computation ===
class CancunComputation(ShanghaiComputation): ...

# === Internal dependency: eth.vm.message ===
class Message(MessageAPI):
    def __init__(self, gas, to, sender, value, data, code, depth=..., create_address=..., code_address=..., should_transfer_value=..., is_static=..., is_delegation=..., refund=...): ...

# === Internal dependency: eth.vm.opcode_values ===
ADD = 1
MUL = 2
EXP = 10
SHL = 27
SHR = 28
SAR = 29
BALANCE = 49
EXTCODEHASH = 63
CHAINID = 70
SELFBALANCE = 71
BASEFEE = 72
BLOCKHASH = 64
COINBASE = 65
NUMBER = 67
DIFFICULTY = 68
GASLIMIT = 69
SLOAD = 84
PUSH1 = 96
PUSH20 = 115
SELFDESTRUCT = 255

# === Internal dependency: eth.vm.spoof ===
class SpoofTransaction(SpoofAttributes):
    def __init__(self, transaction, **overrides): ...

# === Third-party dependency: eth_utils ===
# Used symbols: ValidationError, decode_hex, encode_hex, hexstr_if_str, int_to_big_endian, to_bytes, to_canonical_address

# === Third-party dependency: pytest ===
# Used symbols: mark, param, raises, warns