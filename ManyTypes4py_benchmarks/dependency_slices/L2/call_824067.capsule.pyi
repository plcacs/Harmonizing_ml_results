# === Internal dependency: eth._utils.address ===
def force_bytes_to_address(value: bytes) -> Address: ...

# === Internal dependency: eth.constants ===
STACK_DEPTH_LIMIT: int
GAS_CALLVALUE: int
GAS_CALLSTIPEND: int
GAS_NEWACCOUNT: int

# === Internal dependency: eth.exceptions ===
class OutOfGas(VMError): ...
class WriteProtection(VMError): ...

# === Internal dependency: eth.vm.opcode ===
class Opcode(Configurable, OpcodeAPI):
    def logger(self) -> ExtendedDebugLogger: ...

# === Third-party dependency: eth_typing ===
# Used symbols: Address