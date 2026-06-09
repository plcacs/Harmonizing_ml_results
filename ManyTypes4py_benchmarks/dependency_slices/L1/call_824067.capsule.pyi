# === Internal dependency: eth._utils.address ===
def force_bytes_to_address(value): ...

# === Internal dependency: eth.constants ===
STACK_DEPTH_LIMIT = 1024
GAS_CALLVALUE = 9000
GAS_CALLSTIPEND = 2300
GAS_NEWACCOUNT = 25000

# === Internal dependency: eth.exceptions ===
class OutOfGas(VMError): ...
class WriteProtection(VMError): ...

# === Internal dependency: eth.vm.opcode ===
class Opcode(Configurable, OpcodeAPI):
    def logger(self): ...

# === Third-party dependency: eth_typing ===
# Used symbols: Address