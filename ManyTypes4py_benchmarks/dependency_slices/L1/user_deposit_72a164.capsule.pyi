from typing import Any

# === Third-party dependency: eth_utils ===
# Used symbols: is_binary_address, to_canonical_address

# === Third-party dependency: gevent.event ===
class AsyncResult(AbstractLinkable):
    def __init__(self) -> Any: ...
    def ready(self) -> Any: ...
    def set_exception(self, exception, exc_info = ...) -> Any: ...

# === Third-party dependency: gevent.thread ===
allocate_lock = LockType

# === Third-party dependency: gevent.threading ===
# re-export: from gevent.thread import allocate_lock as _allocate_lock
Lock = allocate_lock

# === Internal dependency: raiden.constants ===
UINT256_MAX = 2 ** 256 - 1
EMPTY_ADDRESS = b'\x00' * 20
BLOCK_ID_LATEST = 'latest'
BLOCK_ID_PENDING = 'pending'

# === Internal dependency: raiden.exceptions ===
class RaidenRecoverableError(RaidenError): ...
class BrokenPreconditionError(RaidenError): ...

# === Internal dependency: raiden.network.proxies.utils ===
def raise_on_call_returned_empty(given_block_identifier): ...

# === Internal dependency: raiden.network.rpc.client ===
def was_transaction_successfully_mined(transaction): ...
def check_address_has_code_handle_pruned_block(client, address, contract_name, given_block_identifier): ...
class TransactionMined: ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address): ...
def format_block_id(block_id): ...

# === Internal dependency: raiden.utils.typing ===
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Tuple
from eth_typing import Address
from eth_typing import BlockNumber
from raiden_contracts.utils.type_aliases import TokenAmount
T_Balance = int
Balance = NewType(...)
T_TokenAddress = bytes
TokenAddress = NewType(...)
T_MonitoringServiceAddress = bytes
MonitoringServiceAddress = NewType(...)
T_OneToNAddress = bytes
OneToNAddress = NewType(...)
T_TransactionHash = bytes
TransactionHash = NewType(...)

# === Third-party dependency: raiden_contracts.constants ===
CONTRACT_MONITORING_SERVICE: str
CONTRACT_USER_DEPOSIT: str
CONTRACT_ONE_TO_N: str

# === Third-party dependency: raiden_contracts.contract_manager ===
def gas_measurements(version: Optional[str] = ...) -> Dict[str, int]: ...

# === Third-party dependency: structlog ===
# Used symbols: get_logger

# === Unresolved dependency: web3.exceptions ===
# Used unresolved symbols: BadFunctionCallOutput