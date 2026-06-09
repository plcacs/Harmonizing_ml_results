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
BLOCK_ID_LATEST: Literal['latest']
BLOCK_ID_PENDING: Literal['pending']
UINT256_MAX: Any
EMPTY_ADDRESS: Any

# === Internal dependency: raiden.exceptions ===
class RaidenRecoverableError(RaidenError): ...
class BrokenPreconditionError(RaidenError): ...

# === Internal dependency: raiden.network.proxies.utils ===
def raise_on_call_returned_empty(given_block_identifier: BlockIdentifier) -> NoReturn: ...

# === Internal dependency: raiden.network.rpc.client ===
def was_transaction_successfully_mined(transaction: 'TransactionMined') -> bool: ...
def check_address_has_code_handle_pruned_block(client: 'JSONRPCClient', address: Address, contract_name: str, given_block_identifier: BlockIdentifier) -> None: ...
class TransactionMined: ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address: AddressTypes) -> ChecksumAddress: ...
def format_block_id(block_id: BlockIdentifier) -> str: ...

# === Internal dependency: raiden.utils.typing ===
# re-export: from typing import TYPE_CHECKING
# re-export: from typing import Any
# re-export: from typing import Dict
# re-export: from typing import Tuple
# re-export: from eth_typing import Address
# re-export: from eth_typing import BlockNumber
# re-export: from raiden_contracts.utils.type_aliases import TokenAmount
Balance: NewType
TokenAddress: NewType
MonitoringServiceAddress: NewType
OneToNAddress: NewType
TransactionHash: NewType

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