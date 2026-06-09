from typing import Any

# === Third-party dependency: click ===
# Used symbols: secho

# === Internal dependency: raiden.constants ===
class Environment(Enum): ...
class RoutingMode(Enum): ...

# === Internal dependency: raiden.exceptions ===
class PFSReturnedError(ServiceRequestFailed): ...

# === Internal dependency: raiden.network.pathfinding ===
class PFSInfo:
    ...
class PFSConfig:
class PFSProxy:
    def __init__(self, pfs_config: PFSConfig) -> Any: ...

# === Internal dependency: raiden.raiden_service ===
class RaidenService(Runnable): ...

# === Internal dependency: raiden.settings ===
class RaidenConfig:
    ...

# === Internal dependency: raiden.storage.serialization ===
# re-export: from .serializer import JSONSerializer

# === Internal dependency: raiden.storage.sqlite ===
class SerializedSQLiteStorage: ...

# === Internal dependency: raiden.storage.wal ===
class WriteAheadLog(Generic[ST]):
    def __init__(self, state: ST, storage: SerializedSQLiteStorage, state_transition: Callable[[ST, StateChange], TransitionResult[ST]]) -> None: ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_address() -> Address: ...
def make_token_network_registry_address() -> TokenNetworkRegistryAddress: ...
def make_block_hash() -> BlockHash: ...
def make_privkey_address(privatekey: bytes = ...) -> Tuple[PrivateKey, Address]: ...
def make_canonical_identifier(chain_identifier = ..., token_network_address = ..., channel_identifier = ...) -> CanonicalIdentifier: ...
UNIT_CHAIN_ID: ChainID

# === Internal dependency: raiden.tests.utils.transfer ===
def create_route_state_for_route(apps: List[RaidenService], token_address: TokenAddress, fee_estimate: FeeAmount = ...) -> RouteState: ...

# === Internal dependency: raiden.transfer.node ===
def state_transition(chain_state: ChainState, state_change: StateChange) -> TransitionResult[ChainState]: ...

# === Internal dependency: raiden.transfer.state ===
class ChainState(State):
    ...

# === Internal dependency: raiden.transfer.views ===
def get_token_network_by_address(chain_state: ChainState, token_network_address: TokenNetworkAddress) -> Optional[TokenNetworkState]: ...

# === Internal dependency: raiden.utils.keys ===
def privatekey_to_address(private_key_bin: bytes) -> Address: ...

# === Internal dependency: raiden.utils.signer ===
class LocalSigner(Signer):
    def __init__(self, private_key: bytes) -> None: ...

# === Internal dependency: raiden.utils.typing ===
# re-export: from typing import Dict
# re-export: from typing import Tuple
# re-export: from eth_typing import Address
# re-export: from eth_typing import BlockNumber
# re-export: from raiden_contracts.utils.type_aliases import ChainID
# re-export: from raiden_contracts.utils.type_aliases import TokenAmount
BlockTimeout: NewType
TokenAddress: NewType
AddressMetadata: Any

# === Unresolved dependency: raiden_contracts.utils.type_aliases ===
# Used unresolved symbols: ChainID