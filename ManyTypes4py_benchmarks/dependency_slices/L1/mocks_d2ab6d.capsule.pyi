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
    def __init__(self, pfs_config): ...

# === Internal dependency: raiden.raiden_service ===
class RaidenService(Runnable): ...

# === Internal dependency: raiden.settings ===
class RaidenConfig:
    ...

# === Internal dependency: raiden.storage.serialization ===
from .serializer import JSONSerializer

# === Internal dependency: raiden.storage.sqlite ===
class SerializedSQLiteStorage: ...

# === Internal dependency: raiden.storage.wal ===
class WriteAheadLog(Generic[ST]):
    def __init__(self, state, storage, state_transition): ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_address(): ...
def make_token_network_registry_address(): ...
def make_block_hash(): ...
def make_privkey_address(privatekey=...): ...
def make_canonical_identifier(chain_identifier=..., token_network_address=..., channel_identifier=...): ...
UNIT_CHAIN_ID = ChainID(...)

# === Internal dependency: raiden.tests.utils.transfer ===
def create_route_state_for_route(apps, token_address, fee_estimate=...): ...

# === Internal dependency: raiden.transfer.node ===
def state_transition(chain_state, state_change): ...

# === Internal dependency: raiden.transfer.state ===
class ChainState(State):
    ...

# === Internal dependency: raiden.transfer.views ===
def get_token_network_by_address(chain_state, token_network_address): ...

# === Internal dependency: raiden.utils.keys ===
def privatekey_to_address(private_key_bin): ...

# === Internal dependency: raiden.utils.signer ===
class LocalSigner(Signer):
    def __init__(self, private_key): ...

# === Internal dependency: raiden.utils.typing ===
from typing import Dict
from typing import Tuple
from eth_typing import Address
from eth_typing import BlockNumber
from raiden_contracts.utils.type_aliases import ChainID
from raiden_contracts.utils.type_aliases import TokenAmount
T_BlockTimeout = int
BlockTimeout = NewType(...)
T_TokenAddress = bytes
TokenAddress = NewType(...)
T_UserID = str
UserID = NewType(...)
AddressMetadata = Dict[str, Union[UserID, str]]

# === Unresolved dependency: raiden_contracts.utils.type_aliases ===
# Used unresolved symbols: ChainID