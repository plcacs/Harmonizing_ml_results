from typing import Any

# === Third-party dependency: eth_utils ===
# Used symbols: keccak

# === Third-party dependency: gevent ===
# Used symbols: Timeout

# === Third-party dependency: pytest ===
# Used symbols: mark

# === Internal dependency: raiden.api.python ===
class RaidenAPI: ...

# === Internal dependency: raiden.blockchain.events ===
def get_contract_events(proxy_manager: ProxyManager, abi: ABI, contract_address: Address, topics: Optional[List[str]] = ..., from_block: BlockIdentifier = ..., to_block: BlockIdentifier = ...) -> List[Dict]: ...
def get_all_netting_channel_events(proxy_manager: ProxyManager, token_network_address: TokenNetworkAddress, netting_channel_identifier: ChannelID, contract_manager: ContractManager, from_block: BlockIdentifier = ..., to_block: BlockIdentifier = ...) -> List[Dict]: ...

# === Internal dependency: raiden.constants ===
BLOCK_ID_LATEST: Literal['latest']
GENESIS_BLOCK_NUMBER: BlockNumber

# === Internal dependency: raiden.network.proxies.token_network ===
class TokenNetwork: ...

# === Internal dependency: raiden.settings ===
INTERNAL_ROUTING_DEFAULT_FEE_PERC: float
DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS: BlockTimeout

# === Internal dependency: raiden.tests.utils.detect_failure ===
def raise_on_failure(test_function: Callable) -> Callable: ...

# === Internal dependency: raiden.tests.utils.events ===
def search_for_item(item_list: Iterable[T], item_type: Type[T], attributes: Mapping) -> Optional[T]: ...
def must_have_event(event_list: Iterable[TM], dict_data: TM) -> Optional[TM]: ...
def wait_for_state_change(raiden: RaidenService, item_type: Type[SC], attributes: Mapping, retry_timeout: float) -> SC: ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_secret_with_hash(i: int = ...) -> Tuple[Secret, SecretHash]: ...

# === Internal dependency: raiden.tests.utils.network ===
CHAIN: Any

# === Internal dependency: raiden.tests.utils.protocol ===
class HoldRaidenEventHandler(EventHandler): ...

# === Internal dependency: raiden.tests.utils.transfer ===
def get_channelstate(app0: RaidenService, app1: RaidenService, token_network_address: TokenNetworkAddress) -> NettingChannelState: ...
def create_route_state_for_route(apps: List[RaidenService], token_address: TokenAddress, fee_estimate: FeeAmount = ...) -> RouteState: ...
def watch_for_unlock_failures(*apps) -> Any: ...
def assert_synced_channel_state(token_network_address: TokenNetworkAddress, app0: RaidenService, balance0: Balance, pending_locks0: List[HashTimeLockState], app1: RaidenService, balance1: Balance, pending_locks1: List[HashTimeLockState]) -> None: ...
def block_offset_timeout(raiden: RaidenService, error_message: Optional[str] = ..., offset: Optional[BlockOffset] = ..., safety_margin: int = ...) -> BlockTimeout: ...

# === Internal dependency: raiden.transfer.events ===
class ContractSendChannelClose(ContractSendEvent):
    ...

# === Internal dependency: raiden.transfer.mediated_transfer.events ===
class SendLockedTransfer(SendMessageEvent): ...

# === Internal dependency: raiden.transfer.mediated_transfer.state_change ===
class ReceiveSecretReveal(AuthenticatedSenderStateChange): ...

# === Internal dependency: raiden.transfer.state ===
# re-export: from raiden.transfer.architecture import BalanceProofSignedState

# === Internal dependency: raiden.transfer.state_change ===
class ContractReceiveChannelBatchUnlock(ContractReceiveStateChange): ...

# === Internal dependency: raiden.transfer.views ===
def state_from_raiden(raiden: 'RaidenService') -> ChainState: ...
def get_all_messagequeues(chain_state: ChainState) -> QueueIdsToQueues: ...
def get_token_network_address_by_token_address(chain_state: ChainState, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress) -> Optional[TokenNetworkAddress]: ...
def total_token_network_channels(chain_state: ChainState, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress) -> int: ...
def get_token_network_by_address(chain_state: ChainState, token_network_address: TokenNetworkAddress) -> Optional[TokenNetworkState]: ...
def get_channelstate_by_token_network_and_partner(chain_state: ChainState, token_network_address: TokenNetworkAddress, partner_address: Address) -> Optional[NettingChannelState]: ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address: AddressTypes) -> ChecksumAddress: ...

# === Internal dependency: raiden.utils.secrethash ===
def sha256_secrethash(secret: Secret) -> SecretHash: ...

# === Internal dependency: raiden.utils.typing ===
# re-export: from typing import Dict
# re-export: from eth_typing import Address
# re-export: from eth_typing import BlockNumber
Balance: NewType
BlockTimeout: NewType
PaymentID: NewType
PaymentAmount: NewType
FeeAmount: NewType
TargetAddress: NewType
Secret: NewType

# === Internal dependency: raiden.waiting ===
def wait_until(func: Callable, wait_for: float = ..., sleep_for: float = ...) -> Any: ...
def wait_for_newchannel(raiden: 'RaidenService', token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, partner_address: Address, retry_timeout: float) -> None: ...
def wait_single_channel_deposit(app_deposit: 'RaidenService', app_partner: 'RaidenService', registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, total_deposit: TokenAmount, retry_timeout: float) -> None: ...
def wait_for_close(raiden: 'RaidenService', token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, channel_ids: List[ChannelID], retry_timeout: float) -> None: ...

# === Third-party dependency: raiden_contracts.constants ===
class ChannelEvent(str, Enum): ...
CONTRACT_TOKEN_NETWORK_REGISTRY: str
CONTRACT_TOKEN_NETWORK: str
EVENT_TOKEN_NETWORK_CREATED: str

# === Unresolved dependency: web3._utils.events ===
# Used unresolved symbols: construct_event_topic_set