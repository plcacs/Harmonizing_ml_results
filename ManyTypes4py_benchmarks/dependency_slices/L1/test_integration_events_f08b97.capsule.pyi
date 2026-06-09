# === Third-party dependency: eth_utils ===
# Used symbols: keccak

# === Third-party dependency: gevent ===
# Used symbols: Timeout

# === Third-party dependency: pytest ===
# Used symbols: mark

# === Internal dependency: raiden.api.python ===
class RaidenAPI: ...

# === Internal dependency: raiden.blockchain.events ===
def get_contract_events(proxy_manager, abi, contract_address, topics=..., from_block=..., to_block=...): ...
def get_all_netting_channel_events(proxy_manager, token_network_address, netting_channel_identifier, contract_manager, from_block=..., to_block=...): ...

# === Internal dependency: raiden.constants ===
GENESIS_BLOCK_NUMBER = BlockNumber(...)
BLOCK_ID_LATEST = 'latest'

# === Internal dependency: raiden.network.proxies.token_network ===
class TokenNetwork: ...

# === Internal dependency: raiden.settings ===
DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS = BlockTimeout(...)
INTERNAL_ROUTING_DEFAULT_FEE_PERC = 0.02

# === Internal dependency: raiden.tests.utils.detect_failure ===
def raise_on_failure(test_function): ...

# === Internal dependency: raiden.tests.utils.events ===
def search_for_item(item_list, item_type, attributes): ...
def must_have_event(event_list, dict_data): ...
def wait_for_state_change(raiden, item_type, attributes, retry_timeout): ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_secret_with_hash(i=...): ...

# === Internal dependency: raiden.tests.utils.network ===
CHAIN = object(...)

# === Internal dependency: raiden.tests.utils.protocol ===
class HoldRaidenEventHandler(EventHandler): ...

# === Internal dependency: raiden.tests.utils.transfer ===
def get_channelstate(app0, app1, token_network_address): ...
def create_route_state_for_route(apps, token_address, fee_estimate=...): ...
def watch_for_unlock_failures(*apps): ...
def assert_synced_channel_state(token_network_address, app0, balance0, pending_locks0, app1, balance1, pending_locks1): ...
def block_offset_timeout(raiden, error_message=..., offset=..., safety_margin=...): ...

# === Internal dependency: raiden.transfer.events ===
class ContractSendChannelClose(ContractSendEvent):
    ...

# === Internal dependency: raiden.transfer.mediated_transfer.events ===
class SendLockedTransfer(SendMessageEvent): ...

# === Internal dependency: raiden.transfer.mediated_transfer.state_change ===
class ReceiveSecretReveal(AuthenticatedSenderStateChange): ...

# === Internal dependency: raiden.transfer.state ===
from raiden.transfer.architecture import BalanceProofSignedState

# === Internal dependency: raiden.transfer.state_change ===
class ContractReceiveChannelBatchUnlock(ContractReceiveStateChange): ...

# === Internal dependency: raiden.transfer.views ===
def state_from_raiden(raiden): ...
def get_all_messagequeues(chain_state): ...
def get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address): ...
def total_token_network_channels(chain_state, token_network_registry_address, token_address): ...
def get_token_network_by_address(chain_state, token_network_address): ...
def get_channelstate_by_token_network_and_partner(chain_state, token_network_address, partner_address): ...

# === Internal dependency: raiden.utils.formatting ===
def to_checksum_address(address): ...

# === Internal dependency: raiden.utils.secrethash ===
def sha256_secrethash(secret): ...

# === Internal dependency: raiden.utils.typing ===
from typing import Dict
from eth_typing import Address
from eth_typing import BlockNumber
T_Balance = int
Balance = NewType(...)
T_BlockTimeout = int
BlockTimeout = NewType(...)
T_PaymentID = int
PaymentID = NewType(...)
T_PaymentAmount = int
PaymentAmount = NewType(...)
T_FeeAmount = int
FeeAmount = NewType(...)
T_TargetAddress = bytes
TargetAddress = NewType(...)
T_Secret = bytes
Secret = NewType(...)

# === Internal dependency: raiden.waiting ===
def wait_until(func, wait_for=..., sleep_for=...): ...
def wait_for_newchannel(raiden, token_network_registry_address, token_address, partner_address, retry_timeout): ...
def wait_single_channel_deposit(app_deposit, app_partner, registry_address, token_address, total_deposit, retry_timeout): ...
def wait_for_close(raiden, token_network_registry_address, token_address, channel_ids, retry_timeout): ...

# === Third-party dependency: raiden_contracts.constants ===
class ChannelEvent(str, Enum): ...
CONTRACT_TOKEN_NETWORK_REGISTRY: str
CONTRACT_TOKEN_NETWORK: str
EVENT_TOKEN_NETWORK_CREATED: str

# === Unresolved dependency: web3._utils.events ===
# Used unresolved symbols: construct_event_topic_set