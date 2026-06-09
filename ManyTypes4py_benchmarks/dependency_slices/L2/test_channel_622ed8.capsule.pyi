from typing import Any

# === Third-party dependency: eth_utils ===
# Used symbols: to_canonical_address, to_checksum_address

# === Third-party dependency: gevent ===
# Used symbols: Timeout, joinall, spawn

# === Third-party dependency: grequests ===
get: partial
post: partial
put: partial
patch: partial

# === Third-party dependency: pytest ===
# Used symbols: mark

# === Internal dependency: raiden.constants ===
BLOCK_ID_LATEST: Literal['latest']
NULL_ADDRESS_HEX: to_hex_address

# === Internal dependency: raiden.tests.integration.api.rest.test_rest ===
DEPOSIT_FOR_TEST_API_DEPOSIT_LIMIT: Any

# === Internal dependency: raiden.tests.integration.api.rest.utils ===
def get_json_response(response) -> Any: ...
def assert_response_with_code(response, status_code) -> Any: ...
def assert_response_with_error(response, status_code) -> Any: ...
def assert_proper_response(response, status_code = ...) -> Any: ...
def api_url_for(api_server, endpoint, **kwargs) -> Any: ...

# === Internal dependency: raiden.tests.utils.client ===
def burn_eth(rpc_client: JSONRPCClient, amount_to_leave: int = ...) -> None: ...

# === Internal dependency: raiden.tests.utils.detect_failure ===
def raise_on_failure(test_function: Callable) -> Callable: ...

# === Internal dependency: raiden.tests.utils.events ===
def check_dict_nested_attrs(item: Mapping, dict_data: Mapping) -> bool: ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_address() -> Address: ...

# === Internal dependency: raiden.transfer.state ===
class ChannelState(Enum): ...

# === Internal dependency: raiden.transfer.views ===
def state_from_raiden(raiden: 'RaidenService') -> ChainState: ...
def get_token_network_address_by_token_address(chain_state: ChainState, token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress) -> Optional[TokenNetworkAddress]: ...
def get_channelstate_by_token_network_and_partner(chain_state: ChainState, token_network_address: TokenNetworkAddress, partner_address: Address) -> Optional[NettingChannelState]: ...

# === Internal dependency: raiden.utils.formatting ===
def to_hex_address(address: AddressTypes) -> AddressHex: ...

# === Internal dependency: raiden.utils.typing ===
# re-export: from raiden_contracts.utils.type_aliases import TokenAmount

# === Internal dependency: raiden.waiting ===
def wait_for_participant_deposit(raiden: 'RaidenService', token_network_registry_address: TokenNetworkRegistryAddress, token_address: TokenAddress, partner_address: Address, target_address: Address, target_balance: TokenAmount, retry_timeout: float) -> None: ...

# === Third-party dependency: raiden_contracts.constants ===
TEST_SETTLE_TIMEOUT_MIN: int
TEST_SETTLE_TIMEOUT_MAX: int