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
NULL_ADDRESS_BYTES = bytes(...)
NULL_ADDRESS_HEX = to_hex_address(...)
BLOCK_ID_LATEST = 'latest'

# === Internal dependency: raiden.tests.integration.api.rest.test_rest ===
DEPOSIT_FOR_TEST_API_DEPOSIT_LIMIT = RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT + 2

# === Internal dependency: raiden.tests.integration.api.rest.utils ===
def get_json_response(response): ...
def assert_response_with_code(response, status_code): ...
def assert_response_with_error(response, status_code): ...
def assert_proper_response(response, status_code=...): ...
def api_url_for(api_server, endpoint, **kwargs): ...

# === Internal dependency: raiden.tests.utils.client ===
def burn_eth(rpc_client, amount_to_leave=...): ...

# === Internal dependency: raiden.tests.utils.detect_failure ===
def raise_on_failure(test_function): ...

# === Internal dependency: raiden.tests.utils.events ===
def check_dict_nested_attrs(item, dict_data): ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_address(): ...

# === Internal dependency: raiden.transfer.state ===
class ChannelState(Enum): ...

# === Internal dependency: raiden.transfer.views ===
def state_from_raiden(raiden): ...
def get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address): ...
def get_channelstate_by_token_network_and_partner(chain_state, token_network_address, partner_address): ...

# === Internal dependency: raiden.utils.formatting ===
def to_hex_address(address): ...

# === Internal dependency: raiden.utils.typing ===
from raiden_contracts.utils.type_aliases import TokenAmount

# === Internal dependency: raiden.waiting ===
def wait_for_participant_deposit(raiden, token_network_registry_address, token_address, partner_address, target_address, target_balance, retry_timeout): ...

# === Third-party dependency: raiden_contracts.constants ===
TEST_SETTLE_TIMEOUT_MIN: int
TEST_SETTLE_TIMEOUT_MAX: int