# === Third-party dependency: eth_utils ===
# Used symbols: is_checksum_address, to_checksum_address, to_hex

# === Third-party dependency: flask ===
# Used symbols: url_for

# === Third-party dependency: gevent ===
# Used symbols: joinall

# === Third-party dependency: grequests ===
get: partial
post: partial
put: partial
patch: partial

# === Third-party dependency: pytest ===
# Used symbols: mark

# === Internal dependency: raiden.api.python ===
class RaidenAPI: ...

# === Internal dependency: raiden.constants ===
class Environment(Enum): ...
BLOCK_ID_LATEST = 'latest'

# === Internal dependency: raiden.exceptions ===
class InvalidSecret(RaidenError): ...

# === Internal dependency: raiden.messages.transfers ===
class Unlock(EnvelopeMessage): ...
class LockedTransfer(LockedTransferBase): ...

# === Internal dependency: raiden.settings ===
DEFAULT_NUMBER_OF_BLOCK_CONFIRMATIONS = BlockTimeout(...)
INTERNAL_ROUTING_DEFAULT_FEE_PERC = 0.02

# === Internal dependency: raiden.tests.integration.api.rest.utils ===
def get_json_response(response): ...
def assert_response_with_code(response, status_code): ...
def assert_response_with_error(response, status_code): ...
def assert_proper_response(response, status_code=...): ...
def api_url_for(api_server, endpoint, **kwargs): ...

# === Internal dependency: raiden.tests.integration.api.utils ===
def prepare_api_server(raiden_app): ...

# === Internal dependency: raiden.tests.integration.fixtures.smartcontracts ===
RED_EYES_PER_CHANNEL_PARTICIPANT_LIMIT = TokenAmount(...)

# === Internal dependency: raiden.tests.utils.client ===
def burn_eth(rpc_client, amount_to_leave=...): ...

# === Internal dependency: raiden.tests.utils.detect_failure ===
def raise_on_failure(test_function): ...
expect_failure = pytest.mark.expect_failure

# === Internal dependency: raiden.tests.utils.events ===
def must_have_event(event_list, dict_data): ...
def must_have_events(event_list, *args): ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_address(): ...
def make_checksum_address(): ...
def make_secret(i=...): ...
def make_secret_with_hash(i=...): ...

# === Internal dependency: raiden.tests.utils.network ===
CHAIN = object(...)

# === Internal dependency: raiden.tests.utils.protocol ===
class WaitForMessage(MessageHandler):
    def __init__(self): ...
    def wait_for_message(self, message_type, attributes): ...
class HoldRaidenEventHandler(EventHandler): ...

# === Internal dependency: raiden.tests.utils.transfer ===
def create_route_state_for_route(apps, token_address, fee_estimate=...): ...
def watch_for_unlock_failures(*apps): ...
def block_offset_timeout(raiden, error_message=..., offset=..., safety_margin=...): ...

# === Internal dependency: raiden.transfer.mediated_transfer.initiator ===
def calculate_fee_margin(payment_amount, estimated_fee): ...

# === Internal dependency: raiden.transfer.state ===
class ChannelState(Enum): ...

# === Internal dependency: raiden.transfer.views ===
def state_from_raiden(raiden): ...
def get_token_network_address_by_token_address(chain_state, token_network_registry_address, token_address): ...

# === Internal dependency: raiden.utils.secrethash ===
def sha256_secrethash(secret): ...

# === Internal dependency: raiden.utils.system ===
def get_system_spec(): ...

# === Internal dependency: raiden.utils.typing ===
from eth_typing import BlockNumber
from raiden_contracts.utils.type_aliases import TokenAmount
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

# === Internal dependency: raiden.waiting ===
def wait_for_block(raiden, block_number, retry_timeout): ...
def wait_for_token_network(raiden, token_network_registry_address, token_address, retry_timeout): ...
class TransferWaitResult(Enum): ...
def wait_for_received_transfer_result(raiden, payment_identifier, amount, retry_timeout, secrethash): ...

# === Third-party dependency: raiden_contracts.constants ===
CONTRACTS_VERSION: str
CONTRACT_CUSTOM_TOKEN: str