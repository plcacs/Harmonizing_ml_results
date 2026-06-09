from typing import Any

# === Third-party dependency: eth_utils ===
# Used symbols: encode_hex

# === Third-party dependency: gevent ===
# Used symbols: Timeout, joinall, sleep

# === Third-party dependency: gevent.event ===
class Event(AbstractLinkable):
    def __init__(self) -> Any: ...
    def set(self) -> Any: ...
    def wait(self, timeout = ...) -> Any: ...

# === Third-party dependency: matrix_client.errors ===
class MatrixRequestError(MatrixError): ...

# === Third-party dependency: matrix_client.user ===
class User(object):
    def __init__(self, api, user_id, displayname = ...) -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, param

# === Internal dependency: raiden ===
network: Any

# === Internal dependency: raiden.constants ===
class Environment(Enum): ...
class MatrixMessageType(Enum): ...

# === Internal dependency: raiden.messages.transfers ===
class SecretRequest(SignedRetrieableMessage):
    ...

# === Internal dependency: raiden.network.transport ===
from raiden.network.transport.matrix import MatrixTransport

# === Internal dependency: raiden.network.transport.matrix ===
transport: Any

# === Internal dependency: raiden.network.transport.matrix.client ===
class GMatrixHttpApi(MatrixHttpApi): ...
class GMatrixClient(MatrixClient): ...

# === Internal dependency: raiden.network.transport.matrix.rtc.web_rtc ===
class WebRTCManager(Runnable):
    def __init__(self, node_address, process_messages, signaling_send, stop_event): ...

# === Internal dependency: raiden.network.transport.matrix.transport ===
class _RetryQueue(Runnable):
    def __init__(self, transport, receiver): ...
    def enqueue(self, queue_identifier, messages): ...
    def _check_and_send(self): ...
RETRY_QUEUE_IDLE_AFTER = 10

# === Internal dependency: raiden.network.transport.matrix.utils ===
class UserPresence(Enum): ...

# === Internal dependency: raiden.settings ===
class MatrixTransportConfig:
    ...

# === Internal dependency: raiden.storage.serialization.serializer ===
class MessageSerializer(SerializationBase):
    ...

# === Internal dependency: raiden.tests.utils.factories ===
def make_message_identifier(): ...
def make_signer(): ...
UNIT_SECRET = Secret(...)
UNIT_SECRETHASH = sha256_secrethash(...)
HOP1_KEY = b'11111111111111111111111111111111'
HOP1 = InitiatorAddress(...)

# === Internal dependency: raiden.tests.utils.mocks ===
class MockRaidenService: ...

# === Internal dependency: raiden.transfer.identifiers ===
class CanonicalIdentifier:
    ...
class QueueIdentifier:
CANONICAL_IDENTIFIER_UNORDERED_QUEUE = CanonicalIdentifier(...)

# === Internal dependency: raiden.transfer.mediated_transfer.events ===
class SendSecretRequest(SendMessageEvent): ...

# === Internal dependency: raiden.utils.formatting ===
def to_hex_address(address): ...

# === Internal dependency: raiden.utils.secrethash ===
def sha256_secrethash(secret): ...

# === Internal dependency: raiden.utils.signer ===
class LocalSigner(Signer): ...

# === Internal dependency: raiden.utils.typing ===
from eth_typing import Address
from raiden_contracts.utils.type_aliases import BlockExpiration
T_InitiatorAddress = bytes
InitiatorAddress = NewType(...)
T_PaymentID = int
PaymentID = NewType(...)
T_PaymentAmount = int
PaymentAmount = NewType(...)
T_Secret = bytes
Secret = NewType(...)
T_UserID = str
UserID = NewType(...)
AddressMetadata = Dict[str, Union[UserID, str]]