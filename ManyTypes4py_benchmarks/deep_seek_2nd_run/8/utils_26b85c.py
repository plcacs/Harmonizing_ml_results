import json
import re
import time
from binascii import Error as DecodeError
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from operator import itemgetter
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlparse
import gevent
import structlog
from cachetools import LRUCache, cached
from eth_utils import decode_hex, encode_hex, to_canonical_address, to_checksum_address, to_normalized_address
from gevent.lock import Semaphore
from matrix_client.errors import MatrixError, MatrixRequestError
from raiden.api.v1.encoding import CapabilitiesSchema
from raiden.constants import DeviceIDs
from raiden.exceptions import InvalidSignature, SerializationError, TransportError
from raiden.messages.abstract import Message, RetrieableMessage, SignedMessage
from raiden.messages.synchronization import Processed
from raiden.network.transport.matrix.client import GMatrixClient, MatrixMessage, User, node_address_from_userid
from raiden.network.utils import get_average_http_response_time
from raiden.storage.serialization.serializer import MessageSerializer
from raiden.utils.gevent import spawn_named
from raiden.utils.signer import Signer, recover
from raiden.utils.typing import Address, AddressMetadata, MessageID, Signature, T_UserID, UserID, cast, typecheck

log: Any = structlog.get_logger(__name__)
cached_deserialize: Callable = lru_cache()(MessageSerializer.deserialize)
JOIN_RETRIES: int = 10
USERID_RE: re.Pattern = re.compile('^@(0x[0-9a-f]{40})(?:\\.[0-9a-f]{8})?(?::.+)?$')
DISPLAY_NAME_HEX_RE: re.Pattern = re.compile('^0x[0-9a-fA-F]{130}$')
MATRIX_MAX_BATCH_SIZE: int = 50000
JSONResponse = Dict[str, Any]
capabilities_schema: CapabilitiesSchema = CapabilitiesSchema()

class UserPresence(Enum):
    ONLINE = 'online'
    UNAVAILABLE = 'unavailable'
    OFFLINE = 'offline'
    UNKNOWN = 'unknown'
    SERVER_ERROR = 'server_error'

class AddressReachability(Enum):
    REACHABLE = 1
    UNREACHABLE = 2
    UNKNOWN = 3

@dataclass
class ReachabilityState:
    reachability: AddressReachability
    time: datetime

EPOCH: datetime = datetime(1970, 1, 1)
UNKNOWN_REACHABILITY_STATE: ReachabilityState = ReachabilityState(AddressReachability.UNKNOWN, EPOCH)
USER_PRESENCE_REACHABLE_STATES: Set[UserPresence] = {UserPresence.ONLINE, UserPresence.UNAVAILABLE}
USER_PRESENCE_TO_ADDRESS_REACHABILITY: Dict[UserPresence, AddressReachability] = {
    UserPresence.ONLINE: AddressReachability.REACHABLE,
    UserPresence.UNAVAILABLE: AddressReachability.REACHABLE,
    UserPresence.OFFLINE: AddressReachability.UNREACHABLE,
    UserPresence.UNKNOWN: AddressReachability.UNKNOWN,
    UserPresence.SERVER_ERROR: AddressReachability.UNKNOWN
}

def address_from_userid(user_id: str) -> Optional[Address]:
    match = USERID_RE.match(user_id)
    if not match:
        return None
    encoded_address = match.group(1)
    address = to_canonical_address(encoded_address)
    return address

def is_valid_userid_for_address(user_id: str, address: Address) -> bool:
    try:
        typecheck(user_id, T_UserID)
    except ValueError:
        return False
    user_id_address = address_from_userid(user_id)
    if not user_id_address:
        return False
    return address == user_id_address

def get_user_id_from_metadata(address: Address, address_metadata: Optional[AddressMetadata] = None) -> Optional[UserID]:
    """Get user-id from the address-metadata, if it is valid and present."""
    if address_metadata is not None:
        user_id = address_metadata.get('user_id')
        if is_valid_userid_for_address(user_id, address):
            user_id = cast(UserID, user_id)
            return user_id
    return None

class DisplayNameCache:
    def __init__(self) -> None:
        self.userid_to_displayname: Dict[UserID, str] = {}

    def warm_users(self, users: List[User]) -> None:
        for user in users:
            user_id = user.user_id
            cached_displayname = self.userid_to_displayname.get(user_id)
            if cached_displayname is None:
                if not user.displayname:
                    try:
                        user.get_display_name()
                    except MatrixRequestError as ex:
                        log.error('Ignoring Matrix error in `get_display_name`', exc_info=ex, user_id=user.user_id)
                if user.displayname is not None:
                    self.userid_to_displayname[user.user_id] = user.displayname
            elif user.displayname is None:
                user.displayname = cached_displayname
            elif user.displayname != cached_displayname:
                log.debug('User displayname changed!', cached=cached_displayname, current=user.displayname)
                self.userid_to_displayname[user.user_id] = user.displayname

class MessageAckTimingKeeper:
    def __init__(self) -> None:
        self._seen_messages: Set[MessageID] = set()
        self._messages_in_flight: Dict[MessageID, float] = {}
        self._durations: List[float] = []

    def add_message(self, message: Message) -> None:
        if message.message_identifier in self._seen_messages:
            return
        self._messages_in_flight[message.message_identifier] = time.monotonic()
        self._seen_messages.add(message.message_identifier)

    def finalize_message(self, message: Message) -> None:
        start_time = self._messages_in_flight.pop(message.message_identifier, None)
        if start_time is None:
            return
        self._durations.append(time.monotonic() - start_time)

    def generate_report(self) -> List[float]:
        if not self._durations:
            return []
        return sorted(self._durations)

def first_login(
    client: GMatrixClient,
    signer: Signer,
    username: str,
    capabilities: Dict[str, Any],
    device_id: DeviceIDs
) -> User:
    """Login within a server."""
    server_url = client.api.base_url
    server_name = urlparse(server_url).netloc
    password = encode_hex(signer.sign(server_name.encode()))
    client.login(username, password, sync=False, device_id=device_id.value)
    client.api.disable_push_notifications()
    signature_bytes = signer.sign(client.user_id.encode())
    signature_hex = encode_hex(signature_bytes)
    user = client.get_user(client.user_id)
    try:
        current_display_name = user.get_display_name()
    except MatrixRequestError as ex:
        log.error('Ignoring Matrix error in `get_display_name`', exc_info=ex, user_id=user.user_id)
        current_display_name = ''
    if current_display_name != signature_hex:
        user.set_display_name(signature_hex)
    try:
        current_capabilities = client.api.get_avatar_url(client.user_id) or ''
    except MatrixRequestError as ex:
        log.error('Ignoring Matrix error in `get_avatar_url`', exc_info=ex, user_id=user.user_id)
        current_capabilities = ''
    cap_str = capabilities.get('capabilities', 'mxc://')
    if current_capabilities != cap_str:
        user.set_avatar_url(cap_str)
    log.debug('Logged in', node=to_checksum_address(username), homeserver=server_name, server_url=server_url, capabilities=capabilities)
    return user

def is_valid_username(username: str, server_name: str, user_id: str) -> bool:
    _match_user = re.match(f'^@{re.escape(username)}:{re.escape(server_name)}$', user_id)
    return bool(_match_user)

def login_with_token(client: GMatrixClient, user_id: str, access_token: str) -> Optional[User]:
    """Reuse an existing authentication code."""
    client.set_access_token(user_id=user_id, token=access_token)
    try:
        client.api.get_devices()
    except MatrixRequestError as ex:
        log.debug("Couldn't use previous login credentials", node=node_address_from_userid(client.user_id), prev_user_id=user_id, _exception=ex)
        return None
    log.debug('Success. Valid previous credentials', node=node_address_from_userid(client.user_id), user_id=user_id)
    return client.get_user(client.user_id)

def login(
    client: GMatrixClient,
    signer: Signer,
    device_id: DeviceIDs,
    prev_auth_data: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None
) -> User:
    """Login with a matrix server."""
    if capabilities is None:
        capabilities = {}
    server_url = client.api.base_url
    server_name = urlparse(server_url).netloc
    username = str(to_normalized_address(signer.address))
    if prev_auth_data and prev_auth_data.count('/') == 1:
        user_id, _, access_token = prev_auth_data.partition('/')
        if is_valid_username(username, server_name, user_id):
            user = login_with_token(client, user_id, access_token)
            if user is not None:
                return user
        else:
            log.debug('Auth data is invalid, discarding', node=username, user_id=user_id, server_name=server_name)
    try:
        cap = capabilities_schema.dump({'capabilities': capabilities})
    except ValueError:
        raise Exception('error serializing')
    return first_login(client, signer, username, cap, device_id)

def validate_userid_signature(user: User) -> Optional[Address]:
    """Validate a userId format and signature on displayName, and return its address"""
    return validate_user_id_signature(user.user_id, user.displayname)

@cached(cache=LRUCache(128), lock=Semaphore())
def validate_user_id_signature(user_id: str, displayname: Optional[str]) -> Optional[Address]:
    match = USERID_RE.match(user_id)
    if not match:
        log.warning('Invalid user id', user=user_id)
        return None
    if displayname is None:
        log.warning('Displayname not set', user=user_id)
        return None
    encoded_address = match.group(1)
    address = to_canonical_address(encoded_address)
    try:
        if DISPLAY_NAME_HEX_RE.match(displayname):
            signature_bytes = decode_hex(displayname)
        else:
            log.warning('Displayname invalid format', user=user_id, displayname=displayname)
            return None
        recovered = recover(data=user_id.encode(), signature=Signature(signature_bytes))
        if not (address and recovered and (recovered == address)):
            log.warning('Unexpected signer of displayname', user=user_id)
            return None
    except (DecodeError, TypeError, InvalidSignature, MatrixRequestError, json.decoder.JSONDecodeError):
        return None
    return address

def sort_servers_closest(
    servers: Sequence[str],
    max_timeout: float = 3.0,
    samples_per_server: int = 3,
    sample_delay: float = 0.125
) -> Dict[str, float]:
    """Sorts a list of servers by http round-trip time"""
    if not {urlparse(url).scheme for url in servers}.issubset({'http', 'https'}):
        raise TransportError('Invalid server urls')
    rtt_greenlets = set((spawn_named('get_average_http_response_time', get_average_http_response_time, url=server_url, samples=samples_per_server, sample_delay=sample_delay) for server_url in servers))
    total_timeout = samples_per_server * (max_timeout + sample_delay)
    results = []
    for greenlet in gevent.iwait(rtt_greenlets, timeout=total_timeout):
        result = greenlet.get()
        if result is not None:
            results.append(result)
    gevent.killall(rtt_greenlets)
    if not results:
        raise TransportError(f'No Matrix server available with good latency, requests takes more than {max_timeout} seconds.')
    server_url_to_rtt = dict(sorted(results, key=itemgetter(1)))
    log.debug('Available Matrix homeservers', servers=server_url_to_rtt)
    return server_url_to_rtt

def make_client(
    handle_messages_callback: Callable,
    servers: Sequence[str],
    *args: Any,
    **kwargs: Any
) -> GMatrixClient:
    """Given a list of possible servers, chooses the closest available and create a GMatrixClient"""
    if len(servers) > 1:
        sorted_servers = sort_servers_closest(servers)
        log.debug('Selecting best matrix server', sorted_servers=sorted_servers)
    elif len(servers) == 1:
        sorted_servers = {servers[0]: 0}
    else:
        raise TransportError('No valid servers list given')
    last_ex = None
    for server_url, rtt in sorted_servers.items():
        client = GMatrixClient(handle_messages_callback, server_url, *args, **kwargs)
        retries = 3
        while retries:
            retries -= 1
            try:
                client.api._send('GET', '/versions', api_path='/_matrix/client')
            except MatrixRequestError as ex:
                log.warning('Matrix server returned an error, retrying', server_url=server_url, _exception=ex)
                last_ex = ex
            except MatrixError as ex:
                log.warning('Selected server not usable', server_url=server_url, _exception=ex)
                last_ex = ex
                retries = 0
            else:
                log.info('Using Matrix server', server_url=server_url, server_ident=client.api.server_ident, average_rtt=rtt)
                return client
    raise TransportError('Unable to find a reachable Matrix server. Please check your network connectivity.') from last_ex

def validate_and_parse_message(data: Union[str, bytes], peer_address: Address) -> List[SignedMessage]:
    messages = []
    if not isinstance(data, str):
        log.warning('Received Message body not a string', message_data=data, peer_address=to_checksum_address(peer_address))
        return []
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            message = cached_deserialize(line, peer_address)
        except SerializationError as ex:
            log.warning('Not a valid Message', message_data=line, peer_address=to_checksum_address(peer_address), _exc=ex)
            continue
        if not isinstance(message, SignedMessage):
            log.warning('Message not a SignedMessage!', message=message, peer_address=to_checksum_address(peer_address))
            continue
        if message.sender != peer_address:
            log.warning('Message not signed by sender!', message=message, signer=message.sender, peer_address=to_checksum_address(peer_address))
            continue
        messages.append(message)
    return messages

def my_place_or_yours(our_address: Address, partner_address: Address) -> Address:
    """Convention to compare two addresses."""
    if our_address == partner_address:
        raise ValueError('Addresses to compare must differ')
    sorted_addresses = sorted([our_address, partner_address])
    return sorted_addresses[0]

def make_message_batches(message_texts: List[str], _max_batch_size: int = MATRIX_MAX_BATCH_SIZE) -> Generator[str, None, None]:
    """Group messages into newline separated batches not exceeding ``_max_batch_size``."""
    current_batch = []
    size = 0
    for message_text in message_texts:
        if size + len(message_text) > _max_batch_size:
            if size == 0:
                raise TransportError(f'Message exceeds batch size. Size: {len(message_text)}, Max: {MATRIX_MAX_BATCH_SIZE}, Message: {message_text}')
            yield '\n'.join(current_batch)
            current_batch = []
            size = 0
        current_batch.append(message_text)
        size += len(message_text)
    if current_batch:
        yield '\n'.join(current_batch)

def make_user_id(address: Address, home_server: str) -> UserID:
    return UserID(f'@{to_normalized_address(address)}:{home_server}')
