import itertools
import time
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from itertools import repeat
from typing import Any, Callable, Container, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import quote
from uuid import UUID, uuid4
import gevent
import structlog
from eth_typing import HexStr
from gevent import Greenlet
from gevent.event import Event
from gevent.lock import Semaphore
from gevent.pool import Pool
from matrix_client.api import MatrixHttpApi
from matrix_client.client import CACHE, MatrixClient
from matrix_client.errors import MatrixHttpLibError, MatrixRequestError
from matrix_client.user import User
from requests import Response
from requests.adapters import HTTPAdapter
from raiden.constants import Environment
from raiden.exceptions import MatrixSyncMaxTimeoutReached, TransportError
from raiden.messages.abstract import Message
from raiden.network.transport.matrix.sync_progress import SyncProgress
from raiden.utils.debugging import IDLE
from raiden.utils.typing import Address, AddressHex, AddressMetadata
from raiden.utils.notifying_queue import NotifyingQueue

log = structlog.get_logger(__name__)

SHUTDOWN_TIMEOUT: int = 35

MatrixMessage: Dict[str, Any] = Dict[str, Any]
JSONResponse: Dict[str, Any] = Dict[str, Any]

@dataclass
class _ReceivedMessageBase:
    pass

@dataclass
class ReceivedRaidenMessage(_ReceivedMessageBase):
    sender_metadata: Optional[Dict[str, Any]]

@dataclass
class ReceivedCallMessage(_ReceivedMessageBase):
    pass

def node_address_from_userid(user_id: Optional[str]) -> Optional[AddressHex]:
    if user_id:
        return AddressHex(HexStr(user_id.split(':', 1)[0][1:]))
    return None

class GMatrixHttpApi(MatrixHttpApi):
    # ...

    def __init__(self, 
                 base_url: str, 
                 token: Optional[str], 
                 pool_maxsize: int = 10, 
                 retry_timeout: int = 60, 
                 retry_delay: Optional[Callable[[], Iterable[int]]] = None, 
                 long_paths: Tuple[str, ...] = (), 
                 user_agent: Optional[str] = None) -> None:
        # ...

class GMatrixClient(MatrixClient):
    # ...

    def __init__(self, 
                 handle_messages_callback: Callable[[List[Message]], None], 
                 base_url: str, 
                 token: Optional[str], 
                 user_id: Optional[str], 
                 valid_cert_check: bool, 
                 sync_filter_limit: int, 
                 cache_level: CACHE, 
                 http_pool_maxsize: int = 10, 
                 http_retry_timeout: int = 60, 
                 http_retry_delay: Optional[Callable[[], Iterable[int]]] = None, 
                 environment: Environment = Environment.PRODUCTION, 
                 user_agent: Optional[str] = None) -> None:
        # ...

    def create_sync_filter(self, limit: int) -> str:
        # ...

    def listen_forever(self, 
                       timeout_ms: int, 
                       latency_ms: int, 
                       exception_handler: Optional[Callable[[Exception], None]] = None, 
                       bad_sync_timeout: int = 5) -> None:
        # ...

    def start_listener_thread(self, 
                             timeout_ms: int, 
                             latency_ms: int, 
                             exception_handler: Optional[Callable[[Exception], None]] = None) -> None:
        # ...

    def stop_listener_thread(self) -> None:
        # ...

    def blocking_sync(self, 
                      timeout_ms: int, 
                      latency_ms: int) -> None:
        # ...

    def _sync(self, 
              timeout_ms: int, 
              latency_ms: int) -> None:
        # ...

    def _handle_message(self, 
                        response_queue: NotifyingQueue, 
                        stop_event: Event) -> None:
        # ...

    def _handle_responses(self, responses: List[Dict[str, Any]]) -> None:
        # ...

    def set_access_token(self, 
                         user_id: str, 
                         token: str) -> None:
        # ...

    def set_sync_filter_id(self, sync_filter_id: str) -> str:
        # ...

@wraps(User.__repr__)
def user__repr__(self) -> str:
    return f'<User id={self.user_id!r}>'
User.__repr__ = user__repr__
