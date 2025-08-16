from typing import IO, Any, Callable, List, Optional, Tuple, Union
import structlog
from gevent import subprocess
from mirakuru.base import ENV_UUID
from mirakuru.exceptions import AlreadyRunning, ProcessExitedWithError, TimeoutExpired
from mirakuru.http import HTTPConnection, HTTPException, HTTPExecutor as MiHTTPExecutor
from requests.adapters import HTTPAdapter
from raiden.utils.typing import Endpoint, Host, HostPort, Port

T_IO_OR_INT = Union[IO, int]
EXECUTOR_IO = Union[int, Tuple[T_IO_OR_INT, T_IO_OR_INT, T_IO_OR_INT]]

log: structlog.types.WrappedLogger = structlog.get_logger(__name__)

def split_endpoint(endpoint: str) -> Tuple[Host, Port]:
    ...

class TimeoutHTTPAdapter(HTTPAdapter):

    def __init__(self, *args, timeout: int = 0, **kwargs):
        ...

    def send(self, request, **kwargs) -> Any:
        ...

class HTTPExecutor(MiHTTPExecutor):
    ...

class JSONRPCExecutor(HTTPExecutor):

    def __init__(self, command: str, url: str, jsonrpc_method: str, jsonrpc_params: Optional[List[Any]] = None, status: str = '^2\\d\\d$', result_validator: Optional[Callable[[Any], Tuple[bool, str]]] = None, io: Optional[EXECUTOR_IO] = None, cwd: Optional[str] = None, verify_tls: bool = True, **kwargs):
        ...

    def _send_request(self, conn) -> None:
        ...

    def _validate_response(self, response) -> bool:
        ...

    def stop(self) -> None:
        ...
