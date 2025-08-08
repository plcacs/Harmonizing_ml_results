from typing import IO, Any, Callable, List, Optional, Tuple, Union
from os import PathLike

T_IO_OR_INT = Union[IO, int]
EXECUTOR_IO = Union[int, Tuple[T_IO_OR_INT, T_IO_OR_INT, T_IO_OR_INT]]

def split_endpoint(endpoint: str) -> Tuple[str, int]:
    ...

class TimeoutHTTPAdapter(HTTPAdapter):

    def __init__(self, *args, timeout: int = 0, **kwargs):
        ...

    def send(self, request, **kwargs):
        ...

class HTTPExecutor(MiHTTPExecutor):

    def __init__(self, command: str, url: str, status: str = '^2\\d\\d$', method: str = 'HEAD', io: Optional[EXECUTOR_IO] = None, cwd: Optional[str] = None, verify_tls: bool = True, **kwargs):
        ...

    def after_start_check(self) -> bool:
        ...

    def start(self) -> 'HTTPExecutor':
        ...

    def kill(self):
        ...

    def wait_for(self, wait_for: Callable[[], bool]) -> 'HTTPExecutor':
        ...

    def running(self) -> bool:
        ...

    def _send_request(self, conn):
        ...

    def _validate_response(self, response) -> bool:
        ...

class JSONRPCExecutor(HTTPExecutor):

    def __init__(self, command: str, url: str, jsonrpc_method: str, jsonrpc_params: Optional[List[Any]] = None, status: str = '^2\\d\\d$', result_validator: Optional[Callable[[Any], Tuple[bool, str]]] = None, io: Optional[EXECUTOR_IO] = None, cwd: Optional[str] = None, verify_tls: bool = True, **kwargs):
        ...

    def _send_request(self, conn):
        ...

    def _validate_response(self, response) -> bool:
        ...

    def stop(self):
        ...
