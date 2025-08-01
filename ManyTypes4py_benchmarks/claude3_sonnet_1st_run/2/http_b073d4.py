import json
import os
import platform
import re
import socket
import ssl
import time
from http.client import HTTPSConnection
from json import JSONDecodeError
from os import PathLike
from typing import IO, Any, Callable, Dict, List, Optional, Pattern, Tuple, Union, cast
import structlog
from gevent import subprocess
from mirakuru.base import ENV_UUID
from mirakuru.exceptions import AlreadyRunning, ProcessExitedWithError, TimeoutExpired
from mirakuru.http import HTTPConnection, HTTPException, HTTPExecutor as MiHTTPExecutor
from requests.adapters import HTTPAdapter
from raiden.utils.typing import Endpoint, Host, HostPort, Port

T_IO_OR_INT = Union[IO, int]
EXECUTOR_IO = Union[int, Tuple[T_IO_OR_INT, T_IO_OR_INT, T_IO_OR_INT]]
log = structlog.get_logger(__name__)

def split_endpoint(endpoint: str) -> Tuple[Host, Port]:
    match = re.match('(?:[a-z0-9]*:?//)?([^:/]+)(?::(\\d+))?', endpoint, re.I)
    if not match:
        raise ValueError('Invalid endpoint', endpoint)
    host, port = match.groups()
    if not port:
        port = '0'
    return (Host(host), Port(int(port)))

class TimeoutHTTPAdapter(HTTPAdapter):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout: int = 0
        if 'timeout' in kwargs:
            self.timeout = kwargs['timeout']
            del kwargs['timeout']
        super().__init__(*args, **kwargs)

    def send(self, request: Any, **kwargs: Any) -> Any:
        timeout = kwargs.get('timeout')
        if timeout is None:
            kwargs['timeout'] = self.timeout
        return super().send(request, **kwargs)

class HTTPExecutor(MiHTTPExecutor):
    """Subclass off mirakuru.HTTPExecutor, which allows other methods than HEAD"""

    def __init__(
        self, 
        command: Union[str, List[str]], 
        url: str, 
        status: str = '^2\\d\\d$', 
        method: str = 'HEAD', 
        io: Optional[EXECUTOR_IO] = None, 
        cwd: Optional[Union[str, PathLike]] = None, 
        verify_tls: bool = True, 
        **kwargs: Any
    ) -> None:
        super().__init__(command, url, status, **kwargs)
        self.method: str = method
        self.stdio: Optional[EXECUTOR_IO] = io
        self.cwd: Optional[Union[str, PathLike]] = cwd
        self.verify_tls: bool = verify_tls

    def after_start_check(self) -> bool:
        """Check if defined URL returns expected status to a <method> request."""
        try:
            if self.url.scheme == 'http':
                conn = HTTPConnection(self.host, self.port)
            elif self.url.scheme == 'https':
                ssl_context = None
                if not self.verify_tls:
                    ssl_context = ssl._create_unverified_context()
                conn = HTTPSConnection(self.host, self.port, context=ssl_context)
            else:
                raise ValueError(f'Unsupported URL scheme: "{self.url.scheme}"')
            self._send_request(conn)
            response = conn.getresponse()
            status = str(response.status)
            if not self._validate_response(response):
                return False
            if status == self.status or self.status_re.match(status):
                conn.close()
                return True
        except (HTTPException, socket.timeout, socket.error) as ex:
            log.debug('Executor process not healthy yet', command=self.command, error=ex)
            time.sleep(0.1)
            return False
        return False

    def start(self) -> 'HTTPExecutor':
        """
        Reimplements Executor and SimpleExecutor start to allow setting stdin/stdout/stderr/cwd

        It may break input/output/communicate, but will ensure child output redirects won't
        break parent process by filling the PIPE.
        Also, catches ProcessExitedWithError and raise FileNotFoundError if exitcode was 127
        """
        if self.pre_start_check():
            raise AlreadyRunning(self)
        if self.process is None:
            command = self.command
            if not self._shell:
                command = self.command_parts
            if isinstance(self.stdio, (list, tuple)):
                stdin, stdout, stderr = self.stdio
            else:
                stdin = stdout = stderr = self.stdio
            env = os.environ.copy()
            env[ENV_UUID] = self._uuid
            popen_kwargs: Dict[str, Any] = {
                'shell': self._shell, 
                'stdin': stdin, 
                'stdout': stdout, 
                'stderr': stderr, 
                'universal_newlines': True, 
                'env': env, 
                'cwd': self.cwd
            }
            if platform.system() != 'Windows':
                popen_kwargs['preexec_fn'] = os.setsid
            self.process = subprocess.Popen(command, **popen_kwargs)
        self._set_timeout()
        try:
            self.wait_for(self.check_subprocess)
        except ProcessExitedWithError as e:
            if e.exit_code == 127:
                raise FileNotFoundError(f'Can not execute {command!r}, check that the executable exists.') from e
            else:
                output_file_names = {io.name for io in (stdout, stderr) if hasattr(io, 'name')}
                if output_file_names:
                    log.warning('Process output file(s)', output_files=output_file_names)
            raise
        return self

    def kill(self) -> None:
        STDOUT = subprocess.STDOUT
        ps_param = 'fax'
        if platform.system() == 'Darwin':
            ps_param = 'ax'
        ps_fax = subprocess.check_output(['ps', ps_param], stderr=STDOUT)
        log.debug('Executor process: killing process', command=self.command)
        log.debug('Executor process: current processes', ps_fax=ps_fax)
        super().kill()
        ps_fax = subprocess.check_output(['ps', ps_param], stderr=STDOUT)
        log.debug('Executor process: process killed', command=self.command)
        log.debug('Executor process: current processes', ps_fax=ps_fax)

    def wait_for(self, wait_for: Callable[[], bool]) -> 'HTTPExecutor':
        while self.check_timeout():
            log.debug('Executor process: waiting', command=self.command, until=self._endtime, now=time.time())
            if wait_for():
                log.debug('Executor process: waiting ended', command=self.command, until=self._endtime, now=time.time())
                return self
            time.sleep(self._sleep)
        self.kill()
        raise TimeoutExpired(self, timeout=self._timeout)

    def running(self) -> bool:
        """Include pre_start_check in running, so stop will wait for the underlying listener"""
        return super().running() or self.pre_start_check()

    def _send_request(self, conn: Union[HTTPConnection, HTTPSConnection]) -> None:
        conn.request(self.method, self.url.path)

    def _validate_response(self, response: Any) -> bool:
        return True

class JSONRPCExecutor(HTTPExecutor):

    def __init__(
        self, 
        command: Union[str, List[str]], 
        url: str, 
        jsonrpc_method: str, 
        jsonrpc_params: Optional[List[Any]] = None, 
        status: str = '^2\\d\\d$', 
        result_validator: Optional[Callable[[Any], Tuple[bool, str]]] = None, 
        io: Optional[EXECUTOR_IO] = None, 
        cwd: Optional[Union[str, PathLike]] = None, 
        verify_tls: bool = True, 
        **kwargs: Any
    ) -> None:
        super().__init__(command, url, status, 'POST', io, cwd, verify_tls, **kwargs)
        self.jsonrpc_method: str = jsonrpc_method
        self.jsonrpc_params: List[Any] = jsonrpc_params if jsonrpc_method and jsonrpc_params else []
        self.result_validator: Optional[Callable[[Any], Tuple[bool, str]]] = result_validator

    def _send_request(self, conn: Union[HTTPConnection, HTTPSConnection]) -> None:
        req_body = {'jsonrpc': '2.0', 'method': self.jsonrpc_method, 'params': self.jsonrpc_params, 'id': repr(self)}
        conn.request(method=self.method, url=self.url.path, body=json.dumps(req_body), headers={'Accept': 'application/json', 'Content-Type': 'application/json'})

    def _validate_response(self, response: Any) -> bool:
        try:
            response_data = json.loads(response.read())
            error = response_data.get('error')
            if error:
                log.warning('Executor process: error response', command=self.command, error=error)
                return False
            assert response_data['jsonrpc'] == '2.0', 'invalid jsonrpc version'
            assert 'id' in response_data, 'no id in jsonrpc response'
            result = response_data['result']
            if self.result_validator:
                result_valid, reason = self.result_validator(result)
                if not result_valid:
                    log.warning('Executor process: invalid response', command=self.command, result=result, reason=reason)
                    return False
        except (AssertionError, KeyError, UnicodeDecodeError, JSONDecodeError) as ex:
            log.warning('Executor process: invalid response', command=self.command, error=ex)
            return False
        return True

    def stop(self) -> None:
        log.debug('Executor process: stopping process', command=self.command)
        super().stop()
        log.debug('Executor process: process stopped', command=self.command)
