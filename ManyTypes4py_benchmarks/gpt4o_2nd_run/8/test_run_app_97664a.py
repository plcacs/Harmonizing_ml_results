import asyncio
import contextlib
import logging
import os
import platform
import signal
import socket
import ssl
import subprocess
import sys
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Coroutine, Dict, Iterator, List, NoReturn, Optional, Set, Tuple
from unittest import mock
from uuid import uuid4
import pytest
from pytest_mock import MockerFixture
from aiohttp import ClientConnectorError, ClientSession, ClientTimeout, WSCloseCode, web
from aiohttp.log import access_logger
from aiohttp.test_utils import make_mocked_coro
from aiohttp.web_protocol import RequestHandler
from aiohttp.web_runner import BaseRunner

_has_unix_domain_socks: bool = hasattr(socket, 'AF_UNIX')
if _has_unix_domain_socks:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as _abstract_path_sock:
        try:
            _abstract_path_sock.bind(b'\x00' + uuid4().hex.encode('ascii'))
        except FileNotFoundError:
            _abstract_path_failed: bool = True
        else:
            _abstract_path_failed = False
        finally:
            del _abstract_path_sock
else:
    _abstract_path_failed = True

skip_if_no_abstract_paths = pytest.mark.skipif(_abstract_path_failed, reason='Linux-style abstract paths are not supported.')
skip_if_no_unix_socks = pytest.mark.skipif(not _has_unix_domain_socks, reason='Unix domain sockets are not supported')
del _has_unix_domain_socks, _abstract_path_failed

HAS_IPV6: bool = socket.has_ipv6
if HAS_IPV6:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM):
            pass
    except OSError:
        HAS_IPV6 = False

def skip_if_on_windows() -> None:
    if platform.system() == 'Windows':
        pytest.skip('the test is not valid for Windows')

@pytest.fixture
def patched_loop(loop: asyncio.AbstractEventLoop) -> Iterator[asyncio.AbstractEventLoop]:
    server = mock.create_autospec(asyncio.Server, spec_set=True, instance=True)
    server.wait_closed.return_value = None
    unix_server = mock.create_autospec(asyncio.Server, spec_set=True, instance=True)
    unix_server.wait_closed.return_value = None
    with mock.patch.object(loop, 'create_server', autospec=True, spec_set=True, return_value=server):
        with mock.patch.object(loop, 'create_unix_server', autospec=True, spec_set=True, return_value=unix_server):
            asyncio.set_event_loop(loop)
            yield loop

def stopper(loop: asyncio.AbstractEventLoop) -> Callable[..., None]:
    def raiser() -> None:
        raise KeyboardInterrupt

    def f(*args: Any) -> None:
        loop.call_soon(raiser)
    return f

def test_run_app_http(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    startup_handler = make_mocked_coro()
    app.on_startup.append(startup_handler)
    cleanup_handler = make_mocked_coro()
    app.on_cleanup.append(cleanup_handler)
    web.run_app(app, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(mock.ANY, None, 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None)
    startup_handler.assert_called_once_with(app)
    cleanup_handler.assert_called_once_with(app)

def test_run_app_close_loop(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    web.run_app(app, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(mock.ANY, None, 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None)
    assert patched_loop.is_closed()

mock_unix_server_single: List[mock.call] = [mock.call(mock.ANY, '/tmp/testsock1.sock', ssl=None, backlog=128)]
mock_unix_server_multi: List[mock.call] = [mock.call(mock.ANY, '/tmp/testsock1.sock', ssl=None, backlog=128), mock.call(mock.ANY, '/tmp/testsock2.sock', ssl=None, backlog=128)]
mock_server_single: List[mock.call] = [mock.call(mock.ANY, '127.0.0.1', 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None)]
mock_server_multi: List[mock.call] = [mock.call(mock.ANY, '127.0.0.1', 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None), mock.call(mock.ANY, '192.168.1.1', 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None)]
mock_server_default_8989: List[mock.call] = [mock.call(mock.ANY, None, 8989, ssl=None, backlog=128, reuse_address=None, reuse_port=None)]
mock_socket: mock.Mock = mock.Mock(getsockname=lambda: ('mock-socket', 123))

mixed_bindings_tests: Tuple[Tuple[str, Dict[str, Any], List[mock.call], List[mock.call]], ...] = (
    ('Nothing Specified', {}, [mock.call(mock.ANY, None, 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None)], []),
    ('Port Only', {'port': 8989}, mock_server_default_8989, []),
    ('Multiple Hosts', {'host': ('127.0.0.1', '192.168.1.1')}, mock_server_multi, []),
    ('Multiple Paths', {'path': ('/tmp/testsock1.sock', '/tmp/testsock2.sock')}, [], mock_unix_server_multi),
    ('Multiple Paths, Port', {'path': ('/tmp/testsock1.sock', '/tmp/testsock2.sock'), 'port': 8989}, mock_server_default_8989, mock_unix_server_multi),
    ('Multiple Paths, Single Host', {'path': ('/tmp/testsock1.sock', '/tmp/testsock2.sock'), 'host': '127.0.0.1'}, mock_server_single, mock_unix_server_multi),
    ('Single Path, Single Host', {'path': '/tmp/testsock1.sock', 'host': '127.0.0.1'}, mock_server_single, mock_unix_server_single),
    ('Single Path, Multiple Hosts', {'path': '/tmp/testsock1.sock', 'host': ('127.0.0.1', '192.168.1.1')}, mock_server_multi, mock_unix_server_single),
    ('Single Path, Port', {'path': '/tmp/testsock1.sock', 'port': 8989}, mock_server_default_8989, mock_unix_server_single),
    ('Multiple Paths, Multiple Hosts, Port', {'path': ('/tmp/testsock1.sock', '/tmp/testsock2.sock'), 'host': ('127.0.0.1', '192.168.1.1'), 'port': 8000}, [mock.call(mock.ANY, '127.0.0.1', 8000, ssl=None, backlog=128, reuse_address=None, reuse_port=None), mock.call(mock.ANY, '192.168.1.1', 8000, ssl=None, backlog=128, reuse_address=None, reuse_port=None)], mock_unix_server_multi),
    ('Only socket', {'sock': [mock_socket]}, [mock.call(mock.ANY, ssl=None, sock=mock_socket, backlog=128)], []),
    ('Socket, port', {'sock': [mock_socket], 'port': 8765}, [mock.call(mock.ANY, None, 8765, ssl=None, backlog=128, reuse_address=None, reuse_port=None), mock.call(mock.ANY, sock=mock_socket, ssl=None, backlog=128)], []),
    ('Socket, Host, No port', {'sock': [mock_socket], 'host': 'localhost'}, [mock.call(mock.ANY, 'localhost', 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None), mock.call(mock.ANY, sock=mock_socket, ssl=None, backlog=128)], []),
    ('reuse_port', {'reuse_port': True}, [mock.call(mock.ANY, None, 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=True)], []),
    ('reuse_address', {'reuse_address': False}, [mock.call(mock.ANY, None, 8080, ssl=None, backlog=128, reuse_address=False, reuse_port=None)], []),
    ('reuse_port, reuse_address', {'reuse_address': True, 'reuse_port': True}, [mock.call(mock.ANY, None, 8080, ssl=None, backlog=128, reuse_address=True, reuse_port=True)], []),
    ('Port, reuse_port', {'port': 8989, 'reuse_port': True}, [mock.call(mock.ANY, None, 8989, ssl=None, backlog=128, reuse_address=None, reuse_port=True)], []),
    ('Multiple Hosts, reuse_port', {'host': ('127.0.0.1', '192.168.1.1'), 'reuse_port': True}, [mock.call(mock.ANY, '127.0.0.1', 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=True), mock.call(mock.ANY, '192.168.1.1', 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=True)], []),
    ('Multiple Paths, Port, reuse_address', {'path': ('/tmp/testsock1.sock', '/tmp/testsock2.sock'), 'port': 8989, 'reuse_address': False}, [mock.call(mock.ANY, None, 8989, ssl=None, backlog=128, reuse_address=False, reuse_port=None)], mock_unix_server_multi),
    ('Multiple Paths, Single Host, reuse_address, reuse_port', {'path': ('/tmp/testsock1.sock', '/tmp/testsock2.sock'), 'host': '127.0.0.1', 'reuse_address': True, 'reuse_port': True}, [mock.call(mock.ANY, '127.0.0.1', 8080, ssl=None, backlog=128, reuse_address=True, reuse_port=True)], mock_unix_server_multi)
)

mixed_bindings_test_ids: List[str] = [test[0] for test in mixed_bindings_tests]
mixed_bindings_test_params: List[Tuple[Dict[str, Any], List[mock.call], List[mock.call]]] = [test[1:] for test in mixed_bindings_tests]

@pytest.mark.parametrize('run_app_kwargs, expected_server_calls, expected_unix_server_calls', mixed_bindings_test_params, ids=mixed_bindings_test_ids)
def test_run_app_mixed_bindings(run_app_kwargs: Dict[str, Any], expected_server_calls: List[mock.call], expected_unix_server_calls: List[mock.call], patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    web.run_app(app, print=stopper(patched_loop), **run_app_kwargs, loop=patched_loop)
    assert patched_loop.create_unix_server.mock_calls == expected_unix_server_calls
    assert patched_loop.create_server.mock_calls == expected_server_calls

def test_run_app_https(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    ssl_context = ssl.create_default_context()
    web.run_app(app, ssl_context=ssl_context, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(mock.ANY, None, 8443, ssl=ssl_context, backlog=128, reuse_address=None, reuse_port=None)

def test_run_app_nondefault_host_port(patched_loop: asyncio.AbstractEventLoop, unused_port_socket: socket.socket) -> None:
    port = unused_port_socket.getsockname()[1]
    host = '127.0.0.1'
    app = web.Application()
    web.run_app(app, host=host, port=port, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(mock.ANY, host, port, ssl=None, backlog=128, reuse_address=None, reuse_port=None)

def test_run_app_with_sock(patched_loop: asyncio.AbstractEventLoop, unused_port_socket: socket.socket) -> None:
    sock = unused_port_socket
    app = web.Application()
    web.run_app(app, sock=sock, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(mock.ANY, sock=sock, ssl=None, backlog=128)

def test_run_app_multiple_hosts(patched_loop: asyncio.AbstractEventLoop) -> None:
    hosts = ('127.0.0.1', '127.0.0.2')
    app = web.Application()
    web.run_app(app, host=hosts, print=stopper(patched_loop), loop=patched_loop)
    calls = map(lambda h: mock.call(mock.ANY, h, 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None), hosts)
    patched_loop.create_server.assert_has_calls(calls)

def test_run_app_custom_backlog(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    web.run_app(app, backlog=10, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(mock.ANY, None, 8080, ssl=None, backlog=10, reuse_address=None, reuse_port=None)

def test_run_app_custom_backlog_unix(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    web.run_app(app, path='/tmp/tmpsock.sock', backlog=10, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_unix_server.assert_called_with(mock.ANY, '/tmp/tmpsock.sock', ssl=None, backlog=10)

@skip_if_no_unix_socks
def test_run_app_http_unix_socket(patched_loop: asyncio.AbstractEventLoop, unix_sockname: str) -> None:
    app = web.Application()
    printer = mock.Mock(wraps=stopper(patched_loop))
    web.run_app(app, path=unix_sockname, print=printer, loop=patched_loop)
    patched_loop.create_unix_server.assert_called_with(mock.ANY, unix_sockname, ssl=None, backlog=128)
    assert f'http://unix:{unix_sockname}:' in printer.call_args[0][0]

@skip_if_no_unix_socks
def test_run_app_https_unix_socket(patched_loop: asyncio.AbstractEventLoop, unix_sockname: str) -> None:
    app = web.Application()
    ssl_context = ssl.create_default_context()
    printer = mock.Mock(wraps=stopper(patched_loop))
    web.run_app(app, path=unix_sockname, ssl_context=ssl_context, print=printer, loop=patched_loop)
    patched_loop.create_unix_server.assert_called_with(mock.ANY, unix_sockname, ssl=ssl_context, backlog=128)
    assert f'https://unix:{unix_sockname}:' in printer.call_args[0][0]

@pytest.mark.skipif(not hasattr(socket, 'AF_UNIX'), reason='requires UNIX sockets')
@skip_if_no_abstract_paths
def test_run_app_abstract_linux_socket(patched_loop: asyncio.AbstractEventLoop) -> None:
    sock_path = b'\x00' + uuid4().hex.encode('ascii')
    app = web.Application()
    web.run_app(app, path=sock_path.decode('ascii', 'ignore'), print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_unix_server.assert_called_with(mock.ANY, sock_path.decode('ascii'), ssl=None, backlog=128)

def test_run_app_preexisting_inet_socket(patched_loop: asyncio.AbstractEventLoop, mocker: MockerFixture) -> None:
    app = web.Application()
    sock = socket.socket()
    with contextlib.closing(sock):
        sock.bind(('127.0.0.1', 0))
        _, port = sock.getsockname()
        printer = mock.Mock(wraps=stopper(patched_loop))
        web.run_app(app, sock=sock, print=printer, loop=patched_loop)
        patched_loop.create_server.assert_called_with(mock.ANY, sock=sock, backlog=128, ssl=None)
        assert f'http://127.0.0.1:{port}' in printer.call_args[0][0]

@pytest.mark.skipif(not HAS_IPV6, reason='IPv6 is not available')
def test_run_app_preexisting_inet6_socket(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    sock = socket.socket(socket.AF_INET6)
    with contextlib.closing(sock):
        sock.bind(('::1', 0))
        port = sock.getsockname()[1]
        printer = mock.Mock(wraps=stopper(patched_loop))
        web.run_app(app, sock=sock, print=printer, loop=patched_loop)
        patched_loop.create_server.assert_called_with(mock.ANY, sock=sock, backlog=128, ssl=None)
        assert f'http://[::1]:{port}' in printer.call_args[0][0]

@skip_if_no_unix_socks
def test_run_app_preexisting_unix_socket(patched_loop: asyncio.AbstractEventLoop, mocker: MockerFixture) -> None:
    app = web.Application()
    sock_path = '/tmp/test_preexisting_sock1'
    sock = socket.socket(socket.AF_UNIX)
    with contextlib.closing(sock):
        sock.bind(sock_path)
        os.unlink(sock_path)
        printer = mock.Mock(wraps=stopper(patched_loop))
        web.run_app(app, sock=sock, print=printer, loop=patched_loop)
        patched_loop.create_server.assert_called_with(mock.ANY, sock=sock, backlog=128, ssl=None)
        assert f'http://unix:{sock_path}:' in printer.call_args[0][0]

def test_run_app_multiple_preexisting_sockets(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    sock1 = socket.socket()
    sock2 = socket.socket()
    with contextlib.closing(sock1), contextlib.closing(sock2):
        sock1.bind(('localhost', 0))
        _, port1 = sock1.getsockname()
        sock2.bind(('localhost', 0))
        _, port2 = sock2.getsockname()
        printer = mock.Mock(wraps=stopper(patched_loop))
        web.run_app(app, sock=(sock1, sock2), print=printer, loop=patched_loop)
        patched_loop.create_server.assert_has_calls([mock.call(mock.ANY, sock=sock1, backlog=128, ssl=None), mock.call(mock.ANY, sock=sock2, backlog=128, ssl=None)])
        assert f'http://127.0.0.1:{port1}' in printer.call_args[0][0]
        assert f'http://127.0.0.1:{port2}' in printer.call_args[0][0]

_script_test_signal: str = '\nfrom aiohttp import web\n\napp = web.Application()\nweb.run_app(app, host=())\n'

def test_sigint() -> None:
    skip_if_on_windows()
    with subprocess.Popen([sys.executable, '-u', '-c', _script_test_signal], stdout=subprocess.PIPE) as proc:
        for line in proc.stdout:
            if line.startswith(b'======== Running on'):
                break
        proc.send_signal(signal.SIGINT)
        assert proc.wait() == 0

def test_sigterm() -> None:
    skip_if_on_windows()
    with subprocess.Popen([sys.executable, '-u', '-c', _script_test_signal], stdout=subprocess.PIPE) as proc:
        for line in proc.stdout:
            if line.startswith(b'======== Running on'):
                break
        proc.terminate()
        assert proc.wait() == 0

def test_startup_cleanup_signals_even_on_failure(patched_loop: asyncio.AbstractEventLoop) -> None:
    patched_loop.create_server.side_effect = RuntimeError()
    app = web.Application()
    startup_handler = make_mocked_coro()
    app.on_startup.append(startup_handler)
    cleanup_handler = make_mocked_coro()
    app.on_cleanup.append(cleanup_handler)
    with pytest.raises(RuntimeError):
        web.run_app(app, print=stopper(patched_loop), loop=patched_loop)
    startup_handler.assert_called_once_with(app)
    cleanup_handler.assert_called_once_with(app)

def test_run_app_coro(patched_loop: asyncio.AbstractEventLoop) -> None:
    startup_handler = cleanup_handler = None

    async def make_app() -> web.Application:
        nonlocal startup_handler, cleanup_handler
        app = web.Application()
        startup_handler = make_mocked_coro()
        app.on_startup.append(startup_handler)
        cleanup_handler = make_mocked_coro()
        app.on_cleanup.append(cleanup_handler)
        return app

    web.run_app(make_app(), print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(mock.ANY, None, 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None)
    assert startup_handler is not None
    assert cleanup_handler is not None
    startup_handler.assert_called_once_with(mock.ANY)
    cleanup_handler.assert_called_once_with(mock.ANY)

def test_run_app_default_logger(monkeypatch: pytest.MonkeyPatch, patched_loop: asyncio.AbstractEventLoop) -> None:
    logger = access_logger
    attrs = {'hasHandlers.return_value': False, 'level': logging.NOTSET, 'name': 'aiohttp.access'}
    mock_logger = mock.create_autospec(logger, name='mock_access_logger')
    mock_logger.configure_mock(**attrs)
    app = web.Application()
    web.run_app(app, debug=True, print=stopper(patched_loop), access_log=mock_logger, loop=patched_loop)
    mock_logger.setLevel.assert_any_call(logging.DEBUG)
    mock_logger.hasHandlers.assert_called_with()
    assert isinstance(mock_logger.addHandler.call_args[0][0], logging.StreamHandler)

def test_run_app_default_logger_setup_requires_debug(patched_loop: asyncio.AbstractEventLoop) -> None:
    logger = access_logger
    attrs = {'hasHandlers.return_value': False, 'level': logging.NOTSET, 'name': 'aiohttp.access'}
    mock_logger = mock.create_autospec(logger, name='mock_access_logger')
    mock_logger.configure_mock(**attrs)
    app = web.Application()
    web.run_app(app, debug=False, print=stopper(patched_loop), access_log=mock_logger, loop=patched_loop)
    mock_logger.setLevel.assert_not_called()
    mock_logger.hasHandlers.assert_not_called()
    mock_logger.addHandler.assert_not_called()

def test_run_app_default_logger_setup_requires_default_logger(patched_loop: asyncio.AbstractEventLoop) -> None:
    logger = access_logger
    attrs = {'hasHandlers.return_value': False, 'level': logging.NOTSET, 'name': None}
    mock_logger = mock.create_autospec(logger, name='mock_access_logger')
    mock_logger.configure_mock(**attrs)
    app = web.Application()
    web.run_app(app, debug=True, print=stopper(patched_loop), access_log=mock_logger, loop=patched_loop)
    mock_logger.setLevel.assert_not_called()
    mock_logger.hasHandlers.assert_not_called()
    mock_logger.addHandler.assert_not_called()

def test_run_app_default_logger_setup_only_if_unconfigured(patched_loop: asyncio.AbstractEventLoop) -> None:
    logger = access_logger
    attrs = {'hasHandlers.return_value': True, 'level': None, 'name': 'aiohttp.access'}
    mock_logger = mock.create_autospec(logger, name='mock_access_logger')
    mock_logger.configure_mock(**attrs)
    app = web.Application()
    web.run_app(app, debug=True, print=stopper(patched_loop), access_log=mock_logger, loop=patched_loop)
    mock_logger.setLevel.assert_not_called()
    mock_logger.hasHandlers.assert_called_with()
    mock_logger.addHandler.assert_not_called()

def test_run_app_cancels_all_pending_tasks(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    task: Optional[asyncio.Task] = None

    async def on_startup(app: web.Application) -> None:
        nonlocal task
        loop = asyncio.get_event_loop()
        task = loop.create_task(asyncio.sleep(1000))

    app.on_startup.append(on_startup)
    web.run_app(app, print=stopper(patched_loop), loop=patched_loop)
    assert task is not None
    assert task.cancelled()

def test_run_app_cancels_done_tasks(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    task: Optional[asyncio.Task] = None

    async def coro() -> int:
        return 123

    async def on_startup(app: web.Application) -> None:
        nonlocal task
        loop = asyncio.get_event_loop()
        task = loop.create_task(coro())

    app.on_startup.append(on_startup)
    web.run_app(app, print=stopper(patched_loop), loop=patched_loop)
    assert task is not None
    assert task.done()

def test_run_app_cancels_failed_tasks(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    task: Optional[asyncio.Task] = None
    exc = RuntimeError('FAIL')

    async def fail() -> None:
        try:
            await asyncio.sleep(1000)
        except asyncio.CancelledError:
            raise exc

    async def on_startup(app: web.Application) -> None:
        nonlocal task
        loop = asyncio.get_event_loop()
        task = loop.create_task(fail())
        await asyncio.sleep(0.01)

    app.on_startup.append(on_startup)
    exc_handler = mock.Mock()
    patched_loop.set_exception_handler(exc_handler)
    web.run_app(app, print=stopper(patched_loop), loop=patched_loop)
    assert task is not None
    assert task.done()
    msg = {'message': 'unhandled exception during asyncio.run() shutdown', 'exception': exc, 'task': task}
    exc_handler.assert_called_with(patched_loop, msg)

def test_run_app_keepalive_timeout(patched_loop: asyncio.AbstractEventLoop, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    new_timeout = 1234
    base_runner_init_orig = BaseRunner.__init__

    def base_runner_init_spy(self: BaseRunner, *args: Any, **kwargs: Any) -> None:
        assert kwargs['keepalive_timeout'] == new_timeout
        base_runner_init_orig(self, *args, **kwargs)

    app = web.Application()
    monkeypatch.setattr(BaseRunner, '__init__', base_runner_init_spy)
    web.run_app(app, keepalive_timeout=new_timeout, print=stopper(patched_loop), loop=patched_loop)

def test_run_app_context_vars(patched_loop: asyncio.AbstractEventLoop) -> None:
    from contextvars import ContextVar
    count = 0
    VAR = ContextVar('VAR', default='default')

    async def on_startup(app: web.Application) -> None:
        nonlocal count
        assert 'init' == VAR.get()
        VAR.set('on_startup')
        count += 1

    async def on_cleanup(app: web.Application) -> None:
        nonlocal count
        assert 'on_startup' == VAR.get()
        count += 1

    async def init() -> web.Application:
        nonlocal count
        assert 'default' == VAR.get()
        VAR.set('init')
        app = web.Application()
        app.on_startup.append(on_startup)
        app.on_cleanup.append(on_cleanup)
        count += 1
        return app

    web.run_app(init(), print=stopper(patched_loop), loop=patched_loop)
    assert count == 3

def test_run_app_raises_exception(patched_loop: asyncio.AbstractEventLoop) -> None:
    async def context(app: web.Application) -> AsyncIterator[None]:
        raise RuntimeError('foo')
        yield

    app = web.Application()
    app.cleanup_ctx.append(context)
    with mock.patch.object(patched_loop, 'call_exception_handler', autospec=True, spec_set=True) as m:
        with pytest.raises(RuntimeError, match='foo'):
            web.run_app(app, loop=patched_loop)
    assert not m.called

class TestShutdown:
    def raiser(self) -> None:
        raise KeyboardInterrupt

    async def stop(self, request: web.Request) -> web.Response:
        asyncio.get_running_loop().call_soon(self.raiser)
        return web.Response()

    def run_app(self, sock: socket.socket, timeout: float, task: Callable[[], Awaitable[None]], extra_test: Optional[Callable[[ClientSession], Awaitable[None]]] = None) -> Tuple[asyncio.Task, int]:
        num_connections = -1
        t = test_task = None
        port = sock.getsockname()[1]

        class DictRecordClear(Dict[RequestHandler[web.Request], asyncio.Transport]):
            def clear(self) -> None:
                nonlocal num_connections
                num_connections = len(self)
                super().clear()

        class ServerWithRecordClear(web.Server[web.Request]):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._connections = DictRecordClear()

        async def test() -> None:
            await asyncio.sleep(0.5)
            async with ClientSession() as sess:
                for _ in range(5):
                    try:
                        with pytest.raises(asyncio.TimeoutError):
                            async with sess.get(f'http://127.0.0.1:{port}/', timeout=ClientTimeout(total=0.1)):
                                pass
                    except ClientConnectorError:
                        await asyncio.sleep(0.5)
                    else:
                        break
                async with sess.get(f'http://127.0.0.1:{port}/stop'):
                    pass
                if extra_test:
                    await extra_test(sess)

        async def run_test(app: web.Application) -> AsyncIterator[None]:
            nonlocal test_task
            test_task = asyncio.create_task(test())
            yield
            await test_task

        async def handler(request: web.Request) -> web.Response:
            nonlocal t
            t = asyncio.create_task(task())
            await t
            return web.Response(text='FOO')

        app = web.Application()
        app.cleanup_ctx.append(run_test)
        app.router.add_get('/', handler)
        app.router.add_get('/stop', self.stop)
        with mock.patch('aiohttp.web_runner.Server', ServerWithRecordClear):
            web.run_app(app, sock=sock, shutdown_timeout=timeout)
        assert test_task is not None
        assert test_task.exception() is None
        assert t is not None
        return (t, num_connections)

    def test_shutdown_wait_for_handler(self, unused_port_socket: socket.socket) -> None:
        sock = unused_port_socket
        finished = False

        async def task() -> None:
            nonlocal finished
            await asyncio.sleep(2)
            finished = True

        t, connection_count = self.run_app(sock, 3, task)
        assert finished is True
        assert t.done()
        assert not t.cancelled()
        assert connection_count == 0

    def test_shutdown_timeout_handler(self, unused_port_socket: socket.socket) -> None:
        sock = unused_port_socket
        finished = False

        async def task() -> None:
            nonlocal finished
            await asyncio.sleep(2)
            finished = True

        t, connection_count = self.run_app(sock, 1, task)
        assert finished is False
        assert t.done()
        assert t.cancelled()
        assert connection_count == 1

    def test_shutdown_timeout_not_reached(self, unused_port_socket: socket.socket) -> None:
        sock = unused_port_socket
        finished = False

        async def task() -> None:
            nonlocal finished
            await asyncio.sleep(1)
            finished = True

        start_time = time.time()
        t, connection_count = self.run_app(sock, 15, task)
        assert finished is True
        assert t.done()
        assert connection_count == 0
        assert time.time() - start_time < 10

    def test_shutdown_new_conn_rejected(self, unused_port_socket: socket.socket) -> None:
        sock = unused_port_socket
        port = sock.getsockname()[1]
        finished = False

        async def task() -> None:
            nonlocal finished
            await asyncio.sleep(9)
            finished = True

        async def test(sess: ClientSession) -> None:
            await asyncio.sleep(1)
            with pytest.raises(ClientConnectorError):
                async with ClientSession() as sess:
                    async with sess.get(f'http://127.0.0.1:{port}/'):
                        pass
            assert finished is False

        t, connection_count = self.run_app(sock, 10, task, test)
        assert finished is True
        assert t.done()
        assert connection_count == 0

    def test_shutdown_pending_handler_responds(self, unused_port_socket: socket.socket) -> None:
        sock = unused_port_socket
        port = sock.getsockname()[1]
        finished = False
        t = None

        async def test() -> None:
            async def test_resp(sess: ClientSession) -> None:
                async with sess.get(f'http://127.0.0.1:{port}/') as resp:
                    assert await resp.text() == 'FOO'

            await asyncio.sleep(1)
            async with ClientSession() as sess:
                t = asyncio.create_task(test_resp(sess))
                await asyncio.sleep(1)
                async with sess.get(f'http://127.0.0.1:{port}/stop'):
                    pass
                assert finished is False
                await t

        async def run_test(app: web.Application) -> AsyncIterator[None]:
            nonlocal t
            t = asyncio.create_task(test())
            yield
            await t

        async def handler(request: web.Request) -> web.Response:
            nonlocal finished
            await asyncio.sleep(3)
            finished = True
            return web.Response(text='FOO')

        app = web.Application()
        app.cleanup_ctx.append(run_test)
        app.router.add_get('/', handler)
        app.router.add_get('/stop', self.stop)
        web.run_app(app, sock=sock, shutdown_timeout=5)
        assert t is not None
        assert t.exception() is None
        assert finished is True

    def test_shutdown_close_idle_keepalive(self, unused_port_socket: socket.socket) -> None:
        sock = unused_port_socket
        port = sock.getsockname()[1]
        t = None

        async def test() -> None:
            await asyncio.sleep(1)
            async with ClientSession() as sess:
                async with sess.get(f'http://127.0.0.1:{port}/stop'):
                    pass
                await asyncio.sleep(5)

        async def run_test(app: web.Application) -> AsyncIterator[None]:
            nonlocal t
            t = asyncio.create_task(test())
            yield
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

        app = web.Application()
        app.cleanup_ctx.append(run_test)
        app.router.add_get('/stop', self.stop)
        web.run_app(app, sock=sock, shutdown_timeout=10)
        assert t is not None
        assert t.cancelled()

    def test_shutdown_close_websockets(self, unused_port_socket: socket.socket) -> None:
        sock = unused_port_socket
        port = sock.getsockname()[1]
        WS = web.AppKey('ws', Set[web.WebSocketResponse])
        client_finished = server_finished = False
        t = None

        async def ws_handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            request.app[WS].add(ws)
            async for msg in ws:
                pass
            nonlocal server_finished
            server_finished = True
            return ws

        async def close_websockets(app: web.Application) -> None:
            for ws in app[WS]:
                await ws.close(code=WSCloseCode.GOING_AWAY)

        async def test() -> None:
            await asyncio.sleep(1)
            async with ClientSession() as sess:
                async with sess.ws_connect(f'http://127.0.0.1:{port}/ws') as ws:
                    async with sess.get(f'http://127.0.0.1:{port}/stop'):
                        pass
                    async for msg in ws:
                        pass
                    nonlocal client_finished
                    client_finished = True

        async def run_test(app: web.Application) -> AsyncIterator[None]:
            nonlocal t
            t = asyncio.create_task(test())
            yield
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

        app = web.Application()
        app[WS] = set()
        app.on_shutdown.append(close_websockets)
        app.cleanup_ctx.append(run_test)
        app.router.add_get('/ws', ws_handler)
        app.router.add_get('/stop', self.stop)
        start = time.time()
        web.run_app(app, sock=sock, shutdown_timeout=10)
        assert time.time() - start < 5
        assert client_finished
        assert server_finished

    def test_shutdown_handler_cancellation_suppressed(self, unused_port_socket: socket.socket) -> None:
        sock = unused_port_socket
        port = sock.getsockname()[1]
        actions: List[str] = []
        t = None

        async def test() -> None:
            async def test_resp(sess: ClientSession) -> None:
                t = ClientTimeout(total=0.4)
                with pytest.raises(asyncio.TimeoutError):
                    async with sess.get(f'http://127.0.0.1:{port}/', timeout=t) as resp:
                        assert await resp.text() == 'FOO'
                actions.append('CANCELLED')

            async with ClientSession() as sess:
                t = asyncio.create_task(test_resp(sess))
                await asyncio.sleep(0.5)
                actions.append('PRESTOP')
                async with sess.get(f'http://127.0.0.1:{port}/stop'):
                    pass
                actions.append('STOPPING')
                await t

        async def run_test(app: web.Application) -> AsyncIterator[None]:
            nonlocal t
            t = asyncio.create_task(test())
            yield
            await t

        async def handler(request: web.Request) -> web.Response:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                actions.append('SUPPRESSED')
                await asyncio.sleep(2)
                actions.append('DONE')
            return web.Response(text='FOO')

        app = web.Application()
        app.cleanup_ctx.append(run_test)
        app.router.add_get('/', handler)
        app.router.add_get('/stop', self.stop)
        web.run_app(app, sock=sock, shutdown_timeout=2, handler_cancellation=True)
        assert t is not None
        assert t.exception() is None
        assert actions == ['CANCELLED', 'SUPPRESSED', 'PRESTOP', 'STOPPING', 'DONE']
