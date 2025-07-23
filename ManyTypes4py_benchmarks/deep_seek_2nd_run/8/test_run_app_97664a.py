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
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    NoReturn,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from unittest import mock
from uuid import uuid4
import pytest
from pytest_mock import MockerFixture
from aiohttp import (
    ClientConnectorError,
    ClientSession,
    ClientTimeout,
    WSCloseCode,
    web,
)
from aiohttp.log import access_logger
from aiohttp.test_utils import make_mocked_coro
from aiohttp.web_protocol import RequestHandler
from aiohttp.web_runner import BaseRunner
from aiohttp.typedefs import AppKey
from contextvars import ContextVar

_has_unix_domain_socks = hasattr(socket, "AF_UNIX")
if _has_unix_domain_socks:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as _abstract_path_sock:
        try:
            _abstract_path_sock.bind(b"\x00" + uuid4().hex.encode("ascii"))
        except FileNotFoundError:
            _abstract_path_failed = True
        else:
            _abstract_path_failed = False
        finally:
            del _abstract_path_sock
else:
    _abstract_path_failed = True

skip_if_no_abstract_paths = pytest.mark.skipif(
    _abstract_path_failed, reason="Linux-style abstract paths are not supported."
)
skip_if_no_unix_socks = pytest.mark.skipif(
    not _has_unix_domain_socks, reason="Unix domain sockets are not supported"
)
del _has_unix_domain_socks, _abstract_path_failed

HAS_IPV6 = socket.has_ipv6
if HAS_IPV6:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM):
            pass
    except OSError:
        HAS_IPV6 = False


def skip_if_on_windows() -> None:
    if platform.system() == "Windows":
        pytest.skip("the test is not valid for Windows")


@pytest.fixture
def patched_loop(loop: asyncio.AbstractEventLoop) -> Iterator[asyncio.AbstractEventLoop]:
    server = mock.create_autospec(asyncio.Server, spec_set=True, instance=True)
    server.wait_closed.return_value = None
    unix_server = mock.create_autospec(asyncio.Server, spec_set=True, instance=True)
    unix_server.wait_closed.return_value = None
    with mock.patch.object(
        loop, "create_server", autospec=True, spec_set=True, return_value=server
    ):
        with mock.patch.object(
            loop,
            "create_unix_server",
            autospec=True,
            spec_set=True,
            return_value=unix_server,
        ):
            asyncio.set_event_loop(loop)
            yield loop


def stopper(loop: asyncio.AbstractEventLoop) -> Callable[..., None]:
    def raiser() -> NoReturn:
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
    patched_loop.create_server.assert_called_with(
        mock.ANY,
        None,
        8080,
        ssl=None,
        backlog=128,
        reuse_address=None,
        reuse_port=None,
    )
    startup_handler.assert_called_once_with(app)
    cleanup_handler.assert_called_once_with(app)


def test_run_app_close_loop(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    web.run_app(app, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(
        mock.ANY,
        None,
        8080,
        ssl=None,
        backlog=128,
        reuse_address=None,
        reuse_port=None,
    )
    assert patched_loop.is_closed()


mock_unix_server_single = [
    mock.call(mock.ANY, "/tmp/testsock1.sock", ssl=None, backlog=128)
]
mock_unix_server_multi = [
    mock.call(mock.ANY, "/tmp/testsock1.sock", ssl=None, backlog=128),
    mock.call(mock.ANY, "/tmp/testsock2.sock", ssl=None, backlog=128),
]
mock_server_single = [
    mock.call(
        mock.ANY,
        "127.0.0.1",
        8080,
        ssl=None,
        backlog=128,
        reuse_address=None,
        reuse_port=None,
    )
]
mock_server_multi = [
    mock.call(
        mock.ANY,
        "127.0.0.1",
        8080,
        ssl=None,
        backlog=128,
        reuse_address=None,
        reuse_port=None,
    ),
    mock.call(
        mock.ANY,
        "192.168.1.1",
        8080,
        ssl=None,
        backlog=128,
        reuse_address=None,
        reuse_port=None,
    ),
]
mock_server_default_8989 = [
    mock.call(
        mock.ANY,
        None,
        8989,
        ssl=None,
        backlog=128,
        reuse_address=None,
        reuse_port=None,
    )
]
mock_socket = mock.Mock(getsockname=lambda: ("mock-socket", 123))
mixed_bindings_tests = (
    (
        "Nothing Specified",
        {},
        [
            mock.call(
                mock.ANY,
                None,
                8080,
                ssl=None,
                backlog=128,
                reuse_address=None,
                reuse_port=None,
            )
        ],
        [],
    ),
    ("Port Only", {"port": 8989}, mock_server_default_8989, []),
    (
        "Multiple Hosts",
        {"host": ("127.0.0.1", "192.168.1.1")},
        mock_server_multi,
        [],
    ),
    (
        "Multiple Paths",
        {"path": ("/tmp/testsock1.sock", "/tmp/testsock2.sock")},
        [],
        mock_unix_server_multi,
    ),
    (
        "Multiple Paths, Port",
        {"path": ("/tmp/testsock1.sock", "/tmp/testsock2.sock"), "port": 8989},
        mock_server_default_8989,
        mock_unix_server_multi,
    ),
    (
        "Multiple Paths, Single Host",
        {"path": ("/tmp/testsock1.sock", "/tmp/testsock2.sock"), "host": "127.0.0.1"},
        mock_server_single,
        mock_unix_server_multi,
    ),
    (
        "Single Path, Single Host",
        {"path": "/tmp/testsock1.sock", "host": "127.0.0.1"},
        mock_server_single,
        mock_unix_server_single,
    ),
    (
        "Single Path, Multiple Hosts",
        {"path": "/tmp/testsock1.sock", "host": ("127.0.0.1", "192.168.1.1")},
        mock_server_multi,
        mock_unix_server_single,
    ),
    (
        "Single Path, Port",
        {"path": "/tmp/testsock1.sock", "port": 8989},
        mock_server_default_8989,
        mock_unix_server_single,
    ),
    (
        "Multiple Paths, Multiple Hosts, Port",
        {
            "path": ("/tmp/testsock1.sock", "/tmp/testsock2.sock"),
            "host": ("127.0.0.1", "192.168.1.1"),
            "port": 8000,
        },
        [
            mock.call(
                mock.ANY,
                "127.0.0.1",
                8000,
                ssl=None,
                backlog=128,
                reuse_address=None,
                reuse_port=None,
            ),
            mock.call(
                mock.ANY,
                "192.168.1.1",
                8000,
                ssl=None,
                backlog=128,
                reuse_address=None,
                reuse_port=None,
            ),
        ],
        mock_unix_server_multi,
    ),
    (
        "Only socket",
        {"sock": [mock_socket]},
        [mock.call(mock.ANY, ssl=None, sock=mock_socket, backlog=128)],
        [],
    ),
    (
        "Socket, port",
        {"sock": [mock_socket], "port": 8765},
        [
            mock.call(
                mock.ANY,
                None,
                8765,
                ssl=None,
                backlog=128,
                reuse_address=None,
                reuse_port=None,
            ),
            mock.call(mock.ANY, sock=mock_socket, ssl=None, backlog=128),
        ],
        [],
    ),
    (
        "Socket, Host, No port",
        {"sock": [mock_socket], "host": "localhost"},
        [
            mock.call(
                mock.ANY,
                "localhost",
                8080,
                ssl=None,
                backlog=128,
                reuse_address=None,
                reuse_port=None,
            ),
            mock.call(mock.ANY, sock=mock_socket, ssl=None, backlog=128),
        ],
        [],
    ),
    (
        "reuse_port",
        {"reuse_port": True},
        [
            mock.call(
                mock.ANY,
                None,
                8080,
                ssl=None,
                backlog=128,
                reuse_address=None,
                reuse_port=True,
            )
        ],
        [],
    ),
    (
        "reuse_address",
        {"reuse_address": False},
        [
            mock.call(
                mock.ANY,
                None,
                8080,
                ssl=None,
                backlog=128,
                reuse_address=False,
                reuse_port=None,
            )
        ],
        [],
    ),
    (
        "reuse_port, reuse_address",
        {"reuse_address": True, "reuse_port": True},
        [
            mock.call(
                mock.ANY,
                None,
                8080,
                ssl=None,
                backlog=128,
                reuse_address=True,
                reuse_port=True,
            )
        ],
        [],
    ),
    (
        "Port, reuse_port",
        {"port": 8989, "reuse_port": True},
        [
            mock.call(
                mock.ANY,
                None,
                8989,
                ssl=None,
                backlog=128,
                reuse_address=None,
                reuse_port=True,
            )
        ],
        [],
    ),
    (
        "Multiple Hosts, reuse_port",
        {"host": ("127.0.0.1", "192.168.1.1"), "reuse_port": True},
        [
            mock.call(
                mock.ANY,
                "127.0.0.1",
                8080,
                ssl=None,
                backlog=128,
                reuse_address=None,
                reuse_port=True,
            ),
            mock.call(
                mock.ANY,
                "192.168.1.1",
                8080,
                ssl=None,
                backlog=128,
                reuse_address=None,
                reuse_port=True,
            ),
        ],
        [],
    ),
    (
        "Multiple Paths, Port, reuse_address",
        {
            "path": ("/tmp/testsock1.sock", "/tmp/testsock2.sock"),
            "port": 8989,
            "reuse_address": False,
        },
        [
            mock.call(
                mock.ANY,
                None,
                8989,
                ssl=None,
                backlog=128,
                reuse_address=False,
                reuse_port=None,
            )
        ],
        mock_unix_server_multi,
    ),
    (
        "Multiple Paths, Single Host, reuse_address, reuse_port",
        {
            "path": ("/tmp/testsock1.sock", "/tmp/testsock2.sock"),
            "host": "127.0.0.1",
            "reuse_address": True,
            "reuse_port": True,
        },
        [
            mock.call(
                mock.ANY,
                "127.0.0.1",
                8080,
                ssl=None,
                backlog=128,
                reuse_address=True,
                reuse_port=True,
            )
        ],
        mock_unix_server_multi,
    ),
)
mixed_bindings_test_ids = [test[0] for test in mixed_bindings_tests]
mixed_bindings_test_params = [test[1:] for test in mixed_bindings_tests]


@pytest.mark.parametrize(
    "run_app_kwargs, expected_server_calls, expected_unix_server_calls",
    mixed_bindings_test_params,
    ids=mixed_bindings_test_ids,
)
def test_run_app_mixed_bindings(
    run_app_kwargs: Dict[str, Any],
    expected_server_calls: List[mock._Call],
    expected_unix_server_calls: List[mock._Call],
    patched_loop: asyncio.AbstractEventLoop,
) -> None:
    app = web.Application()
    web.run_app(
        app, print=stopper(patched_loop), **run_app_kwargs, loop=patched_loop
    )
    assert patched_loop.create_unix_server.mock_calls == expected_unix_server_calls
    assert patched_loop.create_server.mock_calls == expected_server_calls


def test_run_app_https(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    ssl_context = ssl.create_default_context()
    web.run_app(
        app,
        ssl_context=ssl_context,
        print=stopper(patched_loop),
        loop=patched_loop,
    )
    patched_loop.create_server.assert_called_with(
        mock.ANY,
        None,
        8443,
        ssl=ssl_context,
        backlog=128,
        reuse_address=None,
        reuse_port=None,
    )


def test_run_app_nondefault_host_port(
    patched_loop: asyncio.AbstractEventLoop, unused_port_socket: socket.socket
) -> None:
    port = unused_port_socket.getsockname()[1]
    host = "127.0.0.1"
    app = web.Application()
    web.run_app(
        app,
        host=host,
        port=port,
        print=stopper(patched_loop),
        loop=patched_loop,
    )
    patched_loop.create_server.assert_called_with(
        mock.ANY,
        host,
        port,
        ssl=None,
        backlog=128,
        reuse_address=None,
        reuse_port=None,
    )


def test_run_app_with_sock(
    patched_loop: asyncio.AbstractEventLoop, unused_port_socket: socket.socket
) -> None:
    sock = unused_port_socket
    app = web.Application()
    web.run_app(app, sock=sock, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(
        mock.ANY, sock=sock, ssl=None, backlog=128
    )


def test_run_app_multiple_hosts(patched_loop: asyncio.AbstractEventLoop) -> None:
    hosts = ("127.0.0.1", "127.0.0.2")
    app = web.Application()
    web.run_app(app, host=hosts, print=stopper(patched_loop), loop=patched_loop)
    calls = map(
        lambda h: mock.call(
            mock.ANY,
            h,
            8080,
            ssl=None,
            backlog=128,
            reuse_address=None,
            reuse_port=None,
        ),
        hosts,
    )
    patched_loop.create_server.assert_has_calls(calls)


def test_run_app_custom_backlog(patched_loop: asyncio.AbstractEventLoop) -> None:
    app = web.Application()
    web.run_app(
        app, backlog=10, print=stopper(patched_loop), loop=patched_loop
    )
    patched_loop.create_server.assert_called_with(
        mock.ANY,
        None,
        8080,
        ssl=None,
        backlog=10,
        reuse_address=None,
        reuse_port=None,
    )


def test_run_app_custom_backlog_unix(
    patched_loop: asyncio.AbstractEventLoop,
) -> None:
    app = web.Application()
    web.run_app(
        app,
        path="/tmp/tmpsock.sock",
        backlog=10,
        print=stopper(patched_loop),
        loop=patched_loop,
    )
    patched_loop.create_unix_server.assert_called_with(
        mock.ANY, "/tmp/tmpsock.sock", ssl=None, backlog=10
    )


@skip_if_no_unix_socks
def test_run_app_http_unix_socket(
    patched_loop: asyncio.AbstractEventLoop, unix_sockname: str
) -> None:
    app = web.Application()
    printer = mock.Mock(wraps=stopper(patched_loop))
    web.run_app(app, path=unix_sockname, print=printer, loop=patched_loop)
    patched_loop.create_unix_server.assert_called_with(
        mock.ANY, unix_sockname, ssl=None, backlog=128
    )
    assert f"http://unix:{unix_sockname}:" in printer.call_args[0][0]


@skip_if_no_unix_socks
def test_run_app_https_unix_socket(
    patched_loop: asyncio.AbstractEventLoop, unix_sockname: str
) -> None:
    app = web.Application()
    ssl_context = ssl.create_default_context()
    printer = mock.Mock(wraps=stopper(patched_loop))
    web.run_app(
        app,
        path=unix_sockname,
        ssl_context=