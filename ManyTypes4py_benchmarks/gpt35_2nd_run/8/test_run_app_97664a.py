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
            _abstract_path_failed: bool = False
        finally:
            del _abstract_path_sock
else:
    _abstract_path_failed: bool = True

skip_if_no_abstract_paths: Any = pytest.mark.skipif(_abstract_path_failed, reason='Linux-style abstract paths are not supported.')
skip_if_no_unix_socks: Any = pytest.mark.skipif(not _has_unix_domain_socks, reason='Unix domain sockets are not supported')
del _has_unix_domain_socks, _abstract_path_failed

HAS_IPV6: bool = socket.has_ipv6
if HAS_IPV6:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM):
            pass
    except OSError:
        HAS_IPV6: bool = False

def skip_if_on_windows() -> None:
    if platform.system() == 'Windows':
        pytest.skip('the test is not valid for Windows')

@pytest.fixture
def patched_loop(loop: Any) -> Any:
    server: Any = mock.create_autospec(asyncio.Server, spec_set=True, instance=True)
    server.wait_closed.return_value = None
    unix_server: Any = mock.create_autospec(asyncio.Server, spec_set=True, instance=True)
    unix_server.wait_closed.return_value = None
    with mock.patch.object(loop, 'create_server', autospec=True, spec_set=True, return_value=server):
        with mock.patch.object(loop, 'create_unix_server', autospec=True, spec_set=True, return_value=unix_server):
            asyncio.set_event_loop(loop)
            yield loop

def stopper(loop: Any) -> Callable:
    def raiser() -> None:
        raise KeyboardInterrupt

    def f(*args: Any) -> None:
        loop.call_soon(raiser)
    return f

def test_run_app_http(patched_loop: Any) -> None:
    app: web.Application = web.Application()
    startup_handler: Any = make_mocked_coro()
    app.on_startup.append(startup_handler)
    cleanup_handler: Any = make_mocked_coro()
    app.on_cleanup.append(cleanup_handler)
    web.run_app(app, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(mock.ANY, None, 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None)
    startup_handler.assert_called_once_with(app)
    cleanup_handler.assert_called_once_with(app)

def test_run_app_close_loop(patched_loop: Any) -> None:
    app: web.Application = web.Application()
    web.run_app(app, print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_server.assert_called_with(mock.ANY, None, 8080, ssl=None, backlog=128, reuse_address=None, reuse_port=None)
    assert patched_loop.is_closed()

# More annotated functions follow...
