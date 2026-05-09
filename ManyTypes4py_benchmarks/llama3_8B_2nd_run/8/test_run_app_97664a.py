import asyncio
import contextlib
import logging
import os
import platform
import socket
import ssl
import subprocess
import sys
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Coroutine, Dict, Iterator, List, NoReturn, Optional, Set, Tuple
from unittest import mock
from uuid import uuid4
import pytest
from aiohttp import ClientConnectorError, ClientSession, ClientTimeout, WSCloseCode, web
from aiohttp.log import access_logger
from aiohttp.test_utils import make_mocked_coro
from aiohttp.web_protocol import RequestHandler
from aiohttp.web_runner import BaseRunner

@skip_if_no_abstract_paths
def test_run_app_abstract_linux_socket(patched_loop):
    sock_path = b'\x00' + uuid4().hex.encode('ascii')
    app = web.Application()
    web.run_app(app, path=sock_path.decode('ascii', 'ignore'), print=stopper(patched_loop), loop=patched_loop)
    patched_loop.create_unix_server.assert_called_with(mock.ANY, sock_path.decode('ascii'), ssl=None, backlog=128)

def test_run_app_cancels_all_pending_tasks(patched_loop: asyncio.BaseEventLoop) -> None:
    # ...

def test_run_app_context_vars(patched_loop: asyncio.BaseEventLoop) -> Tuple[asyncio.Task, int]:
    # ...

def test_shutdown_wait_for_handler(unused_port_socket: socket.socket) -> Tuple[asyncio.Task, int]:
    # ...

def test_shutdown_timeout_handler(unused_port_socket: socket.socket) -> Tuple[asyncio.Task, int]:
    # ...

def test_shutdown_timeout_not_reached(unused_port_socket: socket.socket) -> Tuple[asyncio.Task, int]:
    # ...

def test_shutdown_new_conn_rejected(unused_port_socket: socket.socket) -> Tuple[asyncio.Task, int]:
    # ...

def test_shutdown_pending_handler_responds(unused_port_socket: socket.socket) -> None:
    # ...

def test_shutdown_close_idle_keepalive(unused_port_socket: socket.socket) -> None:
    # ...

def test_shutdown_close_websockets(unused_port_socket: socket.socket) -> None:
    # ...

def test_shutdown_handler_cancellation_suppressed(unused_port_socket: socket.socket) -> None:
    # ...
