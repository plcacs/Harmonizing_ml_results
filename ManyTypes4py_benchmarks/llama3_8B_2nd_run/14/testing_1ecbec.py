import asyncio
from collections.abc import Generator
import functools
import inspect
import logging
import os
import re
import signal
import socket
import sys
import unittest
import warnings
from tornado import gen
from tornado.httpclient import AsyncHTTPClient, HTTPResponse
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop, TimeoutError
from tornado.platform.asyncio import AsyncIOMainLoop
from tornado.process import Subprocess
from tornado.log import app_log
from tornado.util import raise_exc_info, basestring_type
from typing import Tuple, Any, Callable, Type, Dict, Union, Optional, Coroutine
from types import TracebackType
if typing.TYPE_CHECKING:
    _ExcInfoTuple = Tuple[Optional[Type[BaseException]], Optional[BaseException], Optional[TracebackType]]

def bind_unused_port(reuse_port=False, address='127.0.0.1') -> Tuple[socket.socket, int]:
    """Binds a server socket to an available port on localhost."""
    # ...

class AsyncTestCase(unittest.TestCase):
    """`~unittest.TestCase` subclass for testing `.IOLoop`-based asynchronous code."""

    def __init__(self, methodName: str = 'runTest') -> None:
        # ...

    def get_new_ioloop(self) -> IOLoop:
        """Returns the `.IOLoop` to use for this test."""
        # ...

    def stop(self, _arg: Any = None, **kwargs: Any) -> None:
        """Stops the `.IOLoop`."""
        # ...

    def wait(self, condition: Optional[Callable[[], bool]] = None, timeout: Optional[float] = None) -> Any:
        """Runs the `.IOLoop` until stop is called or timeout has passed."""
        # ...

class AsyncHTTPTestCase(AsyncTestCase):
    """A test case that starts up an HTTP server."""

    def get_app(self) -> Application:
        """Should be overridden by subclasses to return a `tornado.web.Application` or other `.HTTPServer` callback."""
        # ...

    def get_http_client(self) -> AsyncHTTPClient:
        """Returns an `AsyncHTTPClient` instance."""
        # ...

    def get_http_server(self) -> HTTPServer:
        """Returns an `HTTPServer` instance."""
        # ...

    def get_url(self, path: str) -> str:
        """Returns an absolute URL for the given path on the test server."""
        # ...

@typing.overload
def gen_test(*, timeout: Optional[float] = None) -> Callable[[Any], Any]:
    pass

@typing.overload
def gen_test(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    pass

def gen_test(func: Optional[Callable[[Any], Any]] = None, timeout: Optional[float] = None) -> Callable[[Any], Any]:
    """Testing equivalent of `@gen.coroutine`, to be applied to test methods."""
    # ...

class ExpectLog(logging.Filter):
    """Context manager to capture and suppress expected log output."""

    def __init__(self, logger: logging.Logger, regex: str, required: bool = True, level: Optional[int] = None) -> None:
        """Constructs an ExpectLog context manager."""
        # ...

def setup_with_context_manager(testcase: Any, cm: Generator) -> Any:
    """Use a context manager to setUp a test case."""
    # ...

def main(**kwargs: Any) -> None:
    """A simple test runner."""
    # ...
