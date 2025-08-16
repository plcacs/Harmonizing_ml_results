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
from tornado import netutil
from tornado.platform.asyncio import AsyncIOMainLoop
from tornado.process import Subprocess
from tornado.log import app_log
from tornado.util import raise_exc_info, basestring_type
from tornado.web import Application
import typing
from typing import Tuple, Any, Callable, Type, Dict, Union, Optional, Coroutine
from types import TracebackType

if typing.TYPE_CHECKING:
    _ExcInfoTuple = Tuple[Optional[Type[BaseException]], Optional[BaseException], Optional[TracebackType]
_NON_OWNED_IOLOOPS = AsyncIOMainLoop

def bind_unused_port(reuse_port: bool = False, address: str = '127.0.0.1') -> Tuple[socket.socket, int]:
    ...

def get_async_test_timeout() -> float:
    ...

class AsyncTestCase(unittest.TestCase):
    ...

    def __init__(self, methodName: str = 'runTest') -> None:
        ...

    def setUp(self) -> None:
        ...

    def tearDown(self) -> None:
        ...

    def get_new_ioloop(self) -> IOLoop:
        ...

    def _handle_exception(self, typ: Type[BaseException], value: BaseException, tb: TracebackType) -> bool:
        ...

    def __rethrow(self) -> None:
        ...

    def run(self, result: Optional[unittest.TestResult] = None) -> Optional[unittest.TestResult]:
        ...

    def _callTestMethod(self, method: Callable) -> None:
        ...

    def stop(self, _arg: Optional[Any] = None, **kwargs: Any) -> None:
        ...

    def wait(self, condition: Optional[Callable[[], bool]] = None, timeout: Optional[float] = None) -> Any:
        ...

class AsyncHTTPTestCase(AsyncTestCase):
    ...

    def setUp(self) -> None:
        ...

    def get_http_client(self) -> AsyncHTTPClient:
        ...

    def get_http_server(self) -> HTTPServer:
        ...

    def get_app(self) -> Application:
        ...

    def fetch(self, path: str, raise_error: bool = False, **kwargs: Any) -> HTTPResponse:
        ...

    def get_httpserver_options(self) -> Dict[str, Any]:
        ...

    def get_http_port(self) -> int:
        ...

    def get_protocol(self) -> str:
        ...

    def get_url(self, path: str) -> str:
        ...

    def tearDown(self) -> None:
        ...

class AsyncHTTPSTestCase(AsyncHTTPTestCase):
    ...

    def get_http_client(self) -> AsyncHTTPClient:
        ...

    def get_httpserver_options(self) -> Dict[str, Any]:
        ...

    def get_ssl_options(self) -> Dict[str, str]:
        ...

    def get_protocol(self) -> str:
        ...

@typing.overload
def gen_test(*, timeout: Optional[float] = None) -> None:
    ...

@typing.overload
def gen_test(func: Callable) -> Callable:
    ...

def gen_test(func: Optional[Callable] = None, timeout: Optional[float] = None) -> Callable:
    ...

class ExpectLog(logging.Filter):
    ...

    def __init__(self, logger: Union[logging.Logger, str], regex: str, required: bool = True, level: Optional[int] = None) -> None:
        ...

    def filter(self, record: logging.LogRecord) -> bool:
        ...

    def __enter__(self) -> 'ExpectLog':
        ...

    def __exit__(self, typ: Optional[Type[BaseException]], value: BaseException, tb: TracebackType) -> None:
        ...

def setup_with_context_manager(testcase: unittest.TestCase, cm: typing.ContextManager) -> Any:
    ...

def main(**kwargs: Any) -> None:
    ...
