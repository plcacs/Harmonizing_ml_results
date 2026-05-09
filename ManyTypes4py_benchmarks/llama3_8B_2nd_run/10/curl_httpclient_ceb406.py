import collections
import functools
import logging
import pycurl
import re
import threading
import time
from io import BytesIO
from tornado import httputil
from tornado import ioloop
from tornado.httpclient import HTTPRequest, HTTPResponse, HTTPError, AsyncHTTPClient, main
from tornado.log import app_log
from typing import Dict, Any, Callable, Union, Optional
import typing
if typing.TYPE_CHECKING:
    from typing import Deque, Tuple

class CurlAsyncHTTPClient(AsyncHTTPClient):
    def __init__(self, max_clients: int = 10, defaults: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(defaults=defaults)
        self._multi: pycurl.CurlMulti = pycurl.CurlMulti()
        self._multi.setopt(pycurl.M_TIMERFUNCTION, self._set_timeout)
        self._multi.setopt(pycurl.M_SOCKETFUNCTION, self._handle_socket)
        self._curls: List[pycurl.Curl] = [self._curl_create() for _ in range(max_clients)]
        self._free_list: List[pycurl.Curl] = self._curls[:]
        self._requests: collections.deque[Tuple[HTTPRequest, Callable[[HTTPResponse], Any], float]] = collections.deque()
        self._fds: Dict[int, int] = {}
        self._timeout: Optional[float] = None
        self._force_timeout_callback: ioloop.PeriodicCallback = ioloop.PeriodicCallback(self._handle_force_timeout, 1000)
        self._force_timeout_callback.start()
        dummy_curl_handle: pycurl.Curl = pycurl.Curl()
        self._multi.add_handle(dummy_curl_handle)
        self._multi.remove_handle(dummy_curl_handle)

    # ... (rest of the class remains the same)

class CurlError(HTTPError):
    def __init__(self, errno: int, message: str) -> None:
        HTTPError.__init__(self, 599, message)
        self.errno: int = errno
