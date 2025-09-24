"""Support classes for automated testing.

* `AsyncTestCase` and `AsyncHTTPTestCase`:  Subclasses of unittest.TestCase
  with additional support for testing asynchronous (`.IOLoop`-based) code.

* `ExpectLog`: Make test logs less spammy.

* `main()`: A simple test runner (wrapper around unittest.main()) with support
  for the tornado.autoreload module to rerun the tests when code changes.
"""
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
from typing import Tuple, Any, Callable, Type, Dict, Union, Optional, Coroutine, List, Iterator
from types import TracebackType
if typing.TYPE_CHECKING:
    _ExcInfoTuple = Tuple[Optional[Type[BaseException]], Optional[BaseException], Optional[TracebackType]]
_NON_OWNED_IOLOOPS = AsyncIOMainLoop

def bind_unused_port(reuse_port: bool = False, address: str = '127.0.0.1') -> Tuple[socket.socket, int]:
    """Binds a server socket to an available port on localhost.

    Returns a tuple (socket, port).

    .. versionchanged:: 4.4
       Always binds to ``127.0.0.1`` without resolving the name
       ``localhost``.

    .. versionchanged:: 6.2
       Added optional ``address`` argument to
       override the default "127.0.0.1".
    """
    sock = netutil.bind_sockets(0, address, family=socket.AF_INET, reuse_port=reuse_port)[0]
    port = sock.getsockname()[1]
    return (sock, port)

def get_async_test_timeout() -> float:
    """Get the global timeout setting for async tests.

    Returns a float, the timeout in seconds.

    .. versionadded:: 3.1
    """
    env = os.environ.get('ASYNC_TEST_TIMEOUT')
    if env is not None:
        try:
            return float(env)
        except ValueError:
            pass
    return 5

class AsyncTestCase(unittest.TestCase):
    """`~unittest.TestCase` subclass for testing `.IOLoop`-based
    asynchronous code.

    The unittest framework is synchronous, so the test must be
    complete by the time the test method returns. This means that
    asynchronous code cannot be used in quite the same way as usual
    and must be adapted to fit. To write your tests with coroutines,
    decorate your test methods with `tornado.testing.gen_test` instead
    of `tornado.gen.coroutine`.

    This class also provides the (deprecated) `stop()` and `wait()`
    methods for a more manual style of testing. The test method itself
    must call ``self.wait()``, and asynchronous callbacks should call
    ``self.stop()`` to signal completion.

    By default, a new `.IOLoop` is constructed for each test and is available
    as ``self.io_loop``.  If the code being tested requires a
    reused global `.IOLoop`, subclasses should override `get_new_ioloop` to return it,
    although this is deprecated as of Tornado 6.3.

    The `.IOLoop`'s ``start`` and ``stop`` methods should not be
    called directly.  Instead, use `self.stop <stop>` and `self.wait
    <wait>`.  Arguments passed to ``self.stop`` are returned from
    ``self.wait``.  It is possible to have multiple ``wait``/``stop``
    cycles in the same test.

    Example::

        # This test uses coroutine style.
        class MyTestCase(AsyncTestCase):
            @tornado.testing.gen_test
            def test_http_fetch(self):
                client = AsyncHTTPClient()
                response = yield client.fetch("http://www.tornadoweb.org")
                # Test contents of response
                self.assertIn("FriendFeed", response.body)

        # This test uses argument passing between self.stop and self.wait.
        class MyTestCase2(AsyncTestCase):
            def test_http_fetch(self):
                client = AsyncHTTPClient()
                client.fetch("http://www.tornadoweb.org/", self.stop)
                response = self.wait()
                # Test contents of response
                self.assertIn("FriendFeed", response.body)
    """

    def __init__(self, methodName: str = 'runTest') -> None:
        super().__init__(methodName)
        self.__stopped: bool = False
        self.__running: bool = False
        self.__failure: Optional[_ExcInfoTuple] = None
        self.__stop_args: Optional[Any] = None
        self.__timeout: Optional[object] = None
        self._test_generator: Optional[Union[Generator, Coroutine]] = None
        self.io_loop: IOLoop

    def setUp(self) -> None:
        py_ver = sys.version_info
        if (3, 10, 0) <= py_ver < (3, 10, 9) or (3, 11, 0) <= py_ver <= (3, 11, 1):
            setup_with_context_manager(self, warnings.catch_warnings())
            warnings.filterwarnings('ignore', message='There is no current event loop', category=DeprecationWarning, module='tornado\\..*')
        super().setUp()
        if type(self).get_new_ioloop is not AsyncTestCase.get_new_ioloop:
            warnings.warn('get_new_ioloop is deprecated', DeprecationWarning)
        self.io_loop = self.get_new_ioloop()
        asyncio.set_event_loop(self.io_loop.asyncio_loop)

    def tearDown(self) -> None:
        asyncio_loop = self.io_loop.asyncio_loop
        tasks = asyncio.all_tasks(asyncio_loop)
        tasks = [t for t in tasks if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            (done, pending) = self.io_loop.run_sync(lambda: asyncio.wait(tasks))
            assert not pending
            for f in done:
                try:
                    f.result()
                except asyncio.CancelledError:
                    pass
        Subprocess.uninitialize()
        asyncio.set_event_loop(None)
        if not isinstance(self.io_loop, _NON_OWNED_IOLOOPS):
            self.io_loop.close(all_fds=True)
        super().tearDown()
        self.__rethrow()

    def get_new_ioloop(self) -> IOLoop:
        """Returns the `.IOLoop` to use for this test.

        By default, a new `.IOLoop` is created for each test.
        Subclasses may override this method to return
        `.IOLoop.current()` if it is not appropriate to use a new
        `.IOLoop` in each tests (for example, if there are global
        singletons using the default `.IOLoop`) or if a per-test event
        loop is being provided by another system (such as
        ``pytest-asyncio``).

        .. deprecated:: 6.3
           This method will be removed in Tornado 7.0.
        """
        return IOLoop(make_current=False)

    def _handle_exception(self, typ: Type[Exception], value: Exception, tb: TracebackType) -> bool:
        if self.__failure is None:
            self.__failure = (typ, value, tb)
        else:
            app_log.error('multiple unhandled exceptions in test', exc_info=(typ, value, tb))
        self.stop()
        return True

    def __rethrow(self) -> None:
        if self.__failure is not None:
            failure = self.__failure
            self.__failure = None
            raise_exc_info(failure)

    def run(self, result: Optional[unittest.TestResult] = None) -> Optional[unittest.TestResult]:
        ret = super().run(result)
        self.__rethrow()
        return ret

    def _callTestMethod(self, method: Callable) -> None:
        """Run the given test method, raising an error if it returns non-None.

        Failure to decorate asynchronous test methods with ``@gen_test`` can lead to tests
        incorrectly passing.

        Remove this override when Python 3.10 support is dropped. This check (in the form of a
        DeprecationWarning) became a part of the standard library in 3.11.

        Note that ``_callTestMethod`` is not documented as a public interface. However, it is
        present in all supported versions of Python (3.8+), and if it goes away in the future that's
        OK because we can just remove this override as noted above.
        """
        result = method()
        if isinstance(result, Generator) or inspect.iscoroutine(result):
            raise TypeError('Generator and coroutine test methods should be decorated with tornado.testing.gen_test')
        elif result is not None:
            raise ValueError('Return value from test method ignored: %r' % result)

    def stop(self, _arg: Any = None, **kwargs: Any) -> None:
        """Stops the `.IOLoop`, causing one pending (or future) call to `wait()`
        to return.

        Keyword arguments or a single positional argument passed to `stop()` are
        saved and will be returned by `wait()`.

        .. deprecated:: 5.1

           `stop` and `wait` are deprecated; use ``@gen_test`` instead.
        """
        assert _arg is None or not kwargs
        self.__stop_args = kwargs or _arg
        if self.__running:
            self.io_loop.stop()
            self.__running = False
        self.__stopped = True

    def wait(self, condition: Optional[Callable[..., bool]] = None, timeout: Optional[float] = None) -> Any:
        """Runs the `.IOLoop` until stop is called or timeout has passed.

        In the event of a timeout, an exception will be thrown. The
        default timeout is 5 seconds; it may be overridden with a
        ``timeout`` keyword argument or globally with the
        ``ASYNC_TEST_TIMEOUT`` environment variable.

        If ``condition`` is not ``None``, the `.IOLoop` will be restarted
        after `stop()` until ``condition()`` returns ``True``.

        .. versionchanged:: 3.1
           Added the ``ASYNC_TEST_TIMEOUT`` environment variable.

        .. deprecated:: 5.1

           `stop` and `wait` are deprecated; use ``@gen_test`` instead.
        """
        if timeout is None:
            timeout = get_async_test_timeout()
        if not self.__stopped:
            if timeout:

                def timeout_func() -> None:
                    try:
                        raise self.failureException('Async operation timed out after %s seconds' % timeout)
                    except Exception:
                        self.__failure = sys.exc_info()
                    self.stop()
                self.__timeout = self.io_loop.add_timeout(self.io_loop.time() + timeout, timeout_func)
            while True:
                self.__running = True
                self.io_loop.start()
                if self.__failure is not None or condition is None or condition():
                    break
            if self.__timeout is not None:
                self.io_loop.remove_timeout(self.__timeout)
                self.__timeout = None
        assert self.__stopped
        self.__stopped = False
        self.__rethrow()
        result = self.__stop_args
        self.__stop_args = None
        return result

class AsyncHTTPTestCase(AsyncTestCase):
    """A test case that starts up an HTTP server.

    Subclasses must override `get_app()`, which returns the
    `tornado.web.Application` (or other `.HTTPServer` callback) to be tested.
    Tests will typically use the provided ``self.http_client`` to fetch
    URLs from this server.

    Example, assuming the "Hello, world" example from the user guide is in
    ``hello.py``::

        import hello

        class TestHelloApp(AsyncHTTPTestCase):
            def get_app(self):
                return hello.make_app()

            def test_homepage(self):
                response = self.fetch('/')
                self.assertEqual(response.code, 200)
                self.assertEqual(response.body, 'Hello, world')

    That call to ``self.fetch()`` is equivalent to ::

        self.http_client.fetch(self.get_url('/'), self.stop)
        response = self.wait()

    which illustrates how AsyncTestCase can turn an asynchronous operation,
    like ``http_client.fetch()``, into a synchronous operation. If you need
    to do other asynchronous operations in tests, you'll probably need to use
    ``stop()`` and ``wait()`` yourself.
    """

    def setUp(self) -> None:
        super().setUp()
        (sock, port) = bind_unused_port()
        self.__port: int = port
        self.http_client: AsyncHTTPClient = self.get_http_client()
        self._app: Application = self.get_app()
        self.http_server: HTTPServer = self.get_http_server()
        self.http_server.add_sockets([sock])

    def get_http_client(self) -> AsyncHTTPClient:
        return AsyncHTTPClient()

    def get_http_server(self) -> HTTPServer:
        return HTTPServer(self._app, **self.get_httpserver_options())

    def get_app(self) -> Application:
        """Should be overridden by subclasses to return a
        `tornado.web.Application` or other `.HTTPServer` callback.
        """
        raise NotImplementedError()

    def fetch(self, path: str, raise_error: bool = False, **kwargs: Any) -> HTTPResponse:
        """Convenience method to synchronously fetch a URL.

        The given path will be appended to the local server's host and
        port.  Any additional keyword arguments will be passed directly to
        `.AsyncHTTPClient.fetch` (and so could be used to pass
        ``method="POST"``, ``body="..."``, etc).

        If the path begins with http:// or https://, it will be treated as a
        full URL and will be fetched as-is.

        If ``raise_error`` is ``True``, a `tornado.httpclient.HTTPError` will
        be raised if the response code is not 200. This is the same behavior
        as the ``raise_error`` argument to `.AsyncHTTPClient.fetch`, but
        the default is ``False`` here (it's ``True`` in `.AsyncHTTPClient`)
        because tests often need to deal with non-200 response codes.

        .. versionchanged:: 5.0
           Added support for absolute URLs.

        .. versionchanged:: 5.1

           Added the ``raise_error`` argument.

        .. deprecated:: 5.1

           This method currently turns any exception into an
           `.HTTPResponse` with status code 599. In Tornado 6.0,
           errors other than `tornado.httpclient.HTTPError` will be
           passed through, and ``raise_error=False`` will only
           suppress errors that would be raised due to non-200
           response codes.

        """
        if path.lower().startswith(('http://', 'https://')):
            url = path
        else:
            url = self.get_url(path)
        return self.io_loop.run_sync(lambda: self.http_client.fetch(url, raise_error=raise_error, **kwargs), timeout=get_async_test_timeout())

    def get_httpserver_options(self) -> Dict[str, Any]:
        """May be overridden by subclasses to return additional
        keyword arguments for the server.
        """
        return {}

    def get_http_port(self) -> int:
        """Returns the port used by the server.

        A new port is chosen for each test.
        """
        return self.__port

    def get_protocol(self) -> str:
        return 'http'

    def get_url(self, path: str) -> str:
        """Returns an absolute url for the given path on the test server."""
        return f'{self.get_protocol()}://127.0.0.1:{self.get_http_port()}{path}'

    def tearDown(self) -> None:
        self.http_server.stop()
        self.io_loop.run_sync(self.http_server.close_all_connections, timeout=get_async_test_timeout())
        self.http_client.close()
        del self.http_server
        del self._app
        super().tearDown()

class AsyncHTTPSTestCase(AsyncHTTPTestCase):
    """A test case that starts an HTTPS server.

    Interface is generally the same as `AsyncHTTPTestCase`.
    """

    def get_http_client(self) -> AsyncHTTPClient:
        return AsyncHTTPClient(force_instance=True, defaults=dict(validate_cert=False))

    def get_httpserver_options(self) -> Dict[str, Any]:
        return dict(ssl_options=self.get_ssl_options())

    def get_ssl_options(self) -> Dict[str, Any]:
        """May be overridden by subclasses to select SSL options.

        By default includes a self-signed testing certificate.
        """
        return AsyncHTTPSTestCase.default_ssl_options()

    @staticmethod
    def default_ssl_options() -> Dict[str, Any]:
        module_dir = os.path.dirname(__file__)
        return dict(certfile=os.path.join(module_dir, 'test', 'test.crt'), keyfile=os.path.join(module_dir, 'test', 'test.key'))

    def get_protocol(self) -> str:
        return 'https'

@typing.overload
def gen_test(*, timeout: Optional[float] = None) -> Callable[[Callable[..., Union[Generator, Coroutine]]], Callable[..., None]]:
    pass

@typing.overload
def gen_test(func: Callable[..., Union[Generator, Coroutine]]) -> Callable[..., None]:
    pass

def gen_test(func: Optional[Callable[..., Union[Generator, Coroutine]]] = None, timeout: Optional[float] = None) -> Union[Callable[[Callable[..., Union[Generator, Coroutine