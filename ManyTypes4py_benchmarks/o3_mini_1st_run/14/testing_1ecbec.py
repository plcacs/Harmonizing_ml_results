#!/usr/bin/env python3
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
from typing import Tuple, Any, Callable, Type, Dict, Union, Optional, Coroutine, ContextManager

if typing.TYPE_CHECKING:
    _ExcInfoTuple = Tuple[Optional[Type[BaseException]], Optional[BaseException], Optional[Any]]
_NON_OWNED_IOLOOPS: Any = AsyncIOMainLoop

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
    sock: socket.socket = netutil.bind_sockets(0, address, family=socket.AF_INET, reuse_port=reuse_port)[0]
    port: int = sock.getsockname()[1]
    return (sock, port)

def get_async_test_timeout() -> float:
    """Get the global timeout setting for async tests.

    Returns a float, the timeout in seconds.

    .. versionadded:: 3.1
    """
    env: Optional[str] = os.environ.get('ASYNC_TEST_TIMEOUT')
    if env is not None:
        try:
            return float(env)
        except ValueError:
            pass
    return 5.0

class AsyncTestCase(unittest.TestCase):
    """`~unittest.TestCase` subclass for testing `.IOLoop`-based
    asynchronous code.
    """
    def __init__(self, methodName: str = 'runTest') -> None:
        super().__init__(methodName)
        self.__stopped: bool = False
        self.__running: bool = False
        self.__failure: Optional[_ExcInfoTuple] = None
        self.__stop_args: Any = None
        self.__timeout: Any = None
        self._test_generator: Optional[Union[Generator, Coroutine]] = None

    def setUp(self) -> None:
        py_ver = sys.version_info
        if (3, 10, 0) <= py_ver < (3, 10, 9) or (3, 11, 0) <= py_ver <= (3, 11, 1):
            setup_with_context_manager(self, warnings.catch_warnings())
            warnings.filterwarnings('ignore', message='There is no current event loop', category=DeprecationWarning, module='tornado\\..*')
        super().setUp()
        if type(self).get_new_ioloop is not AsyncTestCase.get_new_ioloop:
            warnings.warn('get_new_ioloop is deprecated', DeprecationWarning)
        self.io_loop: IOLoop = self.get_new_ioloop()
        asyncio.set_event_loop(self.io_loop.asyncio_loop)

    def tearDown(self) -> None:
        asyncio_loop = self.io_loop.asyncio_loop
        tasks = asyncio.all_tasks(asyncio_loop)
        tasks = [t for t in tasks if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            done, pending = self.io_loop.run_sync(lambda: asyncio.wait(tasks))
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

    def _handle_exception(self, typ: Any, value: Any, tb: Any) -> bool:
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

    def run(self, result: Optional[unittest.TestResult] = None) -> Any:
        ret = super().run(result)
        self.__rethrow()
        return ret

    def _callTestMethod(self, method: Callable[..., Any]) -> None:
        """Run the given test method, raising an error if it returns non-None.

        Failure to decorate asynchronous test methods with ``@gen_test`` can lead to tests
        incorrectly passing.
        """
        result = method()
        if isinstance(result, Generator) or inspect.iscoroutine(result):
            raise TypeError('Generator and coroutine test methods should be decorated with tornado.testing.gen_test')
        elif result is not None:
            raise ValueError('Return value from test method ignored: %r' % result)

    def stop(self, _arg: Any = None, **kwargs: Any) -> None:
        """Stops the `.IOLoop`, causing one pending (or future) call to `wait()`
        to return.
        """
        assert _arg is None or not kwargs
        self.__stop_args = kwargs or _arg
        if self.__running:
            self.io_loop.stop()
            self.__running = False
        self.__stopped = True

    def wait(self, condition: Optional[Callable[[], bool]] = None, timeout: Optional[float] = None) -> Any:
        """Runs the `.IOLoop` until stop is called or timeout has passed.
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
    """
    def setUp(self) -> None:
        super().setUp()
        sock, port = bind_unused_port()
        self.__port: int = port
        self.http_client: AsyncHTTPClient = self.get_http_client()
        self._app: Any = self.get_app()
        self.http_server: HTTPServer = self.get_http_server()
        self.http_server.add_sockets([sock])

    def get_http_client(self) -> AsyncHTTPClient:
        return AsyncHTTPClient()

    def get_http_server(self) -> HTTPServer:
        return HTTPServer(self._app, **self.get_httpserver_options())

    def get_app(self) -> Any:
        """Should be overridden by subclasses to return a
        `tornado.web.Application` or other `.HTTPServer` callback.
        """
        raise NotImplementedError()

    def fetch(self, path: str, raise_error: bool = False, **kwargs: Any) -> HTTPResponse:
        """Convenience method to synchronously fetch a URL.
        """
        if path.lower().startswith(('http://', 'https://')):
            url: str = path
        else:
            url = self.get_url(path)
        return self.io_loop.run_sync(
            lambda: self.http_client.fetch(url, raise_error=raise_error, **kwargs),
            timeout=get_async_test_timeout()
        )

    def get_httpserver_options(self) -> Dict[str, Any]:
        """May be overridden by subclasses to return additional
        keyword arguments for the server.
        """
        return {}

    def get_http_port(self) -> int:
        """Returns the port used by the server.
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
    """
    def get_http_client(self) -> AsyncHTTPClient:
        return AsyncHTTPClient(force_instance=True, defaults=dict(validate_cert=False))

    def get_httpserver_options(self) -> Dict[str, Any]:
        return dict(ssl_options=self.get_ssl_options())

    def get_ssl_options(self) -> Dict[str, Any]:
        """May be overridden by subclasses to select SSL options.
        """
        return AsyncHTTPSTestCase.default_ssl_options()

    @staticmethod
    def default_ssl_options() -> Dict[str, Any]:
        module_dir: str = os.path.dirname(__file__)
        return dict(
            certfile=os.path.join(module_dir, 'test', 'test.crt'),
            keyfile=os.path.join(module_dir, 'test', 'test.key')
        )

    def get_protocol(self) -> str:
        return 'https'

@typing.overload
def gen_test(*, timeout: Optional[float] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...

@typing.overload
def gen_test(func: Callable[..., Any]) -> Callable[..., Any]:
    ...

def gen_test(func: Optional[Callable[..., Any]] = None, timeout: Optional[float] = None) -> Any:
    """Testing equivalent of ``@gen.coroutine``, to be applied to test methods.
    """
    if timeout is None:
        timeout = get_async_test_timeout()

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(f)
        def pre_coroutine(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = f(self, *args, **kwargs)
            if isinstance(result, Generator) or inspect.iscoroutine(result):
                self._test_generator = result
            else:
                self._test_generator = None
            return result
        if inspect.iscoroutinefunction(f):
            coro: Callable[..., Any] = pre_coroutine
        else:
            coro = gen.coroutine(pre_coroutine)

        @functools.wraps(coro)
        def post_coroutine(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return self.io_loop.run_sync(functools.partial(coro, self, *args, **kwargs), timeout=timeout)
            except TimeoutError as e:
                if self._test_generator is not None and getattr(self._test_generator, 'cr_running', True):
                    self._test_generator.throw(e)
                raise
        return post_coroutine
    if func is not None:
        return wrap(func)
    else:
        return wrap
gen_test.__test__ = False

class ExpectLog(logging.Filter):
    """Context manager to capture and suppress expected log output.
    """
    def __init__(self, logger: Union[logging.Logger, str], regex: str, required: bool = True, level: Optional[int] = None) -> None:
        """Constructs an ExpectLog context manager.
        """
        if isinstance(logger, basestring_type):
            logger = logging.getLogger(logger)
        self.logger: logging.Logger = logger
        self.regex: re.Pattern = re.compile(regex)
        self.required: bool = required
        self.matched: int = 0
        self.deprecated_level_matched: int = 0
        self.logged_stack: bool = False
        self.level: Optional[int] = level
        self.orig_level: Optional[int] = None

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            self.logged_stack = True
        message: str = record.getMessage()
        if self.regex.match(message):
            if self.level is None and record.levelno < logging.WARNING:
                self.deprecated_level_matched += 1
            if self.level is not None and record.levelno != self.level:
                app_log.warning('Got expected log message %r at unexpected level (%s vs %s)' %
                                (message, logging.getLevelName(self.level), record.levelname))
                return True
            self.matched += 1
            return False
        return True

    def __enter__(self) -> "ExpectLog":
        if self.level is not None and self.level < self.logger.getEffectiveLevel():
            self.orig_level = self.logger.level
            self.logger.setLevel(self.level)
        self.logger.addFilter(self)
        return self

    def __exit__(self, typ: Optional[Type[BaseException]], value: Optional[BaseException], tb: Optional[Any]) -> Optional[bool]:
        if self.orig_level is not None:
            self.logger.setLevel(self.orig_level)
        self.logger.removeFilter(self)
        if not typ and self.required and (not self.matched):
            raise Exception('did not get expected log message')
        if not typ and self.required and (self.deprecated_level_matched >= self.matched):
            warnings.warn('ExpectLog matched at INFO or below without level argument', DeprecationWarning)
        return None

def setup_with_context_manager(testcase: unittest.TestCase, cm: ContextManager[Any]) -> Any:
    """Use a context manager to setUp a test case.
    """
    val = cm.__enter__()
    testcase.addCleanup(cm.__exit__, None, None, None)
    return val

def main(**kwargs: Any) -> None:
    """A simple test runner.
    """
    from tornado.options import define, options, parse_command_line
    define('exception_on_interrupt', type=bool, default=True,
           help='If true (default), ctrl-c raises a KeyboardInterrupt exception.  This prints a stack trace but cannot interrupt certain operations.  If false, the process is more reliably killed, but does not print a stack trace.')
    define('verbose', type=bool)
    define('quiet', type=bool)
    define('failfast', type=bool)
    define('catch', type=bool)
    define('buffer', type=bool)
    argv = [sys.argv[0]] + parse_command_line(sys.argv)
    if not options.exception_on_interrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    if options.verbose is not None:
        kwargs['verbosity'] = 2
    if options.quiet is not None:
        kwargs['verbosity'] = 0
    if options.failfast is not None:
        kwargs['failfast'] = True
    if options.catch is not None:
        kwargs['catchbreak'] = True
    if options.buffer is not None:
        kwargs['buffer'] = True
    if __name__ == '__main__' and len(argv) == 1:
        print('No tests specified', file=sys.stderr)
        sys.exit(1)
    if len(argv) > 1:
        unittest.main(module=None, argv=argv, **kwargs)
    else:
        unittest.main(defaultTest='all', argv=argv, **kwargs)

if __name__ == '__main__':
    main()