import os
import socket
import time
import contextlib
from threading import Thread
from threading import Event
from threading import Lock
import json
import subprocess
from contextlib import contextmanager
from typing import Generator, Callable, Any, Optional, Tuple

import pytest
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from chalice import app
from chalice.local import create_local_server
from chalice.config import Config
from chalice.utils import OSUtils


APPS_DIR: str = os.path.dirname(os.path.abspath(__file__))
ENV_APP_DIR: str = os.path.join(APPS_DIR, 'envapp')
BASIC_APP: str = os.path.join(APPS_DIR, 'basicapp')


NEW_APP_VERSION: str = """
from chalice import Chalice

app = Chalice(app_name='basicapp')


@app.route('/')
def index():
    return {'version': 'reloaded'}
"""


@contextmanager
def cd(path: str) -> Generator[None, None, None]:
    try:
        original_dir: str = os.getcwd()
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


@pytest.fixture()
def basic_app(tmpdir: Any) -> str:
    tmpdir_path: str = str(tmpdir.mkdir('basicapp'))
    OSUtils().copytree(BASIC_APP, tmpdir_path)
    return tmpdir_path


class ThreadedLocalServer(Thread):
    def __init__(self, port: int, host: str = 'localhost') -> None:
        super(ThreadedLocalServer, self).__init__()
        self._app_object: Optional[Any] = None
        self._config: Optional[Config] = None
        self._host: str = host
        self._port: int = port
        self._server: Optional[Any] = None
        self._server_ready: Event = Event()

    def wait_for_server_ready(self) -> None:
        self._server_ready.wait()

    def configure(self, app_object: Any, config: Config) -> None:
        self._app_object = app_object
        self._config = config

    def run(self) -> None:
        self._server = create_local_server(
            self._app_object, self._config, self._host, self._port)
        self._server_ready.set()
        self._server.serve_forever()

    def make_call(self, method: Callable[..., requests.Response], path: str, port: int, timeout: float = 0.5) -> requests.Response:
        self._server_ready.wait()
        url: str = f'http://{self._host}:{port}{path}'
        return method(url, timeout=timeout)

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.server.shutdown()


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture()
def unused_tcp_port() -> int:
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class HTTPFetcher:
    def __init__(self, session: requests.Session) -> None:
        self.session: requests.Session = session

    def json_get(self, url: str) -> Any:
        response: requests.Response = self.session.get(url)
        response.raise_for_status()
        return json.loads(response.content)


@pytest.fixture()
def http_session() -> HTTPFetcher:
    session: requests.Session = requests.Session()
    retry: Retry = Retry(
        # How many connection-related errors to retry on.
        connect=10,
        # A backoff factor to apply between attempts after the second try.
        backoff_factor=2,
        allowed_methods=['GET', 'POST', 'PUT'],
    )
    session.mount('http://', HTTPAdapter(max_retries=retry))
    return HTTPFetcher(session)


@pytest.fixture()
def local_server_factory(unused_tcp_port: int) -> Callable[[Any, Config], Tuple[ThreadedLocalServer, int]]:
    threaded_server: ThreadedLocalServer = ThreadedLocalServer(unused_tcp_port)

    def create_server(app_object: Any, config: Config) -> Tuple[ThreadedLocalServer, int]:
        threaded_server.configure(app_object, config)
        threaded_server.start()
        return threaded_server, unused_tcp_port

    try:
        yield create_server
    finally:
        threaded_server.shutdown()


@pytest.fixture
def sample_app() -> app.Chalice:
    demo: app.Chalice = app.Chalice('demo-app')

    thread_safety_check: list = []
    lock: Lock = Lock()

    @demo.route('/', methods=['GET'])
    def index() -> dict:
        return {'hello': 'world'}

    @demo.route('/test-cors', methods=['POST'], cors=True)
    def test_cors() -> dict:
        return {'hello': 'world'}

    @demo.route('/count', methods=['POST'])
    def record_counter() -> None:
        # An extra delay helps ensure we consistently fail if we're
        # not thread safe.
        time.sleep(0.001)
        count: int = int(demo.current_request.json_body['counter'])
        with lock:
            thread_safety_check.append(count)

    @demo.route('/count', methods=['GET'])
    def get_record_counter() -> list:
        return thread_safety_check[:]

    return demo


def test_has_thread_safe_current_request(config: Config, sample_app: app.Chalice,
                                         local_server_factory: Callable[[Any, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    local_server, port = local_server_factory(sample_app, config)
    local_server.wait_for_server_ready()

    num_requests: int = 25
    num_threads: int = 5

    # The idea here is that each requests.post() has a unique 'counter'
    # integer.  If the current request is thread safe we should see a number
    # for each 0 - (num_requests * num_threads).  If it's not thread safe
    # we'll see missing numbers and/or duplicates.
    def make_requests(counter_start: int) -> None:
        for i in range(counter_start * num_requests,
                       (counter_start + 1) * num_requests):
            # We're slowing the sending rate down a bit.  The threaded
            # http server is good, but not great.  You can still overwhelm
            # it pretty easily.
            time.sleep(0.001)
            requests.post(
                f'http://localhost:{port}/count', json={'counter': i})

    threads: list = []
    for i in range(num_threads):
        threads.append(Thread(target=make_requests, args=(i,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    response: requests.Response = requests.get(f'http://localhost:{port}/count')
    assert len(response.json()) == len(range(num_requests * num_threads))
    assert sorted(response.json()) == list(range(num_requests * num_threads))


def test_can_accept_get_request(config: Config, sample_app: app.Chalice, local_server_factory: Callable[[Any, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    local_server, port = local_server_factory(sample_app, config)
    response: requests.Response = local_server.make_call(requests.get, '/', port)
    assert response.status_code == 200
    assert response.text == '{"hello":"world"}'


def test_can_get_unicode_string_content_length(
        config: Config, local_server_factory: Callable[[Any, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/')
    def index_view() -> str:
        return u'\u2713'

    local_server, port = local_server_factory(demo, config)
    response: requests.Response = local_server.make_call(requests.get, '/', port)
    assert response.headers['Content-Length'] == '3'


def test_can_accept_options_request(config: Config, sample_app: app.Chalice, local_server_factory: Callable[[Any, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    local_server, port = local_server_factory(sample_app, config)
    response: requests.Response = local_server.make_call(requests.options, '/test-cors', port)
    assert response.headers['Content-Length'] == '0'
    assert response.headers['Access-Control-Allow-Methods'] == 'POST,OPTIONS'
    assert response.text == ''


def test_can_accept_multiple_options_request(config: Config, sample_app: app.Chalice,
                                             local_server_factory: Callable[[Any, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    local_server, port = local_server_factory(sample_app, config)

    response: requests.Response = local_server.make_call(requests.options, '/test-cors', port)
    assert response.headers['Content-Length'] == '0'
    assert response.headers['Access-Control-Allow-Methods'] == 'POST,OPTIONS'
    assert response.text == ''

    response = local_server.make_call(requests.options, '/test-cors', port)
    assert response.headers['Content-Length'] == '0'
    assert response.headers['Access-Control-Allow-Methods'] == 'POST,OPTIONS'
    assert response.text == ''


def test_can_accept_multiple_connections(config: Config, sample_app: app.Chalice,
                                         local_server_factory: Callable[[Any, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    # When a GET request is made to Chalice from a browser, it will send the
    # connection keep-alive header in order to hold the connection open and
    # reuse it for subsequent requests. If the conncetion close header is sent
    # back by the server the connection will be closed, but the browser will
    # reopen a new connection just in order to have it ready when needed.
    # In this case, since it does not send any content we do not have the
    # opportunity to send a connection close header back in a response to
    # force it to close the socket.
    # This is an issue in Chalice since the single threaded local server will
    # now be blocked waiting for IO from the browser socket. If a request from
    # any other source is made it will be blocked until the browser sends
    # another request through, giving us a chance to read from another socket.
    local_server, port = local_server_factory(sample_app, config)
    local_server.wait_for_server_ready()
    # We create a socket here to emulate a browser's open connection and then
    # make a request. The request should succeed.
    with socket.create_connection(('localhost', port), timeout=1) as sock:
        try:
            response: requests.Response = local_server.make_call(requests.get, '/', port)
        except requests.exceptions.ReadTimeout:
            assert False, (
                'Read timeout occurred, the socket is blocking the next request '
                'from going though.'
            )
    assert response.status_code == 200
    assert response.text == '{"hello":"world"}'


def test_can_import_env_vars(unused_tcp_port: int, http_session: HTTPFetcher) -> None:
    with cd(ENV_APP_DIR):
        p: subprocess.Popen = subprocess.Popen(['chalice', 'local', '--port',
                                                str(unused_tcp_port)],
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
        _wait_for_server_ready(p)
        try:
            _assert_env_var_loaded(unused_tcp_port, http_session)
        finally:
            p.terminate()


def _wait_for_server_ready(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        raise AssertionError(
            f'Local server immediately exited with rc: {process.poll()}'
        )


def _assert_env_var_loaded(port_number: int, http_session: HTTPFetcher) -> None:
    response: Any = http_session.json_get(f'http://localhost:{port_number}/')
    assert response == {'hello': 'bar'}


def test_can_reload_server(unused_tcp_port: int, basic_app: str, http_session: HTTPFetcher) -> None:
    with cd(basic_app):
        p: subprocess.Popen = subprocess.Popen(['chalice', 'local', '--port',
                                                str(unused_tcp_port)],
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
        _wait_for_server_ready(p)
        url: str = f'http://localhost:{unused_tcp_port}/'
        try:
            assert http_session.json_get(url) == {'version': 'original'}
            # Updating the app should trigger a reload.
            with open(os.path.join(basic_app, 'app.py'), 'w') as f:
                f.write(NEW_APP_VERSION)
            time.sleep(2)
            assert http_session.json_get(url) == {'version': 'reloaded'}
        finally:
            p.terminate()
