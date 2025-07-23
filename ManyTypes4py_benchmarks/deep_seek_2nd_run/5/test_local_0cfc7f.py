import os
import socket
import time
import contextlib
from threading import Thread
from threading import Event
from threading import Lock
from typing import Any, Dict, List, Generator, Iterator, Tuple, Optional, Callable, Union
import json
import subprocess
from contextlib import contextmanager
import pytest
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from chalice import app
from chalice.local import create_local_server
from chalice.config import Config
from chalice.utils import OSUtils
from requests import Session, Response
from _pytest.fixtures import FixtureRequest
from _pytest.tmpdir import TempPathFactory
from chalice.app import Chalice

APPS_DIR: str = os.path.dirname(os.path.abspath(__file__))
ENV_APP_DIR: str = os.path.join(APPS_DIR, 'envapp')
BASIC_APP: str = os.path.join(APPS_DIR, 'basicapp')
NEW_APP_VERSION: str = "\nfrom chalice import Chalice\n\napp = Chalice(app_name='basicapp')\n\n\n@app.route('/')\ndef index():\n    return {'version': 'reloaded'}\n"

@contextmanager
def cd(path: str) -> Generator[None, None, None]:
    try:
        original_dir: str = os.getcwd()
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)

@pytest.fixture()
def basic_app(tmpdir: TempPathFactory) -> str:
    tmpdir_str: str = str(tmpdir.mkdir('basicapp'))
    OSUtils().copytree(BASIC_APP, tmpdir_str)
    return tmpdir_str

class ThreadedLocalServer(Thread):
    def __init__(self, port: int, host: str = 'localhost') -> None:
        super(ThreadedLocalServer, self).__init__()
        self._app_object: Optional[Chalice] = None
        self._config: Optional[Config] = None
        self._host: str = host
        self._port: int = port
        self._server: Any = None
        self._server_ready: Event = Event()

    def wait_for_server_ready(self) -> None:
        self._server_ready.wait()

    def configure(self, app_object: Chalice, config: Config) -> None:
        self._app_object = app_object
        self._config = config

    def run(self) -> None:
        if self._app_object is None or self._config is None:
            raise ValueError("App object and config must be configured")
        self._server = create_local_server(self._app_object, self._config, self._host, self._port)
        self._server_ready.set()
        self._server.serve_forever()

    def make_call(self, method: Callable[..., Response], path: str, port: int, timeout: float = 0.5) -> Response:
        self._server_ready.wait()
        return method('http://{host}:{port}{path}'.format(path=path, host=self._host, port=port), timeout=timeout)

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

@pytest.fixture()
def http_session() -> 'HTTPFetcher':
    session: Session = requests.Session()
    retry: Retry = Retry(connect=10, backoff_factor=2, allowed_methods=['GET', 'POST', 'PUT'])
    session.mount('http://', HTTPAdapter(max_retries=retry))
    return HTTPFetcher(session)

class HTTPFetcher:
    def __init__(self, session: Session) -> None:
        self.session: Session = session

    def json_get(self, url: str) -> Dict[str, Any]:
        response: Response = self.session.get(url)
        response.raise_for_status()
        return json.loads(response.content)

@pytest.fixture()
def local_server_factory(unused_tcp_port: int) -> Generator[Callable[[Chalice, Config], Tuple[ThreadedLocalServer, int]], None, None]:
    threaded_server: ThreadedLocalServer = ThreadedLocalServer(unused_tcp_port)

    def create_server(app_object: Chalice, config: Config) -> Tuple[ThreadedLocalServer, int]:
        threaded_server.configure(app_object, config)
        threaded_server.start()
        return (threaded_server, unused_tcp_port)
    try:
        yield create_server
    finally:
        threaded_server.shutdown()

@pytest.fixture
def sample_app() -> Chalice:
    demo: Chalice = app.Chalice('demo-app')
    thread_safety_check: List[int] = []
    lock: Lock = Lock()

    @demo.route('/', methods=['GET'])
    def index() -> Dict[str, str]:
        return {'hello': 'world'}

    @demo.route('/test-cors', methods=['POST'], cors=True)
    def test_cors() -> Dict[str, str]:
        return {'hello': 'world'}

    @demo.route('/count', methods=['POST'])
    def record_counter() -> None:
        time.sleep(0.001)
        count: int = int(demo.current_request.json_body['counter'])
        with lock:
            thread_safety_check.append(count)

    @demo.route('/count', methods=['GET'])
    def get_record_counter() -> List[int]:
        return thread_safety_check[:]
    return demo

def test_has_thread_safe_current_request(config: Config, sample_app: Chalice, local_server_factory: Callable[[Chalice, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    local_server, port = local_server_factory(sample_app, config)
    local_server.wait_for_server_ready()
    num_requests: int = 25
    num_threads: int = 5

    def make_requests(counter_start: int) -> None:
        for i in range(counter_start * num_requests, (counter_start + 1) * num_requests):
            time.sleep(0.001)
            requests.post('http://localhost:%s/count' % port, json={'counter': i})
    threads: List[Thread] = []
    for i in range(num_threads):
        threads.append(Thread(target=make_requests, args=(i,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    response: Response = requests.get('http://localhost:%s/count' % port)
    assert len(response.json()) == len(range(num_requests * num_threads))
    assert sorted(response.json()) == list(range(num_requests * num_threads))

def test_can_accept_get_request(config: Config, sample_app: Chalice, local_server_factory: Callable[[Chalice, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    local_server, port = local_server_factory(sample_app, config)
    response: Response = local_server.make_call(requests.get, '/', port)
    assert response.status_code == 200
    assert response.text == '{"hello":"world"}'

def test_can_get_unicode_string_content_length(config: Config, local_server_factory: Callable[[Chalice, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    demo: Chalice = app.Chalice('app-name')

    @demo.route('/')
    def index_view() -> str:
        return u'✓'
    local_server, port = local_server_factory(demo, config)
    response: Response = local_server.make_call(requests.get, '/', port)
    assert response.headers['Content-Length'] == '3'

def test_can_accept_options_request(config: Config, sample_app: Chalice, local_server_factory: Callable[[Chalice, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    local_server, port = local_server_factory(sample_app, config)
    response: Response = local_server.make_call(requests.options, '/test-cors', port)
    assert response.headers['Content-Length'] == '0'
    assert response.headers['Access-Control-Allow-Methods'] == 'POST,OPTIONS'
    assert response.text == ''

def test_can_accept_multiple_options_request(config: Config, sample_app: Chalice, local_server_factory: Callable[[Chalice, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    local_server, port = local_server_factory(sample_app, config)
    response: Response = local_server.make_call(requests.options, '/test-cors', port)
    assert response.headers['Content-Length'] == '0'
    assert response.headers['Access-Control-Allow-Methods'] == 'POST,OPTIONS'
    assert response.text == ''
    response = local_server.make_call(requests.options, '/test-cors', port)
    assert response.headers['Content-Length'] == '0'
    assert response.headers['Access-Control-Allow-Methods'] == 'POST,OPTIONS'
    assert response.text == ''

def test_can_accept_multiple_connections(config: Config, sample_app: Chalice, local_server_factory: Callable[[Chalice, Config], Tuple[ThreadedLocalServer, int]]) -> None:
    local_server, port = local_server_factory(sample_app, config)
    local_server.wait_for_server_ready()
    socket.create_connection(('localhost', port), timeout=1)
    try:
        response: Response = local_server.make_call(requests.get, '/', port)
    except requests.exceptions.ReadTimeout:
        assert False, 'Read timeout occurred, the socket is blocking the next request from going though.'
    assert response.status_code == 200
    assert response.text == '{"hello":"world"}'

def test_can_import_env_vars(unused_tcp_port: int, http_session: HTTPFetcher) -> None:
    with cd(ENV_APP_DIR):
        p: subprocess.Popen = subprocess.Popen(['chalice', 'local', '--port', str(unused_tcp_port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _wait_for_server_ready(p)
        try:
            _assert_env_var_loaded(unused_tcp_port, http_session)
        finally:
            p.terminate()

def _wait_for_server_ready(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        raise AssertionError('Local server immediately exited with rc: %s' % process.poll())

def _assert_env_var_loaded(port_number: int, http_session: HTTPFetcher) -> None:
    response: Dict[str, str] = http_session.json_get('http://localhost:%s/' % port_number)
    assert response == {'hello': 'bar'}

def test_can_reload_server(unused_tcp_port: int, basic_app: str, http_session: HTTPFetcher) -> None:
    with cd(basic_app):
        p: subprocess.Popen = subprocess.Popen(['chalice', 'local', '--port', str(unused_tcp_port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _wait_for_server_ready(p)
        url: str = 'http://localhost:%s/' % unused_tcp_port
        try:
            assert http_session.json_get(url) == {'version': 'original'}
            with open(os.path.join(basic_app, 'app.py'), 'w') as f:
                f.write(NEW_APP_VERSION)
            time.sleep(2)
            assert http_session.json_get(url) == {'version': 'reloaded'}
        finally:
            p.terminate()
