import os
import socket
import time
import contextlib
from threading import Thread, Event, Lock
import json
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, Callable, Iterator
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
NEW_APP_VERSION: str = "\nfrom chalice import Chalice\n\napp = Chalice(app_name='basicapp')\n\n\n@app.route('/')\ndef index():\n    return {'version': 'reloaded'}\n"

@contextmanager
def cd(path):
    try:
        original_dir: str = os.getcwd()
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)

@pytest.fixture()
def basic_app(tmpdir):
    tmpdir: str = str(tmpdir.mkdir('basicapp'))
    OSUtils().copytree(BASIC_APP, tmpdir)
    return tmpdir

class ThreadedLocalServer(Thread):

    def __init__(self, port, host='localhost'):
        super(ThreadedLocalServer, self).__init__()
        self._app_object: Optional[app.Chalice] = None
        self._config: Optional[Config] = None
        self._host: str = host
        self._port: int = port
        self._server: Optional[Any] = None
        self._server_ready: Event = Event()

    def wait_for_server_ready(self):
        self._server_ready.wait()

    def configure(self, app_object, config):
        self._app_object = app_object
        self._config = config

    def run(self):
        self._server = create_local_server(self._app_object, self._config, self._host, self._port)
        self._server_ready.set()
        self._server.serve_forever()

    def make_call(self, method, path, port, timeout=0.5):
        self._server_ready.wait()
        return method('http://{host}:{port}{path}'.format(path=path, host=self._host, port=port), timeout=timeout)

    def shutdown(self):
        if self._server is not None:
            self._server.server.shutdown()

@pytest.fixture
def config():
    return Config()

@pytest.fixture()
def unused_tcp_port():
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]

@pytest.fixture()
def http_session():
    session: requests.Session = requests.Session()
    retry: Retry = Retry(connect=10, backoff_factor=2, allowed_methods=['GET', 'POST', 'PUT'])
    session.mount('http://', HTTPAdapter(max_retries=retry))
    return HTTPFetcher(session)

class HTTPFetcher:

    def __init__(self, session):
        self.session: requests.Session = session

    def json_get(self, url):
        response: requests.Response = self.session.get(url)
        response.raise_for_status()
        return json.loads(response.content)

@pytest.fixture()
def local_server_factory(unused_tcp_port):
    threaded_server: ThreadedLocalServer = ThreadedLocalServer(unused_tcp_port)

    def create_server(app_object, config):
        threaded_server.configure(app_object, config)
        threaded_server.start()
        return (threaded_server, unused_tcp_port)
    try:
        yield create_server
    finally:
        threaded_server.shutdown()

@pytest.fixture
def sample_app():
    demo: app.Chalice = app.Chalice('demo-app')
    thread_safety_check: List[int] = []
    lock: Lock = Lock()

    @demo.route('/', methods=['GET'])
    def index():
        return {'hello': 'world'}

    @demo.route('/test-cors', methods=['POST'], cors=True)
    def test_cors():
        return {'hello': 'world'}

    @demo.route('/count', methods=['POST'])
    def record_counter():
        time.sleep(0.001)
        count: int = int(demo.current_request.json_body['counter'])
        with lock:
            thread_safety_check.append(count)

    @demo.route('/count', methods=['GET'])
    def get_record_counter():
        return thread_safety_check[:]
    return demo

def test_has_thread_safe_current_request(config, sample_app, local_server_factory):
    local_server: ThreadedLocalServer
    port: int
    local_server, port = local_server_factory(sample_app, config)
    local_server.wait_for_server_ready()
    num_requests: int = 25
    num_threads: int = 5

    def make_requests(counter_start):
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
    response: requests.Response = requests.get('http://localhost:%s/count' % port)
    assert len(response.json()) == len(range(num_requests * num_threads))
    assert sorted(response.json()) == list(range(num_requests * num_threads))

def test_can_accept_get_request(config, sample_app, local_server_factory):
    local_server: ThreadedLocalServer
    port: int
    local_server, port = local_server_factory(sample_app, config)
    response: requests.Response = local_server.make_call(requests.get, '/', port)
    assert response.status_code == 200
    assert response.text == '{"hello":"world"}'

def test_can_get_unicode_string_content_length(config, local_server_factory):
    demo: app.Chalice = app.Chalice('app-name')

    @demo.route('/')
    def index_view():
        return u'✓'
    local_server: ThreadedLocalServer
    port: int
    local_server, port = local_server_factory(demo, config)
    response: requests.Response = local_server.make_call(requests.get, '/', port)
    assert response.headers['Content-Length'] == '3'

def test_can_accept_options_request(config, sample_app, local_server_factory):
    local_server: ThreadedLocalServer
    port: int
    local_server, port = local_server_factory(sample_app, config)
    response: requests.Response = local_server.make_call(requests.options, '/test-cors', port)
    assert response.headers['Content-Length'] == '0'
    assert response.headers['Access-Control-Allow-Methods'] == 'POST,OPTIONS'
    assert response.text == ''

def test_can_accept_multiple_options_request(config, sample_app, local_server_factory):
    local_server: ThreadedLocalServer
    port: int
    local_server, port = local_server_factory(sample_app, config)
    response: requests.Response = local_server.make_call(requests.options, '/test-cors', port)
    assert response.headers['Content-Length'] == '0'
    assert response.headers['Access-Control-Allow-Methods'] == 'POST,OPTIONS'
    assert response.text == ''
    response = local_server.make_call(requests.options, '/test-cors', port)
    assert response.headers['Content-Length'] == '0'
    assert response.headers['Access-Control-Allow-Methods'] == 'POST,OPTIONS'
    assert response.text == ''

def test_can_accept_multiple_connections(config, sample_app, local_server_factory):
    local_server: ThreadedLocalServer
    port: int
    local_server, port = local_server_factory(sample_app, config)
    local_server.wait_for_server_ready()
    socket.create_connection(('localhost', port), timeout=1)
    try:
        response: requests.Response = local_server.make_call(requests.get, '/', port)
    except requests.exceptions.ReadTimeout:
        assert False, 'Read timeout occurred, the socket is blocking the next request from going though.'
    assert response.status_code == 200
    assert response.text == '{"hello":"world"}'

def test_can_import_env_vars(unused_tcp_port, http_session):
    with cd(ENV_APP_DIR):
        p: subprocess.Popen = subprocess.Popen(['chalice', 'local', '--port', str(unused_tcp_port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _wait_for_server_ready(p)
        try:
            _assert_env_var_loaded(unused_tcp_port, http_session)
        finally:
            p.terminate()

def _wait_for_server_ready(process):
    if process.poll() is not None:
        raise AssertionError('Local server immediately exited with rc: %s' % process.poll())

def _assert_env_var_loaded(port_number, http_session):
    response: Dict[str, Any] = http_session.json_get('http://localhost:%s/' % port_number)
    assert response == {'hello': 'bar'}

def test_can_reload_server(unused_tcp_port, basic_app, http_session):
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