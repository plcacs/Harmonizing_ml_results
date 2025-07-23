import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytest
import rapidjson
import uvicorn
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.exceptions import HTTPException
from fastapi.testclient import TestClient
from requests.auth import _basic_auth_str
from sqlalchemy import select
from freqtrade.__init__ import __version__
from freqtrade.enums import CandleType, RunMode, State, TradingMode
from freqtrade.exceptions import DependencyException, ExchangeError, OperationalException
from freqtrade.loggers import setup_logging, setup_logging_pre
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import Trade
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server import ApiServer
from freqtrade.rpc.api_server.api_auth import create_token, get_user_from_token
from freqtrade.rpc.api_server.uvicorn_threaded import UvicornServer
from freqtrade.rpc.api_server.webserver_bgwork import ApiBG
from freqtrade.util.datetime_helpers import format_date
from tests.conftest import CURRENT_TEST_STRATEGY, EXMS, create_mock_trades, create_mock_trades_usdt, generate_test_data, get_mock_coro, get_patched_freqtradebot, log_has, log_has_re, patch_get_signal

BASE_URI: str = '/api/v1'
_TEST_USER: str = 'FreqTrader'
_TEST_PASS: str = 'SuperSecurePassword1!'
_TEST_WS_TOKEN: str = 'secret_Ws_t0ken'

@pytest.fixture
def botclient(default_conf: Dict[str, Any], mocker: Any) -> Any:
    setup_logging_pre()
    setup_logging(default_conf)
    default_conf['runmode'] = RunMode.DRY_RUN
    default_conf.update({'api_server': {'enabled': True, 'listen_ip_address': '127.0.0.1', 'listen_port': 8080, 'CORS_origins': ['http://example.com'], 'username': _TEST_USER, 'password': _TEST_PASS, 'ws_token': _TEST_WS_TOKEN}})
    ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    mocker.patch('freqtrade.rpc.api_server.ApiServer.start_api', MagicMock())
    apiserver = None
    try:
        apiserver = ApiServer(default_conf)
        apiserver.add_rpc_handler(rpc)
        with TestClient(apiserver.app) as client:
            yield (ftbot, client)
    finally:
        if apiserver:
            apiserver.cleanup()
        ApiServer.shutdown()

def client_post(client: TestClient, url: str, data: Optional[Dict[str, Any]] = None) -> Any:
    if data is None:
        data = {}
    return client.post(url, json=data, headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS), 'Origin': 'http://example.com', 'content-type': 'application/json'})

def client_patch(client: TestClient, url: str, data: Optional[Dict[str, Any]] = None) -> Any:
    if data is None:
        data = {}
    return client.patch(url, json=data, headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS), 'Origin': 'http://example.com', 'content-type': 'application/json'})

def client_get(client: TestClient, url: str) -> Any:
    return client.get(url, headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS), 'Origin': 'http://example.com'})

def client_delete(client: TestClient, url: str) -> Any:
    return client.delete(url, headers={'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS), 'Origin': 'http://example.com'})

def assert_response(response: Any, expected_code: int = 200, needs_cors: bool = True) -> None:
    assert response.status_code == expected_code
    assert response.headers.get('content-type') == 'application/json'
    if needs_cors:
        assert ('access-control-allow-credentials', 'true') in response.headers.items()
        assert ('access-control-allow-origin', 'http://example.com') in response.headers.items()

def test_api_not_found(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/invalid_url')
    assert_response(rc, 404)
    assert rc.json() == {'detail': 'Not Found'}

def test_api_ui_fallback(botclient: Tuple[Any, TestClient], mocker: Any) -> None:
    _ftbot, client = botclient
    rc = client_get(client, '/favicon.ico')
    assert rc.status_code == 200
    rc = client_get(client, '/fallback_file.html')
    assert rc.status_code == 200
    assert '`freqtrade install-ui`' in rc.text
    rc = client_get(client, '/something')
    assert rc.status_code == 200
    rc = client_get(client, '/something.js')
    assert rc.status_code == 200
    rc = client_get(client, '%2F%2F%2Fetc/passwd')
    assert rc.status_code == 200
    assert '`freqtrade install-ui`' in rc.text or '<!DOCTYPE html>' in rc.text
    mocker.patch.object(Path, 'is_file', MagicMock(side_effect=[True, False]))
    rc = client_get(client, '%2F%2F%2Fetc/passwd')
    assert rc.status_code == 200
    assert '`freqtrade install-ui`' in rc.text

def test_api_ui_version(botclient: Tuple[Any, TestClient], mocker: Any) -> None:
    _ftbot, client = botclient
    mocker.patch('freqtrade.commands.deploy_ui.read_ui_version', return_value='0.1.2')
    rc = client_get(client, '/ui_version')
    assert rc.status_code == 200
    assert rc.json()['version'] == '0.1.2'

def test_api_auth() -> None:
    with pytest.raises(ValueError):
        create_token({'identity': {'u': 'Freqtrade'}}, 'secret1234', token_type='NotATokenType')
    token = create_token({'identity': {'u': 'Freqtrade'}}, 'secret1234')
    assert isinstance(token, str)
    u = get_user_from_token(token, 'secret1234')
    assert u == 'Freqtrade'
    with pytest.raises(HTTPException):
        get_user_from_token(token, 'secret1234', token_type='refresh')
    token = create_token({'identity': {'u1': 'Freqrade'}}, 'secret1234')
    with pytest.raises(HTTPException):
        get_user_from_token(token, 'secret1234')
    with pytest.raises(HTTPException):
        get_user_from_token(b'not_a_token', 'secret1234')

def test_api_ws_auth(botclient: Tuple[Any, TestClient]) -> None:
    ftbot, client = botclient

    def url(token: str) -> str:
        return f'/api/v1/message/ws?token={token}'
    bad_token = 'bad-ws_token'
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(url(bad_token)):
            pass
    good_token = _TEST_WS_TOKEN
    with client.websocket_connect(url(good_token)):
        pass
    jwt_secret = ftbot.config['api_server'].get('jwt_secret_key', 'super-secret')
    jwt_token = create_token({'identity': {'u': 'Freqtrade'}}, jwt_secret)
    with client.websocket_connect(url(jwt_token)):
        pass

# ... (continue with the rest of the functions, adding type annotations to each)

def test_api_unauthorized(botclient: Tuple[Any, TestClient]) -> None:
    ftbot, client = botclient
    rc = client.get(f'{BASE_URI}/ping')
    assert_response(rc, needs_cors=False)
    assert rc.json() == {'status': 'pong'}
    rc = client.get(f'{BASE_URI}/version')
    assert_response(rc, 401, needs_cors=False)
    assert rc.json() == {'detail': 'Unauthorized'}
    ftbot.config['api_server']['username'] = 'Ftrader'
    rc = client_get(client, f'{BASE_URI}/version')
    assert_response(rc, 401)
    assert rc.json() == {'detail': 'Unauthorized'}
    ftbot.config['api_server']['username'] = _TEST_USER
    ftbot.config['api_server']['password'] = 'WrongPassword'
    rc = client_get(client, f'{BASE_URI}/version')
    assert_response(rc, 401)
    assert rc.json() == {'detail': 'Unauthorized'}
    ftbot.config['api_server']['username'] = 'Ftrader'
    ftbot.config['api_server']['password'] = 'WrongPassword'
    rc = client_get(client, f'{BASE_URI}/version')
    assert_response(rc, 401)
    assert rc.json() == {'detail': 'Unauthorized'}

def test_api_token_login(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client.post(f'{BASE_URI}/token/login', data=None, headers={'Authorization': _basic_auth_str('WRONG_USER', 'WRONG_PASS'), 'Origin': 'http://example.com'})
    assert_response(rc, 401)
    rc = client_post(client, f'{BASE_URI}/token/login')
    assert_response(rc)
    assert 'access_token' in rc.json()
    assert 'refresh_token' in rc.json()
    rc = client.get(f'{BASE_URI}/count', headers={'Authorization': f'Bearer {rc.json()["access_token"]}', 'Origin': 'http://example.com'})
    assert_response(rc)

def test_api_token_refresh(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_post(client, f'{BASE_URI}/token/login')
    assert_response(rc)
    rc = client.post(f'{BASE_URI}/token/refresh', data=None, headers={'Authorization': f'Bearer {rc.json()["refresh_token"]}', 'Origin': 'http://example.com'})
    assert_response(rc)
    assert 'access_token' in rc.json()
    assert 'refresh_token' not in rc.json()

# ... (continue with the rest of the functions, adding type annotations to each)

def test_api_stop_workflow(botclient: Tuple[Any, TestClient]) -> None:
    ftbot, client = botclient
    assert ftbot.state == State.RUNNING
    rc = client_post(client, f'{BASE_URI}/stop')
    assert_response(rc)
    assert rc.json() == {'status': 'stopping trader ...'}
    assert ftbot.state == State.STOPPED
    rc = client_post(client, f'{BASE_URI}/stop')
    assert_response(rc)
    assert rc.json() == {'status': 'already stopped'}
    rc = client_post(client, f'{BASE_URI}/start')
    assert_response(rc)
    assert rc.json() == {'status': 'starting trader ...'}
    assert ftbot.state == State.RUNNING
    rc = client_post(client, f'{BASE_URI}/start')
    assert_response(rc)
    assert rc.json() == {'status': 'already running'}

# ... (continue with the rest of the functions, adding type annotations to each)

def test_api__init__(default_conf: Dict[str, Any], mocker: Any) -> None:
    default_conf.update({'api_server': {'enabled': True, 'listen_ip_address': '127.0.0.1', 'listen_port': 8080, 'username': 'TestUser', 'password': 'testPass'}})
    mocker.patch('freqtrade.rpc.telegram.Telegram._init')
    mocker.patch('freqtrade.rpc.api_server.webserver.ApiServer.start_api', MagicMock())
    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))
    assert apiserver._config == default_conf
    with pytest.raises(OperationalException, match='RPC Handler already attached.'):
        apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))
    apiserver.cleanup()
    ApiServer.shutdown()

# ... (continue with the rest of the functions, adding type annotations to each)

def test_api_UvicornServer(mocker: Any) -> None:
    thread_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.threading.Thread')
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert thread_mock.call_count == 0
    s.started = True
    s.run_in_thread()
    assert thread_mock.call_count == 1
    s.cleanup()
    assert s.should_exit is True

# ... (continue with the rest of the functions, adding type annotations to each)

def test_api_UvicornServer_run(mocker: Any) -> None:
    serve_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.UvicornServer.serve', get_mock_coro(None))
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert serve_mock.call_count == 0
    s.started = True
    s.run()
    assert serve_mock.call_count == 1

# ... (continue with the rest of the functions, adding type annotations to each)

def test_api_UvicornServer_run_no_uvloop(mocker: Any, import_fails: Any) -> None:
    serve_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.UvicornServer.serve', get_mock_coro(None))
    asyncio.set_event_loop(asyncio.new_event_loop())
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert serve_mock.call_count == 0
    s.started = True
    s.run()
    assert serve_mock.call_count == 1

# ... (continue with the rest of the functions, adding type annotations to each)

def test_api_run(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf.update({'api_server': {'enabled': True, 'listen_ip_address': '127.0.0.1', 'listen_port': 8080, 'username': 'TestUser', 'password': 'testPass'}})
    mocker.patch('freqtrade.rpc.telegram.Telegram._init')
    server_inst_mock = MagicMock()
    server_inst_mock.run_in_thread = MagicMock()
    server_inst_mock.run = MagicMock()
    server_mock = MagicMock(return_value=server_inst_mock)
    mocker.patch('freqtrade.rpc.api_server.webserver.UvicornServer', server_mock)
    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))
    assert server_mock.call_count == 1
    assert apiserver._config == default_conf
    apiserver.start_api()
    assert server_mock.call_count == 2
    assert server_inst_mock.run_in_thread.call_count == 2
    assert server_inst_mock.run.call_count == 0
    assert server_mock.call_args_list[0][0][0].host == '127.0.0.1'
    assert server_mock.call_args_list[0][0][0].port == 8080
    assert isinstance(server_mock.call_args_list[0][0][0].app, FastAPI)
    assert log_has('Starting HTTP Server at 127.0.0.1:8080', caplog)
    assert log_has('Starting Local Rest Server.', caplog)
    caplog.clear()
    server_mock.reset_mock()
    apiserver._config.update({'api_server': {'enabled': True, 'listen_ip_address': '0.0.0.0', 'listen_port': 8089, 'password': ''}})
    apiserver.start_api()
    assert server_mock.call_count == 1
    assert server_inst_mock.run_in_thread.call_count == 1
    assert server_inst_mock.run.call_count == 0
    assert server_mock.call_args_list[0][0][0].host == '0.0.0.0'
    assert server_mock.call_args_list[0][0][0].port == 8089
    assert isinstance(server_mock.call_args_list[0][0][0].app, FastAPI)
    assert log_has('Starting HTTP Server at 0.0.0.0:8089', caplog)
    assert log_has('Starting Local Rest Server.', caplog)
    assert log_has('SECURITY WARNING - Local Rest Server listening to external connections', caplog)
    assert log_has('SECURITY WARNING - This is insecure please set to your loopback,e.g 127.0.0.1 in config.json', caplog)
    assert log_has('SECURITY WARNING - No password for local REST Server defined. Please make sure that this is intentional!', caplog)
    assert log_has_re('SECURITY WARNING - `jwt_secret_key` seems to be default.*', caplog)
    server_mock.reset_mock()
    apiserver._standalone = True
    apiserver.start_api()
    assert server_inst_mock.run