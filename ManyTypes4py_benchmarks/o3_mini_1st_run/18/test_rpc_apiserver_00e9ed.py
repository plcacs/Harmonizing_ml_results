#!/usr/bin/env python3
"""
Unit test file for rpc/api_server.py
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import pytest
import rapidjson
import uvicorn
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.exceptions import HTTPException
from fastapi.testclient import TestClient
from requests.auth import _basic_auth_str
from requests.models import Response
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
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    create_mock_trades,
    create_mock_trades_usdt,
    generate_test_data,
    get_mock_coro,
    get_patched_freqtradebot,
    log_has,
    log_has_re,
    patch_get_signal,
)

BASE_URI: str = '/api/v1'
_TEST_USER: str = 'FreqTrader'
_TEST_PASS: str = 'SuperSecurePassword1!'
_TEST_WS_TOKEN: str = 'secret_Ws_t0ken'


@pytest.fixture
def botclient(default_conf: Dict[str, Any], mocker: Any) -> Iterator[Tuple[Any, TestClient]]:
    setup_logging_pre()
    setup_logging(default_conf)
    default_conf['runmode'] = RunMode.DRY_RUN
    default_conf.update({
        'api_server': {
            'enabled': True,
            'listen_ip_address': '127.0.0.1',
            'listen_port': 8080,
            'CORS_origins': ['http://example.com'],
            'username': _TEST_USER,
            'password': _TEST_PASS,
            'ws_token': _TEST_WS_TOKEN
        }
    })
    ftbot = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(ftbot)
    mocker.patch('freqtrade.rpc.api_server.ApiServer.start_api', return_value=None)
    apiserver: Optional[ApiServer] = None
    try:
        apiserver = ApiServer(default_conf)
        apiserver.add_rpc_handler(rpc)
        with TestClient(apiserver.app) as client:
            yield (ftbot, client)
    finally:
        if apiserver:
            apiserver.cleanup()
        ApiServer.shutdown()


def client_post(client: TestClient, url: str, data: Optional[Any] = None) -> Response:
    if data is None:
        data = {}
    return client.post(url, json=data, headers={
        'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
        'Origin': 'http://example.com',
        'content-type': 'application/json'
    })


def client_patch(client: TestClient, url: str, data: Optional[Any] = None) -> Response:
    if data is None:
        data = {}
    return client.patch(url, json=data, headers={
        'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
        'Origin': 'http://example.com',
        'content-type': 'application/json'
    })


def client_get(client: TestClient, url: str) -> Response:
    return client.get(url, headers={
        'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
        'Origin': 'http://example.com'
    })


def client_delete(client: TestClient, url: str) -> Response:
    return client.delete(url, headers={
        'Authorization': _basic_auth_str(_TEST_USER, _TEST_PASS),
        'Origin': 'http://example.com'
    })


def assert_response(response: Response, expected_code: int = 200, needs_cors: bool = True) -> None:
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
    mocker.patch.object(Path, 'is_file', return_value=True)
    mocker.patch.object(Path, 'is_file', side_effect=[True, False])
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
    token: str = create_token({'identity': {'u': 'Freqtrade'}}, 'secret1234')
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

    bad_token: str = 'bad-ws_token'
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(url(bad_token)):
            pass
    good_token: str = _TEST_WS_TOKEN
    with client.websocket_connect(url(good_token)):
        pass
    jwt_secret: str = ftbot.config['api_server'].get('jwt_secret_key', 'super-secret')
    jwt_token: str = create_token({'identity': {'u': 'Freqtrade'}}, jwt_secret)
    with client.websocket_connect(url(jwt_token)):
        pass


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
    rc = client.post(f'{BASE_URI}/token/login', data=None, headers={
        'Authorization': _basic_auth_str('WRONG_USER', 'WRONG_PASS'),
        'Origin': 'http://example.com'
    })
    assert_response(rc, 401)
    rc = client_post(client, f'{BASE_URI}/token/login')
    assert_response(rc)
    assert 'access_token' in rc.json()
    assert 'refresh_token' in rc.json()
    rc = client.get(f'{BASE_URI}/count', headers={
        'Authorization': f'Bearer {rc.json()["access_token"]}',
        'Origin': 'http://example.com'
    })
    assert_response(rc)


def test_api_token_refresh(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_post(client, f'{BASE_URI}/token/login')
    assert_response(rc)
    rc = client.post(f'{BASE_URI}/token/refresh', data=None, headers={
        'Authorization': f'Bearer {rc.json()["refresh_token"]}',
        'Origin': 'http://example.com'
    })
    assert_response(rc)
    assert 'access_token' in rc.json()
    assert 'refresh_token' not in rc.json()


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


def test_api__init__(default_conf: Dict[str, Any], mocker: Any) -> None:
    """
    Test __init__() method
    """
    default_conf.update({
        'api_server': {
            'enabled': True,
            'listen_ip_address': '127.0.0.1',
            'listen_port': 8080,
            'username': 'TestUser',
            'password': 'testPass'
        }
    })
    mocker.patch('freqtrade.rpc.telegram.Telegram._init')
    mocker.patch('freqtrade.rpc.api_server.webserver.ApiServer.start_api', return_value=None)
    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))
    assert apiserver._config == default_conf
    with pytest.raises(OperationalException, match='RPC Handler already attached.'):
        apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))
    apiserver.cleanup()
    ApiServer.shutdown()


def test_api_UvicornServer(mocker: Any) -> None:
    thread_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.threading.Thread')
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert thread_mock.call_count == 0
    s.started = True
    s.run_in_thread()
    assert thread_mock.call_count == 1
    s.cleanup()
    assert s.should_exit is True


def test_api_UvicornServer_run(mocker: Any) -> None:
    serve_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.UvicornServer.serve', get_mock_coro(None))
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert serve_mock.call_count == 0
    s.started = True
    s.run()
    assert serve_mock.call_count == 1


def test_api_UvicornServer_run_no_uvloop(mocker: Any, import_fails: Any) -> None:
    serve_mock = mocker.patch('freqtrade.rpc.api_server.uvicorn_threaded.UvicornServer.serve', get_mock_coro(None))
    asyncio.set_event_loop(asyncio.new_event_loop())
    s = UvicornServer(uvicorn.Config(MagicMock(), port=8080, host='127.0.0.1'))
    assert serve_mock.call_count == 0
    s.started = True
    s.run()
    assert serve_mock.call_count == 1


def test_api_run(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf.update({
        'api_server': {
            'enabled': True,
            'listen_ip_address': '127.0.0.1',
            'listen_port': 8080,
            'username': 'TestUser',
            'password': 'testPass'
        }
    })
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
    apiserver._config.update({
        'api_server': {
            'enabled': True,
            'listen_ip_address': '0.0.0.0',
            'listen_port': 8089,
            'password': ''
        }
    })
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
    assert server_inst_mock.run_in_thread.call_count == 0
    assert server_inst_mock.run.call_count == 1
    apiserver1 = ApiServer(default_conf)
    assert id(apiserver1) == id(apiserver)
    apiserver._standalone = False
    caplog.clear()
    mocker.patch('freqtrade.rpc.api_server.webserver.UvicornServer', return_value=MagicMock(side_effect=Exception))
    apiserver.start_api()
    assert log_has('Api server failed to start.', caplog)
    apiserver.cleanup()
    ApiServer.shutdown()


def test_api_cleanup(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    default_conf.update({
        'api_server': {
            'enabled': True,
            'listen_ip_address': '127.0.0.1',
            'listen_port': 8080,
            'username': 'TestUser',
            'password': 'testPass'
        }
    })
    mocker.patch('freqtrade.rpc.telegram.Telegram._init')
    server_mock = MagicMock()
    server_mock.cleanup = MagicMock()
    mocker.patch('freqtrade.rpc.api_server.webserver.UvicornServer', server_mock)
    apiserver = ApiServer(default_conf)
    apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))
    apiserver.cleanup()
    assert apiserver._server.cleanup.call_count == 1
    assert log_has('Stopping API Server', caplog)
    ApiServer.shutdown()


def test_api_reloadconf(botclient: Tuple[Any, TestClient]) -> None:
    ftbot, client = botclient
    rc = client_post(client, f'{BASE_URI}/reload_config')
    assert_response(rc)
    assert rc.json() == {'status': 'Reloading config ...'}
    assert ftbot.state == State.RELOAD_CONFIG


def test_api_stopentry(botclient: Tuple[Any, TestClient]) -> None:
    ftbot, client = botclient
    assert ftbot.config['max_open_trades'] != 0
    rc = client_post(client, f'{BASE_URI}/stopbuy')
    assert_response(rc)
    assert rc.json() == {'status': 'No more entries will occur from now. Run /reload_config to reset.'}
    assert ftbot.config['max_open_trades'] == 0
    rc = client_post(client, f'{BASE_URI}/stopentry')
    assert_response(rc)
    assert rc.json() == {'status': 'No more entries will occur from now. Run /reload_config to reset.'}
    assert ftbot.config['max_open_trades'] == 0


def test_api_balance(botclient: Tuple[Any, TestClient], mocker: Any, rpc_balance: Any, tickers: Any) -> None:
    ftbot, client = botclient
    ftbot.config['dry_run'] = False
    mocker.patch(f'{EXMS}.get_balances', return_value=rpc_balance)
    mocker.patch(f'{EXMS}.get_tickers', tickers)
    mocker.patch(f'{EXMS}.get_valid_pair_combination', side_effect=lambda a, b: [f'{a}/{b}'])
    ftbot.wallets.update()
    rc = client_get(client, f'{BASE_URI}/balance')
    assert_response(rc)
    response: Dict[str, Any] = rc.json()
    assert 'currencies' in response
    assert len(response['currencies']) == 5
    assert response['currencies'][0] == {'currency': 'BTC', 'free': 12.0, 'balance': 12.0, 'used': 0.0, 'bot_owned': pytest.approx(11.879999), 'est_stake': 12.0, 'est_stake_bot': pytest.approx(11.879999), 'stake': 'BTC', 'is_position': False, 'position': 0.0, 'side': 'long', 'is_bot_managed': True}
    assert response['total'] == 12.159513094
    assert response['total_bot'] == pytest.approx(11.879999)
    assert 'starting_capital' in response
    assert 'starting_capital_fiat' in response
    assert 'starting_capital_pct' in response
    assert 'starting_capital_ratio' in response


@pytest.mark.parametrize('is_short', [True, False])
def test_api_count(botclient: Tuple[Any, TestClient], mocker: Any, ticker: Any, fee: Any, markets: Any, is_short: bool) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value=ticker), fetch_ticker=ticker, get_fee=fee, markets=PropertyMock(return_value=markets))
    rc = client_get(client, f'{BASE_URI}/count')
    assert_response(rc)
    assert rc.json()['current'] == 0
    assert rc.json()['max'] == 1
    create_mock_trades(fee, is_short=is_short)
    rc = client_get(client, f'{BASE_URI}/count')
    assert_response(rc)
    assert rc.json()['current'] == 4
    assert rc.json()['max'] == 1
    ftbot.config['max_open_trades'] = float('inf')
    rc = client_get(client, f'{BASE_URI}/count')
    assert rc.json()['max'] == -1


def test_api_locks(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/locks')
    assert_response(rc)
    assert 'locks' in rc.json()
    assert rc.json()['lock_count'] == 0
    assert rc.json()['lock_count'] == len(rc.json()['locks'])
    rc = client_post(client, f'{BASE_URI}/locks', [
        {
            'pair': 'ETH/BTC',
            'until': f'{format_date(datetime.now(timezone.utc) + timedelta(minutes=4))}Z',
            'reason': 'randreason'
        },
        {
            'pair': 'XRP/BTC',
            'until': f'{format_date(datetime.now(timezone.utc) + timedelta(minutes=20))}Z',
            'reason': 'deadbeef'
        }
    ])
    assert_response(rc)
    assert rc.json()['lock_count'] == 2
    rc = client_get(client, f'{BASE_URI}/locks')
    assert_response(rc)
    assert rc.json()['lock_count'] == 2
    assert rc.json()['lock_count'] == len(rc.json()['locks'])
    assert 'ETH/BTC' in (rc.json()['locks'][0]['pair'], rc.json()['locks'][1]['pair'])
    assert 'randreason' in (rc.json()['locks'][0]['reason'], rc.json()['locks'][1]['reason'])
    assert 'deadbeef' in (rc.json()['locks'][0]['reason'], rc.json()['locks'][1]['reason'])
    rc = client_delete(client, f'{BASE_URI}/locks/1')
    assert_response(rc)
    assert rc.json()['lock_count'] == 1
    rc = client_post(client, f'{BASE_URI}/locks/delete', data={'pair': 'XRP/BTC'})
    assert_response(rc)
    assert rc.json()['lock_count'] == 0


def test_api_show_config(botclient: Tuple[Any, TestClient]) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    rc = client_get(client, f'{BASE_URI}/show_config')
    assert_response(rc)
    response: Dict[str, Any] = rc.json()
    assert 'dry_run' in response
    assert response['exchange'] == 'binance'
    assert response['timeframe'] == '5m'
    assert response['timeframe_ms'] == 300000
    assert response['timeframe_min'] == 5
    assert response['state'] == 'running'
    assert response['bot_name'] == 'freqtrade'
    assert response['trading_mode'] == 'spot'
    assert response['strategy_version'] is None
    assert not response['trailing_stop']
    assert 'entry_pricing' in response
    assert 'exit_pricing' in response
    assert 'unfilledtimeout' in response
    assert 'version' in response
    assert 'api_version' in response
    assert 2.1 <= response['api_version'] < 3.0


def test_api_daily(botclient: Tuple[Any, TestClient], mocker: Any, ticker: Any, fee: Any, markets: Any) -> None:
    ftbot, client = botclient
    ftbot.config['dry_run'] = False
    mocker.patch(f'{EXMS}.get_balances', return_value=ticker)
    mocker.patch(f'{EXMS}.get_tickers', ticker)
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.markets', return_value=markets)
    ftbot.wallets.update()
    rc = client_get(client, f'{BASE_URI}/daily')
    assert_response(rc)
    response: Dict[str, Any] = rc.json()
    assert 'data' in response
    assert len(response['data']) == 7
    assert response['stake_currency'] == 'BTC'
    assert response['fiat_display_currency'] == 'USD'
    assert response['data'][0]['date'] == str(datetime.now(timezone.utc).date())


def test_api_weekly(botclient: Tuple[Any, TestClient], mocker: Any, ticker: Any, fee: Any, markets: Any, time_machine: Any) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value=ticker), fetch_ticker=ticker, get_fee=fee, markets=PropertyMock(return_value=markets))
    time_machine.move_to('2023-03-31 21:45:05 +00:00')
    rc = client_get(client, f'{BASE_URI}/weekly')
    assert_response(rc)
    assert len(rc.json()['data']) == 4
    assert rc.json()['stake_currency'] == 'BTC'
    assert rc.json()['fiat_display_currency'] == 'USD'
    assert rc.json()['data'][0]['date'] == '2023-03-27'
    assert rc.json()['data'][1]['date'] == '2023-03-20'


def test_api_monthly(botclient: Tuple[Any, TestClient], mocker: Any, ticker: Any, fee: Any, markets: Any, time_machine: Any) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value=ticker), fetch_ticker=ticker, get_fee=fee, markets=PropertyMock(return_value=markets))
    time_machine.move_to('2023-03-31 21:45:05 +00:00')
    rc = client_get(client, f'{BASE_URI}/monthly')
    assert_response(rc)
    assert len(rc.json()['data']) == 3
    assert rc.json()['stake_currency'] == 'BTC'
    assert rc.json()['fiat_display_currency'] == 'USD'
    assert rc.json()['data'][0]['date'] == '2023-03-01'
    assert rc.json()['data'][1]['date'] == '2023-02-01'


@pytest.mark.parametrize('is_short', [True, False])
def test_api_trades(botclient: Tuple[Any, TestClient], mocker: Any, fee: Any, markets: Any, is_short: bool) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=markets))
    rc = client_get(client, f'{BASE_URI}/trades')
    assert_response(rc)
    assert len(rc.json()) == 4
    assert rc.json()['trades_count'] == 0
    assert rc.json()['total_trades'] == 0
    assert rc.json()['offset'] == 0
    create_mock_trades(fee, is_short=is_short)
    Trade.session.flush()
    rc = client_get(client, f'{BASE_URI}/trades')
    assert_response(rc)
    assert len(rc.json()['trades']) == 2
    assert rc.json()['trades_count'] == 2
    assert rc.json()['total_trades'] == 2
    assert rc.json()['trades'][0]['is_short'] == is_short
    rc = client_get(client, f'{BASE_URI}/trades?limit=1')
    assert_response(rc)
    assert len(rc.json()['trades']) == 1
    assert rc.json()['trades_count'] == 1
    assert rc.json()['total_trades'] == 2


@pytest.mark.parametrize('is_short', [True, False])
def test_api_trade_single(botclient: Tuple[Any, TestClient], mocker: Any, fee: Any, ticker: Any, markets: Any, is_short: bool) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=markets), fetch_ticker=ticker)
    rc = client_get(client, f'{BASE_URI}/trade/3')
    assert_response(rc, 404)
    assert rc.json()['detail'] == 'Trade not found.'
    Trade.rollback()
    create_mock_trades(fee, is_short=is_short)
    rc = client_get(client, f'{BASE_URI}/trade/3')
    assert_response(rc)
    assert rc.json()['trade_id'] == 3
    assert rc.json()['is_short'] == is_short


@pytest.mark.parametrize('is_short', [True, False])
def test_api_delete_trade(botclient: Tuple[Any, TestClient], mocker: Any, fee: Any, markets: Any, is_short: bool) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=markets), cancel_order=cancel_mock, cancel_stoploss_order=stoploss_mock)
    create_mock_trades(fee, is_short=is_short)
    ftbot.strategy.order_types['stoploss_on_exchange'] = True
    trades = list(Trade.session.scalars(select(Trade)))
    Trade.commit()
    assert len(trades) > 2
    rc = client_delete(client, f'{BASE_URI}/trades/1')
    assert_response(rc)
    assert rc.json()['result_msg'] == 'Deleted trade 1. Closed 1 open orders.'
    assert len(trades) - 1 == len(list(Trade.session.scalars(select(Trade))))
    assert cancel_mock.call_count == 1
    cancel_mock.reset_mock()
    rc = client_delete(client, f'{BASE_URI}/trades/1')
    assert_response(rc, 502)
    assert cancel_mock.call_count == 0
    assert len(trades) - 1 == len(list(Trade.session.scalars(select(Trade))))
    rc = client_delete(client, f'{BASE_URI}/trades/5')
    assert_response(rc)
    assert rc.json()['result_msg'] == 'Deleted trade 5. Closed 1 open orders.'
    assert len(trades) - 2 == len(list(Trade.session.scalars(select(Trade))))
    assert stoploss_mock.call_count == 1
    rc = client_delete(client, f'{BASE_URI}/trades/502')
    assert_response(rc, 502)


@pytest.mark.parametrize('is_short', [True, False])
def test_api_delete_open_order(botclient: Tuple[Any, TestClient], mocker: Any, fee: Any, markets: Any, ticker: Any, is_short: bool) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=markets), fetch_ticker=ticker, cancel_order=cancel_mock, cancel_stoploss_order=stoploss_mock)
    rc = client_delete(client, f'{BASE_URI}/trades/10/open-order')
    assert_response(rc, 502)
    assert 'Invalid trade_id.' in rc.json()['error']
    create_mock_trades(fee, is_short=is_short)
    Trade.commit()
    rc = client_delete(client, f'{BASE_URI}/trades/5/open-order')
    assert_response(rc, 502)
    assert 'No open order for trade_id' in rc.json()['error']
    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f'{EXMS}.fetch_order', side_effect=ExchangeError)
    rc = client_delete(client, f'{BASE_URI}/trades/6/open-order')
    assert_response(rc, 502)
    assert 'Order not found.' in rc.json()['error']
    trade = Trade.get_trades([Trade.id == 6]).first()
    mocker.patch(f'{EXMS}.fetch_order', return_value=trade.orders[-1].to_ccxt_object())
    rc = client_delete(client, f'{BASE_URI}/trades/6/open-order')
    assert_response(rc)
    assert cancel_mock.call_count == 1


@pytest.mark.parametrize('is_short', [True, False])
def test_api_trade_reload_trade(botclient: Tuple[Any, TestClient], mocker: Any, fee: Any, markets: Any, ticker: Any, is_short: bool) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    ftbot.handle_onexchange_order = MagicMock()
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=markets), fetch_ticker=ticker, cancel_order=cancel_mock, cancel_stoploss_order=stoploss_mock)
    rc = client_post(client, f'{BASE_URI}/trades/10/reload')
    assert_response(rc, 502)
    assert 'Could not find trade with id 10.' in rc.json()['error']
    assert ftbot.handle_onexchange_order.call_count == 0
    create_mock_trades(fee, is_short=is_short)
    Trade.commit()
    rc = client_post(client, f'{BASE_URI}/trades/5/reload')
    assert ftbot.handle_onexchange_order.call_count == 1


def test_api_logs(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/logs')
    assert_response(rc)
    assert len(rc.json()) == 2
    assert 'logs' in rc.json()
    assert rc.json()['log_count'] > 1
    assert len(rc.json()['logs']) == rc.json()['log_count']
    assert isinstance(rc.json()['logs'][0], list)
    assert isinstance(rc.json()['logs'][0][0], str)
    assert isinstance(rc.json()['logs'][0][1], float)
    assert isinstance(rc.json()['logs'][0][2], str)
    assert isinstance(rc.json()['logs'][0][3], str)
    assert isinstance(rc.json()['logs'][0][4], str)
    rc1 = client_get(client, f'{BASE_URI}/logs?limit=5')
    assert_response(rc1)
    assert len(rc1.json()) == 2
    assert 'logs' in rc1.json()
    if rc1.json()['log_count'] < 5:
        print(f'rc={rc.json()}')
        print(f'rc1={rc1.json()}')
    assert rc1.json()['log_count'] > 2
    assert len(rc1.json()['logs']) == rc1.json()['log_count']


def test_api_edge_disabled(botclient: Tuple[Any, TestClient], mocker: Any, ticker: Any, fee: Any, markets: Any) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value=ticker), fetch_ticker=ticker, get_fee=fee, markets=PropertyMock(return_value=markets))
    rc = client_get(client, f'{BASE_URI}/edge')
    assert_response(rc, 502)
    assert rc.json() == {'error': 'Error querying /api/v1/edge: Edge is not enabled.'}


@pytest.mark.parametrize('is_short,expected', [
    (True, {'best_pair': 'XRP/BTC', 'best_rate': -0.02, 'best_pair_profit_ratio': -0.00018780487, 'best_pair_profit_abs': -0.001155, 'profit_all_coin': 15.382312, 'profit_all_fiat': 189894.6470718, 'profit_all_percent_mean': 49.62, 'profit_all_ratio_mean': 0.49620917, 'profit_all_percent_sum': 198.48, 'profit_all_ratio_sum': 1.98483671, 'profit_all_percent': 1.54, 'profit_all_ratio': 0.01538214, 'profit_closed_coin': -0.00673913, 'profit_closed_fiat': -83.19455985, 'profit_closed_ratio_mean': -0.0075, 'profit_closed_percent_mean': -0.75, 'profit_closed_ratio_sum': -0.015, 'profit_closed_percent_sum': -1.5, 'profit_closed_ratio': -6.739057628404269e-06, 'profit_closed_percent': -0.0, 'winning_trades': 0, 'losing_trades': 2, 'profit_factor': 0.0, 'winrate': 0.0, 'expectancy': -0.0033695635, 'expectancy_ratio': -1.0, 'trading_volume': 75.945}),
    (False, {'best_pair': 'ETC/BTC', 'best_rate': 0.0, 'best_pair_profit_ratio': 3.860975e-05, 'best_pair_profit_abs': 0.000584127, 'profit_all_coin': -15.46546305, 'profit_all_fiat': -190921.14135225, 'profit_all_percent_mean': -49.62, 'profit_all_ratio_mean': -0.49620955, 'profit_all_percent_sum': -198.48, 'profit_all_ratio_sum': -1.9848382, 'profit_all_percent': -1.55, 'profit_all_ratio': -0.0154654126, 'profit_closed_coin': 0.00073913, 'profit_closed_fiat': 9.124559849999999, 'profit_closed_ratio_mean': 0.0075, 'profit_closed_percent_mean': 0.75, 'profit_closed_ratio_sum': 0.015, 'profit_closed_percent_sum': 1.5, 'profit_closed_ratio': 7.391275897987988e-07, 'profit_closed_percent': 0.0, 'winning_trades': 2, 'losing_trades': 0, 'profit_factor': None, 'winrate': 1.0, 'expectancy': 0.0003695635, 'expectancy_ratio': 100, 'trading_volume': 75.945}),
    (None, {'best_pair': 'XRP/BTC', 'best_rate': 0.0, 'best_pair_profit_ratio': 2.5203252e-05, 'best_pair_profit_abs': 0.000155, 'profit_all_coin': -14.87167525, 'profit_all_fiat': -183590.83096125, 'profit_all_percent_mean': 0.13, 'profit_all_ratio_mean': 0.0012538324, 'profit_all_percent_sum': 0.5, 'profit_all_ratio_sum': 0.005015329, 'profit_all_percent': -1.49, 'profit_all_ratio': -0.014871535, 'profit_closed_coin': -0.00542913, 'profit_closed_fiat': -67.02260985, 'profit_closed_ratio_mean': 0.0025, 'profit_closed_percent_mean': 0.25, 'profit_closed_ratio_sum': 0.005, 'profit_closed_percent_sum': 0.5, 'profit_closed_ratio': -5.429078808526421e-06, 'profit_closed_percent': -0.0, 'winning_trades': 1, 'losing_trades': 1, 'profit_factor': 0.02775724835771106, 'winrate': 0.5, 'expectancy': -0.0027145635000000003, 'expectancy_ratio': -0.48612137582114445, 'trading_volume': 75.945})
])
def test_api_profit(botclient: Tuple[Any, TestClient], mocker: Any, ticker: Any, fee: Any, markets: Any, is_short: Optional[bool], expected: Dict[str, Any]) -> None:
    ftbot, client = botclient
    ftbot.config['tradable_balance_ratio'] = 1
    patch_get_signal(ftbot)
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value=ticker), fetch_ticker=ticker, get_fee=fee, markets=PropertyMock(return_value=markets))
    rc = client_get(client, f'{BASE_URI}/profit')
    assert_response(rc, 200)
    assert rc.json()['trade_count'] == 0
    create_mock_trades(fee, is_short=is_short)
    rc = client_get(client, f'{BASE_URI}/profit')
    assert_response(rc)
    assert rc.json() == {
        'avg_duration': Any,
        'best_pair': expected['best_pair'],
        'best_pair_profit_ratio': pytest.approx(expected['best_pair_profit_ratio']),
        'best_pair_profit_abs': expected['best_pair_profit_abs'],
        'best_rate': expected['best_rate'],
        'first_trade_date': Any,
        'first_trade_humanized': Any,
        'first_trade_timestamp': Any,
        'latest_trade_date': Any,
        'latest_trade_humanized': '5 minutes ago',
        'latest_trade_timestamp': Any,
        'profit_all_coin': pytest.approx(expected['profit_all_coin']),
        'profit_all_fiat': pytest.approx(expected['profit_all_fiat']),
        'profit_all_percent_mean': pytest.approx(expected['profit_all_percent_mean']),
        'profit_all_ratio_mean': pytest.approx(expected['profit_all_ratio_mean']),
        'profit_all_percent_sum': pytest.approx(expected['profit_all_percent_sum']),
        'profit_all_ratio_sum': pytest.approx(expected['profit_all_ratio_sum']),
        'profit_all_percent': pytest.approx(expected['profit_all_percent']),
        'profit_all_ratio': pytest.approx(expected['profit_all_ratio']),
        'profit_closed_coin': pytest.approx(expected['profit_closed_coin']),
        'profit_closed_fiat': pytest.approx(expected['profit_closed_fiat']),
        'profit_closed_ratio_mean': pytest.approx(expected['profit_closed_ratio_mean']),
        'profit_closed_percent_mean': pytest.approx(expected['profit_closed_percent_mean']),
        'profit_closed_ratio_sum': pytest.approx(expected['profit_closed_ratio_sum']),
        'profit_closed_percent_sum': pytest.approx(expected['profit_closed_percent_sum']),
        'profit_closed_ratio': pytest.approx(expected['profit_closed_ratio']),
        'profit_closed_percent': pytest.approx(expected['profit_closed_percent']),
        'trade_count': 6,
        'closed_trade_count': 2,
        'winning_trades': expected['winning_trades'],
        'losing_trades': expected['losing_trades'],
        'profit_factor': expected['profit_factor'],
        'winrate': expected['winrate'],
        'expectancy': expected['expectancy'],
        'expectancy_ratio': expected['expectancy_ratio'],
        'max_drawdown': Any,
        'max_drawdown_abs': Any,
        'max_drawdown_start': Any,
        'max_drawdown_start_timestamp': Any,
        'max_drawdown_end': Any,
        'max_drawdown_end_timestamp': Any,
        'trading_volume': expected['trading_volume'],
        'bot_start_timestamp': 0,
        'bot_start_date': ''
    }


@pytest.mark.parametrize('is_short', [True, False])
def test_api_stats(botclient: Tuple[Any, TestClient], mocker: Any, ticker: Any, fee: Any, markets: Any, is_short: bool) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot, enter_long=not is_short, enter_short=is_short)
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value=ticker), fetch_ticker=ticker, get_fee=fee, markets=PropertyMock(return_value=markets))
    rc = client_get(client, f'{BASE_URI}/stats')
    assert_response(rc, 200)
    assert 'durations' in rc.json()
    assert 'exit_reasons' in rc.json()
    create_mock_trades(fee, is_short=is_short)
    rc = client_get(client, f'{BASE_URI}/stats')
    assert_response(rc, 200)
    assert 'durations' in rc.json()
    assert 'exit_reasons' in rc.json()
    assert 'wins' in rc.json()['durations']
    assert 'losses' in rc.json()['durations']
    assert 'draws' in rc.json()['durations']


def test_api_performance(botclient: Tuple[Any, TestClient], fee: Any) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    create_mock_trades_usdt(fee)
    rc = client_get(client, f'{BASE_URI}/performance')
    assert_response(rc)
    assert len(rc.json()) == 3
    assert rc.json() == [
        {
            'count': 1,
            'pair': 'NEO/USDT',
            'profit': 1.99,
            'profit_pct': 1.99,
            'profit_ratio': 0.0199375,
            'profit_abs': 3.9875
        },
        {
            'count': 1,
            'pair': 'XRP/USDT',
            'profit': 9.47,
            'profit_abs': 2.8425,
            'profit_pct': 9.47,
            'profit_ratio': pytest.approx(0.094749999)
        },
        {
            'count': 1,
            'pair': 'LTC/USDT',
            'profit': -20.45,
            'profit_abs': -4.09,
            'profit_pct': -20.45,
            'profit_ratio': -0.2045
        }
    ]


def test_api_entries(botclient: Tuple[Any, TestClient], fee: Any) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    rc = client_get(client, f'{BASE_URI}/entries')
    assert_response(rc)
    assert len(rc.json()) == 0
    create_mock_trades(fee)
    rc = client_get(client, f'{BASE_URI}/entries')
    assert_response(rc)
    response = rc.json()
    assert len(response) == 2
    resp = response[0]
    assert resp['enter_tag'] == 'TEST1'
    assert resp['count'] == 1
    assert resp['profit_pct'] == 0.0
    assert pytest.approx(resp['profit_ratio']) == 3.8609756e-05


def test_api_exits(botclient: Tuple[Any, TestClient], fee: Any) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    rc = client_get(client, f'{BASE_URI}/exits')
    assert_response(rc)
    assert len(rc.json()) == 0
    create_mock_trades(fee)
    rc = client_get(client, f'{BASE_URI}/exits')
    assert_response(rc)
    response = rc.json()
    assert len(response) == 2
    resp = response[0]
    assert resp['exit_reason'] == 'sell_signal'
    assert resp['count'] == 1
    assert resp['profit_pct'] == 0.0
    assert pytest.approx(resp['profit_ratio']) == 3.8609756e-05


def test_api_mix_tag(botclient: Tuple[Any, TestClient], fee: Any) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    rc = client_get(client, f'{BASE_URI}/mix_tags')
    assert_response(rc)
    assert len(rc.json()) == 0
    create_mock_trades(fee)
    rc = client_get(client, f'{BASE_URI}/mix_tags')
    assert_response(rc)
    response = rc.json()
    assert len(response) == 2
    resp = response[0]
    assert resp['mix_tag'] == 'TEST1 sell_signal'
    assert resp['count'] == 1
    assert resp['profit_pct'] == 0.5


@pytest.mark.parametrize('is_short,current_rate,open_trade_value', [
    (True, 1.098e-05, 6.134625),
    (False, 1.099e-05, 6.165375)
])
def test_api_status(botclient: Tuple[Any, TestClient], mocker: Any, ticker: Any, fee: Any, markets: Any, is_short: bool, current_rate: float, open_trade_value: float) -> None:
    ftbot, client = botclient
    patch_get_signal(ftbot)
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value=ticker), fetch_ticker=ticker, get_fee=fee, markets=PropertyMock(return_value=markets), fetch_order=MagicMock(return_value={}))
    rc = client_get(client, f'{BASE_URI}/status')
    assert_response(rc, 200)
    assert rc.json() == []
    create_mock_trades(fee, is_short=is_short)
    rc = client_get(client, f'{BASE_URI}/status')
    assert_response(rc)
    assert len(rc.json()) == 4
    assert rc.json()[0] == {
        'amount': 50.0,
        'amount_requested': 123.0,
        'close_date': None,
        'close_timestamp': None,
        'close_profit': None,
        'close_profit_pct': None,
        'close_profit_abs': None,
        'close_rate': None,
        'profit_ratio': Any,
        'profit_pct': Any,
        'profit_abs': Any,
        'profit_fiat': Any,
        'total_profit_abs': Any,
        'total_profit_fiat': Any,
        'total_profit_ratio': Any,
        'realized_profit': 0.0,
        'realized_profit_ratio': None,
        'current_rate': current_rate,
        'open_date': Any,
        'open_timestamp': Any,
        'open_fill_date': Any,
        'open_fill_timestamp': Any,
        'open_rate': 0.123,
        'pair': 'ETH/BTC',
        'base_currency': 'ETH',
        'quote_currency': 'BTC',
        'stake_amount': 0.001,
        'max_stake_amount': Any,
        'stop_loss_abs': Any,
        'stop_loss_pct': Any,
        'stop_loss_ratio': Any,
        'stoploss_last_update': Any,
        'stoploss_last_update_timestamp': Any,
        'initial_stop_loss_abs': 0.0,
        'initial_stop_loss_pct': Any,
        'initial_stop_loss_ratio': Any,
        'stoploss_current_dist': Any,
        'stoploss_current_dist_ratio': Any,
        'stoploss_current_dist_pct': Any,
        'stoploss_entry_dist': Any,
        'stoploss_entry_dist_ratio': Any,
        'trade_id': 1,
        'close_rate_requested': Any,
        'fee_close': 0.0025,
        'fee_close_cost': None,
        'fee_close_currency': None,
        'fee_open': 0.0025,
        'fee_open_cost': None,
        'fee_open_currency': None,
        'is_open': True,
        'is_short': is_short,
        'max_rate': Any,
        'min_rate': Any,
        'open_rate_requested': Any,
        'open_trade_value': open_trade_value,
        'exit_reason': None,
        'exit_order_status': None,
        'strategy': CURRENT_TEST_STRATEGY,
        'enter_tag': None,
        'timeframe': 5,
        'exchange': 'binance',
        'leverage': 1.0,
        'interest_rate': 0.0,
        'liquidation_price': None,
        'funding_fees': None,
        'trading_mode': Any,
        'amount_precision': None,
        'price_precision': None,
        'precision_mode': None,
        'has_open_orders': True,
        'orders': [Any]
    }
    mocker.patch(f'{EXMS}.get_rate', MagicMock(side_effect=ExchangeError("Pair 'ETH/BTC' not available")))
    rc = client_get(client, f'{BASE_URI}/status')
    assert_response(rc)
    resp_values = rc.json()
    assert len(resp_values) == 4
    assert resp_values[0]['profit_abs'] == 0.0


def test_api_version(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/version')
    assert_response(rc)
    assert rc.json() == {'version': __version__}


def test_api_blacklist(botclient: Tuple[Any, TestClient], mocker: Any) -> None:
    _ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/blacklist')
    assert_response(rc)
    assert rc.json() == {'blacklist': ['DOGE/BTC', 'HOT/BTC'], 'blacklist_expanded': [], 'length': 2, 'method': ['StaticPairList'], 'errors': {}}
    rc = client_post(client, f'{BASE_URI}/blacklist', data={'blacklist': ['ETH/BTC']})
    assert_response(rc)
    assert rc.json() == {'blacklist': ['DOGE/BTC', 'HOT/BTC', 'ETH/BTC'], 'blacklist_expanded': ['ETH/BTC'], 'length': 3, 'method': ['StaticPairList'], 'errors': {}}
    rc = client_post(client, f'{BASE_URI}/blacklist', data={'blacklist': ['XRP/.*']})
    assert_response(rc)
    assert rc.json() == {'blacklist': ['DOGE/BTC', 'HOT/BTC', 'ETH/BTC', 'XRP/.*'], 'blacklist_expanded': ['ETH/BTC', 'XRP/BTC', 'XRP/USDT'], 'length': 4, 'method': ['StaticPairList'], 'errors': {}}
    rc = client_delete(client, f'{BASE_URI}/blacklist?pairs_to_delete=DOGE/BTC')
    assert_response(rc)
    assert rc.json() == {'blacklist': ['HOT/BTC', 'ETH/BTC', 'XRP/.*'], 'blacklist_expanded': ['ETH/BTC', 'XRP/BTC', 'XRP/USDT'], 'length': 3, 'method': ['StaticPairList'], 'errors': {}}
    rc = client_delete(client, f'{BASE_URI}/blacklist?pairs_to_delete=NOTHING/BTC')
    assert_response(rc)
    assert rc.json() == {'blacklist': ['HOT/BTC', 'ETH/BTC', 'XRP/.*'], 'blacklist_expanded': ['ETH/BTC', 'XRP/BTC', 'XRP/USDT'], 'length': 3, 'method': ['StaticPairList'], 'errors': {'NOTHING/BTC': {'error_msg': 'Pair NOTHING/BTC is not in the current blacklist.'}}}
    rc = client_delete(client, f'{BASE_URI}/blacklist?pairs_to_delete=HOT/BTC&pairs_to_delete=ETH/BTC')
    assert_response(rc)
    assert rc.json() == {'blacklist': ['XRP/.*'], 'blacklist_expanded': ['XRP/BTC', 'XRP/USDT'], 'length': 1, 'method': ['StaticPairList'], 'errors': {}}


def test_api_whitelist(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/whitelist')
    assert_response(rc)
    assert rc.json() == {'whitelist': ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC'], 'length': 4, 'method': ['StaticPairList']}


@pytest.mark.parametrize('endpoint', ['forcebuy', 'forceenter'])
def test_api_force_entry(botclient: Tuple[Any, TestClient], mocker: Any, fee: Any, endpoint: str) -> None:
    ftbot, client = botclient
    rc = client_post(client, f'{BASE_URI}/{endpoint}', data={'pair': 'ETH/BTC'})
    assert_response(rc, 502)
    assert rc.json() == {'error': f'Error querying /api/v1/{endpoint}: Force_entry not enabled.'}
    ftbot.config['force_entry_enable'] = True
    fbuy_mock = MagicMock(return_value=None)
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_entry', fbuy_mock)
    rc = client_post(client, f'{BASE_URI}/{endpoint}', data={'pair': 'ETH/BTC'})
    assert_response(rc)
    assert rc.json() == {'status': 'Error entering long trade for pair ETH/BTC.'}
    fbuy_mock = MagicMock(return_value=Trade(
        pair='ETH/BTC',
        amount=1,
        amount_requested=1,
        exchange='binance',
        stake_amount=1,
        open_rate=0.245441,
        open_date=datetime.now(timezone.utc),
        is_open=False,
        is_short=False,
        fee_close=fee.return_value,
        fee_open=fee.return_value,
        close_rate=0.265441,
        id=22,
        timeframe=5,
        strategy=CURRENT_TEST_STRATEGY,
        trading_mode=TradingMode.SPOT
    ))
    mocker.patch('freqtrade.rpc.rpc.RPC._rpc_force_entry', fbuy_mock)
    rc = client_post(client, f'{BASE_URI}/{endpoint}', data={'pair': 'ETH/BTC'})
    assert_response(rc)
    assert rc.json() == {
        'amount': 1.0,
        'amount_requested': 1.0,
        'trade_id': 22,
        'close_date': None,
        'close_timestamp': None,
        'close_rate': 0.265441,
        'open_date': Any,
        'open_timestamp': Any,
        'open_fill_date': Any,
        'open_fill_timestamp': Any,
        'open_rate': 0.245441,
        'pair': 'ETH/BTC',
        'base_currency': 'ETH',
        'quote_currency': 'BTC',
        'stake_amount': 1,
        'max_stake_amount': Any,
        'stop_loss_abs': None,
        'stop_loss_pct': None,
        'stop_loss_ratio': None,
        'stoploss_last_update': None,
        'stoploss_last_update_timestamp': None,
        'initial_stop_loss_abs': None,
        'initial_stop_loss_pct': None,
        'initial_stop_loss_ratio': None,
        'close_profit': None,
        'close_profit_pct': None,
        'close_profit_abs': None,
        'close_rate_requested': None,
        'profit_ratio': None,
        'profit_pct': None,
        'profit_abs': None,
        'profit_fiat': None,
        'realized_profit': 0.0,
        'realized_profit_ratio': None,
        'fee_close': 0.0025,
        'fee_close_cost': None,
        'fee_close_currency': None,
        'fee_open': 0.0025,
        'fee_open_cost': None,
        'fee_open_currency': None,
        'is_open': False,
        'is_short': False,
        'max_rate': None,
        'min_rate': None,
        'open_rate_requested': None,
        'open_trade_value': 0.2460546,
        'exit_reason': None,
        'exit_order_status': None,
        'strategy': CURRENT_TEST_STRATEGY,
        'enter_tag': None,
        'timeframe': 5,
        'exchange': 'binance',
        'leverage': None,
        'interest_rate': None,
        'liquidation_price': None,
        'funding_fees': None,
        'trading_mode': 'spot',
        'amount_precision': None,
        'price_precision': None,
        'precision_mode': None,
        'has_open_orders': False,
        'orders': []
    }


def test_api_forceexit(botclient: Tuple[Any, TestClient], mocker: Any, ticker: Any, fee: Any, markets: Any) -> None:
    ftbot, client = botclient
    mocker.patch.multiple(EXMS, get_balances=MagicMock(return_value=ticker), fetch_ticker=ticker, get_fee=fee, markets=PropertyMock(return_value=markets), _dry_is_price_crossed=MagicMock(return_value=True))
    patch_get_signal(ftbot)
    rc = client_post(client, f'{BASE_URI}/forceexit', data={'tradeid': '1'})
    assert_response(rc, 502)
    assert rc.json() == {'error': 'Error querying /api/v1/forceexit: invalid argument'}
    Trade.rollback()
    create_mock_trades(fee)
    trade = Trade.get_trades([Trade.id == 5]).first()
    assert pytest.approx(trade.amount) == 123
    rc = client_post(client, f'{BASE_URI}/forceexit', data={'tradeid': '5', 'ordertype': 'market', 'amount': 23})
    assert_response(rc)
    assert rc.json() == {'result': 'Created exit order for trade 5.'}
    Trade.rollback()
    trade = Trade.get_trades([Trade.id == 5]).first()
    assert pytest.approx(trade.amount) == 100
    assert trade.is_open is True
    rc = client_post(client, f'{BASE_URI}/forceexit', data={'tradeid': '5'})
    assert_response(rc)
    assert rc.json() == {'result': 'Created exit order for trade 5.'}
    Trade.rollback()
    trade = Trade.get_trades([Trade.id == 5]).first()
    assert trade.is_open is False


def test_api_pair_candles(botclient: Tuple[Any, TestClient], ohlcv_history: pd.DataFrame) -> None:
    ftbot, client = botclient
    timeframe: str = '5m'
    amount: int = 3
    rc = client_get(client, f'{BASE_URI}/pair_candles?limit={amount}&timeframe={timeframe}')
    assert_response(rc, 422)
    rc = client_get(client, f'{BASE_URI}/pair_candles?pair=XRP%2FBTC')
    assert_response(rc, 422)
    rc = client_get(client, f'{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}')
    assert_response(rc)
    assert 'columns' in rc.json()
    assert 'data_start_ts' in rc.json()
    assert 'data_start' in rc.json()
    assert 'data_stop' in rc.json()
    assert 'data_stop_ts' in rc.json()
    assert len(rc.json()['data']) == 0
    ohlcv_history['sma'] = ohlcv_history['close'].rolling(2).mean()
    ohlcv_history['sma2'] = ohlcv_history['close'].rolling(2).mean()
    ohlcv_history['enter_long'] = 0
    ohlcv_history.loc[1, 'enter_long'] = 1
    ohlcv_history['exit_long'] = 0
    ohlcv_history['enter_short'] = 0
    ohlcv_history['exit_short'] = 0
    ftbot.dataprovider._set_cached_df('XRP/BTC', timeframe, ohlcv_history, CandleType.SPOT)
    for call in ('get', 'post'):
        if call == 'get':
            rc = client_get(client, f'{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}')
        else:
            rc = client_post(client, f'{BASE_URI}/pair_candles', data={'pair': 'XRP/BTC', 'timeframe': timeframe, 'limit': amount, 'columns': ['sma']})
        assert_response(rc)
        resp = rc.json()
        assert 'strategy' in resp
        assert resp['strategy'] == CURRENT_TEST_STRATEGY
        assert 'columns' in resp
        assert 'data_start_ts' in resp
        assert 'data_start' in resp
        assert 'data_stop' in resp
        assert 'data_stop_ts' in resp
        assert resp['data_start'] == '2017-11-26 08:50:00+00:00'
        assert resp['data_start_ts'] == 1511686200000
        assert resp['data_stop'] == '2017-11-26 09:00:00+00:00'
        assert resp['data_stop_ts'] == 1511686800000
        assert isinstance(resp['columns'], list)
        base_cols = {'date', 'open', 'high', 'low', 'close', 'volume', 'sma', 'enter_long', 'exit_long', 'enter_short', 'exit_short', '__date_ts', '_enter_long_signal_close', '_exit_long_signal_close', '_enter_short_signal_close', '_exit_short_signal_close'}
        if call == 'get':
            assert set(resp['columns']) == base_cols.union({'sma2'})
        else:
            assert set(resp['columns']) == base_cols
        assert set(resp['all_columns']) == {'date', 'open', 'high', 'low', 'close', 'volume', 'sma', 'sma2', 'enter_long', 'exit_long', 'enter_short', 'exit_short'}
        assert 'pair' in resp
        assert resp['pair'] == 'XRP/BTC'
        assert 'data' in resp
        assert len(resp['data']) == amount
        if call == 'get':
            assert len(resp['data'][0]) == 17
            assert resp['data'] == [
                ['2017-11-26T08:50:00Z', 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 0.0877869, None, None, 0, 0, 0, 0, 1511686200000, None, None, None, None],
                ['2017-11-26T08:55:00Z', 8.88e-05, 8.942e-05, 8.88e-05, 8.893e-05, 0.05874751, 8.886500000000001e-05, 8.886500000000001e-05, 1, 0, 0, 0, 1511686500000, 8.893e-05, None, None, None],
                ['2017-11-26T09:00:00Z', 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05, 0.7039405, 8.885e-05, 8.885e-05, 0, 0, 0, 0, 1511686800000, None, None, None, None]
            ]
        else:
            assert len(resp['data'][0]) == 16
            assert resp['data'] == [
                ['2017-11-26T08:50:00Z', 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 0.0877869, None, 0, 0, 0, 0, 1511686200000, None, None, None, None],
                ['2017-11-26T08:55:00Z', 8.88e-05, 8.942e-05, 8.88e-05, 8.893e-05, 0.05874751, 8.886500000000001e-05, 1, 0, 0, 0, 1511686500000, 8.893e-05, None, None, None],
                ['2017-11-26T09:00:00Z', 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05, 0.7039405, 8.885e-05, 0, 0, 0, 0, 1511686800000, None, None, None, None]
            ]
    ohlcv_history['exit_long'] = ohlcv_history['exit_long'].astype('float64')
    ohlcv_history.at[0, 'exit_long'] = float('inf')
    ohlcv_history['date1'] = ohlcv_history['date']
    ohlcv_history.at[0, 'date1'] = pd.NaT
    ftbot.dataprovider._set_cached_df('XRP/BTC', timeframe, ohlcv_history, CandleType.SPOT)
    rc = client_get(client, f'{BASE_URI}/pair_candles?limit={amount}&pair=XRP%2FBTC&timeframe={timeframe}')
    assert_response(rc)
    assert rc.json()['data'] == [
        ['2017-11-26T08:50:00Z', 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 0.0877869, None, None, 0, None, 0, 0, None, 1511686200000, None, None, None, None],
        ['2017-11-26T08:55:00Z', 8.88e-05, 8.942e-05, 8.88e-05, 8.893e-05, 0.05874751, 8.886500000000001e-05, 8.886500000000001e-05, 1, 0.0, 0, 0, '2017-11-26T08:55:00Z', 1511686500000, 8.893e-05, None, None, None],
        ['2017-11-26T09:00:00Z', 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05, 0.7039405, 8.885e-05, 8.885e-05, 0, 0.0, 0, 0, '2017-11-26T09:00:00Z', 1511686800000, None, None, None, None]
    ]


def test_api_pair_history(botclient: Tuple[Any, TestClient], tmp_path: Path, mocker: Any) -> None:
    _ftbot, client = botclient
    _ftbot.config['user_data_dir'] = tmp_path
    timeframe: str = '5m'
    lfm = mocker.patch('freqtrade.strategy.interface.IStrategy.load_freqAI_model')
    rc = client_get(client, f'{BASE_URI}/pair_history?timeframe={timeframe}&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}')
    assert_response(rc, 503)
    _ftbot.config['runmode'] = RunMode.WEBSERVER
    rc = client_get(client, f'{BASE_URI}/pair_history?timeframe={timeframe}&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}')
    assert_response(rc, 422)
    rc = client_get(client, f'{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}')
    assert_response(rc, 422)
    rc = client_get(client, f'{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}&strategy={CURRENT_TEST_STRATEGY}')
    assert_response(rc, 422)
    rc = client_get(client, f'{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}&timerange=20180111-20180112')
    assert_response(rc, 422)
    rc = client_get(client, f'{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}&timerange=20180111-20180112&strategy={{CURRENT_TEST_STRATEGY}}11')
    assert_response(rc, 502)
    for call in ('get', 'post'):
        if call == 'get':
            rc = client_get(client, f'{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}&timerange=20180111-20180112&strategy={CURRENT_TEST_STRATEGY}')
        else:
            rc = client_post(client, f'{BASE_URI}/pair_history', data={
                'pair': 'UNITTEST/BTC',
                'timeframe': timeframe,
                'timerange': '20180111-20180112',
                'strategy': CURRENT_TEST_STRATEGY,
                'columns': ['rsi', 'fastd', 'fastk']
            })
        assert_response(rc, 200)
        result = rc.json()
        assert result['length'] == 289
        assert len(result['data']) == result['length']
        assert 'columns' in result
        assert 'data' in result
        data = result['data']
        assert len(data) == 289
        col_count: int = 30 if call == 'get' else 18
        assert len(result['columns']) == col_count
        assert len(result['all_columns']) == 25
        assert len(data[0]) == col_count
        date_col_idx: int = [idx for idx, c in enumerate(result['columns']) if c == 'date'][0]
        rsi_col_idx: int = [idx for idx, c in enumerate(result['columns']) if c == 'rsi'][0]
        assert data[0][date_col_idx] == '2018-01-11T00:00:00Z'
        assert data[0][rsi_col_idx] is not None
        assert data[0][rsi_col_idx] > 0
        assert lfm.call_count == 1
        assert result['pair'] == 'UNITTEST/BTC'
        assert result['strategy'] == CURRENT_TEST_STRATEGY
        assert result['data_start'] == '2018-01-11 00:00:00+00:00'
        assert result['data_start_ts'] == 1515628800000
        assert result['data_stop'] == '2018-01-12 00:00:00+00:00'
        assert result['data_stop_ts'] == 1515715200000
        lfm.reset_mock()
        if call == 'get':
            rc = client_get(client, f'{BASE_URI}/pair_history?pair=UNITTEST%2FBTC&timeframe={timeframe}&timerange=20200111-20200112&strategy={CURRENT_TEST_STRATEGY}')
        else:
            rc = client_post(client, f'{BASE_URI}/pair_history', data={
                'pair': 'UNITTEST/BTC',
                'timeframe': timeframe,
                'timerange': '20200111-20200112',
                'strategy': CURRENT_TEST_STRATEGY,
                'columns': ['rsi', 'fastd', 'fastk']
            })
        assert_response(rc, 502)
        assert rc.json()['detail'] == 'No data for UNITTEST/BTC, 5m in 20200111-20200112 found.'
    rc = client_post(client, f'{BASE_URI}/pair_history', data={
        'pair': 'UNITTEST/BTC',
        'timeframe': timeframe,
        'timerange': '20180111-20180112',
        'columns': ['rsi', 'fastd', 'fastk']
    })
    assert_response(rc, 200)
    result = rc.json()
    assert result['length'] == 289
    assert len(result['data']) == result['length']
    assert 'columns' in result
    assert 'data' in result
    assert 'enter_long' not in result['columns']
    assert result['columns'] == ['date', 'open', 'high', 'low', 'close', 'volume', '__date_ts']


def test_api_pair_history_live_mode(botclient: Tuple[Any, TestClient], tmp_path: Path, mocker: Any) -> None:
    _ftbot, client = botclient
    _ftbot.config['user_data_dir'] = tmp_path
    _ftbot.config['runmode'] = RunMode.WEBSERVER
    mocker.patch('freqtrade.strategy.interface.IStrategy.load_freqAI_model')
    gho = mocker.patch('freqtrade.exchange.binance.Binance.get_historic_ohlcv', return_value=generate_test_data('1h', 100))
    rc = client_post(client, f'{BASE_URI}/pair_history', data={
        'pair': 'UNITTEST/BTC',
        'timeframe': '1h',
        'timerange': '20240101-',
        'columns': ['rsi', 'fastd', 'fastk'],
        'live_mode': True
    })
    assert_response(rc, 200)
    result = rc.json()
    assert result['length'] == 100
    assert len(result['data']) == result['length']
    assert result['columns'] == ['date', 'open', 'high', 'low', 'close', 'volume', '__date_ts']
    assert gho.call_count == 1
    gho.reset_mock()
    rc = client_post(client, f'{BASE_URI}/pair_history', data={
        'pair': 'UNITTEST/BTC',
        'timeframe': '1h',
        'timerange': '20240101-',
        'strategy': CURRENT_TEST_STRATEGY,
        'columns': ['rsi', 'fastd', 'fastk'],
        'live_mode': True
    })
    assert_response(rc, 200)
    result = rc.json()
    assert result['length'] == 100 - 20
    assert len(result['data']) == result['length']
    assert 'rsi' in result['columns']
    assert 'enter_long' in result['columns']
    assert 'fastd' in result['columns']
    assert 'date' in result['columns']
    assert gho.call_count == 1


def test_api_plot_config(botclient: Tuple[Any, TestClient], mocker: Any, tmp_path: Path) -> None:
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path
    rc = client_get(client, f'{BASE_URI}/plot_config')
    assert_response(rc)
    assert rc.json() == {}
    ftbot.strategy.plot_config = {'main_plot': {'sma': {}}, 'subplots': {'RSI': {'rsi': {'color': 'red'}}}}
    rc = client_get(client, f'{BASE_URI}/plot_config')
    assert_response(rc)
    assert rc.json() == ftbot.strategy.plot_config
    assert isinstance(rc.json()['main_plot'], dict)
    assert isinstance(rc.json()['subplots'], dict)
    ftbot.strategy.plot_config = {'main_plot': {'sma': {}}}
    rc = client_get(client, f'{BASE_URI}/plot_config')
    assert_response(rc)
    assert isinstance(rc.json()['main_plot'], dict)
    assert isinstance(rc.json()['subplots'], dict)
    rc = client_get(client, f'{BASE_URI}/plot_config?strategy=freqai_test_classifier')
    assert_response(rc)
    res = rc.json()
    assert 'target_roi' in res['subplots']
    assert 'do_predict' in res['subplots']
    rc = client_get(client, f'{BASE_URI}/plot_config?strategy=HyperoptableStrategy')
    assert_response(rc)
    assert rc.json()['subplots'] == {}
    rc = client_get(client, f'{BASE_URI}/plot_config?strategy=NotAStrategy')
    assert_response(rc, 502)
    assert rc.json()['detail'] is not None
    mocker.patch('freqtrade.rpc.api_server.api_v1.get_rpc_optional', return_value=None)
    rc = client_get(client, f'{BASE_URI}/plot_config')
    assert_response(rc)


def test_api_strategies(botclient: Tuple[Any, TestClient], tmp_path: Path) -> None:
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path
    rc = client_get(client, f'{BASE_URI}/strategies')
    assert_response(rc)
    assert rc.json() == {'strategies': [
        'HyperoptableStrategy',
        'HyperoptableStrategyV2',
        'InformativeDecoratorTest',
        'StrategyTestV2',
        'StrategyTestV3',
        'StrategyTestV3CustomEntryPrice',
        'StrategyTestV3Futures',
        'freqai_rl_test_strat',
        'freqai_test_classifier',
        'freqai_test_multimodel_classifier_strat',
        'freqai_test_multimodel_strat',
        'freqai_test_strat',
        'strategy_test_v3_recursive_issue'
    ]}


def test_api_strategy(botclient: Tuple[Any, TestClient], tmp_path: Path, mocker: Any) -> None:
    _ftbot, client = botclient
    _ftbot.config['user_data_dir'] = tmp_path
    rc = client_get(client, f'{BASE_URI}/strategy/{CURRENT_TEST_STRATEGY}')
    assert_response(rc)
    assert rc.json()['strategy'] == CURRENT_TEST_STRATEGY
    data = (Path(__file__).parents[1] / 'strategy/strats/strategy_test_v3.py').read_text()
    assert rc.json()['code'] == data
    rc = client_get(client, f'{BASE_URI}/strategy/NoStrat')
    assert_response(rc, 404)
    rc = client_get(client, f'{BASE_URI}/strategy/xx:cHJpbnQoImhlbGxvIHdvcmxkIik=')
    assert_response(rc, 500)
    mocker.patch('freqtrade.resolvers.strategy_resolver.StrategyResolver._load_strategy', side_effect=Exception('Test'))
    rc = client_get(client, f'{BASE_URI}/strategy/NoStrat')
    assert_response(rc, 502)


def test_api_exchanges(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/exchanges')
    assert_response(rc)
    response = rc.json()
    assert isinstance(response['exchanges'], list)
    assert len(response['exchanges']) > 20
    okx = [x for x in response['exchanges'] if x['classname'] == 'okx'][0]
    assert okx == {
        'classname': 'okx',
        'name': 'OKX',
        'valid': True,
        'supported': True,
        'comment': '',
        'dex': False,
        'is_alias': False,
        'alias_for': None,
        'trade_modes': [
            {'trading_mode': 'spot', 'margin_mode': ''},
            {'trading_mode': 'futures', 'margin_mode': 'isolated'}
        ]
    }
    mexc = [x for x in response['exchanges'] if x['classname'] == 'mexc'][0]
    assert mexc == {
        'classname': 'mexc',
        'name': 'MEXC Global',
        'valid': True,
        'supported': False,
        'dex': False,
        'comment': '',
        'is_alias': False,
        'alias_for': None,
        'trade_modes': [{'trading_mode': 'spot', 'margin_mode': ''}]
    }
    waves = [x for x in response['exchanges'] if x['classname'] == 'wavesexchange'][0]
    assert waves == {
        'classname': 'wavesexchange',
        'name': 'Waves.Exchange',
        'valid': True,
        'supported': False,
        'dex': True,
        'comment': Any,
        'is_alias': False,
        'alias_for': None,
        'trade_modes': [{'trading_mode': 'spot', 'margin_mode': ''}]
    }


def test_list_hyperoptloss(botclient: Tuple[Any, TestClient], tmp_path: Path) -> None:
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path
    rc = client_get(client, f'{BASE_URI}/hyperoptloss')
    assert_response(rc)
    response = rc.json()
    assert isinstance(response['loss_functions'], list)
    assert len(response['loss_functions']) > 0
    sharpeloss = [r for r in response['loss_functions'] if r['name'] == 'SharpeHyperOptLoss']
    assert len(sharpeloss) == 1
    assert 'Sharpe Ratio calculation' in sharpeloss[0]['description']
    assert len([r for r in response['loss_functions'] if r['name'] == 'SortinoHyperOptLoss']) == 1


def test_api_freqaimodels(botclient: Tuple[Any, TestClient], tmp_path: Path, mocker: Any) -> None:
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path
    mocker.patch('freqtrade.resolvers.freqaimodel_resolver.FreqaiModelResolver.search_all_objects', return_value=[
        {'name': 'LightGBMClassifier'},
        {'name': 'LightGBMClassifierMultiTarget'},
        {'name': 'LightGBMRegressor'},
        {'name': 'LightGBMRegressorMultiTarget'},
        {'name': 'ReinforcementLearner'},
        {'name': 'ReinforcementLearner_multiproc'},
        {'name': 'SKlearnRandomForestClassifier'},
        {'name': 'XGBoostClassifier'},
        {'name': 'XGBoostRFClassifier'},
        {'name': 'XGBoostRFRegressor'},
        {'name': 'XGBoostRegressor'},
        {'name': 'XGBoostRegressorMultiTarget'}
    ])
    rc = client_get(client, f'{BASE_URI}/freqaimodels')
    assert_response(rc)
    assert rc.json() == {'freqaimodels': [
        'LightGBMClassifier',
        'LightGBMClassifierMultiTarget',
        'LightGBMRegressor',
        'LightGBMRegressorMultiTarget',
        'ReinforcementLearner',
        'ReinforcementLearner_multiproc',
        'SKlearnRandomForestClassifier',
        'XGBoostClassifier',
        'XGBoostRFClassifier',
        'XGBoostRFRegressor',
        'XGBoostRegressor',
        'XGBoostRegressorMultiTarget'
    ]}


def test_api_pairlists_available(botclient: Tuple[Any, TestClient], tmp_path: Path) -> None:
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path
    rc = client_get(client, f'{BASE_URI}/pairlists/available')
    assert_response(rc, 503)
    assert rc.json()['detail'] == 'Bot is not in the correct state.'
    ftbot.config['runmode'] = RunMode.WEBSERVER
    rc = client_get(client, f'{BASE_URI}/pairlists/available')
    assert_response(rc)
    response = rc.json()
    assert isinstance(response['pairlists'], list)
    assert len(response['pairlists']) > 0
    assert len([r for r in response['pairlists'] if r['name'] == 'AgeFilter']) == 1
    assert len([r for r in response['pairlists'] if r['name'] == 'VolumePairList']) == 1
    assert len([r for r in response['pairlists'] if r['name'] == 'StaticPairList']) == 1
    volumepl = [r for r in response['pairlists'] if r['name'] == 'VolumePairList'][0]
    assert volumepl['is_pairlist_generator'] is True
    assert len(volumepl['params']) > 1
    age_pl = [r for r in response['pairlists'] if r['name'] == 'AgeFilter'][0]
    assert age_pl['is_pairlist_generator'] is False
    assert len(volumepl['params']) > 2


def test_api_pairlists_evaluate(botclient: Tuple[Any, TestClient], tmp_path: Path, mocker: Any) -> None:
    ftbot, client = botclient
    ftbot.config['user_data_dir'] = tmp_path
    rc = client_get(client, f'{BASE_URI}/pairlists/evaluate/randomJob')
    assert_response(rc, 503)
    assert rc.json()['detail'] == 'Bot is not in the correct state.'
    ftbot.config['runmode'] = RunMode.WEBSERVER
    rc = client_get(client, f'{BASE_URI}/pairlists/evaluate/randomJob')
    assert_response(rc, 404)
    assert rc.json()['detail'] == 'Job not found.'
    body = {'pairlists': [{'method': 'StaticPairList'}], 'blacklist': [], 'stake_currency': 'BTC'}
    ApiBG.pairlist_running = True
    rc = client_post(client, f'{BASE_URI}/pairlists/evaluate', body)
    assert_response(rc, 400)
    assert rc.json()['detail'] == 'Pairlist evaluation is already running.'
    ApiBG.pairlist_running = False
    rc = client_post(client, f'{BASE_URI}/pairlists/evaluate', body)
    assert_response(rc)
    assert rc.json()['status'] == 'Pairlist evaluation started in background.'
    job_id = rc.json()['job_id']
    rc = client_get(client, f'{BASE_URI}/background/RandomJob')
    assert_response(rc, 404)
    assert rc.json()['detail'] == 'Job not found.'
    rc = client_get(client, f'{BASE_URI}/background')
    assert_response(rc)
    response = rc.json()
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0]['job_id'] == job_id
    rc = client_get(client, f'{BASE_URI}/background/{job_id}')
    assert_response(rc)
    response = rc.json()
    assert response['job_id'] == job_id
    assert response['job_category'] == 'pairlist'
    rc = client_get(client, f'{BASE_URI}/pairlists/evaluate/{job_id}')
    assert_response(rc)
    response = rc.json()
    assert response['result']['whitelist'] == ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC']
    assert response['result']['length'] == 4
    body['pairlists'].append({'method': 'OffsetFilter', 'number_assets': 2})
    rc = client_post(client, f'{BASE_URI}/pairlists/evaluate', body)
    assert_response(rc)
    assert rc.json()['status'] == 'Pairlist evaluation started in background.'
    job_id = rc.json()['job_id']
    rc = client_get(client, f'{BASE_URI}/pairlists/evaluate/{job_id}')
    assert_response(rc)
    response = rc.json()
    assert response['result']['whitelist'] == ['ETH/BTC', 'LTC/BTC']
    assert response['result']['length'] == 2
    plm = mocker.patch('freqtrade.rpc.api_server.api_pairlists.__run_pairlist', return_value=None)
    body = {'pairlists': [{'method': 'StaticPairList'}], 'blacklist': [], 'stake_currency': 'BTC', 'exchange': 'randomExchange', 'trading_mode': 'futures', 'margin_mode': 'isolated'}
    rc = client_post(client, f'{BASE_URI}/pairlists/evaluate', body)
    assert_response(rc)
    assert plm.call_count == 1
    call_config = plm.call_args_list[0][0][1]
    assert call_config['exchange']['name'] == 'randomExchange'
    assert call_config['trading_mode'] == 'futures'
    assert call_config['margin_mode'] == 'isolated'


def test_list_available_pairs(botclient: Tuple[Any, TestClient]) -> None:
    ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/available_pairs')
    assert_response(rc)
    assert rc.json()['length'] == 12
    assert isinstance(rc.json()['pairs'], list)
    rc = client_get(client, f'{BASE_URI}/available_pairs?timeframe=5m')
    assert_response(rc)
    assert rc.json()['length'] == 12
    rc = client_get(client, f'{BASE_URI}/available_pairs?stake_currency=ETH')
    assert_response(rc)
    assert rc.json()['length'] == 1
    assert rc.json()['pairs'] == ['XRP/ETH']
    assert len(rc.json()['pair_interval']) == 2
    rc = client_get(client, f'{BASE_URI}/available_pairs?stake_currency=ETH&timeframe=5m')
    assert_response(rc)
    assert rc.json()['length'] == 1
    assert rc.json()['pairs'] == ['XRP/ETH']
    assert len(rc.json()['pair_interval']) == 1
    ftbot.config['trading_mode'] = 'futures'
    rc = client_get(client, f'{BASE_URI}/available_pairs?timeframe=1h')
    assert_response(rc)
    assert rc.json()['length'] == 1
    assert rc.json()['pairs'] == ['XRP/USDT:USDT']
    rc = client_get(client, f'{BASE_URI}/available_pairs?timeframe=1h&candletype=mark')
    assert_response(rc)
    assert rc.json()['length'] == 2
    assert rc.json()['pairs'] == ['UNITTEST/USDT:USDT', 'XRP/USDT:USDT']
    assert len(rc.json()['pair_interval']) == 2


def test_sysinfo(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/sysinfo')
    assert_response(rc)
    result = rc.json()
    assert 'cpu_pct' in result
    assert 'ram_pct' in result


def test_api_backtesting(botclient: Tuple[Any, TestClient], mocker: Any, fee: Any, caplog: Any, tmp_path: Path) -> None:
    try:
        ftbot, client = botclient
        mocker.patch(f'{EXMS}.get_fee', fee)
        rc = client_get(client, f'{BASE_URI}/backtest')
        assert_response(rc, 503)
        assert rc.json()['detail'] == 'Bot is not in the correct state.'
        ftbot.config['runmode'] = RunMode.WEBSERVER
        rc = client_get(client, f'{BASE_URI}/backtest')
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'not_started'
        assert not result['running']
        assert result['status_msg'] == 'Backtest not yet executed'
        assert result['progress'] == 0
        rc = client_delete(client, f'{BASE_URI}/backtest')
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'reset'
        assert not result['running']
        assert result['status_msg'] == 'Backtest reset'
        ftbot.config['export'] = 'trades'
        ftbot.config['backtest_cache'] = 'day'
        ftbot.config['user_data_dir'] = tmp_path
        ftbot.config['exportfilename'] = tmp_path / 'backtest_results'
        ftbot.config['exportfilename'].mkdir()
        data = {
            'strategy': CURRENT_TEST_STRATEGY,
            'timeframe': '5m',
            'timerange': '20180110-20180111',
            'max_open_trades': 3,
            'stake_amount': 100,
            'dry_run_wallet': 1000,
            'enable_protections': False
        }
        rc = client_post(client, f'{BASE_URI}/backtest', data=data)
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'running'
        assert result['progress'] == 0
        assert result['running']
        assert result['status_msg'] == 'Backtest started'
        rc = client_get(client, f'{BASE_URI}/backtest')
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'ended'
        assert not result['running']
        assert result['status_msg'] == 'Backtest ended'
        assert result['progress'] == 1
        assert result['backtest_result']
        rc = client_get(client, f'{BASE_URI}/backtest/abort')
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'not_running'
        assert not result['running']
        assert result['status_msg'] == 'Backtest ended'
        ApiBG.bgtask_running = True
        rc = client_get(client, f'{BASE_URI}/backtest/abort')
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'stopping'
        assert not result['running']
        assert result['status_msg'] == 'Backtest ended'
        rc = client_get(client, f'{BASE_URI}/backtest')
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'running'
        assert result['running']
        assert result['step'] == 'backtest'
        assert result['status_msg'] == 'Backtest running'
        rc = client_delete(client, f'{BASE_URI}/backtest')
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'running'
        rc = client_post(client, f'{BASE_URI}/backtest', data=data)
        assert_response(rc, 502)
        result = rc.json()
        assert 'Bot Background task already running' in result['error']
        ApiBG.bgtask_running = False
        rc = client_post(client, f'{BASE_URI}/backtest', data=data)
        assert_response(rc)
        result = rc.json()
        assert log_has_re('Reusing result of previous backtest.*', caplog)
        data['stake_amount'] = 101
        mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest_one_strategy', side_effect=DependencyException('DeadBeef'))
        rc = client_post(client, f'{BASE_URI}/backtest', data=data)
        assert log_has('Backtesting caused an error: DeadBeef', caplog)
        rc = client_get(client, f'{BASE_URI}/backtest')
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'error'
        assert 'Backtest failed' in result['status_msg']
        rc = client_delete(client, f'{BASE_URI}/backtest')
        assert_response(rc)
        result = rc.json()
        assert result['status'] == 'reset'
        assert not result['running']
        assert result['status_msg'] == 'Backtest reset'
        data['strategy'] = 'xx:cHJpbnQoImhlbGxvIHdvcmxkIik='
        rc = client_post(client, f'{BASE_URI}/backtest', data=data)
        assert_response(rc, 500)
    finally:
        Backtesting.cleanup()


def test_api_backtest_history(botclient: Tuple[Any, TestClient], mocker: Any, testdatadir: Path) -> None:
    ftbot, client = botclient
    mocker.patch('freqtrade.data.btanalysis._get_backtest_files', return_value=[
        testdatadir / 'backtest_results/backtest-result_multistrat.json',
        testdatadir / 'backtest_results/backtest-result.json'
    ])
    rc = client_get(client, f'{BASE_URI}/backtest/history')
    assert_response(rc, 503)
    assert rc.json()['detail'] == 'Bot is not in the correct state.'
    ftbot.config['user_data_dir'] = testdatadir
    ftbot.config['runmode'] = RunMode.WEBSERVER
    rc = client_get(client, f'{BASE_URI}/backtest/history')
    assert_response(rc)
    result = rc.json()
    assert len(result) == 3
    fn = result[0]['filename']
    assert fn == 'backtest-result_multistrat'
    assert result[0]['notes'] == ''
    strategy = result[0]['strategy']
    rc = client_get(client, f'{BASE_URI}/backtest/history/result?filename={fn}&strategy={strategy}')
    assert_response(rc)
    result2 = rc.json()
    assert result2
    assert result2['status'] == 'ended'
    assert not result2['running']
    assert result2['progress'] == 1
    assert len(result2['backtest_result']['strategy']) == 1
    assert result2['backtest_result']['strategy'][strategy]


def test_api_delete_backtest_history_entry(botclient: Tuple[Any, TestClient], tmp_path: Path) -> None:
    ftbot, client = botclient
    bt_results_base: Path = tmp_path / 'backtest_results'
    bt_results_base.mkdir()
    file_path: Path = bt_results_base / 'test.json'
    file_path.touch()
    meta_path: Path = file_path.with_suffix('.meta.json')
    meta_path.touch()
    market_change_path: Path = file_path.with_name(file_path.stem + '_market_change.feather')
    market_change_path.touch()
    rc = client_delete(client, f'{BASE_URI}/backtest/history/randomFile.json')
    assert_response(rc, 503)
    assert rc.json()['detail'] == 'Bot is not in the correct state.'
    ftbot.config['user_data_dir'] = tmp_path
    ftbot.config['runmode'] = RunMode.WEBSERVER
    rc = client_delete(client, f'{BASE_URI}/backtest/history/randomFile.json')
    assert rc.status_code == 404
    assert rc.json()['detail'] == 'File not found.'
    rc = client_delete(client, f'{BASE_URI}/backtest/history/{file_path.name}')
    assert rc.status_code == 200
    assert not file_path.exists()
    assert not meta_path.exists()
    assert not market_change_path.exists()


def test_api_patch_backtest_history_entry(botclient: Tuple[Any, TestClient], tmp_path: Path) -> None:
    ftbot, client = botclient
    bt_results_base: Path = tmp_path / 'backtest_results'
    bt_results_base.mkdir()
    file_path: Path = bt_results_base / 'test.json'
    file_path.touch()
    meta_path: Path = file_path.with_suffix('.meta.json')
    with meta_path.open('w') as metafile:
        rapidjson.dump({CURRENT_TEST_STRATEGY: {
            'run_id': '6e542efc8d5e62cef6e5be0ffbc29be81a6e751d',
            'backtest_start_time': 1690176003
        }}, metafile)
    def read_metadata() -> Dict[str, Any]:
        with meta_path.open('r') as metafile:
            return rapidjson.load(metafile)
    rc = client_patch(client, f'{BASE_URI}/backtest/history/randomFile.json')
    assert_response(rc, 503)
    ftbot.config['user_data_dir'] = tmp_path
    ftbot.config['runmode'] = RunMode.WEBSERVER
    rc = client_patch(client, f'{BASE_URI}/backtest/history/randomFile.json', {'strategy': CURRENT_TEST_STRATEGY})
    assert rc.status_code == 404
    rc = client_patch(client, f'{BASE_URI}/backtest/history/{file_path.name}', {'strategy': f'{CURRENT_TEST_STRATEGY}xxx'})
    assert rc.status_code == 400
    assert rc.json()['detail'] == 'Strategy not in metadata.'
    rc = client_patch(client, f'{BASE_URI}/backtest/history/{file_path.name}', {'strategy': CURRENT_TEST_STRATEGY})
    assert rc.status_code == 200
    res = rc.json()
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]['strategy'] == CURRENT_TEST_STRATEGY
    assert res[0]['notes'] == ''
    fileres = read_metadata()
    assert fileres[CURRENT_TEST_STRATEGY]['run_id'] == res[0]['run_id']
    assert fileres[CURRENT_TEST_STRATEGY]['notes'] == ''
    rc = client_patch(client, f'{BASE_URI}/backtest/history/{file_path.name}', {'strategy': CURRENT_TEST_STRATEGY, 'notes': 'FooBar'})
    assert rc.status_code == 200
    res = rc.json()
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0]['strategy'] == CURRENT_TEST_STRATEGY
    assert res[0]['notes'] == 'FooBar'
    fileres = read_metadata()
    assert fileres[CURRENT_TEST_STRATEGY]['run_id'] == res[0]['run_id']
    assert fileres[CURRENT_TEST_STRATEGY]['notes'] == 'FooBar'


def test_api_patch_backtest_market_change(botclient: Tuple[Any, TestClient], tmp_path: Path) -> None:
    ftbot, client = botclient
    bt_results_base: Path = tmp_path / 'backtest_results'
    bt_results_base.mkdir()
    file_path: Path = bt_results_base / 'test_22_market_change.feather'
    df = pd.DataFrame({'date': ['2018-01-01T00:00:00Z', '2018-01-01T00:05:00Z'], 'count': [2, 4], 'mean': [2555, 2556], 'rel_mean': [0, 0.022]})
    df['date'] = pd.to_datetime(df['date'])
    df.to_feather(file_path, compression_level=9, compression='lz4')
    rc = client_get(client, f'{BASE_URI}/backtest/history/randomFile.json/market_change')
    assert_response(rc, 503)
    ftbot.config['user_data_dir'] = tmp_path
    ftbot.config['runmode'] = RunMode.WEBSERVER
    rc = client_get(client, f'{BASE_URI}/backtest/history/randomFile.json/market_change')
    assert_response(rc, 404)
    rc = client_get(client, f'{BASE_URI}/backtest/history/test_22/market_change')
    assert_response(rc, 200)
    result = rc.json()
    assert result['length'] == 2
    assert result['columns'] == ['date', 'count', 'mean', 'rel_mean', '__date_ts']
    assert result['data'] == [
        ['2018-01-01T00:00:00Z', 2, 2555, 0.0, 1514764800000],
        ['2018-01-01T00:05:00Z', 4, 2556, 0.022, 1514765100000]
    ]


def test_health(botclient: Tuple[Any, TestClient]) -> None:
    _ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/health')
    assert_response(rc)
    ret = rc.json()
    assert ret['last_process_ts'] is None
    assert ret['last_process'] is None


def test_api_ws_subscribe(botclient: Tuple[Any, TestClient], mocker: Any) -> None:
    _ftbot, client = botclient
    ws_url: str = f'/api/v1/message/ws?token={_TEST_WS_TOKEN}'
    sub_mock = mocker.patch('freqtrade.rpc.api_server.ws.WebSocketChannel.set_subscriptions')
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({'type': 'subscribe', 'data': ['whitelist']})
        time.sleep(0.2)
    assert sub_mock.call_count == 1
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({'type': 'subscribe', 'data': 'whitelist'})
        time.sleep(0.2)
    assert sub_mock.call_count == 1


def test_api_ws_requests(botclient: Tuple[Any, TestClient], caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    _ftbot, client = botclient
    ws_url: str = f'/api/v1/message/ws?token={_TEST_WS_TOKEN}'
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({'type': 'whitelist', 'data': None})
        response = ws.receive_json()
    assert log_has_re('Request of type whitelist from.+', caplog)
    assert response['type'] == 'whitelist'
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({'type': 'analyzed_df', 'data': {}})
        response = ws.receive_json()
    assert log_has_re('Request of type analyzed_df from.+', caplog)
    assert response['type'] == 'analyzed_df'
    caplog.clear()
    with client.websocket_connect(ws_url) as ws:
        ws.send_json({'type': 'analyzed_df', 'data': {'limit': 100}})
        response = ws.receive_json()
    assert log_has_re('Request of type analyzed_df from.+', caplog)
    assert response['type'] == 'analyzed_df'


def test_api_ws_send_msg(default_conf: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    try:
        caplog.set_level(logging.DEBUG)
        default_conf.update({
            'api_server': {
                'enabled': True,
                'listen_ip_address': '127.0.0.1',
                'listen_port': 8080,
                'CORS_origins': ['http://example.com'],
                'username': _TEST_USER,
                'password': _TEST_PASS,
                'ws_token': _TEST_WS_TOKEN
            }
        })
        mocker.patch('freqtrade.rpc.telegram.Telegram._init')
        mocker.patch('freqtrade.rpc.api_server.ApiServer.start_api', return_value=None)
        apiserver = ApiServer(default_conf)
        apiserver.add_rpc_handler(RPC(get_patched_freqtradebot(mocker, default_conf)))
        with TestClient(apiserver.app):
            test_message = {'type': 'status', 'data': 'test'}
            first_waiter = apiserver._message_stream._waiter
            apiserver.send_msg(test_message)
            assert first_waiter.result()[0] == test_message
            second_waiter = apiserver._message_stream._waiter
            apiserver.send_msg(test_message)
            assert first_waiter != second_waiter
    finally:
        ApiServer.shutdown()
        ApiServer.shutdown()


def test_api_download_data(botclient: Tuple[Any, TestClient], mocker: Any, tmp_path: Path) -> None:
    ftbot, client = botclient
    rc = client_post(client, f'{BASE_URI}/download_data', data={})
    assert_response(rc, 503)
    assert rc.json()['detail'] == 'Bot is not in the correct state.'
    ftbot.config['runmode'] = RunMode.WEBSERVER
    ftbot.config['user_data_dir'] = tmp_path
    body = {'pairs': ['ETH/BTC', 'XRP/BTC'], 'timeframes': ['5m']}
    ApiBG.download_data_running = True
    rc = client_post(client, f'{BASE_URI}/download_data', body)
    assert_response(rc, 400)
    assert rc.json()['detail'] == 'Data Download is already running.'
    ApiBG.download_data_running = False
    mocker.patch('freqtrade.data.history.history_utils.download_data', return_value=None)
    rc = client_post(client, f'{BASE_URI}/download_data', body)
    assert_response(rc)
    assert rc.json()['status'] == 'Data Download started in background.'
    job_id = rc.json()['job_id']
    rc = client_get(client, f'{BASE_URI}/background/{job_id}')
    assert_response(rc)
    response = rc.json()
    assert response['job_id'] == job_id
    assert response['job_category'] == 'download_data'
    assert response['status'] == 'success'
    rc = client_get(client, f'{BASE_URI}/background')
    assert_response(rc)
    response = rc.json()
    assert isinstance(response, list)
    assert len(response) == 1
    assert response[0]['job_id'] == job_id
    ApiBG.download_data_running = False
    mocker.patch('freqtrade.data.history.history_utils.download_data', side_effect=OperationalException('Download error'))
    rc = client_post(client, f'{BASE_URI}/download_data', body)
    assert_response(rc)
    assert rc.json()['status'] == 'Data Download started in background.'
    job_id = rc.json()['job_id']
    rc = client_get(client, f'{BASE_URI}/background/{job_id}')
    assert_response(rc)
    response = rc.json()
    assert response['job_id'] == job_id
    assert response['job_category'] == 'download_data'
    assert response['status'] == 'failed'
    assert response['error'] == 'Download error'


def test_api_markets_live(botclient: Tuple[Any, TestClient]) -> None:
    ftbot, client = botclient
    rc = client_get(client, f'{BASE_URI}/markets')
    assert_response(rc, 200)
    response = rc.json()
    assert 'markets' in response
    assert len(response['markets']) >= 0
    assert response['markets']['XRP/USDT'] == {'base': 'XRP', 'quote': 'USDT', 'symbol': 'XRP/USDT', 'spot': True, 'swap': False}
    assert 'BTC/USDT' in response['markets']
    assert 'XRP/BTC' in response['markets']
    rc = client_get(client, f'{BASE_URI}/markets?base=XRP')
    assert_response(rc, 200)
    response = rc.json()
    assert 'XRP/USDT' in response['markets']
    assert 'XRP/BTC' in response['markets']
    assert 'BTC/USDT' not in response['markets']


def test_api_markets_webserver(botclient: Tuple[Any, TestClient]) -> None:
    ApiBG.exchanges = {}
    ftbot, client = botclient
    ftbot.config['runmode'] = RunMode.WEBSERVER
    rc = client_get(client, f'{BASE_URI}/markets?exchange=binance')
    assert_response(rc, 200)
    response = rc.json()
    assert 'markets' in response
    assert len(response['markets']) >= 0
    assert response['exchange_id'] == 'binance'
    rc = client_get(client, f'{BASE_URI}/markets?exchange=hyperliquid')
    assert_response(rc, 200)
    assert 'hyperliquid_spot' in ApiBG.exchanges
    assert 'binance_spot' in ApiBG.exchanges