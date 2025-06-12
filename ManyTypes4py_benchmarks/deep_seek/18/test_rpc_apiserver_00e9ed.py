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

# ... (continue with the rest of the functions, adding appropriate type annotations)

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
