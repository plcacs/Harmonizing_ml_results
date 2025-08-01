import json
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, PropertyMock
import pytest
import requests
from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.RemotePairList import RemotePairList
from freqtrade.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, get_patched_exchange, get_patched_freqtradebot, log_has

@pytest.fixture(scope='function')
def rpl_config(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    default_conf['stake_currency'] = 'USDT'
    default_conf['exchange']['pair_whitelist'] = ['ETH/USDT', 'XRP/USDT']
    default_conf['exchange']['pair_blacklist'] = ['BLK/USDT']
    return default_conf

def test_gen_pairlist_with_local_file(mocker: Any, rpl_config: Dict[str, Any]) -> None:
    mock_file: Any = MagicMock()
    mock_file.read.return_value = '{"pairs": ["TKN/USDT","ETH/USDT","NANO/USDT"]}'
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.open', return_value=mock_file)
    mock_file_path: Any = mocker.patch('freqtrade.plugins.pairlist.RemotePairList.Path')
    mock_file_path.exists.return_value = True
    jsonparse: Dict[str, Any] = json.loads(mock_file.read.return_value)
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.rapidjson.load', return_value=jsonparse)
    rpl_config['pairlists'] = [{
        'method': 'RemotePairList',
        'number_assets': 2,
        'refresh_period': 1800,
        'keep_pairlist_on_failure': True,
        'pairlist_url': 'file:///pairlist.json',
        'bearer_token': '',
        'read_timeout': 60
    }]
    exchange: Any = get_patched_exchange(mocker, rpl_config)
    pairlistmanager: PairListManager = PairListManager(exchange, rpl_config)
    remote_pairlist: RemotePairList = RemotePairList(exchange, pairlistmanager, rpl_config, rpl_config['pairlists'][0], 0)
    result: List[str] = remote_pairlist.gen_pairlist([])
    assert result == ['TKN/USDT', 'ETH/USDT']

def test_fetch_pairlist_mock_response_html(mocker: Any, rpl_config: Dict[str, Any]) -> None:
    mock_response: Any = MagicMock()
    mock_response.headers = {'content-type': 'text/html'}
    rpl_config['pairlists'] = [{
        'method': 'RemotePairList',
        'pairlist_url': 'http://example.com/pairlist',
        'number_assets': 10,
        'read_timeout': 10,
        'keep_pairlist_on_failure': True
    }]
    exchange: Any = get_patched_exchange(mocker, rpl_config)
    pairlistmanager: PairListManager = PairListManager(exchange, rpl_config)
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.requests.get', return_value=mock_response)
    remote_pairlist: RemotePairList = RemotePairList(exchange, pairlistmanager, rpl_config, rpl_config['pairlists'][0], 0)
    with pytest.raises(OperationalException, match='RemotePairList is not of type JSON.'):
        remote_pairlist.fetch_pairlist()

def test_fetch_pairlist_timeout_keep_last_pairlist(mocker: Any, rpl_config: Dict[str, Any], caplog: Any) -> None:
    rpl_config['pairlists'] = [{
        'method': 'RemotePairList',
        'pairlist_url': 'http://example.com/pairlist',
        'number_assets': 10,
        'read_timeout': 10,
        'keep_pairlist_on_failure': True
    }]
    exchange: Any = get_patched_exchange(mocker, rpl_config)
    pairlistmanager: PairListManager = PairListManager(exchange, rpl_config)
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.requests.get', side_effect=requests.exceptions.RequestException)
    remote_pairlist: RemotePairList = RemotePairList(exchange, pairlistmanager, rpl_config, rpl_config['pairlists'][0], 0)
    remote_pairlist._last_pairlist = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT']
    remote_pairlist._init_done = True
    pairlist_url: str = rpl_config['pairlists'][0]['pairlist_url']
    pairs, _time_elapsed = remote_pairlist.fetch_pairlist()
    assert log_has(f'Error: Was not able to fetch pairlist from: {pairlist_url}', caplog)
    assert log_has('Keeping last fetched pairlist', caplog)
    assert pairs == ['BTC/USDT', 'ETH/USDT', 'LTC/USDT']

def test_remote_pairlist_init_no_pairlist_url(mocker: Any, rpl_config: Dict[str, Any]) -> None:
    rpl_config['pairlists'] = [{
        'method': 'RemotePairList',
        'number_assets': 10,
        'keep_pairlist_on_failure': True
    }]
    get_patched_exchange(mocker, rpl_config)
    with pytest.raises(OperationalException, match='`pairlist_url` not specified. Please check your configuration for "pairlist.config.pairlist_url"'):
        get_patched_freqtradebot(mocker, rpl_config)

def test_remote_pairlist_init_no_number_assets(mocker: Any, rpl_config: Dict[str, Any]) -> None:
    rpl_config['pairlists'] = [{
        'method': 'RemotePairList',
        'pairlist_url': 'http://example.com/pairlist',
        'keep_pairlist_on_failure': True
    }]
    get_patched_exchange(mocker, rpl_config)
    with pytest.raises(OperationalException, match='`number_assets` not specified. Please check your configuration for "pairlist.config.number_assets"'):
        get_patched_freqtradebot(mocker, rpl_config)

def test_fetch_pairlist_mock_response_valid(mocker: Any, rpl_config: Dict[str, Any]) -> None:
    rpl_config['pairlists'] = [{
        'method': 'RemotePairList',
        'pairlist_url': 'http://example.com/pairlist',
        'number_assets': 10,
        'refresh_period': 10,
        'read_timeout': 10,
        'keep_pairlist_on_failure': True
    }]
    mock_response: Any = MagicMock()
    mock_response.json.return_value = {'pairs': ['ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'EOS/USDT'], 'refresh_period': 60}
    mock_response.headers = {'content-type': 'application/json'}
    mock_response.elapsed.total_seconds.return_value = 0.4
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.requests.get', return_value=mock_response)
    exchange: Any = get_patched_exchange(mocker, rpl_config)
    pairlistmanager: PairListManager = PairListManager(exchange, rpl_config)
    remote_pairlist: RemotePairList = RemotePairList(exchange, pairlistmanager, rpl_config, rpl_config['pairlists'][0], 0)
    pairs, time_elapsed = remote_pairlist.fetch_pairlist()
    assert pairs == ['ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'EOS/USDT']
    assert time_elapsed == 0.4
    assert remote_pairlist._refresh_period == 60

def test_remote_pairlist_init_wrong_mode(mocker: Any, rpl_config: Dict[str, Any]) -> None:
    rpl_config['pairlists'] = [{
        'method': 'RemotePairList',
        'mode': 'blacklis',
        'number_assets': 20,
        'pairlist_url': 'http://example.com/pairlist',
        'keep_pairlist_on_failure': True
    }]
    with pytest.raises(OperationalException, match='`mode` not configured correctly. Supported Modes are "whitelist","blacklist"'):
        get_patched_freqtradebot(mocker, rpl_config)
    rpl_config['pairlists'] = [{
        'method': 'RemotePairList',
        'mode': 'blacklist',
        'number_assets': 20,
        'pairlist_url': 'http://example.com/pairlist',
        'keep_pairlist_on_failure': True
    }]
    with pytest.raises(OperationalException, match='A `blacklist` mode RemotePairList can not be.*first.*'):
        get_patched_freqtradebot(mocker, rpl_config)

def test_remote_pairlist_init_wrong_proc_mode(mocker: Any, rpl_config: Dict[str, Any]) -> None:
    rpl_config['pairlists'] = [{
        'method': 'RemotePairList',
        'processing_mode': 'filler',
        'mode': 'whitelist',
        'number_assets': 20,
        'pairlist_url': 'http://example.com/pairlist',
        'keep_pairlist_on_failure': True
    }]
    get_patched_exchange(mocker, rpl_config)
    with pytest.raises(OperationalException, match='`processing_mode` not configured correctly. Supported Modes are "filter","append"'):
        get_patched_freqtradebot(mocker, rpl_config)

def test_remote_pairlist_blacklist(mocker: Any, rpl_config: Dict[str, Any], caplog: Any, markets: Any, tickers: Any) -> None:
    mock_response: Any = MagicMock()
    mock_response.json.return_value = {'pairs': ['XRP/USDT'], 'refresh_period': 60}
    mock_response.headers = {'content-type': 'application/json'}
    rpl_config['pairlists'] = [
        {'method': 'StaticPairList'},
        {
            'method': 'RemotePairList',
            'mode': 'blacklist',
            'pairlist_url': 'http://example.com/pairlist',
            'number_assets': 3
        }
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers
    )
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.requests.get', return_value=mock_response)
    exchange: Any = get_patched_exchange(mocker, rpl_config)
    pairlistmanager: PairListManager = PairListManager(exchange, rpl_config)
    remote_pairlist: RemotePairList = RemotePairList(exchange, pairlistmanager, rpl_config, rpl_config['pairlists'][1], 1)
    pairs, _time_elapsed = remote_pairlist.fetch_pairlist()
    assert pairs == ['XRP/USDT']
    whitelist: List[str] = remote_pairlist.filter_pairlist(rpl_config['exchange']['pair_whitelist'], {})
    assert whitelist == ['ETH/USDT']
    assert log_has(f'Blacklist - Filtered out pairs: {pairs}', caplog)

@pytest.mark.parametrize('processing_mode', ['filter', 'append'])
def test_remote_pairlist_whitelist(mocker: Any, rpl_config: Dict[str, Any], processing_mode: str, markets: Any, tickers: Any) -> None:
    mock_response: Any = MagicMock()
    mock_response.json.return_value = {'pairs': ['XRP/USDT'], 'refresh_period': 60}
    mock_response.headers = {'content-type': 'application/json'}
    rpl_config['pairlists'] = [
        {'method': 'StaticPairList'},
        {
            'method': 'RemotePairList',
            'mode': 'whitelist',
            'processing_mode': processing_mode,
            'pairlist_url': 'http://example.com/pairlist',
            'number_assets': 3
        }
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers
    )
    mocker.patch('freqtrade.plugins.pairlist.RemotePairList.requests.get', return_value=mock_response)
    exchange: Any = get_patched_exchange(mocker, rpl_config)
    pairlistmanager: PairListManager = PairListManager(exchange, rpl_config)
    remote_pairlist: RemotePairList = RemotePairList(exchange, pairlistmanager, rpl_config, rpl_config['pairlists'][1], 1)
    pairs, _time_elapsed = remote_pairlist.fetch_pairlist()
    assert pairs == ['XRP/USDT']
    whitelist: List[str] = remote_pairlist.filter_pairlist(rpl_config['exchange']['pair_whitelist'], {})
    if processing_mode == 'filter':
        assert whitelist == ['XRP/USDT']
    else:
        assert whitelist == ['ETH/USDT', 'XRP/USDT']