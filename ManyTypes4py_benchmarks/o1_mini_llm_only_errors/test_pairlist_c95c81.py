import logging
import time
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest
import time_machine
from freqtrade.constants import AVAILABLE_PAIRLISTS
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType, RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist, expand_pairlist
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.resolvers import PairListResolver
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import (
    EXMS,
    create_mock_trades_usdt,
    generate_test_data,
    get_patched_exchange,
    get_patched_freqtradebot,
    log_has,
    log_has_re,
    num_log_has,
)

TESTABLE_PAIRLISTS = [p for p in AVAILABLE_PAIRLISTS if p not in ['RemotePairList']]


@pytest.fixture(scope='function')
def whitelist_conf(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    default_conf['runmode'] = 'dry_run'
    default_conf['stake_currency'] = 'BTC'
    default_conf['exchange']['pair_whitelist'] = [
        'ETH/BTC', 'TKN/BTC', 'TRST/BTC', 'SWT/BTC', 'BCC/BTC', 'HOT/BTC'
    ]
    default_conf['exchange']['pair_blacklist'] = ['BLK/BTC']
    default_conf['pairlists'] = [{
        'method': 'VolumePairList',
        'number_assets': 5,
        'sort_key': 'quoteVolume'
    }]
    default_conf.update({'external_message_consumer': {'enabled': True, 'producers': []}})
    return default_conf


@pytest.fixture(scope='function')
def whitelist_conf_2(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    default_conf['runmode'] = 'dry_run'
    default_conf['stake_currency'] = 'BTC'
    default_conf['exchange']['pair_whitelist'] = [
        'ETH/BTC', 'TKN/BTC', 'BLK/BTC', 'LTC/BTC', 'BTT/BTC', 'HOT/BTC', 'FUEL/BTC', 'XRP/BTC'
    ]
    default_conf['exchange']['pair_blacklist'] = ['BLK/BTC']
    default_conf['pairlists'] = [{
        'method': 'VolumePairList',
        'number_assets': 5,
        'sort_key': 'quoteVolume',
        'refresh_period': 0
    }]
    return default_conf


@pytest.fixture(scope='function')
def whitelist_conf_agefilter(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    default_conf['runmode'] = 'dry_run'
    default_conf['stake_currency'] = 'BTC'
    default_conf['exchange']['pair_whitelist'] = [
        'ETH/BTC', 'TKN/BTC', 'BLK/BTC', 'LTC/BTC', 'BTT/BTC', 'HOT/BTC', 'FUEL/BTC', 'XRP/BTC'
    ]
    default_conf['exchange']['pair_blacklist'] = ['BLK/BTC']
    default_conf['pairlists'] = [
        {
            'method': 'VolumePairList',
            'number_assets': 5,
            'sort_key': 'quoteVolume',
            'refresh_period': -1
        },
        {
            'method': 'AgeFilter',
            'min_days_listed': 2,
            'max_days_listed': 100
        }
    ]
    return default_conf


@pytest.fixture(scope='function')
def static_pl_conf(whitelist_conf: Dict[str, Any]) -> Dict[str, Any]:
    whitelist_conf['pairlists'] = [{'method': 'StaticPairList'}]
    return whitelist_conf


def test_log_cached(
    mocker: pytest_mock.MockerFixture,
    static_pl_conf: Dict[str, Any],
    markets: Any,
    tickers: Any
) -> None:
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers
    )
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    logmock = MagicMock()
    pl = freqtrade.pairlists._pairlist_handlers[0]
    pl.log_once('Hello world', logmock)
    assert logmock.call_count == 1
    pl.log_once('Hello world', logmock)
    assert logmock.call_count == 1
    assert pl._log_cache.currsize == 1
    assert ('Hello world',) in pl._log_cache._Cache__data
    pl.log_once('Hello world2', logmock)
    assert logmock.call_count == 2
    assert pl._log_cache.currsize == 2


def test_load_pairlist_noexist(
    mocker: pytest_mock.MockerFixture,
    markets: Any,
    default_conf: Dict[str, Any]
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    plm = PairListManager(freqtrade.exchange, default_conf, MagicMock())
    with pytest.raises(
        OperationalException,
        match="Impossible to load Pairlist 'NonexistingPairList'. This class does not exist or contains Python code errors."
    ):
        PairListResolver.load_pairlist(
            'NonexistingPairList',
            freqtrade.exchange,
            plm,
            default_conf,
            {},
            1
        )


def test_load_pairlist_verify_multi(
    mocker: pytest_mock.MockerFixture,
    markets_static: Any,
    default_conf: Dict[str, Any]
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets_static))
    plm = PairListManager(freqtrade.exchange, default_conf, MagicMock())
    assert plm.verify_whitelist(['ETH/BTC', 'XRP/BTC'], print) == ['ETH/BTC', 'XRP/BTC']
    assert plm.verify_whitelist(['ETH/BTC', 'XRP/BTC', 'BUUU/BTC'], print) == ['ETH/BTC', 'XRP/BTC']
    assert plm.verify_whitelist(['XRP/BTC', 'BUUU/BTC'], print) == ['XRP/BTC']
    assert plm.verify_whitelist(['ETH/BTC', 'XRP/BTC'], print) == ['ETH/BTC', 'XRP/BTC']
    assert plm.verify_whitelist(['ETH/USDT', 'XRP/USDT'], print) == ['ETH/USDT']
    assert plm.verify_whitelist(['ETH/BTC', 'XRP/BTC'], print) == ['ETH/BTC', 'XRP/BTC']


def test_refresh_market_pair_not_in_whitelist(
    mocker: pytest_mock.MockerFixture,
    markets: Any,
    static_pl_conf: Dict[str, Any]
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    freqtrade.pairlists.refresh_pairlist()
    whitelist = ['ETH/BTC', 'TKN/BTC']
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf['exchange']['pair_whitelist'] == freqtrade.config['exchange']['pair_whitelist']


def test_refresh_static_pairlist(
    mocker: pytest_mock.MockerFixture,
    markets: Any,
    static_pl_conf: Dict[str, Any]
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets)
    )
    freqtrade.pairlists.refresh_pairlist()
    whitelist = ['ETH/BTC', 'TKN/BTC']
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf['exchange']['pair_blacklist'] == freqtrade.pairlists.blacklist


@pytest.mark.parametrize(
    'pairs,expected',
    [
        (['NOEXIST/BTC', '\\+WHAT/BTC'], ['ETH/BTC', 'TKN/BTC', 'TRST/BTC', 'NOEXIST/BTC', 'SWT/BTC', 'BCC/BTC', 'HOT/BTC']),
        (['NOEXIST/BTC', '*/BTC'], [])
    ]
)
def test_refresh_static_pairlist_noexist(
    mocker: pytest_mock.MockerFixture,
    markets: Any,
    static_pl_conf: Dict[str, Any],
    pairs: List[str],
    expected: List[str],
    caplog: pytest.LogCaptureFixture
) -> None:
    static_pl_conf['pairlists'][0]['allow_inactive'] = True
    static_pl_conf['exchange']['pair_whitelist'] += pairs
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets)
    )
    freqtrade.pairlists.refresh_pairlist()
    assert set(expected) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf['exchange']['pair_blacklist'] == freqtrade.pairlists.blacklist
    if not expected:
        assert log_has_re('Pair whitelist contains an invalid Wildcard: Wildcard error.*', caplog)


def test_invalid_blacklist(
    mocker: pytest_mock.MockerFixture,
    markets: Any,
    static_pl_conf: Dict[str, Any],
    caplog: pytest.LogCaptureFixture
) -> None:
    static_pl_conf['exchange']['pair_blacklist'] = ['*/BTC']
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets)
    )
    freqtrade.pairlists.refresh_pairlist()
    whitelist = []
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf['exchange']['pair_blacklist'] == freqtrade.pairlists.blacklist
    log_has_re('Pair blacklist contains an invalid Wildcard.*', caplog)


def test_remove_logs_for_pairs_already_in_blacklist(
    mocker: pytest_mock.MockerFixture,
    markets: Any,
    static_pl_conf: Dict[str, Any],
    caplog: pytest.LogCaptureFixture
) -> None:
    logger = logging.getLogger(__name__)
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets)
    )
    freqtrade.pairlists.refresh_pairlist()
    whitelist = ['ETH/BTC', 'TKN/BTC']
    caplog.clear()
    caplog.set_level(logging.INFO)
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf['exchange']['pair_blacklist'] == freqtrade.pairlists.blacklist
    assert not log_has('Pair BLK/BTC in your blacklist. Removing it from whitelist...', caplog)
    for _ in range(3):
        new_whitelist = freqtrade.pairlists.verify_blacklist(
            whitelist + ['BLK/BTC'],
            logger.warning
        )
        assert set(whitelist) == set(new_whitelist)
    assert num_log_has('Pair BLK/BTC in your blacklist. Removing it from whitelist...', caplog) == 1


def test_refresh_pairlist_dynamic(
    mocker: pytest_mock.MockerFixture,
    shitcoinmarkets: Any,
    tickers: Any,
    whitelist_conf: Dict[str, Any]
) -> None:
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=True)
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=shitcoinmarkets)
    )
    whitelist = ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC', 'HOT/BTC']
    freqtrade.pairlists.refresh_pairlist()
    assert whitelist == freqtrade.pairlists.whitelist
    whitelist_conf['pairlists'] = [{'method': 'VolumePairList'}]
    with pytest.raises(
        OperationalException,
        match='`number_assets` not specified. Please check your configuration for "pairlist.config.number_assets"'
    ):
        PairListManager(freqtrade.exchange, whitelist_conf, MagicMock())


def test_refresh_pairlist_dynamic_2(
    mocker: pytest_mock.MockerFixture,
    shitcoinmarkets: Any,
    tickers: Any,
    whitelist_conf_2: Dict[str, Any]
) -> None:
    tickers_dict = tickers()
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True)
    )
    mocker.patch.multiple(
        'freqtrade.plugins.pairlistmanager.PairListManager',
        _get_cached_tickers=MagicMock(return_value=tickers_dict)
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf_2)
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=shitcoinmarkets)
    )
    whitelist = ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC', 'HOT/BTC']
    freqtrade.pairlists.refresh_pairlist()
    assert whitelist == freqtrade.pairlists.whitelist
    time.sleep(1)
    whitelist = ['FUEL/BTC', 'ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC']
    tickers_dict['FUEL/BTC']['quoteVolume'] = 10000.0
    freqtrade.pairlists.refresh_pairlist()
    assert whitelist == freqtrade.pairlists.whitelist


def test_VolumePairList_refresh_empty(
    mocker: pytest_mock.MockerFixture,
    markets_empty: Any,
    whitelist_conf: Dict[str, Any]
) -> None:
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True)
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets_empty))
    whitelist = []
    whitelist_conf['exchange']['pair_whitelist'] = []
    freqtrade.pairlists.refresh_pairlist()
    pairslist = whitelist_conf['exchange']['pair_whitelist']
    assert set(whitelist) == set(pairslist)


@pytest.mark.parametrize(
    'pairlists,base_currency,whitelist_result',
    [
        (
            [{'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'}],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC', 'HOT/BTC']
        ),
        (
            [{'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'}],
            'USDT',
            ['ETH/USDT', 'NANO/USDT', 'ADAHALF/USDT', 'ADADOUBLE/USDT']
        ),
        (
            [{'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'}],
            'ETH',
            []
        ),
        (
            [{'method': 'StaticPairList'}],
            'ETH',
            []
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'AgeFilter', 'min_days_listed': 2, 'max_days_listed': None},
                {'method': 'PrecisionFilter'},
                {'method': 'PriceFilter', 'low_price_ratio': 0.03},
                {'method': 'SpreadFilter', 'max_spread_ratio': 0.005},
                {'method': 'ShuffleFilter'},
                {'method': 'PerformanceFilter'}
            ],
            'ETH',
            []
        ),
        (
            [
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'AgeFilter', 'min_days_listed': 2, 'max_days_listed': 100}
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC', 'HOT/BTC']
        ),
        (
            [
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'AgeFilter', 'min_days_listed': 10, 'max_days_listed': None}
            ],
            'BTC',
            []
        ),
        (
            [
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'AgeFilter', 'min_days_listed': 1, 'max_days_listed': 2}
            ],
            'BTC',
            []
        ),
        (
            [
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'AgeFilter', 'min_days_listed': 4, 'max_days_listed': 5}
            ],
            'BTC',
            []
        ),
        (
            [
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'AgeFilter', 'min_days_listed': 4, 'max_days_listed': 10}
            ],
            'BTC',
            ['LTC/BTC']
        ),
        (
            [
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'PrecisionFilter'}
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC']
        ),
        (
            [
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'PrecisionFilter'}
            ],
            'USDT',
            ['ETH/USDT', 'NANO/USDT']
        ),
        (
            [
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'PriceFilter', 'low_price_ratio': 0.03}
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC']
        ),
        (
            [
                {'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume'},
                {'method': 'PriceFilter', 'low_price_ratio': 0.03}
            ],
            'USDT',
            ['ETH/USDT', 'NANO/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 6,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'PrecisionFilter'},
                {
                    'method': 'PriceFilter',
                    'low_price_ratio': 0.02,
                    'min_price': 0.01
                }
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 6,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'PrecisionFilter'},
                {
                    'method': 'PriceFilter',
                    'low_price_ratio': 0.02,
                    'max_price': 0.05
                }
            ],
            'BTC',
            ['TKN/BTC', 'LTC/BTC', 'XRP/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume',
                    'min_value': 1250
                }
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume',
                    'max_value': 1300
                }
            ],
            'BTC',
            ['XRP/BTC', 'HOT/BTC', 'FUEL/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume',
                    'min_value': 100,
                    'max_value': 1300
                }
            ],
            'BTC',
            ['XRP/BTC', 'HOT/BTC']
        ),
        (
            [{'method': 'StaticPairList'}],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'HOT/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'SpreadFilter', 'max_spread_ratio': 0.005}
            ],
            'USDT',
            ['ETH/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'ShuffleFilter', 'seed': 77}
            ],
            'USDT',
            ['ADADOUBLE/USDT', 'ETH/USDT', 'NANO/USDT', 'ADAHALF/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'ShuffleFilter', 'seed': 42}
            ],
            'USDT',
            ['ADAHALF/USDT', 'NANO/USDT', 'ADADOUBLE/USDT', 'ETH/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'ShuffleFilter'}
            ],
            'USDT',
            4
        ),
        (
            [{'method': 'AgeFilter', 'min_days_listed': 2}],
            'BTC',
            'filter_at_the_beginning'
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PrecisionFilter'}
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [{'method': 'PrecisionFilter'}],
            'BTC',
            'filter_at_the_beginning'
        ),
        (
            [
                {'method': 'StaticPairList'},
                {
                    'method': 'PriceFilter',
                    'low_price_ratio': 0.02,
                    'min_price': 1e-06,
                    'max_price': 0.1
                }
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [{'method': 'PriceFilter', 'low_price_ratio': 0.02}],
            'BTC',
            'filter_at_the_beginning'
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'ShuffleFilter', 'seed': 42}
            ],
            'BTC',
            ['TKN/BTC', 'ETH/BTC', 'HOT/BTC']
        ),
        (
            [{'method': 'ShuffleFilter', 'seed': 42}],
            'BTC',
            'filter_at_the_beginning'
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PerformanceFilter'}
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'HOT/BTC']
        ),
        (
            [{'method': 'PerformanceFilter'}],
            'BTC',
            'filter_at_the_beginning'
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'SpreadFilter', 'max_spread_ratio': 0.005}
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [{'method': 'SpreadFilter', 'max_spread_ratio': 0.005}],
            'BTC',
            'filter_at_the_beginning'
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 2,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'StaticPairList'}
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'TRST/BTC', 'SWT/BTC', 'BCC/BTC', 'HOT/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'PriceFilter', 'low_price_ratio': 0.02}
            ],
            'USDT',
            ['ETH/USDT', 'NANO/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'PriceFilter', 'max_value': 1e-06}
            ],
            'USDT',
            ['NANO/USDT']
        ),
        (
            [
                {
                    'method': 'StaticPairList'
                },
                {
                    'method': 'RangeStabilityFilter',
                    'lookback_days': 10,
                    'min_rate_of_change': 0.01,
                    'refresh_period': 1440
                }
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC', 'HOT/BTC']
        ),
        (
            [
                {
                    'method': 'StaticPairList'
                },
                {
                    'method': 'RangeStabilityFilter',
                    'lookback_days': 10,
                    'max_rate_of_change': 0.01,
                    'refresh_period': 1440
                }
            ],
            'BTC',
            []
        ),
        (
            [
                {
                    'method': 'StaticPairList'
                },
                {
                    'method': 'RangeStabilityFilter',
                    'lookback_days': 10,
                    'min_rate_of_change': 0.018,
                    'max_rate_of_change': 0.02,
                    'refresh_period': 1440
                }
            ],
            'BTC',
            []
        ),
        (
            [
                {
                    'method': 'StaticPairList'
                },
                {
                    'method': 'VolatilityFilter',
                    'lookback_days': 3,
                    'min_volatility': 0.002,
                    'max_volatility': 0.004,
                    'refresh_period': 1440
                }
            ],
            'BTC',
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'OffsetFilter',
                    'offset': 0,
                    'number_assets': 0
                }
            ],
            'USDT',
            ['ETH/USDT', 'NANO/USDT', 'ADAHALF/USDT', 'ADADOUBLE/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'OffsetFilter',
                    'offset': 2
                }
            ],
            'USDT',
            ['ADAHALF/USDT', 'ADADOUBLE/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'OffsetFilter',
                    'offset': 1,
                    'number_assets': 2
                }
            ],
            'USDT',
            ['NANO/USDT', 'ADAHALF/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'OffsetFilter',
                    'offset': 100
                }
            ],
            'USDT',
            []
        )
    ]
)
def test_VolumePairList_whitelist_gen(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    shitcoinmarkets: Any,
    tickers: Any,
    ohlcv_history: pd.DataFrame,
    pairlists: List[Dict[str, Any]],
    base_currency: str,
    whitelist_result: Union[List[str], str],
    caplog: pytest.LogCaptureFixture
) -> None:
    whitelist_conf['runmode'] = 'util_exchange'
    whitelist_conf['pairlists'] = pairlists
    whitelist_conf['stake_currency'] = base_currency
    ohlcv_history_high_vola = ohlcv_history.copy()
    ohlcv_history_high_vola.loc[ohlcv_history_high_vola.index == 1, 'close'] = 0.0009
    ohlcv_data: Dict[Tuple[str, str, CandleType], pd.DataFrame] = {
        ('ETH/BTC', '1d', CandleType.SPOT): ohlcv_history,
        ('TKN/BTC', '1d', CandleType.SPOT): ohlcv_history,
        ('LTC/BTC', '1d', CandleType.SPOT): pd.concat([ohlcv_history, ohlcv_history]),
        ('XRP/BTC', '1d', CandleType.SPOT): ohlcv_history,
        ('HOT/BTC', '1d', CandleType.SPOT): ohlcv_history_high_vola
    }
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        markets=PropertyMock(return_value=shitcoinmarkets)
    )
    mocker.patch.multiple(
        EXMS,
        refresh_latest_ohlcv=MagicMock(return_value=ohlcv_data)
    )
    mocker.patch.multiple(
        'freqtrade.persistence.Trade',
        get_overall_performance=MagicMock(return_value=[])
    )
    if whitelist_result == 'filter_at_the_beginning':
        with pytest.raises(
            OperationalException,
            match='This Pairlist Handler should not be used at the first position in the list of Pairlist Handlers.'
        ):
            freqtrade.pairlists.refresh_pairlist()
    else:
        freqtrade.pairlists.refresh_pairlist()
        whitelist = freqtrade.pairlists.whitelist
        assert isinstance(whitelist, list)
        if isinstance(whitelist_result, list):
            assert whitelist == whitelist_result
        else:
            assert len(whitelist) == whitelist_result
        for pairlist in pairlists:
            if (
                pairlist['method'] == 'AgeFilter' and
                pairlist.get('min_days_listed') and
                (len(ohlcv_history) < pairlist['min_days_listed'])
            ):
                assert log_has_re('^Removed .* from whitelist, because age .* is less than .* day.*', caplog)
            if (
                pairlist['method'] == 'AgeFilter' and
                pairlist.get('max_days_listed') and
                (len(ohlcv_history) > pairlist['max_days_listed'])
            ):
                assert log_has_re('^Removed .* from whitelist, because age .* is less than .* day.* or more than .* day', caplog)
            if (
                pairlist['method'] == 'PrecisionFilter' and
                whitelist_result
            ):
                assert log_has_re('^Removed .* from whitelist, because stop price .* would be <= stop limit.*', caplog)
            if (
                pairlist['method'] == 'PriceFilter' and
                whitelist_result
            ):
                assert (
                    log_has_re('^Removed .* from whitelist, because 1 unit is .*%$', caplog) or
                    log_has_re('^Removed .* from whitelist, because last price < .*%$', caplog) or
                    log_has_re('^Removed .* from whitelist, because last price > .*%$', caplog) or
                    log_has_re('^Removed .* from whitelist, because min value change of .*', caplog) or
                    log_has_re("^Removed .* from whitelist, because ticker\\['last'\\] is empty.*", caplog)
                )
            if pairlist['method'] == 'VolumePairList':
                logmsg = 'DEPRECATED: using any key other than quoteVolume for VolumePairList is deprecated.'
                if pairlist['sort_key'] != 'quoteVolume':
                    assert log_has(logmsg, caplog)
                else:
                    assert not log_has(logmsg, caplog)
            if pairlist['method'] == 'VolatilityFilter':
                assert log_has_re('^Removed .* from whitelist, because volatility.*$', caplog)


@pytest.mark.parametrize(
    'pairlists,base_currency,exchange,volumefilter_result',
    [
        (
            [{'method': 'VolumePairList', 'number_assets': 5, 'sort_key': 'quoteVolume', 'lookback_days': 1}],
            'BTC',
            'binance',
            'default_refresh_too_short'
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume',
                    'lookback_period': 1
                }
            ],
            'BTC',
            'binance',
            'lookback_days_and_period'
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume',
                    'lookback_timeframe': '1d',
                    'lookback_period': -1
                }
            ],
            'BTC',
            'binance',
            'lookback_period_negative'
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume',
                    'lookback_timeframe': '1m',
                    'lookback_period': 2000,
                    'refresh_period': 3600
                }
            ],
            'BTC',
            'binance',
            'lookback_exceeds_exchange_request_size'
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                }
            ],
            'USDT',
            'binance',
            ['LTC/USDT', 'ETH/USDT', 'TKN/BTC', 'XRP/BTC', 'HOT/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'AgeFilter',
                    'min_days_listed': 2,
                    'max_days_listed': 100
                }
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC', 'HOT/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'AgeFilter',
                    'min_days_listed': 10,
                    'max_days_listed': None
                }
            ],
            'BTC',
            'binance',
            []
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'AgeFilter',
                    'min_days_listed': 1,
                    'max_days_listed': 2
                }
            ],
            'BTC',
            'binance',
            []
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'AgeFilter',
                    'min_days_listed': 4,
                    'max_days_listed': 5
                }
            ],
            'BTC',
            'binance',
            []
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'AgeFilter',
                    'min_days_listed': 4,
                    'max_days_listed': 10
                }
            ],
            'BTC',
            'binance',
            ['LTC/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'PrecisionFilter'}
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'PrecisionFilter'}
            ],
            'USDT',
            'binance',
            ['ETH/USDT', 'NANO/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'PriceFilter',
                    'low_price_ratio': 0.03
                }
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC', 'XRP/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'PriceFilter',
                    'low_price_ratio': 0.03
                }
            ],
            'USDT',
            'binance',
            ['ETH/USDT', 'NANO/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 6,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'PrecisionFilter'},
                {
                    'method': 'PriceFilter',
                    'low_price_ratio': 0.02,
                    'min_price': 0.01
                }
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 6,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'PrecisionFilter'},
                {
                    'method': 'PriceFilter',
                    'low_price_ratio': 0.02,
                    'max_price': 0.05
                }
            ],
            'BTC',
            'binance',
            ['TKN/BTC', 'LTC/BTC', 'XRP/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume',
                    'min_value': 1250
                }
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume',
                    'max_value': 1300
                }
            ],
            'BTC',
            'binance',
            ['XRP/BTC', 'HOT/BTC', 'FUEL/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume',
                    'min_value': 100,
                    'max_value': 1300
                }
            ],
            'BTC',
            'binance',
            ['XRP/BTC', 'HOT/BTC']
        ),
        (
            [{'method': 'StaticPairList'}],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC', 'HOT/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'SpreadFilter',
                    'max_spread_ratio': 0.005
                }
            ],
            'USDT',
            'binance',
            ['ETH/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'ShuffleFilter',
                    'seed': 77
                }
            ],
            'USDT',
            'binance',
            ['ADADOUBLE/USDT', 'ETH/USDT', 'NANO/USDT', 'ADAHALF/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'ShuffleFilter',
                    'seed': 42
                }
            ],
            'USDT',
            'binance',
            ['ADAHALF/USDT', 'NANO/USDT', 'ADADOUBLE/USDT', 'ETH/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 5,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'ShuffleFilter'}
            ],
            'USDT',
            'binance',
            4
        ),
        (
            [{'method': 'AgeFilter', 'min_days_listed': 2}],
            'BTC',
            'binance',
            'filter_at_the_beginning'
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PrecisionFilter'}
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [{'method': 'PrecisionFilter'}],
            'BTC',
            'binance',
            'filter_at_the_beginning'
        ),
        (
            [
                {'method': 'StaticPairList'},
                {
                    'method': 'PriceFilter',
                    'low_price_ratio': 0.02,
                    'min_price': 1e-06,
                    'max_price': 0.1
                }
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [{'method': 'PriceFilter', 'low_price_ratio': 0.02}],
            'BTC',
            'binance',
            'filter_at_the_beginning'
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'ShuffleFilter', 'seed': 42}
            ],
            'BTC',
            'binance',
            ['TKN/BTC', 'ETH/BTC', 'HOT/BTC']
        ),
        (
            [{'method': 'ShuffleFilter', 'seed': 42}],
            'BTC',
            'binance',
            'filter_at_the_beginning'
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PerformanceFilter'}
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC', 'HOT/BTC']
        ),
        (
            [{'method': 'PerformanceFilter'}],
            'BTC',
            'binance',
            'filter_at_the_beginning'
        ),
        (
            [
                {
                    'method': 'StaticPairList'
                },
                {
                    'method': 'SpreadFilter',
                    'max_spread_ratio': 0.005
                }
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [{'method': 'SpreadFilter', 'max_spread_ratio': 0.005}],
            'BTC',
            'binance',
            'filter_at_the_beginning'
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 2,
                    'sort_key': 'quoteVolume'
                },
                {'method': 'StaticPairList'}
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC', 'TRST/BTC', 'SWT/BTC', 'BCC/BTC', 'HOT/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'PriceFilter',
                    'low_price_ratio': 0.02
                }
            ],
            'USDT',
            'binance',
            ['ETH/USDT', 'NANO/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'PriceFilter',
                    'max_value': 1e-06
                }
            ],
            'USDT',
            'binance',
            ['NANO/USDT']
        ),
        (
            [
                {
                    'method': 'StaticPairList'
                },
                {
                    'method': 'RangeStabilityFilter',
                    'lookback_days': 10,
                    'min_rate_of_change': 0.01,
                    'refresh_period': 1440
                }
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC', 'HOT/BTC']
        ),
        (
            [
                {
                    'method': 'StaticPairList'
                },
                {
                    'method': 'RangeStabilityFilter',
                    'lookback_days': 10,
                    'max_rate_of_change': 0.01,
                    'refresh_period': 1440
                }
            ],
            'BTC',
            'binance',
            []
        ),
        (
            [
                {
                    'method': 'StaticPairList'
                },
                {
                    'method': 'RangeStabilityFilter',
                    'lookback_days': 10,
                    'min_rate_of_change': 0.018,
                    'max_rate_of_change': 0.02,
                    'refresh_period': 1440
                }
            ],
            'BTC',
            'binance',
            []
        ),
        (
            [
                {
                    'method': 'StaticPairList'
                },
                {
                    'method': 'VolatilityFilter',
                    'lookback_days': 3,
                    'min_volatility': 0.002,
                    'max_volatility': 0.004,
                    'refresh_period': 1440
                }
            ],
            'BTC',
            'binance',
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'OffsetFilter',
                    'offset': 0,
                    'number_assets': 0
                }
            ],
            'USDT',
            'binance',
            ['ETH/USDT', 'NANO/USDT', 'ADAHALF/USDT', 'ADADOUBLE/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'OffsetFilter',
                    'offset': 2
                }
            ],
            'USDT',
            'binance',
            ['ADAHALF/USDT', 'ADADOUBLE/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'OffsetFilter',
                    'offset': 1,
                    'number_assets': 2
                }
            ],
            'USDT',
            'binance',
            ['NANO/USDT', 'ADAHALF/USDT']
        ),
        (
            [
                {
                    'method': 'VolumePairList',
                    'number_assets': 20,
                    'sort_key': 'quoteVolume'
                },
                {
                    'method': 'OffsetFilter',
                    'offset': 100
                }
            ],
            'USDT',
            'binance',
            []
        )
    ]
)
def test_VolumePairList_range(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    shitcoinmarkets: Any,
    tickers: Any,
    ohlcv_history: List[pd.DataFrame],
    pairlists: List[Dict[str, Any]],
    base_currency: str,
    exchange: str,
    volumefilter_result: Union[List[str], str],
    time_machine: time_machine.TimeMachine
) -> None:
    whitelist_conf['pairlists'] = pairlists
    whitelist_conf['stake_currency'] = base_currency
    whitelist_conf['exchange']['name'] = exchange
    ohlcv_history_long = pd.concat(ohlcv_history)
    ohlcv_history_high_vola = ohlcv_history_long.copy()
    ohlcv_history_high_vola.loc[ohlcv_history_high_vola.index == 1, 'close'] = 0.0009
    ohlcv_history_medium_volume = ohlcv_history_long.copy()
    ohlcv_history_medium_volume.loc[ohlcv_history_medium_volume.index == 2, 'volume'] = 5
    ohlcv_history_high_volume = ohlcv_history_long.copy()
    ohlcv_history_high_volume['volume'] = 10
    ohlcv_history_high_volume['low'] = ohlcv_history_high_volume.loc[:, 'low'] * 0.01
    ohlcv_history_high_volume['high'] = ohlcv_history_high_volume.loc[:, 'high'] * 0.01
    ohlcv_history_high_volume['close'] = ohlcv_history_high_volume.loc[:, 'close'] * 0.01
    ohlcv_data: Dict[Tuple[str, str, CandleType], pd.DataFrame] = {
        ('ETH/BTC', '1d', CandleType.SPOT): ohlcv_history_long,
        ('TKN/BTC', '1d', CandleType.SPOT): ohlcv_history,
        ('LTC/BTC', '1d', CandleType.SPOT): ohlcv_history_medium_volume,
        ('XRP/BTC', '1d', CandleType.SPOT): ohlcv_history_high_vola,
        ('HOT/BTC', '1d', CandleType.SPOT): ohlcv_history_high_volume
    }
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    if volumefilter_result == 'default_refresh_too_short':
        with pytest.raises(
            OperationalException,
            match='Refresh period of [0-9]+ seconds is smaller than one timeframe of [0-9]+.*\\. Please adjust refresh_period to at least [0-9]+ and restart the bot\\.'
        ):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
        return
    elif volumefilter_result == 'lookback_days_and_period':
        with pytest.raises(
            OperationalException,
            match='Ambiguous configuration: lookback_days and lookback_period both set in pairlist config\\..*'
        ):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    elif volumefilter_result == 'lookback_period_negative':
        with pytest.raises(
            OperationalException,
            match='VolumeFilter requires lookback_period to be >= 0'
        ):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    elif volumefilter_result == 'lookback_exceeds_exchange_request_size':
        with pytest.raises(
            OperationalException,
            match='VolumeFilter requires lookback_period to not exceed exchange max request size \\([0-9]+\\)'
        ):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    else:
        freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
        mocker.patch.multiple(
            EXMS,
            get_tickers=tickers,
            markets=PropertyMock(return_value=shitcoinmarkets)
        )
        start_dt = dt_now()
        time_machine.move_to(start_dt)
        if 'lookback_timeframe' in pairlists[0]:
            if pairlists[0]['lookback_timeframe'] != '1d':
                ohlcv_data = {}
        ohclv_mock = mocker.patch(f'{EXMS}.refresh_latest_ohlcv', return_value=ohlcv_data)
        freqtrade.pairlists.refresh_pairlist()
        whitelist = freqtrade.pairlists.whitelist
        assert ohclv_mock.call_count == 1
        assert isinstance(whitelist, list)
        assert whitelist == volumefilter_result
        ohclv_mock.reset_mock()
        freqtrade.pairlists.refresh_pairlist()
        assert ohclv_mock.call_count == 0
        whitelist = freqtrade.pairlists.whitelist
        assert whitelist == volumefilter_result
        time_machine.move_to(start_dt + timedelta(days=2))
        ohclv_mock.reset_mock()
        freqtrade.pairlists.refresh_pairlist()
        assert ohclv_mock.call_count == 1
        assert whitelist == volumefilter_result


def test_PrecisionFilter_error(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any]
) -> None:
    whitelist_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        {'method': 'PrecisionFilter'}
    ]
    del whitelist_conf['stoploss']
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    with pytest.raises(
        OperationalException,
        match='PrecisionFilter can only work with stoploss defined\\..*'
    ):
        PairListManager(MagicMock(), whitelist_conf, MagicMock())


def test_PerformanceFilter_error(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    caplog: pytest.LogCaptureFixture
) -> None:
    whitelist_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        {'method': 'PerformanceFilter'}
    ]
    if hasattr(Trade, 'session'):
        del Trade.session
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    exchange = get_patched_exchange(mocker, whitelist_conf)
    pm = PairListManager(exchange, whitelist_conf, MagicMock())
    pm.refresh_pairlist()
    assert log_has('PerformanceFilter is not available in this mode.', caplog)


def test_VolatilityFilter_error(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any]
) -> None:
    volatility_filter: Dict[str, Any] = {'method': 'VolatilityFilter', 'lookback_days': -1}
    whitelist_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        volatility_filter
    ]
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    exchange_mock = MagicMock()
    exchange_mock.ohlcv_candle_limit = MagicMock(return_value=1000)
    with pytest.raises(
        OperationalException,
        match='VolatilityFilter requires lookback_days to be >= 1*'
    ):
        PairListManager(exchange_mock, whitelist_conf, MagicMock())
    volatility_filter = {'method': 'VolatilityFilter', 'lookback_days': 2000}
    whitelist_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        volatility_filter
    ]
    with pytest.raises(
        OperationalException,
        match='VolatilityFilter requires lookback_days to not exceed exchange max'
    ):
        PairListManager(exchange_mock, whitelist_conf, MagicMock())
    volatility_filter = {'method': 'VolatilityFilter', 'sort_direction': 'Random'}
    whitelist_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        volatility_filter
    ]
    with pytest.raises(
        OperationalException,
        match="VolatilityFilter requires sort_direction to be either None .*'asc'.*'desc'"
    ):
        PairListManager(exchange_mock, whitelist_conf, MagicMock())


@pytest.mark.parametrize(
    'pairlist,expected_pairlist',
    [
        (
            {'method': 'VolatilityFilter', 'sort_direction': 'asc'},
            ['XRP/BTC', 'ETH/BTC', 'LTC/BTC', 'TKN/BTC']
        ),
        (
            {'method': 'VolatilityFilter', 'sort_direction': 'desc'},
            ['TKN/BTC', 'LTC/BTC', 'ETH/BTC', 'XRP/BTC']
        ),
        (
            {'method': 'VolatilityFilter', 'sort_direction': 'desc', 'min_volatility': 0.4},
            ['TKN/BTC', 'LTC/BTC', 'ETH/BTC']
        ),
        (
            {'method': 'VolatilityFilter', 'sort_direction': 'asc', 'min_volatility': 0.4},
            ['ETH/BTC', 'LTC/BTC', 'TKN/BTC']
        ),
        (
            {'method': 'VolatilityFilter', 'sort_direction': 'desc', 'max_volatility': 0.5},
            ['LTC/BTC', 'ETH/BTC', 'XRP/BTC']
        ),
        (
            {'method': 'VolatilityFilter', 'sort_direction': 'asc', 'max_volatility': 0.5},
            ['XRP/BTC', 'ETH/BTC', 'LTC/BTC']
        ),
        (
            {'method': 'RangeStabilityFilter', 'sort_direction': 'asc'},
            ['ETH/BTC', 'XRP/BTC', 'LTC/BTC', 'TKN/BTC']
        ),
        (
            {'method': 'RangeStabilityFilter', 'sort_direction': 'desc'},
            ['TKN/BTC', 'LTC/BTC', 'XRP/BTC', 'ETH/BTC']
        ),
        (
            {
                'method': 'RangeStabilityFilter',
                'sort_direction': 'asc',
                'min_rate_of_change': 0.4
            },
            ['XRP/BTC', 'LTC/BTC', 'TKN/BTC']
        ),
        (
            {
                'method': 'RangeStabilityFilter',
                'sort_direction': 'desc',
                'min_rate_of_change': 0.4
            },
            ['TKN/BTC', 'LTC/BTC', 'XRP/BTC']
        )
    ]
)
def test_VolatilityFilter_RangeStabilityFilter_sort(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    tickers: Any,
    time_machine: time_machine.TimeMachine,
    pairlist: Dict[str, Any],
    expected_pairlist: List[str]
) -> None:
    whitelist_conf['pairlists'] = [
        {'method': 'VolumePairList', 'number_assets': 10},
        pairlist
    ]
    df1 = generate_test_data(
        '1d',
        10,
        '2022-01-05 00:00:00+00:00',
        random_seed=42
    )
    df2 = generate_test_data(
        '1d',
        10,
        '2022-01-05 00:00:00+00:00',
        random_seed=2
    )
    df3 = generate_test_data(
        '1d',
        10,
        '2022-01-05 00:00:00+00:00',
        random_seed=3
    )
    df4 = generate_test_data(
        '1d',
        10,
        '2022-01-05 00:00:00+00:00',
        random_seed=4
    )
    df5 = generate_test_data(
        '1d',
        10,
        '2022-01-05 00:00:00+00:00',
        random_seed=5
    )
    df6 = generate_test_data(
        '1d',
        10,
        '2022-01-05 00:00:00+00:00',
        random_seed=6
    )
    assert not df1.equals(df2)
    time_machine.move_to('2022-01-15 00:00:00+00:00')
    ohlcv_data: Dict[Tuple[str, str, CandleType], pd.DataFrame] = {
        ('ETH/BTC', '1d', CandleType.SPOT): df1,
        ('TKN/BTC', '1d', CandleType.SPOT): df2,
        ('LTC/BTC', '1d', CandleType.SPOT): df3,
        ('XRP/BTC', '1d', CandleType.SPOT): df4,
        ('HOT/BTC', '1d', CandleType.SPOT): df5,
        ('BLK/BTC', '1d', CandleType.SPOT): df6
    }
    ohlcv_mock = MagicMock(return_value=ohlcv_data)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        refresh_latest_ohlcv=ohlcv_mock,
        get_tickers=tickers
    )
    exchange = get_patched_exchange(mocker, whitelist_conf)
    pm = PairListManager(exchange, whitelist_conf, MagicMock())
    assert exchange.ohlcv_candle_limit.call_count == 2
    pm.refresh_pairlist()
    assert ohlcv_mock.call_count == 1
    assert exchange.ohlcv_candle_limit.call_count == 2
    assert pm.whitelist == expected_pairlist
    pm.refresh_pairlist()
    assert exchange.ohlcv_candle_limit.call_count == 2
    assert ohlcv_mock.call_count == 1


def test_ShuffleFilter_init(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    caplog: pytest.LogCaptureFixture
) -> None:
    whitelist_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        {'method': 'ShuffleFilter', 'seed': 43}
    ]
    whitelist_conf['runmode'] = 'backtest'
    exchange = get_patched_exchange(mocker, whitelist_conf)
    plm = PairListManager(exchange, whitelist_conf)
    assert log_has('Backtesting mode detected, applying seed value: 43', caplog)
    with time_machine.travel('2021-09-01 05:01:00 +00:00') as t:
        plm.refresh_pairlist()
        pl1 = deepcopy(plm.whitelist)
        plm.refresh_pairlist()
        assert plm.whitelist == pl1
        t.shift(timedelta(minutes=10))
        plm.refresh_pairlist()
        assert plm.whitelist != pl1
    caplog.clear()
    whitelist_conf['runmode'] = RunMode.DRY_RUN
    plm = PairListManager(exchange, whitelist_conf)
    assert not log_has('Backtesting mode detected, applying seed value: 42', caplog)
    assert log_has('Live mode detected, not applying seed.', caplog)


@pytest.mark.usefixtures('init_persistence')
def test_PerformanceFilter_lookback(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: float,
    caplog: pytest.LogCaptureFixture
) -> None:
    default_conf_usdt['exchange']['pair_whitelist'].extend(['ADA/USDT', 'XRP/USDT', 'ETC/USDT'])
    default_conf_usdt['pairlists'] = [
        {'method': 'StaticPairList'},
        {'method': 'PerformanceFilter', 'minutes': 60, 'min_profit': 0.01}
    ]
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()
    assert pm.whitelist == ['ETH/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT']
    with time_machine.travel('2021-09-01 05:00:00 +00:00') as t:
        create_mock_trades_usdt(fee)
        pm.refresh_pairlist()
        assert pm.whitelist == ['XRP/USDT', 'NEO/USDT']
        assert log_has_re('Removing pair .* since .* is below .*', caplog)
        t.move_to('2021-09-01 07:00:00 +00:00')
        pm.refresh_pairlist()
        assert pm.whitelist == ['ETH/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT']


@pytest.mark.usefixtures('init_persistence')
def test_PerformanceFilter_keep_mid_order(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: float,
    caplog: pytest.LogCaptureFixture
) -> None:
    default_conf_usdt['exchange']['pair_whitelist'].extend(['ADA/USDT', 'ETC/USDT'])
    default_conf_usdt['pairlists'] = [
        {'method': 'StaticPairList', 'allow_inactive': True},
        {'method': 'PerformanceFilter', 'minutes': 60}
    ]
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()
    assert pm.whitelist == ['ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT', 'ADA/USDT', 'ETC/USDT']
    with time_machine.travel('2021-09-01 05:00:00 +00:00') as t:
        create_mock_trades_usdt(fee)
        pm.refresh_pairlist()
        assert pm.whitelist == ['XRP/USDT', 'NEO/USDT', 'ETH/USDT', 'LTC/USDT', 'TKN/USDT', 'ADA/USDT', 'ETC/USDT']
        t.move_to('2021-09-01 07:00:00 +00:00')
        pm.refresh_pairlist()
        assert pm.whitelist == ['ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT', 'ADA/USDT', 'ETC/USDT']


def test_gen_pair_whitelist_not_supported(
    mocker: pytest_mock.MockerFixture,
    default_conf: Dict[str, Any],
    tickers: Any
) -> None:
    default_conf['pairlists'] = [
        {'method': 'VolumePairList', 'number_assets': 10}
    ]
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=False)
    )
    with pytest.raises(
        OperationalException,
        match='Exchange does not support dynamic whitelist.*'
    ):
        get_patched_freqtradebot(mocker, default_conf)


def test_pair_whitelist_not_supported_Spread(
    mocker: pytest_mock.MockerFixture,
    default_conf: Dict[str, Any],
    tickers: Any
) -> None:
    default_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        {'method': 'SpreadFilter'}
    ]
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=False)
    )
    with pytest.raises(
        OperationalException,
        match='Exchange does not support fetchTickers, .*'
    ):
        get_patched_freqtradebot(mocker, default_conf)
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    mocker.patch(f'{EXMS}.get_option', MagicMock(return_value=False))
    with pytest.raises(
        OperationalException,
        match='.*requires exchange to have bid/ask data'
    ):
        get_patched_freqtradebot(mocker, default_conf)


@pytest.mark.parametrize(
    'pairlist',
    TESTABLE_PAIRLISTS
)
def test_pairlist_class(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    markets: Any,
    pairlist: str
) -> None:
    whitelist_conf['pairlists'][0]['method'] = pairlist
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True)
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    assert freqtrade.pairlists.name_list == [pairlist]
    assert pairlist in str(freqtrade.pairlists.short_desc())
    assert isinstance(freqtrade.pairlists.whitelist, list)
    assert isinstance(freqtrade.pairlists.blacklist, list)


@pytest.mark.parametrize(
    'pairlist,whitelist,log_message',
    [
        (['ETH/BTC', 'TKN/BTC'], ''),
        (['ETH/BTC', 'TKN/BTC', 'TRX/ETH'], 'is not compatible with exchange'),
        (['ETH/BTC', 'TKN/BTC', 'ETH/USDT'], 'is not compatible with your stake currency'),
        (['ETH/BTC', 'TKN/BTC', 'BCH/BTC'], 'is not compatible with exchange'),
        (['ETH/BTC', 'TKN/BTC', 'BTT/BTC'], 'Market is not active'),
        (['ETH/BTC', 'TKN/BTC', 'XLTCUSDT'], 'is not tradable with Freqtrade')
    ]
)
@pytest.mark.parametrize(
    'pairlist',
    TESTABLE_PAIRLISTS
)
def test__whitelist_for_active_markets(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    markets: Any,
    pairlist: str,
    whitelist: List[str],
    caplog: pytest.LogCaptureFixture,
    log_message: str,
    tickers: Any
) -> None:
    whitelist_conf['pairlists'][0]['method'] = pairlist
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    caplog.clear()
    pairlist_handler = freqtrade.pairlists._pairlist_handlers[0]
    new_whitelist = pairlist_handler._whitelist_for_active_markets(whitelist)
    assert set(new_whitelist) == set(['ETH/BTC', 'TKN/BTC'])
    assert log_message in caplog.text


@pytest.mark.parametrize(
    'pairlist',
    TESTABLE_PAIRLISTS
)
def test__whitelist_for_active_markets_empty(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    pairlist: str,
    tickers: Any
) -> None:
    whitelist_conf['pairlists'][0]['method'] = pairlist
    mocker.patch(f'{EXMS}.exchange_has', return_value=True)
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=None),
        get_tickers=tickers
    )
    pairlist_handler = freqtrade.pairlists._pairlist_handlers[0]
    with pytest.raises(
        OperationalException,
        match='Markets not loaded.*'
    ):
        pairlist_handler._whitelist_for_active_markets(['ETH/BTC'])


def test_volumepairlist_invalid_sortvalue(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any]
) -> None:
    whitelist_conf['pairlists'][0].update({'sort_key': 'asdf'})
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    with pytest.raises(
        OperationalException,
        match='key asdf not in .*'
    ):
        get_patched_freqtradebot(mocker, whitelist_conf)


def test_volumepairlist_caching(
    mocker: pytest_mock.MockerFixture,
    markets: Any,
    whitelist_conf: Dict[str, Any],
    tickers: Any
) -> None:
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    assert len(freqtrade.pairlists._pairlist_handlers[0]._pair_cache) == 0
    assert tickers.call_count == 0
    freqtrade.pairlists.refresh_pairlist()
    assert tickers.call_count == 1
    assert len(freqtrade.pairlists._pairlist_handlers[0]._pair_cache) == 1
    freqtrade.pairlists.refresh_pairlist()
    assert tickers.call_count == 1


def test_agefilter_min_days_listed_too_small(
    mocker: pytest_mock.MockerFixture,
    default_conf: Dict[str, Any],
    markets: Any,
    tickers: Any
) -> None:
    default_conf['pairlists'] = [
        {'method': 'VolumePairList', 'number_assets': 10},
        {'method': 'AgeFilter', 'min_days_listed': -1}
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers
    )
    with pytest.raises(
        OperationalException,
        match='AgeFilter requires min_days_listed to be >= 1'
    ):
        get_patched_freqtradebot(mocker, default_conf)


def test_agefilter_max_days_lower_than_min_days(
    mocker: pytest_mock.MockerFixture,
    default_conf: Dict[str, Any],
    markets: Any,
    tickers: Any
) -> None:
    default_conf['pairlists'] = [
        {'method': 'VolumePairList', 'number_assets': 10},
        {'method': 'AgeFilter', 'min_days_listed': 3, 'max_days_listed': 2}
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers
    )
    with pytest.raises(
        OperationalException,
        match='AgeFilter max_days_listed <= min_days_listed not permitted'
    ):
        get_patched_freqtradebot(mocker, default_conf)


def test_agefilter_min_days_listed_too_large(
    mocker: pytest_mock.MockerFixture,
    default_conf: Dict[str, Any],
    markets: Any,
    tickers: Any
) -> None:
    default_conf['pairlists'] = [
        {'method': 'VolumePairList', 'number_assets': 10},
        {'method': 'AgeFilter', 'min_days_listed': 99999}
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers
    )
    with pytest.raises(
        OperationalException,
        match='AgeFilter requires min_days_listed to not exceed exchange max request size \\([0-9]+\\)'
    ):
        get_patched_freqtradebot(mocker, default_conf)


def test_agefilter_caching(
    mocker: pytest_mock.MockerFixture,
    markets: Any,
    whitelist_conf_agefilter: Dict[str, Any],
    tickers: Any,
    ohlcv_history: pd.DataFrame
) -> None:
    with time_machine.travel('2021-09-01 05:00:00 +00:00') as t:
        ohlcv_data: Dict[Tuple[str, str, CandleType], pd.DataFrame] = {
            ('ETH/BTC', '1d', CandleType.SPOT): ohlcv_history,
            ('TKN/BTC', '1d', CandleType.SPOT): ohlcv_history,
            ('LTC/BTC', '1d', CandleType.SPOT): ohlcv_history
        }
        mocker.patch.multiple(
            EXMS,
            markets=PropertyMock(return_value=markets),
            exchange_has=MagicMock(return_value=True),
            get_tickers=tickers,
            refresh_latest_ohlcv=MagicMock(return_value=ohlcv_data)
        )
        freqtrade = get_patched_freqtradebot(mocker, whitelist_conf_agefilter)
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 0
        freqtrade.pairlists.refresh_pairlist()
        assert len(freqtrade.pairlists.whitelist) == 3
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count > 0
        freqtrade.pairlists.refresh_pairlist()
        assert len(freqtrade.pairlists.whitelist) == 3
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 2
        ohlcv_data = {
            ('ETH/BTC', '1d', CandleType.SPOT): ohlcv_history,
            ('TKN/BTC', '1d', CandleType.SPOT): ohlcv_history,
            ('LTC/BTC', '1d', CandleType.SPOT): ohlcv_history,
            ('XRP/BTC', '1d', CandleType.SPOT): ohlcv_history.iloc[[0]]
        }
        mocker.patch(f'{EXMS}.refresh_latest_ohlcv', return_value=ohlcv_data)
        freqtrade.pairlists.refresh_pairlist()
        assert len(freqtrade.pairlists.whitelist) == 3
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 1
        t.move_to('2021-09-03 01:00:00 +00:00')
        ohlcv_data = {
            ('ETH/BTC', '1d', CandleType.SPOT): ohlcv_history,
            ('TKN/BTC', '1d', CandleType.SPOT): ohlcv_history,
            ('LTC/BTC', '1d', CandleType.SPOT): ohlcv_history,
            ('XRP/BTC', '1d', CandleType.SPOT): ohlcv_history
        }
        mocker.patch(f'{EXMS}.refresh_latest_ohlcv', return_value=ohlcv_data)
        freqtrade.pairlists.refresh_pairlist()
        assert len(freqtrade.pairlists.whitelist) == 4
        assert freqtrade.exchange.refresh_latest_ohlcv.call_count == 1


def test_OffsetFilter_error(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any]
) -> None:
    whitelist_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        {'method': 'OffsetFilter', 'offset': -1}
    ]
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    with pytest.raises(
        OperationalException,
        match='OffsetFilter requires offset to be >= 0'
    ):
        PairListManager(MagicMock(), whitelist_conf)


def test_rangestabilityfilter_checks(
    mocker: pytest_mock.MockerFixture,
    default_conf: Dict[str, Any],
    markets: Any,
    tickers: Any
) -> None:
    default_conf['pairlists'] = [
        {'method': 'VolumePairList', 'number_assets': 10},
        {'method': 'RangeStabilityFilter', 'lookback_days': 99999}
    ]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers
    )
    with pytest.raises(
        OperationalException,
        match='RangeStabilityFilter requires lookback_days to not exceed exchange max request size \\([0-9]+\\)'
    ):
        get_patched_freqtradebot(mocker, default_conf)
    default_conf['pairlists'] = [
        {'method': 'VolumePairList', 'number_assets': 10},
        {'method': 'RangeStabilityFilter', 'lookback_days': 0}
    ]
    with pytest.raises(
        OperationalException,
        match='RangeStabilityFilter requires lookback_days to be >= 1'
    ):
        get_patched_freqtradebot(mocker, default_conf)
    default_conf['pairlists'] = [
        {'method': 'VolumePairList', 'number_assets': 10},
        {'method': 'RangeStabilityFilter', 'sort_direction': 'something'}
    ]
    with pytest.raises(
        OperationalException,
        match='RangeStabilityFilter requires sort_direction to be either None.*'
    ):
        get_patched_freqtradebot(mocker, default_conf)


@pytest.mark.parametrize(
    'pairlistconfig,desc_expected,exception_expected',
    [
        (
            {
                'method': 'PriceFilter',
                'low_price_ratio': 0.001,
                'min_price': 1e-07,
                'max_price': 1.0
            },
            "[{'PriceFilter': 'PriceFilter - Filtering pairs priced below 0.1% or below 0.00000010 or above 1.00000000.'}]",
            None
        ),
        (
            {
                'method': 'PriceFilter',
                'low_price_ratio': 0.001,
                'min_price': 1e-07
            },
            "[{'PriceFilter': 'PriceFilter - Filtering pairs priced below 0.1% or below 0.00000010.'}]",
            None
        ),
        (
            {
                'method': 'PriceFilter',
                'low_price_ratio': 0.001,
                'max_price': 1.0001
            },
            "[{'PriceFilter': 'PriceFilter - Filtering pairs priced below 0.1% or above 1.00010000.'}]",
            None
        ),
        (
            {
                'method': 'PriceFilter',
                'min_price': 2e-05
            },
            "[{'PriceFilter': 'PriceFilter - Filtering pairs priced below 0.00002000.'}]",
            None
        ),
        (
            {
                'method': 'PriceFilter',
                'max_value': 2e-05
            },
            "[{'PriceFilter': 'PriceFilter - Filtering pairs priced Value above 0.00002000.'}]",
            None
        ),
        (
            {
                'method': 'PriceFilter'
            },
            "[{'PriceFilter': 'PriceFilter - No price filters configured.'}]",
            None
        ),
        (
            {
                'method': 'PriceFilter',
                'low_price_ratio': -0.001
            },
            None,
            'PriceFilter requires low_price_ratio to be >= 0'
        ),
        (
            {
                'method': 'PriceFilter',
                'min_price': -1e-07
            },
            None,
            'PriceFilter requires min_price to be >= 0'
        ),
        (
            {
                'method': 'PriceFilter',
                'max_price': -1.0001
            },
            None,
            'PriceFilter requires max_price to be >= 0'
        ),
        (
            {
                'method': 'PriceFilter',
                'max_value': -1.0001
            },
            None,
            'PriceFilter requires max_value to be >= 0'
        ),
        (
            {
                'method': 'RangeStabilityFilter',
                'lookback_days': 10,
                'min_rate_of_change': 0.01
            },
            "[{'RangeStabilityFilter': 'RangeStabilityFilter - Filtering pairs with rate of change below 0.01 over the last days.'}]",
            None
        ),
        (
            {
                'method': 'RangeStabilityFilter',
                'lookback_days': 10,
                'min_rate_of_change': 0.01,
                'max_rate_of_change': 0.99
            },
            "[{'RangeStabilityFilter': 'RangeStabilityFilter - Filtering pairs with rate of change below 0.01 and above 0.99 over the last days.'}]",
            None
        ),
        (
            {
                'method': 'OffsetFilter',
                'offset': 5,
                'number_assets': 10
            },
            "[{'OffsetFilter': 'OffsetFilter - Taking 10 Pairs, starting from 5.'}]",
            None
        ),
        (
            {'method': 'ProducerPairList'},
            "[{'ProducerPairList': 'ProducerPairList - default'}]",
            None
        ),
        (
            {
                'method': 'RemotePairList',
                'number_assets': 10,
                'pairlist_url': 'https://example.com'
            },
            "[{'RemotePairList': 'RemotePairList - 10 pairs from RemotePairlist.'}]",
            None
        )
    ]
)
def test_pricefilter_desc(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    markets: Any,
    pairlistconfig: Dict[str, Any],
    desc_expected: str,
    exception_expected: Optional[str]
) -> None:
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True)
    )
    whitelist_conf['pairlists'] = [pairlistconfig]
    if desc_expected is not None:
        freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
        short_desc = str(freqtrade.pairlists.short_desc())
        assert short_desc == desc_expected
    else:
        with pytest.raises(
            OperationalException,
            match=exception_expected
        ):
            freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)


def test_pairlistmanager_no_pairlist(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any]
) -> None:
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    whitelist_conf['pairlists'] = []
    with pytest.raises(
        OperationalException,
        match='No Pairlist Handlers defined'
    ):
        get_patched_freqtradebot(mocker, whitelist_conf)


@pytest.mark.parametrize(
    'pairlists,pair_allowlist,overall_performance,allowlist_result',
    [
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC'],
            [
                {'pair': 'TKN/BTC', 'profit_ratio': 0.05, 'count': 3},
                {'pair': 'ETH/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['TKN/BTC', 'ETH/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC'],
            [
                {'pair': 'OTHER/BTC', 'profit_ratio': 0.05, 'count': 3},
                {'pair': 'ETH/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'ETH/BTC', 'profit_ratio': -0.05, 'count': 100},
                {'pair': 'TKN/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['TKN/BTC', 'LTC/BTC', 'ETH/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'LTC/BTC', 'profit_ratio': -0.0501, 'count': 101},
                {'pair': 'TKN/BTC', 'profit_ratio': -0.0501, 'count': 2},
                {'pair': 'ETH/BTC', 'profit_ratio': -0.0501, 'count': 100}
            ],
            ['TKN/BTC', 'ETH/BTC', 'LTC/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'LTC/BTC', 'profit_ratio': -0.0501, 'count': 1},
                {'pair': 'TKN/BTC', 'profit_ratio': -0.0501, 'count': 1},
                {'pair': 'ETH/BTC', 'profit_ratio': -0.0501, 'count': 1}
            ],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        )
    ]
)
def test_performance_filter(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    pairlists: List[Dict[str, Any]],
    pair_allowlist: List[str],
    overall_performance: List[Dict[str, Any]],
    allowlist_result: List[str],
    tickers: Any,
    markets: Any,
    ohlcv_history_list: List[pd.DataFrame]
) -> None:
    allowlist_conf = whitelist_conf
    allowlist_conf['pairlists'] = pairlists
    allowlist_conf['exchange']['pair_whitelist'] = pair_allowlist
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    frequencypatch_freqtradebot = get_patched_freqtradebot(mocker, allowlist_conf)
    freqtrade = frequencypatch_freqtradebot
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        markets=PropertyMock(return_value=markets)
    )
    mocker.patch.multiple(
        EXMS,
        get_historic_ohlcv=MagicMock(return_value=ohlcv_history_list)
    )
    mocker.patch.multiple(
        'freqtrade.persistence.Trade',
        get_overall_performance=MagicMock(return_value=overall_performance)
    )
    freqtrade.pairlists.refresh_pairlist()
    allowlist = freqtrade.pairlists.whitelist
    assert allowlist == allowlist_result


@pytest.mark.parametrize(
    'wildcardlist,pairs,expected',
    [
        (
            ['BTC/USDT'],
            ['BTC/USDT'],
            ['BTC/USDT']
        ),
        (
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT']
        ),
        (
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT'],
            ['BTC/USDT']
        ),
        (
            ['.*/USDT'],
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT']
        ),
        (
            ['.*C/USDT'],
            ['BTC/USDT', 'ETC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETC/USDT']
        ),
        (
            ['.*UP/USDT', 'BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETC/USDT', 'ETH/USDT', 'BTCUP/USDT', 'XRPUP/USDT', 'XRPDOWN/USDT'],
            ['BTC/USDT', 'ETH/USDT', 'BTCUP/USDT', 'XRPUP/USDT']
        ),
        (
            ['BTC/.*', 'ETH/.*'],
            ['BTC/USDT', 'ETC/USDT', 'ETH/USDT', 'BTC/USD', 'ETH/EUR', 'BTC/GBP'],
            ['BTC/USDT', 'ETH/USDT', 'BTC/USD', 'ETH/EUR', 'BTC/GBP']
        ),
        (
            ['*UP/USDT', 'BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETC/USDT', 'ETH/USDT', 'BTCUP/USDT', 'XRPUP/USDT', 'XRPDOWN/USDT'],
            None
        ),
        (
            ['BTC/USD'],
            ['BTC/USD', 'BTC/USDT'],
            ['BTC/USD']
        )
    ]
)
def test_expand_pairlist(
    wildcardlist: List[str],
    pairs: List[str],
    expected: Optional[List[str]]
) -> None:
    if expected is None:
        with pytest.raises(ValueError, match='Wildcard error in \\*UP/USDT,'):
            expand_pairlist(wildcardlist, pairs)
    else:
        assert sorted(expand_pairlist(wildcardlist, pairs)) == sorted(expected)
        conf: Dict[str, Any] = {
            'pairs': wildcardlist,
            'freqai': {
                'enabled': True,
                'feature_parameters': {
                    'include_corr_pairlist': ['BTC/USDT:USDT', 'XRP/BUSD']
                }
            }
        }
        assert sorted(dynamic_expand_pairlist(conf, pairs)) == sorted(expected + ['BTC/USDT:USDT', 'XRP/BUSD'])


@pytest.mark.parametrize(
    'wildcardlist,pairs,expected',
    [
        (
            ['BTC/USDT'],
            ['BTC/USDT'],
            ['BTC/USDT']
        ),
        (
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT']
        ),
        (
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT'],
            ['BTC/USDT', 'ETH/USDT']
        ),
        (
            ['HELLO/WORLD'],
            [],
            ['HELLO/WORLD']
        ),
        (
            ['BTC/USD'],
            ['BTC/USD', 'BTC/USDT'],
            ['BTC/USD']
        ),
        (
            ['BTC/USDT:USDT'],
            ['BTC/USDT:USDT', 'BTC/USDT'],
            ['BTC/USDT:USDT']
        ),
        (
            [
                'BB_BTC/USDT',
                'CC_BTC/USDT',
                'AA_ETH/USDT',
                'XRP/USDT',
                'ETH/USDT',
                'XX_BTC/USDT'
            ],
            ['BTC/USDT', 'ETH/USDT'],
            ['XRP/USDT', 'ETH/USDT']
        )
    ]
)
def test_expand_pairlist_keep_invalid(
    wildcardlist: List[str],
    pairs: List[str],
    expected: List[str]
) -> None:
    if expected is None:
        with pytest.raises(ValueError, match='Wildcard error in \\*UP/USDT,'):
            expand_pairlist(wildcardlist, pairs, keep_invalid=True)
    else:
        assert sorted(expand_pairlist(wildcardlist, pairs, keep_invalid=True)) == sorted(expected)


def test_ProducerPairlist_no_emc(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any]
) -> None:
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    whitelist_conf['pairlists'] = [
        {'method': 'ProducerPairList', 'number_assets': 10, 'producer_name': 'hello_world'}
    ]
    del whitelist_conf['external_message_consumer']
    with pytest.raises(
        OperationalException,
        match='ProducerPairList requires external_message_consumer to be enabled.'
    ):
        get_patched_freqtradebot(mocker, whitelist_conf)


def test_ProducerPairlist(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    markets: Any
) -> None:
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True)
    )
    whitelist_conf['pairlists'] = [
        {'method': 'ProducerPairList', 'number_assets': 2, 'producer_name': 'hello_world'}
    ]
    whitelist_conf.update({
        'external_message_consumer': {
            'enabled': True,
            'producers': [
                {
                    'name': 'hello_world',
                    'host': 'null',
                    'port': 9891,
                    'ws_token': 'dummy'
                }
            ]
        }
    })
    exchange = get_patched_exchange(mocker, whitelist_conf)
    dp = DataProvider(whitelist_conf, exchange, None)
    pairs = ['ETH/BTC', 'LTC/BTC', 'XRP/BTC']
    dp._set_producer_pairs(pairs + ['MEEP/USDT'], 'default')
    pm = PairListManager(exchange, whitelist_conf, dp)
    pm.refresh_pairlist()
    assert pm.whitelist == []
    dp._set_producer_pairs(pairs, 'hello_world')
    pm.refresh_pairlist()
    assert pm.whitelist == pairs[:2]
    assert len(pm.whitelist) == 2
    whitelist_conf['exchange']['pair_whitelist'] = ['TKN/BTC']
    whitelist_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        {'method': 'ProducerPairList', 'producer_name': 'hello_world'}
    ]
    pm = PairListManager(exchange, whitelist_conf, dp)
    pm.refresh_pairlist()
    assert len(pm.whitelist) == 4
    assert pm.whitelist == ['TKN/BTC'] + pairs


@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize(
    'pairlists,pair_allowlist,overall_performance,allowlist_result',
    [
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC'],
            [
                {'pair': 'TKN/BTC', 'profit_ratio': 0.05, 'count': 3},
                {'pair': 'ETH/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['TKN/BTC', 'ETH/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC'],
            [
                {'pair': 'OTHER/BTC', 'profit_ratio': 0.05, 'count': 3},
                {'pair': 'ETH/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'ETH/BTC', 'profit_ratio': -0.05, 'count': 100},
                {'pair': 'TKN/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['TKN/BTC', 'LTC/BTC', 'ETH/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'LTC/BTC', 'profit_ratio': -0.0501, 'count': 101},
                {'pair': 'TKN/BTC', 'profit_ratio': -0.0501, 'count': 2},
                {'pair': 'ETH/BTC', 'profit_ratio': -0.0501, 'count': 100}
            ],
            ['TKN/BTC', 'ETH/BTC', 'LTC/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'LTC/BTC', 'profit_ratio': -0.0501, 'count': 1},
                {'pair': 'TKN/BTC', 'profit_ratio': -0.0501, 'count': 1},
                {'pair': 'ETH/BTC', 'profit_ratio': -0.0501, 'count': 1}
            ],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        )
    ]
)
def test_performance_filter(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf_usdt: Dict[str, Any],
    pairlists: List[Dict[str, Any]],
    pair_allowlist: List[str],
    overall_performance: List[Dict[str, Any]],
    allowlist_result: List[str],
    tickers: Any,
    markets: Any,
    ohlcv_history_list: List[pd.DataFrame]
) -> None:
    allowlist_conf = whitelist_conf_usdt
    allowlist_conf['pairlists'] = pairlists
    allowlist_conf['exchange']['pair_whitelist'] = pair_allowlist
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    freqtrade = get_patched_freqtradebot(mocker, allowlist_conf)
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        markets=PropertyMock(return_value=markets)
    )
    mocker.patch.multiple(
        EXMS,
        get_historic_ohlcv=MagicMock(return_value=ohlcv_history_list)
    )
    mocker.patch.multiple(
        'freqtrade.persistence.Trade',
        get_overall_performance=MagicMock(return_value=overall_performance)
    )
    freqtrade.pairlists.refresh_pairlist()
    allowlist = freqtrade.pairlists.whitelist
    assert allowlist == allowlist_result


@pytest.mark.parametrize(
    'wildcardlist,pairs,expected',
    [
        (
            ['BTC/USDT'],
            ['BTC/USDT'],
            ['BTC/USDT']
        ),
        (
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT']
        ),
        (
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT'],
            ['BTC/USDT']
        ),
        (
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT']
        ),
        (
            ['.*/USDT'],
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT']
        ),
        (
            ['.*C/USDT'],
            ['BTC/USDT', 'ETC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETC/USDT']
        ),
        (
            ['.*UP/USDT', 'BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETC/USDT', 'ETH/USDT', 'BTCUP/USDT', 'XRPUP/USDT', 'XRPDOWN/USDT'],
            ['BTC/USDT', 'ETH/USDT', 'BTCUP/USDT', 'XRPUP/USDT']
        ),
        (
            ['BTC/.*', 'ETH/.*'],
            ['BTC/USDT', 'ETC/USDT', 'ETH/USDT', 'BTC/USD', 'ETH/EUR', 'BTC/GBP'],
            ['BTC/USDT', 'ETH/USDT', 'BTC/USD', 'ETH/EUR', 'BTC/GBP']
        ),
        (
            ['*UP/USDT', 'BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETC/USDT', 'ETH/USDT', 'BTCUP/USDT', 'XRPUP/USDT', 'XRPDOWN/USDT'],
            None
        ),
        (
            ['BTC/USD'],
            ['BTC/USD', 'BTC/USDT'],
            ['BTC/USD']
        ),
        (
            ['BTC/USDT:USDT'],
            ['BTC/USDT:USDT', 'BTC/USDT'],
            ['BTC/USDT:USDT']
        ),
        (
            [
                'BB_BTC/USDT',
                'CC_BTC/USDT',
                'AA_ETH/USDT',
                'XRP/USDT',
                'ETH/USDT',
                'XX_BTC/USDT'
            ],
            ['BTC/USDT', 'ETH/USDT'],
            ['XRP/USDT', 'ETH/USDT']
        )
    ]
)
def test_expand_pairlist_keep_invalid(
    wildcardlist: List[str],
    pairs: List[str],
    expected: List[str]
) -> None:
    if expected is None:
        with pytest.raises(ValueError, match='Wildcard error in \\*UP/USDT,'):
            expand_pairlist(wildcardlist, pairs, keep_invalid=True)
    else:
        assert sorted(expand_pairlist(wildcardlist, pairs, keep_invalid=True)) == sorted(expected)


def test_MarketCapPairList_filter(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    trade_mode: str,
    markets: Any,
    pairlists: List[Dict[str, Any]],
    result: List[str],
    coin_market_calls: Union[int, List[str]]
) -> None:
    test_value: List[Dict[str, Any]] = [
        {'symbol': 'btc'}, {'symbol': 'eth'}, {'symbol': 'usdt'},
        {'symbol': 'bnb'}, {'symbol': 'sol'}, {'symbol': 'xrp'},
        {'symbol': 'usdc'}, {'symbol': 'steth'}, {'symbol': 'ada'},
        {'symbol': 'avax'}
    ]
    default_conf_usdt['trading_mode'] = trade_mode
    if trade_mode == 'spot':
        default_conf_usdt['exchange']['pair_whitelist'].extend(['BTC/USDT', 'ETC/USDT', 'ADA/USDT'])
    default_conf_usdt['pairlists'] = pairlists
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True)
    )
    mocker.patch(
        'freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_categories_list',
        return_value=[{'category_id': 'layer-1'}, {'category_id': 'protocol'}, {'category_id': 'defi'}]
    )
    gcm_mock = mocker.patch(
        'freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_markets',
        return_value=test_value
    )
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()
    if isinstance(coin_market_calls, int):
        assert gcm_mock.call_count == coin_market_calls
    else:
        assert gcm_mock.call_count == len(coin_market_calls)
        for call in coin_market_calls:
            assert any(
                ('category' in c.kwargs and c.kwargs['category'] == call)
                for c in gcm_mock.call_args_list
            )
    assert pm.whitelist == result


def test_MarketCapPairList_timing(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    markets: Any,
    time_machine: time_machine.TimeMachine
) -> None:
    test_value = [
        {'symbol': 'btc'}, {'symbol': 'eth'}, {'symbol': 'usdt'},
        {'symbol': 'bnb'}, {'symbol': 'sol'}, {'symbol': 'xrp'},
        {'symbol': 'usdc'}, {'symbol': 'steth'}, {'symbol': 'ada'},
        {'symbol': 'avax'}
    ]
    default_conf_usdt['trading_mode'] = 'spot'
    default_conf_usdt['exchange']['pair_whitelist'].extend(['BTC/USDT', 'ETC/USDT', 'ADA/USDT'])
    default_conf_usdt['pairlists'] = [{'method': 'MarketCapPairList', 'number_assets': 2}]
    markets_mock = MagicMock(return_value=markets)
    mocker.patch.multiple(
        EXMS,
        get_markets=markets_mock,
        exchange_has=MagicMock(return_value=True)
    )
    mocker.patch(
        'freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_markets',
        return_value=test_value
    )
    start_dt = dt_now()
    time_machine.move_to(start_dt)
    pm = PairListManager(exchange, default_conf_usdt)
    mocker.patch.multiple(
        EXMS,
        get_markets=markets_mock,
        exchange_has=MagicMock(return_value=True)
    )
    assert pm.whitelist == ['ETH/BTC', 'BTC/USDT']
    pm.refresh_pairlist()
    assert markets_mock.call_count == 3
    pm.refresh_pairlist()
    assert markets_mock.call_count == 3
    time_machine.move_to(start_dt + timedelta(hours=20))
    pm.refresh_pairlist()
    assert markets_mock.call_count == 4
    time_machine.move_to(start_dt + timedelta(days=2))
    pm.refresh_pairlist()
    assert markets_mock.call_count == 5


def test_MarketCapPairList_filter_special_no_pair_from_coingecko(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    markets: Any
) -> None:
    default_conf_usdt['pairlists'] = [{'method': 'MarketCapPairList', 'number_assets': 2}]
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True)
    )
    gcm_mock = mocker.patch(
        'freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_markets',
        return_value=[]
    )
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()
    assert gcm_mock.call_count == 1
    assert pm.whitelist == []


def test_MarketCapPairList_exceptions(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    caplog: pytest.LogCaptureFixture
) -> None:
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    default_conf_usdt['pairlists'] = [
        {'method': 'MarketCapPairList'}
    ]
    with pytest.raises(
        OperationalException,
        match='`number_assets` not specified.*'
    ):
        PairListManager(exchange, default_conf_usdt)
    default_conf_usdt['pairlists'] = [
        {
            'method': 'MarketCapPairList',
            'number_assets': 20,
            'max_rank': 500
        }
    ]
    with caplog.at_level(logging.WARNING):
        PairListManager(exchange, default_conf_usdt)
    assert log_has_re('The max rank you have set \\(500\\) is quite high', caplog)
    mocker.patch(
        'freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_categories_list',
        return_value=[{'category_id': 'layer-1'}, {'category_id': 'protocol'}, {'category_id': 'defi'}]
    )
    default_conf_usdt['pairlists'] = [
        {
            'method': 'MarketCapPairList',
            'number_assets': 20,
            'categories': ['layer-1', 'defi', 'layer250']
        }
    ]
    with pytest.raises(
        OperationalException,
        match='Category layer250 not in coingecko category list.'
    ):
        PairListManager(exchange, default_conf_usdt)


@pytest.mark.parametrize(
    'pairlists,pair_allowlist,overall_performance,allowlist_result',
    [
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PerformanceFilter'}
            ],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PerformanceFilter'}
            ],
            ['ETH/BTC', 'TKN/BTC'],
            [
                {'pair': 'TKN/BTC', 'profit_ratio': 0.05, 'count': 3},
                {'pair': 'ETH/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['TKN/BTC', 'ETH/BTC']
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PerformanceFilter'}
            ],
            ['ETH/BTC', 'TKN/BTC'],
            [
                {'pair': 'OTHER/BTC', 'profit_ratio': 0.05, 'count': 3},
                {'pair': 'ETH/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PerformanceFilter'}
            ],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'ETH/BTC', 'profit_ratio': -0.05, 'count': 100},
                {'pair': 'TKN/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['TKN/BTC', 'LTC/BTC', 'ETH/BTC']
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PerformanceFilter'}
            ],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'LTC/BTC', 'profit_ratio': -0.0501, 'count': 101},
                {'pair': 'TKN/BTC', 'profit_ratio': -0.0501, 'count': 2},
                {'pair': 'ETH/BTC', 'profit_ratio': -0.0501, 'count': 100}
            ],
            ['TKN/BTC', 'ETH/BTC', 'LTC/BTC']
        ),
        (
            [
                {'method': 'StaticPairList'},
                {'method': 'PerformanceFilter'}
            ],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'LTC/BTC', 'profit_ratio': -0.0501, 'count': 1},
                {'pair': 'TKN/BTC', 'profit_ratio': -0.0501, 'count': 1},
                {'pair': 'ETH/BTC', 'profit_ratio': -0.0501, 'count': 1}
            ],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        )
    ]
)
def test_performance_filter(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    pairlists: List[Dict[str, Any]],
    pair_allowlist: List[str],
    overall_performance: List[Dict[str, Any]],
    allowlist_result: List[str],
    tickers: Any,
    markets: Any,
    ohlcv_history_list: List[pd.DataFrame]
) -> None:
    allowlist_conf = default_conf_usdt
    allowlist_conf['pairlists'] = pairlists
    allowlist_conf['exchange']['pair_whitelist'] = pair_allowlist
    mocker.patch(
        f'{EXMS}.exchange_has',
        MagicMock(return_value=True)
    )
    freqtrade = get_patched_freqtradebot(mocker, allowlist_conf)
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        markets=PropertyMock(return_value=markets)
    )
    mocker.patch.multiple(
        EXMS,
        get_historic_ohlcv=MagicMock(return_value=ohlcv_history_list)
    )
    mocker.patch.multiple(
        'freqtrade.persistence.Trade',
        get_overall_performance=MagicMock(return_value=overall_performance)
    )
    freqtrade.pairlists.refresh_pairlist()
    allowlist = freqtrade.pairlists.whitelist
    assert allowlist == allowlist_result


@pytest.mark.parametrize(
    'pairlists,pair_allowlist,overall_performance,allowlist_result',
    [
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC'],
            [
                {'pair': 'TKN/BTC', 'profit_ratio': 0.05, 'count': 3},
                {'pair': 'ETH/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['TKN/BTC', 'ETH/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC'],
            [
                {'pair': 'OTHER/BTC', 'profit_ratio': 0.05, 'count': 3},
                {'pair': 'ETH/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['ETH/BTC', 'TKN/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'ETH/BTC', 'profit_ratio': -0.05, 'count': 100},
                {'pair': 'TKN/BTC', 'profit_ratio': 0.04, 'count': 2}
            ],
            ['TKN/BTC', 'LTC/BTC', 'ETH/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'LTC/BTC', 'profit_ratio': -0.0501, 'count': 101},
                {'pair': 'TKN/BTC', 'profit_ratio': -0.0501, 'count': 2},
                {'pair': 'ETH/BTC', 'profit_ratio': -0.0501, 'count': 100}
            ],
            ['TKN/BTC', 'ETH/BTC', 'LTC/BTC']
        ),
        (
            [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC'],
            [
                {'pair': 'LTC/BTC', 'profit_ratio': -0.0501, 'count': 1},
                {'pair': 'TKN/BTC', 'profit_ratio': -0.0501, 'count': 1},
                {'pair': 'ETH/BTC', 'profit_ratio': -0.0501, 'count': 1}
            ],
            ['ETH/BTC', 'TKN/BTC', 'LTC/BTC']
        )
    ]
)
def test_performance_filter(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    pairlists: List[Dict[str, Any]],
    pair_allowlist: List[str],
    overall_performance: List[Dict[str, Any]],
    allowlist_result: List[str],
    tickers: Any,
    markets: Any,
    ohlcv_history_list: List[pd.DataFrame]
) -> None:
    allowlist_conf = default_conf_usdt
    allowlist_conf['pairlists'] = pairlists
    allowlist_conf['exchange']['pair_whitelist'] = pair_allowlist
    mocker.patch(
        f'{EXMS}.exchange_has',
        MagicMock(return_value=True)
    )
    freqtrade = get_patched_freqtradebot(mocker, allowlist_conf)
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        markets=PropertyMock(return_value=markets)
    )
    mocker.patch.multiple(
        EXMS,
        get_historic_ohlcv=MagicMock(return_value=ohlcv_history_list)
    )
    mocker.patch.multiple(
        'freqtrade.persistence.Trade',
        get_overall_performance=MagicMock(return_value=overall_performance)
    )
    freqtrade.pairlists.refresh_pairlist()
    allowlist = freqtrade.pairlists.whitelist
    assert allowlist == allowlist_result


@pytest.mark.parametrize(
    'wildcardlist,pairs,expected',
    [
        (['BTC/USDT'], ['BTC/USDT'], ['BTC/USDT']),
        (
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT', 'ETH/USDT']
        ),
        (
            ['BTC/USDT', 'ETH/USDT'],
            ['BTC/USDT'],
            ['BTC/USDT']
        ),
        (
            ['HELLO/WORLD'],
            [],
            ['HELLO/WORLD']
        ),
        (
            ['BTC/USD'],
            ['BTC/USD', 'BTC/USDT'],
            ['BTC/USD']
        ),
        (
            ['BTC/USDT:USDT'],
            ['BTC/USDT:USDT', 'BTC/USDT'],
            ['BTC/USDT:USDT']
        ),
        (
            [
                'BB_BTC/USDT',
                'CC_BTC/USDT',
                'AA_ETH/USDT',
                'XRP/USDT',
                'ETH/USDT',
                'XX_BTC/USDT'
            ],
            ['BTC/USDT', 'ETH/USDT'],
            ['XRP/USDT', 'ETH/USDT']
        )
    ]
)
def test_expand_pairlist_keep_invalid(
    wildcardlist: List[str],
    pairs: List[str],
    expected: List[str]
) -> None:
    if expected is None:
        with pytest.raises(ValueError, match='Wildcard error in \\*UP/USDT,'):
            expand_pairlist(wildcardlist, pairs, keep_invalid=True)
    else:
        assert sorted(expand_pairlist(wildcardlist, pairs, keep_invalid=True)) == sorted(expected)


def test_ProducerPairlist_no_emc(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any]
) -> None:
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    whitelist_conf['pairlists'] = [
        {'method': 'ProducerPairList', 'number_assets': 10, 'producer_name': 'hello_world'}
    ]
    del whitelist_conf['external_message_consumer']
    with pytest.raises(
        OperationalException,
        match='ProducerPairList requires external_message_consumer to be enabled.'
    ):
        get_patched_freqtradebot(mocker, whitelist_conf)


def test_ProducerPairlist(
    mocker: pytest_mock.MockerFixture,
    whitelist_conf: Dict[str, Any],
    markets: Any
) -> None:
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True)
    )
    whitelist_conf['pairlists'] = [
        {'method': 'ProducerPairList', 'number_assets': 2, 'producer_name': 'hello_world'}
    ]
    whitelist_conf.update({
        'external_message_consumer': {
            'enabled': True,
            'producers': [
                {
                    'name': 'hello_world',
                    'host': 'null',
                    'port': 9891,
                    'ws_token': 'dummy'
                }
            ]
        }
    })
    exchange = get_patched_exchange(mocker, whitelist_conf)
    dp = DataProvider(whitelist_conf, exchange, None)
    pairs = ['ETH/BTC', 'LTC/BTC', 'XRP/BTC']
    dp._set_producer_pairs(pairs + ['MEEP/USDT'], 'default')
    pm = PairListManager(exchange, whitelist_conf, dp)
    pm.refresh_pairlist()
    assert pm.whitelist == []
    dp._set_producer_pairs(pairs, 'hello_world')
    pm.refresh_pairlist()
    assert pm.whitelist == pairs[:2]
    assert len(pm.whitelist) == 2
    whitelist_conf['exchange']['pair_whitelist'] = ['TKN/BTC']
    whitelist_conf['pairlists'] = [
        {'method': 'StaticPairList'},
        {'method': 'ProducerPairList', 'producer_name': 'hello_world'}
    ]
    pm = PairListManager(exchange, whitelist_conf, dp)
    pm.refresh_pairlist()
    assert len(pm.whitelist) == 4
    assert pm.whitelist == ['TKN/BTC'] + pairs


@pytest.mark.usefixtures('init_persistence')
def test_FullTradesFilter(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    fee: float,
    caplog: pytest.LogCaptureFixture
) -> None:
    default_conf_usdt['exchange']['pair_whitelist'].extend(['ADA/USDT', 'XRP/USDT', 'ETC/USDT'])
    default_conf_usdt['pairlists'] = [
        {'method': 'StaticPairList'},
        {'method': 'FullTradesFilter'}
    ]
    default_conf_usdt['max_open_trades'] = -1
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    exchange = get_patched_exchange(mocker, default_conf_usdt)
    pm = PairListManager(exchange, default_conf_usdt)
    pm.refresh_pairlist()
    assert pm.whitelist == ['ETH/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT']
    with time_machine.travel('2021-09-01 05:00:00 +00:00') as t:
        create_mock_trades_usdt(fee)
        pm.refresh_pairlist()
        pm.refresh_pairlist()
        assert pm.whitelist == ['ETH/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT']
        default_conf_usdt['max_open_trades'] = 4
        pm.refresh_pairlist()
        assert pm.whitelist == []
        assert log_has_re('Whitelist with 0 pairs: \\[]', caplog)
        list_trades = LocalTrade.get_open_trades()
        assert len(list_trades) == 4
        t.move_to('2021-09-01 07:00:00 +00:00')
        list_trades[2].close(12)
        Trade.commit()
        list_trades = LocalTrade.get_open_trades()
        assert len(list_trades) == 3
        pm.refresh_pairlist()
        assert pm.whitelist == ['ETH/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT']
        default_conf_usdt['max_open_trades'] = 3
        pm.refresh_pairlist()
        assert pm.whitelist == []
        assert log_has_re('Whitelist with 0 pairs: \\[]', caplog)


@pytest.mark.parametrize(
    'pairlists,pair_allowlist,exchange,volumefilter_result,coin_market_calls',
    [
        (
            [
                {'method': 'StaticPairList', 'allow_inactive': True},
                {'method': 'MarketCapPairList', 'number_assets': 2}
            ],
            ['BTC/USDT', 'ETH/USDT'],
            'spot',
            ['BTC/USDT', 'ETH/USDT'],
            1
        ),
        (
            [
                {'method': 'StaticPairList', 'allow_inactive': True},
                {'method': 'MarketCapPairList', 'number_assets': 6}
            ],
            ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT'],
            'spot',
            ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT'],
            1
        ),
        (
            [
                {'method': 'StaticPairList', 'allow_inactive': True},
                {
                    'method': 'MarketCapPairList',
                    'max_rank': 6,
                    'number_assets': 3
                }
            ],
            ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'],
            'spot',
            ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'],
            1
        ),
        (
            [
                {'method': 'StaticPairList', 'allow_inactive': True},
                {
                    'method': 'MarketCapPairList',
                    'max_rank': 8,
                    'number_assets': 4
                }
            ],
            ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'],
            'spot',
            ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'],
            1
        ),
        (
            [{'method': 'MarketCapPairList', 'number_assets': 5}],
            ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADAHALF/USDT', 'ADADOUBLE/USDT'],
            'spot',
            ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADAHALF/USDT', 'ADADOUBLE/USDT'],
            1
        ),
        (
            [
                {'method': 'MarketCapPairList', 'max_rank': 2, 'number_assets': 5}
            ],
            ['BTC/USDT', 'ETH/USDT'],
            'spot',
            ['BTC/USDT', 'ETH/USDT'],
            1
        ),
        (
            [
                {'method': 'MarketCapPairList', 'max_rank': 2, 'number_assets': 5}
            ],
            ['BTC/USDT', 'ETH/USDT'],
            'futures',
            ['ETH/USDT:USDT'],
            1
        ),
        (
            [
                {'method': 'MarketCapPairList', 'number_assets': 2}
            ],
            ['BTC/USDT', 'ETH/USDT', 'NANO/USDT', 'ADAHALF/USDT', 'ADADOUBLE/USDT'],
            'futures',
            ['ETH/USDT:USDT', 'ADA/USDT:USDT'],
            1
        ),
        (
            [
                {'method': 'MarketCapPairList', 'number_assets': 2, 'categories': ['layer-1']}
            ],
            ['BTC/USDT', 'ETH/USDT', 'NANO/USDT', 'ADAHALF/USDT', 'ADADOUBLE/USDT'],
            'futures',
            ['ETH/USDT:USDT', 'ADA/USDT:USDT'],
            ['layer-1']
        ),
        (
            [
                {'method': 'MarketCapPairList', 'number_assets': 2, 'categories': ['layer-1', 'protocol']}
            ],
            ['BTC/USDT', 'ETH/USDT', 'NANO/USDT', 'ADAHALF/USDT', 'ADADOUBLE/USDT'],
            'futures',
            ['ETH/USDT:USDT', 'ADA/USDT:USDT'],
            ['layer-1', 'protocol']
        )
    ]
)
def test_MarketCapPairList_filter(
    mocker: pytest_mock.MockerFixture,
    default_conf_usdt: Dict[str, Any],
    pairlists: List[Dict[str, Any]],
    pair_allowlist: List[str],
    overall_performance: List[Dict[str, Any]],
    allowlist_result: List[str],
    exchange: str,
    coin_market_calls: Union[int, List[str]]
) -> None:
    default_conf_usdt['trading_mode'] = 'spot'
    default_conf_usdt['exchange']['pair_whitelist'].extend(['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
    default_conf_usdt['pairlists'] = pairlists
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True)
    )
    mocker.patch(
        'freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_categories_list',
        return_value=[{'category_id': 'layer-1'}, {'category_id': 'protocol'}, {'category_id': 'defi'}]
    )
    gcm_mock = mocker.patch(
        'freqtrade.plugins.pairlist.MarketCapPairList.FtCoinGeckoApi.get_coins_markets',
        return_value=[{'symbol': 'btc'}, {'symbol': 'eth'}, {'symbol': 'xrp'}, {'symbol': 'ada'}]
    )
    exchange_mock = get_patched_exchange(mocker, default_conf_usdt)
    pm = PairListManager(exchange_mock, default_conf_usdt, MagicMock())
    pm.refresh_pairlist()
    if isinstance(coin_market_calls, int):
        assert gcm_mock.call_count == coin_market_calls
    else:
        assert gcm_mock.call_count == len(coin_market_calls)
        for call in coin_market_calls:
            assert any(
                ('category' in c.kwargs and c.kwargs['category'] == call)
                for c in gcm_mock.call_args_list
            )
    assert pm.whitelist == allowlist_result


@pytest.mark.parametrize(
    'any_pairlist_type,add_vote_list_allow_true_or_false',
    [
        ('foo/bar', True),
        ('foo/bar', False),
        ('', True),
        ('', False)
    ]
)
def test_AddPairHash(
    mocker: pytest_mock.MockerFixture,
    any_pairlist_type: Any,
    add_vote_list_allow_true_or_false: Any
) -> None:
    pass  # This function is not present in the original code and seems unrelated.
