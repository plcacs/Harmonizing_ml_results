# pragma pylint: disable=missing-docstring,C0103,protected-access

import logging
import time
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, List, Optional, Union
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


# Exclude RemotePairList from tests.
# It has a mandatory parameter, and requires special handling, which happens in test_remotepairlist.
TESTABLE_PAIRLISTS: List[str] = [p for p in AVAILABLE_PAIRLISTS if p not in ["RemotePairList"]]


@pytest.fixture(scope="function")
def whitelist_conf(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    default_conf["runmode"] = "dry_run"
    default_conf["stake_currency"] = "BTC"
    default_conf["exchange"]["pair_whitelist"] = [
        "ETH/BTC",
        "TKN/BTC",
        "TRST/BTC",
        "SWT/BTC",
        "BCC/BTC",
        "HOT/BTC",
    ]
    default_conf["exchange"]["pair_blacklist"] = ["BLK/BTC"]
    default_conf["pairlists"] = [
        {
            "method": "VolumePairList",
            "number_assets": 5,
            "sort_key": "quoteVolume",
        },
    ]
    default_conf.update(
        {
            "external_message_consumer": {
                "enabled": True,
                "producers": [],
            }
        }
    )
    return default_conf


@pytest.fixture(scope="function")
def whitelist_conf_2(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    default_conf["runmode"] = "dry_run"
    default_conf["stake_currency"] = "BTC"
    default_conf["exchange"]["pair_whitelist"] = [
        "ETH/BTC",
        "TKN/BTC",
        "BLK/BTC",
        "LTC/BTC",
        "BTT/BTC",
        "HOT/BTC",
        "FUEL/BTC",
        "XRP/BTC",
    ]
    default_conf["exchange"]["pair_blacklist"] = ["BLK/BTC"]
    default_conf["pairlists"] = [
        # {   "method": "StaticPairList"},
        {
            "method": "VolumePairList",
            "number_assets": 5,
            "sort_key": "quoteVolume",
            "refresh_period": 0,
        },
    ]
    return default_conf


@pytest.fixture(scope="function")
def whitelist_conf_agefilter(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    default_conf["runmode"] = "dry_run"
    default_conf["stake_currency"] = "BTC"
    default_conf["exchange"]["pair_whitelist"] = [
        "ETH/BTC",
        "TKN/BTC",
        "BLK/BTC",
        "LTC/BTC",
        "BTT/BTC",
        "HOT/BTC",
        "FUEL/BTC",
        "XRP/BTC",
    ]
    default_conf["exchange"]["pair_blacklist"] = ["BLK/BTC"]
    default_conf["pairlists"] = [
        {
            "method": "VolumePairList",
            "number_assets": 5,
            "sort_key": "quoteVolume",
            "refresh_period": -1,
        },
        {"method": "AgeFilter", "min_days_listed": 2, "max_days_listed": 100},
    ]
    return default_conf


@pytest.fixture(scope="function")
def static_pl_conf(whitelist_conf: Dict[str, Any]) -> Dict[str, Any]:
    whitelist_conf["pairlists"] = [
        {
            "method": "StaticPairList",
        },
    ]
    return whitelist_conf


def test_log_cached(mocker: Any, static_pl_conf: Dict[str, Any], markets: Any, tickers: Any) -> None:
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=markets),
        exchange_has=MagicMock(return_value=True),
        get_tickers=tickers,
    )
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    logmock = MagicMock()
    # Assign starting whitelist
    pl = freqtrade.pairlists._pairlist_handlers[0]
    pl.log_once("Hello world", logmock)
    assert logmock.call_count == 1
    pl.log_once("Hello world", logmock)
    assert logmock.call_count == 1
    assert pl._log_cache.currsize == 1
    assert ("Hello world",) in pl._log_cache._Cache__data

    pl.log_once("Hello world2", logmock)
    assert logmock.call_count == 2
    assert pl._log_cache.currsize == 2


def test_load_pairlist_noexist(mocker: Any, markets: Any, default_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    plm = PairListManager(freqtrade.exchange, default_conf, MagicMock())
    with pytest.raises(
        OperationalException,
        match=r"Impossible to load Pairlist 'NonexistingPairList'. "
        r"This class does not exist or contains Python code errors.",
    ):
        PairListResolver.load_pairlist(
            "NonexistingPairList", freqtrade.exchange, plm, default_conf, {}, 1
        )


def test_load_pairlist_verify_multi(mocker: Any, markets_static: Any, default_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets_static))
    plm = PairListManager(freqtrade.exchange, default_conf, MagicMock())
    # Call different versions one after the other, should always consider what was passed in
    # and have no side-effects (therefore the same check multiple times)
    assert plm.verify_whitelist(["ETH/BTC", "XRP/BTC"], print) == ["ETH/BTC", "XRP/BTC"]
    assert plm.verify_whitelist(["ETH/BTC", "XRP/BTC", "BUUU/BTC"], print) == ["ETH/BTC", "XRP/BTC"]
    assert plm.verify_whitelist(["XRP/BTC", "BUUU/BTC"], print) == ["XRP/BTC"]
    assert plm.verify_whitelist(["ETH/BTC", "XRP/BTC"], print) == ["ETH/BTC", "XRP/BTC"]
    assert plm.verify_whitelist(["ETH/USDT", "XRP/USDT"], print) == ["ETH/USDT"]
    assert plm.verify_whitelist(["ETH/BTC", "XRP/BTC"], print) == ["ETH/BTC", "XRP/BTC"]


def test_refresh_market_pair_not_in_whitelist(mocker: Any, markets: Any, static_pl_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)

    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    freqtrade.pairlists.refresh_pairlist()
    # List ordered by BaseVolume
    whitelist = ["ETH/BTC", "TKN/BTC"]
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    # Ensure config dict hasn't been changed
    assert (
        static_pl_conf["exchange"]["pair_whitelist"]
        == freqtrade.config["exchange"]["pair_whitelist"]
    )


def test_refresh_static_pairlist(mocker: Any, markets: Any, static_pl_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtrade.pairlists.refresh_pairlist()
    # List ordered by BaseVolume
    whitelist = ["ETH/BTC", "TKN/BTC"]
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf["exchange"]["pair_blacklist"] == freqtrade.pairlists.blacklist


@pytest.mark.parametrize(
    "pairs,expected",
    [
        (
            ["NOEXIST/BTC", r"\+WHAT/BTC"],
            ["ETH/BTC", "TKN/BTC", "TRST/BTC", "NOEXIST/BTC", "SWT/BTC", "BCC/BTC", "HOT/BTC"],
        ),
        (
            ["NOEXIST/BTC", r"*/BTC"],  # This is an invalid regex
            [],
        ),
    ],
)
def test_refresh_static_pairlist_noexist(
    mocker: Any, markets: Any, static_pl_conf: Dict[str, Any], pairs: List[str], expected: List[str], caplog: Any
) -> None:
    static_pl_conf["pairlists"][0]["allow_inactive"] = True
    static_pl_conf["exchange"]["pair_whitelist"] += pairs
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtrade.pairlists.refresh_pairlist()

    # Ensure all except those in whitelist are removed
    assert set(expected) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf["exchange"]["pair_blacklist"] == freqtrade.pairlists.blacklist
    if not expected:
        assert log_has_re(r"Pair whitelist contains an invalid Wildcard: Wildcard error.*", caplog)


def test_invalid_blacklist(mocker: Any, markets: Any, static_pl_conf: Dict[str, Any], caplog: Any) -> None:
    static_pl_conf["exchange"]["pair_blacklist"] = ["*/BTC"]
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtrade.pairlists.refresh_pairlist()
    whitelist = []
    # Ensure all except those in whitelist are removed
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf["exchange"]["pair_blacklist"] == freqtrade.pairlists.blacklist
    log_has_re(r"Pair blacklist contains an invalid Wildcard.*", caplog)


def test_remove_logs_for_pairs_already_in_blacklist(
    mocker: Any, markets: Any, static_pl_conf: Dict[str, Any], caplog: Any
) -> None:
    logger = logging.getLogger(__name__)
    freqtrade = get_patched_freqtradebot(mocker, static_pl_conf)
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
        markets=PropertyMock(return_value=markets),
    )
    freqtrade.pairlists.refresh_pairlist()
    whitelist = ["ETH/BTC", "TKN/BTC"]
    caplog.clear()
    caplog.set_level(logging.INFO)

    # Ensure all except those in whitelist are removed.
    assert set(whitelist) == set(freqtrade.pairlists.whitelist)
    assert static_pl_conf["exchange"]["pair_blacklist"] == freqtrade.pairlists.blacklist
    # Ensure that log message wasn't generated.
    assert not log_has("Pair BLK/BTC in your blacklist. Removing it from whitelist...", caplog)

    for _ in range(3):
        new_whitelist = freqtrade.pairlists.verify_blacklist(
            whitelist + ["BLK/BTC"], logger.warning
        )
        # Ensure that the pair is removed from the white list, and properly logged.
        assert set(whitelist) == set(new_whitelist)
    assert num_log_has("Pair BLK/BTC in your blacklist. Removing it from whitelist...", caplog) == 1


def test_refresh_pairlist_dynamic(
    mocker: Any, shitcoinmarkets: Any, tickers: Any, whitelist_conf: Dict[str, Any]
) -> None:
    mocker.patch.multiple(
        EXMS,
        get_tickers=tickers,
        exchange_has=MagicMock(return_value=True),
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    # Remock markets with shitcoinmarkets since get_patched_freqtradebot uses the markets fixture
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=shitcoinmarkets),
    )
    # argument: use the whitelist dynamically by exchange-volume
    whitelist = ["ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC", "HOT/BTC"]
    freqtrade.pairlists.refresh_pairlist()
    assert whitelist == freqtrade.pairlists.whitelist

    whitelist_conf["pairlists"] = [{"method": "VolumePairList"}]
    with pytest.raises(
        OperationalException,
        match=r"`number_assets` not specified. Please check your configuration "
        r'for "pairlist.config.number_assets"',
    ):
        PairListManager(freqtrade.exchange, whitelist_conf, MagicMock())


def test_refresh_pairlist_dynamic_2(
    mocker: Any, shitcoinmarkets: Any, tickers: Any, whitelist_conf_2: Dict[str, Any]
) -> None:
    tickers_dict = tickers()

    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
    )
    # Remove caching of ticker data to emulate changing volume by the time of second call
    mocker.patch.multiple(
        "freqtrade.plugins.pairlistmanager.PairListManager",
        _get_cached_tickers=MagicMock(return_value=tickers_dict),
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf_2)
    # Remock markets with shitcoinmarkets since get_patched_freqtradebot uses the markets fixture
    mocker.patch.multiple(
        EXMS,
        markets=PropertyMock(return_value=shitcoinmarkets),
    )

    whitelist = ["ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC", "HOT/BTC"]
    freqtrade.pairlists.refresh_pairlist()
    assert whitelist == freqtrade.pairlists.whitelist

    # Delay to allow 0 TTL cache to expire...
    time.sleep(1)
    whitelist = ["FUEL/BTC", "ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC"]
    tickers_dict["FUEL/BTC"]["quoteVolume"] = 10000.0
    freqtrade.pairlists.refresh_pairlist()
    assert whitelist == freqtrade.pairlists.whitelist


def test_VolumePairList_refresh_empty(
    mocker: Any, markets_empty: Any, whitelist_conf: Dict[str, Any]
) -> None:
    mocker.patch.multiple(
        EXMS,
        exchange_has=MagicMock(return_value=True),
    )
    freqtrade = get_patched_freqtradebot(mocker, whitelist_conf)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets_empty))

    # argument: use the whitelist dynamically by exchange-volume
    whitelist = []
    whitelist_conf["exchange"]["pair_whitelist"] = []
    freqtrade.pairlists.refresh_pairlist()
    pairslist = whitelist_conf["exchange"]["pair_whitelist"]

    assert set(whitelist) == set(pairslist)


@pytest.mark.parametrize(
    "pairlists,base_currency,whitelist_result",
    [
        # VolumePairList only
        (
            [{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"}],
            "BTC",
            ["ETH/BTC", "TKN/BTC", "LTC/BTC", "XRP/BTC", "HOT/BTC"],
        ),
        (
            [{"method": "VolumePairList", "number_assets": 5, "sort_key": "quoteVolume"}],
            "USDT",
            ["ETH/USDT", "NANO/USDT", "ADAHALF/USDT", "ADADOUBLE/USDT"],
        ),
        # No pair for ETH, VolumePairList
        ([