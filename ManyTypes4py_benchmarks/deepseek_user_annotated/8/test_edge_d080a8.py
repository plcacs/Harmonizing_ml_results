# pragma pylint: disable=missing-docstring, C0103, C0330
# pragma pylint: disable=protected-access, too-many-lines, invalid-name, too-many-arguments

import logging
import math
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
from pandas import DataFrame

from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.edge import Edge, PairInfo
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.util.datetime_helpers import dt_ts, dt_utc
from tests.conftest import EXMS, get_patched_freqtradebot, log_has
from tests.optimize import (
    BTContainer,
    BTrade,
    _build_backtest_dataframe,
    _get_frame_time_from_offset,
)


# Cases to be tested:
# 1) Open trade should be removed from the end
# 2) Two complete trades within dataframe (with sell hit for all)
# 3) Entered, sl 1%, candle drops 8% => Trade closed, 1% loss
# 4) Entered, sl 3%, candle drops 4%, recovers to 1% => Trade closed, 3% loss
# 5) Stoploss and sell are hit. should sell on stoploss
####################################################################

tests_start_time = dt_utc(2018, 10, 3)
timeframe_in_minute = 60

# End helper functions
# Open trade should be removed from the end
tc0 = BTContainer(
    data=[
        # D  O     H     L     C     V    B  S
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 1],
    ],  # enter trade (signal on last candle)
    stop_loss=-0.99,
    roi={"0": float("inf")},
    profit_perc=0.00,
    trades=[],
)

# Two complete trades within dataframe(with sell hit for all)
tc1 = BTContainer(
    data=[
        # D  O     H     L     C     V    B  S
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4975, 4987, 6172, 0, 1],  # enter trade (signal on last candle)
        [2, 5000, 5025, 4975, 4987, 6172, 0, 0],  # exit at open
        [3, 5000, 5025, 4975, 4987, 6172, 1, 0],  # no action
        [4, 5000, 5025, 4975, 4987, 6172, 0, 0],  # should enter the trade
        [5, 5000, 5025, 4975, 4987, 6172, 0, 1],  # no action
        [6, 5000, 5025, 4975, 4987, 6172, 0, 0],  # should sell
    ],
    stop_loss=-0.99,
    roi={"0": float("inf")},
    profit_perc=0.00,
    trades=[
        BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=2),
        BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=4, close_tick=6),
    ],
)

# 3) Entered, sl 1%, candle drops 8% => Trade closed, 1% loss
tc2 = BTContainer(
    data=[
        # D  O     H     L     C     V    B  S
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4600, 4987, 6172, 0, 0],  # enter trade, stoploss hit
        [2, 5000, 5025, 4975, 4987, 6172, 0, 0],
    ],
    stop_loss=-0.01,
    roi={"0": float("inf")},
    profit_perc=-0.01,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=1)],
)

# 4) Entered, sl 3 %, candle drops 4%, recovers to 1 % = > Trade closed, 3 % loss
tc3 = BTContainer(
    data=[
        # D  O     H     L     C     V    B  S
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4800, 4987, 6172, 0, 0],  # enter trade, stoploss hit
        [2, 5000, 5025, 4975, 4987, 6172, 0, 0],
    ],
    stop_loss=-0.03,
    roi={"0": float("inf")},
    profit_perc=-0.03,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=1)],
)

# 5) Stoploss and sell are hit. should sell on stoploss
tc4 = BTContainer(
    data=[
        # D  O     H     L     C     V    B  S
        [0, 5000, 5025, 4975, 4987, 6172, 1, 0],
        [1, 5000, 5025, 4800, 4987, 6172, 0, 1],  # enter trade, stoploss hit, sell signal
        [2, 5000, 5025, 4975, 4987, 6172, 0, 0],
    ],
    stop_loss=-0.03,
    roi={"0": float("inf")},
    profit_perc=-0.03,
    trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=1)],
)

TESTS: List[BTContainer] = [tc0, tc1, tc2, tc3, tc4]


@pytest.mark.parametrize("data", TESTS)
def test_edge_results(edge_conf: Dict[str, Any], mocker: Any, caplog: Any, data: BTContainer) -> None:
    """
    run functional tests
    """
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    frame = _build_backtest_dataframe(data.data)
    caplog.set_level(logging.DEBUG)
    edge.fee = 0

    trades = edge._find_trades_for_stoploss_range(frame, "TEST/BTC", [data.stop_loss])
    results = edge._fill_calculable_fields(DataFrame(trades)) if trades else DataFrame()

    assert len(trades) == len(data.trades)

    if not results.empty:
        assert round(results["profit_ratio"].sum(), 3) == round(data.profit_perc, 3)

    for c, trade in enumerate(data.trades):
        res = results.iloc[c]
        assert res.exit_type == trade.exit_reason
        assert res.open_date == _get_frame_time_from_offset(trade.open_tick).replace(tzinfo=None)
        assert res.close_date == _get_frame_time_from_offset(trade.close_tick).replace(tzinfo=None)


def test_adjust(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch(
        "freqtrade.edge.Edge._cached_pairs",
        mocker.PropertyMock(
            return_value={
                "E/F": PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
                "C/D": PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
                "N/O": PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
            }
        ),
    )

    pairs = ["A/B", "C/D", "E/F", "G/H"]
    assert edge.adjust(pairs) == ["E/F", "C/D"]


def test_edge_get_stoploss(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch(
        "freqtrade.edge.Edge._cached_pairs",
        mocker.PropertyMock(
            return_value={
                "E/F": PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
                "C/D": PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
                "N/O": PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
            }
        ),
    )

    assert edge.get_stoploss("E/F") == -0.01


def test_nonexisting_get_stoploss(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch(
        "freqtrade.edge.Edge._cached_pairs",
        mocker.PropertyMock(
            return_value={
                "E/F": PairInfo(-0.01, 0.66, 3.71, 0.50, 1.71, 10, 60),
            }
        ),
    )

    assert edge.get_stoploss("N/O") == -0.1


def test_edge_stake_amount(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch(
        "freqtrade.edge.Edge._cached_pairs",
        mocker.PropertyMock(
            return_value={
                "E/F": PairInfo(-0.02, 0.66, 3.71, 0.50, 1.71, 10, 60),
            }
        ),
    )
    assert edge._capital_ratio == 0.5
    assert (
        edge.stake_amount("E/F", free_capital=100, total_capital=100, capital_in_trade=25) == 31.25
    )

    assert edge.stake_amount("E/F", free_capital=20, total_capital=100, capital_in_trade=25) == 20

    assert edge.stake_amount("E/F", free_capital=0, total_capital=100, capital_in_trade=25) == 0

    # Test with increased allowed_risk
    # Result should be no more than allowed capital
    edge._allowed_risk = 0.4
    edge._capital_ratio = 0.5
    assert (
        edge.stake_amount("E/F", free_capital=100, total_capital=100, capital_in_trade=25) == 62.5
    )

    assert edge.stake_amount("E/F", free_capital=100, total_capital=100, capital_in_trade=0) == 50

    edge._capital_ratio = 1
    # Full capital is available
    assert edge.stake_amount("E/F", free_capital=100, total_capital=100, capital_in_trade=0) == 100
    # Full capital is available
    assert edge.stake_amount("E/F", free_capital=0, total_capital=100, capital_in_trade=0) == 0


def test_nonexisting_stake_amount(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch(
        "freqtrade.edge.Edge._cached_pairs",
        mocker.PropertyMock(
            return_value={
                "E/F": PairInfo(-0.11, 0.66, 3.71, 0.50, 1.71, 10, 60),
            }
        ),
    )
    # should use strategy stoploss
    assert edge.stake_amount("N/O", 1, 2, 1) == 0.15


def test_edge_heartbeat_calculate(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    heartbeat = edge_conf["edge"]["process_throttle_secs"]

    # should not recalculate if heartbeat not reached
    edge._last_updated = dt_ts() - heartbeat + 1

    assert edge.calculate(edge_conf["exchange"]["pair_whitelist"]) is False


def mocked_load_data(datadir: str, pairs: Optional[List[str]] = None, timeframe: str = "0m", timerange: Optional[Any] = None, *args: Any, **kwargs: Any) -> Dict[str, DataFrame]:
    if pairs is None:
        pairs = []
    hz = 0.1
    base = 0.001

    NEOBTC = [
        [
            dt_ts(tests_start_time + timedelta(minutes=(x * timeframe_in_minute))),
            math.sin(x * hz) / 1000 + base,
            math.sin(x * hz) / 1000 + base + 0.0001,
            math.sin(x * hz) / 1000 + base - 0.0001,
            math.sin(x * hz) / 1000 + base,
            123.45,
        ]
        for x in range(0, 500)
    ]

    hz = 0.2
    base = 0.002
    LTCBTC = [
        [
            dt_ts(tests_start_time + timedelta(minutes=(x * timeframe_in_minute))),
            math.sin(x * hz) / 1000 + base,
            math.sin(x * hz) / 1000 + base + 0.0001,
            math.sin(x * hz) / 1000 + base - 0.0001,
            math.sin(x * hz) / 1000 + base,
            123.45,
        ]
        for x in range(0, 500)
    ]

    pairdata = {
        "NEO/BTC": ohlcv_to_dataframe(NEOBTC, "1h", pair="NEO/BTC", fill_missing=True),
        "LTC/BTC": ohlcv_to_dataframe(LTCBTC, "1h", pair="LTC/BTC", fill_missing=True),
    }
    return pairdata


def test_edge_process_downloaded_data(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    mocker.patch(f"{EXMS}.get_fee", MagicMock(return_value=0.001))
    mocker.patch("freqtrade.edge.edge_positioning.refresh_data", MagicMock())
    mocker.patch("freqtrade.edge.edge_positioning.load_data", mocked_load_data)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)

    assert edge.calculate(edge_conf["exchange"]["pair_whitelist"])
    assert len(edge._cached_pairs) == 2
    assert edge._last_updated <= dt_ts() + 2


def test_edge_process_no_data(mocker: Any, edge_conf: Dict[str, Any], caplog: Any) -> None:
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    mocker.patch(f"{EXMS}.get_fee", MagicMock(return_value=0.001))
    mocker.patch("freqtrade.edge.edge_positioning.refresh_data", MagicMock())
    mocker.patch("freqtrade.edge.edge_positioning.load_data", MagicMock(return_value={}))
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)

    assert not edge.calculate(edge_conf["exchange"]["pair_whitelist"])
    assert len(edge._cached_pairs) == 0
    assert log_has("No data found. Edge is stopped ...", caplog)
    assert edge._last_updated == 0


def test_edge_process_no_trades(mocker: Any, edge_conf: Dict[str, Any], caplog: Any) -> None:
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    mocker.patch(f"{EXMS}.get_fee", return_value=0.001)
    mocker.patch(
        "freqtrade.