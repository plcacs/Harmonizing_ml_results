# pragma pylint: disable=missing-docstring, C0103
# pragma pylint: disable=protected-access, too-many-lines, invalid-name, too-many-arguments

import logging
import time
from copy import deepcopy
from datetime import timedelta
from unittest.mock import ANY, MagicMock, PropertyMock, patch
from typing import Any, Optional, List, Tuple, Dict

import pytest
from pandas import DataFrame
from sqlalchemy import select

from freqtrade.constants import CANCEL_REASON, UNLIMITED_STAKE_AMOUNT
from freqtrade.enums import (
    CandleType,
    ExitCheckTuple,
    ExitType,
    RPCMessageType,
    RunMode,
    SignalDirection,
    State,
)
from freqtrade.exceptions import (
    DependencyException,
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderException,
    OperationalException,
    PricingError,
    TemporaryError,
)
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, PairLocks, Trade
from freqtrade.plugins.protections.iprotection import ProtectionReturn
from freqtrade.util.datetime_helpers import dt_now, dt_utc
from freqtrade.worker import Worker
from tests.conftest import (
    EXMS,
    create_mock_trades,
    create_mock_trades_usdt,
    get_patched_freqtradebot,
    get_patched_worker,
    log_has,
    log_has_re,
    patch_edge,
    patch_exchange,
    patch_get_signal,
    patch_wallet,
    patch_whitelist,
)
from tests.conftest_trades import (
    MOCK_TRADE_COUNT,
    entry_side,
    exit_side,
    mock_order_2,
    mock_order_2_sell,
    mock_order_3,
    mock_order_3_sell,
    mock_order_4,
    mock_order_5_stoploss,
    mock_order_6_sell,
)
from tests.conftest_trades_usdt import mock_trade_usdt_4


def patch_RPCManager(mocker: Any) -> MagicMock:
    """
    This function mock RPC manager to avoid repeating this code in almost every tests
    :param mocker: mocker to patch RPCManager class
    :return: RPCManager.send_msg MagicMock to track if this method is called
    """
    mocker.patch("freqtrade.rpc.telegram.Telegram", MagicMock())
    rpc_mock = mocker.patch("freqtrade.freqtradebot.RPCManager.send_msg", MagicMock())
    return rpc_mock


# Unit tests


def test_freqtradebot_state(mocker: Any, default_conf_usdt: Dict[str, Any], markets: Any) -> None:
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value=markets))
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    assert freqtrade.state is State.RUNNING

    default_conf_usdt.pop("initial_state")
    freqtrade = FreqtradeBot(default_conf_usdt)
    assert freqtrade.state is State.STOPPED


def test_process_stopped(mocker: Any, default_conf_usdt: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    coo_mock = mocker.patch("freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders")
    freqtrade.process_stopped()
    assert coo_mock.call_count == 0

    default_conf_usdt["cancel_open_orders_on_exit"] = True
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.process_stopped()
    assert coo_mock.call_count == 1


def test_process_calls_sendmsg(mocker: Any, default_conf_usdt: Dict[str, Any]) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.process()
    assert freqtrade.rpc.process_msg_queue.call_count == 1


def test_bot_cleanup(mocker: Any, default_conf_usdt: Dict[str, Any], caplog: Any) -> None:
    mock_cleanup = mocker.patch("freqtrade.freqtradebot.Trade.commit")
    coo_mock = mocker.patch("freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders")
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.cleanup()
    assert log_has("Cleaning up modules ...", caplog)
    assert mock_cleanup.call_count == 1
    assert coo_mock.call_count == 0

    freqtrade.config["cancel_open_orders_on_exit"] = True
    freqtrade.cleanup()
    assert coo_mock.call_count == 1


def test_bot_cleanup_db_errors(mocker: Any, default_conf_usdt: Dict[str, Any], caplog: Any) -> None:
    mocker.patch("freqtrade.freqtradebot.Trade.commit", side_effect=OperationalException())
    mocker.patch(
        "freqtrade.freqtradebot.FreqtradeBot.check_for_open_trades",
        side_effect=OperationalException(),
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.emc = MagicMock()
    freqtrade.emc.shutdown = MagicMock()
    freqtrade.cleanup()
    assert freqtrade.emc.shutdown.call_count == 1


@pytest.mark.parametrize("runmode", [RunMode.DRY_RUN, RunMode.LIVE])
def test_order_dict(runmode: RunMode, default_conf_usdt: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf["runmode"] = runmode
    conf["order_types"] = {
        "entry": "market",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": True,
    }
    conf["entry_pricing"]["price_side"] = "ask"

    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re(r".*stoploss_on_exchange .* dry-run", caplog)
    assert freqtrade.strategy.order_types["stoploss_on_exchange"]

    caplog.clear()
    # is left untouched
    conf = default_conf_usdt.copy()
    conf["runmode"] = runmode
    conf["order_types"] = {
        "entry": "market",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types["stoploss_on_exchange"]
    assert not log_has_re(r".*stoploss_on_exchange .* dry-run", caplog)


def test_get_trade_stake_amount(default_conf_usdt: Dict[str, Any], mocker: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)

    freqtrade = FreqtradeBot(default_conf_usdt)

    result = freqtrade.wallets.get_trade_stake_amount("ETH/USDT", 1)
    assert result == default_conf_usdt["stake_amount"]


@pytest.mark.parametrize("runmode", [RunMode.DRY_RUN, RunMode.LIVE])
def test_load_strategy_no_keys(runmode: RunMode, default_conf_usdt: Dict[str, Any], mocker: Any, caplog: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = deepcopy(default_conf_usdt)
    conf["runmode"] = runmode
    erm = mocker.patch("freqtrade.freqtradebot.ExchangeResolver.load_exchange")

    freqtrade = FreqtradeBot(conf)
    strategy_config = freqtrade.strategy.config
    assert id(strategy_config["exchange"]) == id(conf["exchange"])
    # Keys have been removed and are not passed to the exchange
    assert strategy_config["exchange"]["key"] == ""
    assert strategy_config["exchange"]["secret"] == ""

    assert erm.call_count == 1
    ex_conf = erm.call_args_list[0][1]["exchange_config"]
    assert id(ex_conf) != id(conf["exchange"])
    # Keys are still present
    assert ex_conf["key"] != ""
    assert ex_conf["key"] == default_conf_usdt["exchange"]["key"]
    assert ex_conf["secret"] != ""
    assert ex_conf["secret"] == default_conf_usdt["exchange"]["secret"]


@pytest.mark.parametrize(
    "amend_last,wallet,max_open,lsamr,expected",
    [
        (False, 120, 2, 0.5, [60, None]),
        (True, 120, 2, 0.5, [60, 58.8]),
        (False, 180, 3, 0.5, [60, 60, None]),
        (True, 180, 3, 0.5, [60, 60, 58.2]),
        (False, 122, 3, 0.5, [60, 60, None]),
        (True, 122, 3, 0.5, [60, 60, 0.0]),
        (True, 167, 3, 0.5, [60, 60, 45.33]),
        (True, 122, 3, 1, [60, 60, 0.0]),
    ],
)
def test_check_available_stake_amount(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    mocker: Any,
    fee: Any,
    limit_buy_order_usdt_open: Dict[str, Any],
    amend_last: bool,
    wallet: float,
    max_open: int,
    lsamr: float,
    expected: List[Optional[float]],
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    default_conf_usdt["dry_run_wallet"] = wallet

    default_conf_usdt["amend_last_stake_amount"] = amend_last
    default_conf_usdt["last_stake_amount_min_ratio"] = lsamr

    freqtrade = FreqtradeBot(default_conf_usdt)

    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open["id"] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount("ETH/USDT", 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry("ETH/USDT", result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount("ETH/USDT", 1)


def test_edge_called_in_process(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)

    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ["NEO/BTC", "LTC/BTC"]


def test_edge_overrides_stake_amount(mocker: Any, edge_conf: Dict[str, Any]) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf["dry_run_wallet"] = 999.9
    freqtrade = FreqtradeBot(edge_conf)

    assert (
        freqtrade.wallets.get_trade_stake_amount("NEO/BTC", 1, freqtrade.edge)
        == (999.9 * 0.5 * 0.01) / 0.20
    )
    assert (
        freqtrade.wallets.get_trade_stake_amount("LTC/BTC", 1, freqtrade.edge)
        == (999.9 * 0.5 * 0.01) / 0.21
    )


@pytest.mark.parametrize(
    "buy_price_mult,ignore_strat_sl",
    [
        (0.79, False),  # Override stoploss
        (0.85, True),  # Override strategy stoploss
    ],
)
def test_edge_overrides_stoploss(
    mocker: Any,
    limit_order: Dict[str, Any],
    fee: Any,
    caplog: Any,
    buy_price_mult: float,
    ignore_strat_sl: bool,
    edge_conf: Dict[str, Any],
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf["max_open_trades"] = float("inf")

    # Strategy stoploss is -0.1 but Edge imposes a stoploss at -0.2
    # Thus, if price falls 21%, stoploss should be triggered
    #
    # mocking the ticker: price is falling ...
    enter_price = limit_order["buy"]["price"]
    ticker_val = {
        "bid": enter_price,
        "ask": enter_price,
        "last": enter_price,
    }
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value=ticker_val),
        get_fee=fee,
    )
    #############################################

    # Create a trade with "limit_buy_order_usdt" price
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ["NEO/BTC"]
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = False
    assert trade is not None
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.open_rate == enter_price
    assert trade.amount == 30.0

    caplog.clear()
    #############################################
    ticker_val.update(
        {
            "bid": enter_price * buy_price_mult,
            "ask": enter_price * buy_price_mult,
            "last": enter_price * buy_price_mult,
        }
    )

    # stoploss should be hit
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has_re("Exit for NEO/BTC detected. Reason: stop_loss.*", caplog)
        assert trade.exit_reason == ExitType.STOP_LOSS.value
        # Test compatibility ...
        assert trade.sell_reason == ExitType.STOP_LOSS.value


def test_total_open_trades_stakes(mocker: Any, default_conf_usdt: Dict[str, Any], ticker_usdt: Any, fee: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt["max_open_trades"] = 2
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(return_value=False),
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()

    assert trade is not None
    assert trade.stake_amount == 60.0
    assert trade.is_open
    assert trade.open_date is not None

    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade).order_by(Trade.id.desc())).first()

    assert trade is not None
    assert trade.stake_amount == 60.0
    assert trade.is_open
    assert trade.open_date is not None

    assert Trade.total_open_trades_stakes() == 120.0


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize(
    "stake_amount,create,amount_enough,max_open_trades",
    [
        (5.0, True, True, 99),
        (0.042, True, False, 99),  # Amount will be adjusted to min - which is 0.051
        (0, False, True, 99),
        (UNLIMITED_STAKE_AMOUNT, False, True, 0),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    limit_order_open: Dict[str, Any],
    fee: Any,
    mocker: Any,
    stake_amount: float,
    create: bool,
    amount_enough: bool,
    max_open_trades: int,
    caplog: Any,
    is_short: bool,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    enter_mock = MagicMock(return_value=limit_order_open[entry_side(is_short)])
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=enter_mock,
        get_fee=fee,
    )
    default_conf_usdt["max_open_trades"] = max_open_trades
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config["stake_amount"] = stake_amount
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    if create:
        assert freqtrade.create_trade("ETH/USDT")
        if amount_enough:
            rate, amount = enter_mock.call_args[1]["rate"], enter_mock.call_args[1]["amount"]
            assert rate * amount <= default_conf_usdt["stake_amount"]
        else:
            assert log_has_re(r"Stake amount for pair .* is too small.*", caplog)
    else:
        assert not freqtrade.create_trade("ETH/USDT")
        if not max_open_trades:
            assert (
                freqtrade.wallets.get_trade_stake_amount(
                    "ETH/USDT", default_conf_usdt["max_open_trades"], freqtrade.edge
                )
                == 0
            )


def test_create_trade_no_stake_amount(default_conf_usdt: Dict[str, Any], ticker_usdt: Any, fee: Any, mocker: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt["stake_amount"] * 0.5)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    with pytest.raises(DependencyException, match=r".*stake amount.*"):
        freqtrade.create_trade("ETH/USDT")


@pytest.mark.parametrize(
    "whitelist,positions",
    [
        (["ETH/USDT"], 1),  # No pairs left
        ([], 0),  # No pairs in whitelist
    ],
)
def test_enter_positions_no_pairs_left(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    limit_buy_order_usdt_open: Dict[str, Any],
    fee: Any,
    whitelist: List[str],
    positions: int,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    mocker.patch("freqtrade.configuration.config_validation._validate_whitelist")
    default_conf_usdt["exchange"]["pair_whitelist"] = whitelist
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    n = freqtrade.enter_positions()
    assert n == positions
    if positions:
        assert not log_has_re(r"No currency pair in active pair whitelist.*", caplog)
        n = freqtrade.enter_positions()
        assert n == 0
        assert log_has_re(r"No currency pair in active pair whitelist.*", caplog)
    else:
        assert n == 0
        assert log_has("Active pair whitelist is empty.", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_enter_positions_global_pairlock(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    limit_buy_order_usdt: Dict[str, Any],
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={"id": limit_buy_order_usdt["id"]}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = r"Global pairlock active until.* Not creating new trades."
    n = freqtrade.enter_positions()
    # 0 trades, but it's not because of pairlock.
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()

    PairLocks.lock_pair("*", dt_now() + timedelta(minutes=20), "Just because", side="*")
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_protections(mocker: Any, default_conf_usdt: Dict[str, Any], fee: Any, is_short: bool) -> None:
    default_conf_usdt["_strategy_protections"] = [
        {"method": "CooldownPeriod", "stop_duration": 60},
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 24,
            "trade_limit": 4,
            "stop_duration_candles": 4,
            "only_per_pair": False,
        },
    ]

    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.protections._protection_handlers[1].global_stop = MagicMock(
        return_value=ProtectionReturn(True, dt_now() + timedelta(hours=1), "asdf")
    )
    create_mock_trades(fee, is_short=is_short)
    freqtrade.handle_protections("ETC/BTC", "*")
    send_msg_mock = freqtrade.rpc.send_msg
    assert send_msg_mock.call_count == 2
    assert send_msg_mock.call_args_list[0][0][0]["type"] == RPCMessageType.PROTECTION_TRIGGER
    assert send_msg_mock.call_args_list[1][0][0]["type"] == RPCMessageType.PROTECTION_TRIGGER_GLOBAL


def test_create_trade_no_signal(default_conf_usdt: Dict[str, Any], fee: Any, mocker: Any) -> None:
    default_conf_usdt["dry_run"] = True

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        get_fee=fee,
    )
    default_conf_usdt["stake_amount"] = 10
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)

    assert not freqtrade.create_trade("ETH/USDT")


@pytest.mark.parametrize("max_open", range(0, 5))
@pytest.mark.parametrize("tradable_balance_ratio,modifier", [(1.0, 1), (0.99, 0.8), (0.5, 0.5)])
def test_create_trades_multiple_trades(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: Dict[str, Any],
    max_open: int,
    tradable_balance_ratio: float,
    modifier: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt["max_open_trades"] = max_open
    default_conf_usdt["tradable_balance_ratio"] = tradable_balance_ratio
    default_conf_usdt["dry_run_wallet"] = 60.0 * max_open

    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    n = freqtrade.enter_positions()
    trades = Trade.get_open_trades()
    # Expected trades should be max_open * a modified value
    # depending on the configured tradable_balance
    expected_trades = max(int(max_open * modifier), 0)
    assert n == expected_trades
    assert len(trades) == expected_trades


def test_create_trades_preopen(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: Dict[str, Any],
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt["max_open_trades"] = 4
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    # Create 2 existing trades
    freqtrade.execute_entry("ETH/USDT", default_conf_usdt["stake_amount"])
    freqtrade.execute_entry("NEO/BTC", default_conf_usdt["stake_amount"])

    assert len(Trade.get_open_trades()) == 2
    # Change order_id for new orders
    limit_buy_order_usdt_open["id"] = "123444"

    # Create 2 new trades using create_trades
    assert freqtrade.create_trade("ETH/USDT")
    assert freqtrade.create_trade("NEO/BTC")

    trades = Trade.get_open_trades()
    assert len(trades) == 4


@pytest.mark.parametrize("is_short", [False, True])
def test_process_trade_creation(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    limit_order: Dict[str, Any],
    limit_order_open: Dict[str, Any],
    is_short: bool,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    ticker_side = "ask" if is_short else "bid"
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        fetch_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    trades = Trade.get_open_trades()
    assert not trades

    freqtrade.process()

    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == default_conf_usdt["stake_amount"]
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == "binance"
    assert trade.open_rate == ticker_usdt.return_value[ticker_side]
    # Trade opens with 0 amount. Only trade filling will set the amount
    assert pytest.approx(trade.amount) == 0
    assert pytest.approx(trade.amount_requested) == 60 / ticker_usdt.return_value[ticker_side]

    assert log_has(
        f"{'Short' if is_short else 'Long'} signal found: about create a new trade for ETH/USDT "
        "with stake_amount: 60.0 ...",
        caplog,
    )
    mocker.patch("freqtrade.freqtradebot.FreqtradeBot._check_and_execute_exit")

    # Fill trade.
    freqtrade.process()
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == "binance"
    assert trade.open_rate == limit_order[entry_side(is_short)]["price"]
    # Filled trade has amount set to filled order amount
    assert pytest.approx(trade.amount) == limit_order[entry_side(is_short)]["filled"]


def test_process_exchange_failures(default_conf_usdt: Dict[str, Any], ticker_usdt: Any, mocker: Any) -> None:
    # TODO: Move this test to test_worker
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        reload_markets=MagicMock(),
        create_order=MagicMock(side_effect=TemporaryError),
    )
    sleep_mock = mocker.patch("time.sleep")

    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)
    mocker.patch(f"{EXMS}.reload_markets", MagicMock(side_effect=TemporaryError))

    worker._process_running()
    assert sleep_mock.called is True


def test_process_operational_exception(default_conf_usdt: Dict[str, Any], ticker_usdt: Any, mocker: Any) -> None:
    msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(side_effect=OperationalException)
    )
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)

    assert worker.freqtrade.state == State.RUNNING

    worker._process_running()
    assert worker.freqtrade.state == State.STOPPED
    assert "OperationalException" in msg_mock.call_args_list[-1][0][0]["status"]


def test_process_trade_handling(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    limit_buy_order_usdt_open: Dict[str, Any],
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        fetch_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)

    trades = Trade.get_open_trades()
    assert not trades
    freqtrade.process()

    trades = Trade.get_open_trades()
    assert len(trades) == 1

    # Nothing happened ...
    freqtrade.process()
    assert len(trades) == 1


def test_process_trade_no_whitelist_pair(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    limit_buy_order_usdt: Dict[str, Any],
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    """Test process with trade not in pair list"""
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={"id": limit_buy_order_usdt["id"]}),
        fetch_order=MagicMock(return_value=limit_buy_order_usdt),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    pair = "BLK/BTC"
    # Ensure the pair is not in the whitelist!
    assert pair not in default_conf_usdt["exchange"]["pair_whitelist"]

    # create open trade not in whitelist
    Trade.session.add(
        Trade(
            pair=pair,
            stake_amount=0.001,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            is_open=True,
            amount=20,
            open_rate=0.01,
            exchange="binance",
            is_short=False,
            leverage=1,
        )
    )
    Trade.session.add(
        Trade(
            pair="ETH/USDT",
            stake_amount=0.001,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            is_open=True,
            amount=12,
            open_rate=0.001,
            exchange="binance",
            is_short=False,
            leverage=1,
        )
    )
    Trade.commit()

    assert pair not in freqtrade.active_pair_whitelist
    freqtrade.process()
    assert pair in freqtrade.active_pair_whitelist
    # Make sure each pair is only in the list once
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))


def test_process_informative_pairs_added(default_conf_usdt: Dict[str, Any], ticker_usdt: Any, mocker: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)

    refresh_mock = MagicMock()
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=TemporaryError),
        refresh_latest_ohlcv=refresh_mock,
    )
    inf_pairs = MagicMock(
        return_value=[("BTC/ETH", "1m", CandleType.SPOT), ("ETH/USDT", "1h", CandleType.SPOT)]
    )
    mocker.patch.multiple(
        "freqtrade.strategy.interface.IStrategy",
        get_exit_signal=MagicMock(return_value=(False, False)),
        get_entry_signal=MagicMock(return_value=(None, None)),
    )
    mocker.patch("time.sleep", return_value=None)

    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.informative_pairs = inf_pairs
    # patch_get_signal(freqtrade)

    freqtrade.process()
    assert inf_pairs.call_count == 1
    assert refresh_mock.call_count == 1
    assert ("BTC/ETH", "1m", CandleType.SPOT) in refresh_mock.call_args[0][0]
    assert ("ETH/USDT", "1h", CandleType.SPOT) in refresh_mock.call_args[0][0]
    assert ("ETH/USDT", default_conf_usdt["timeframe"], CandleType.SPOT) in refresh_mock.call_args[0][0]


@pytest.mark.parametrize(
    "is_short,trading_mode,exchange_name,margin_mode,liq_buffer,liq_price",
    [
        (False, "spot", "binance", None, 0.0, None),
        (True, "spot", "binance", None, 0.0, None),
        (False, "spot", "gate", None, 0.0, None),
        (True, "spot", "gate", None, 0.0, None),
        (False, "spot", "okx", None, 0.0, None),
        (True, "spot", "okx", None, 0.0, None),
        (True, "futures", "binance", "isolated", 0.0, 11.88151815181518),
        (False, "futures", "binance", "isolated", 0.0, 8.080471380471382),
        (True, "futures", "gate", "isolated", 0.0, 11.87413417771621),
        (False, "futures", "gate", "isolated", 0.0, 8.085708510208207),
        (True, "futures", "binance", "isolated", 0.05, 11.7874422442244),
        (False, "futures", "binance", "isolated", 0.05, 8.17644781144781),
        (True, "futures", "gate", "isolated", 0.05, 11.7804274688304),
        (False, "futures", "gate", "isolated", 0.05, 8.181423084697796),
        (True, "futures", "okx", "isolated", 0.0, 11.87413417771621),
        (False, "futures", "okx", "isolated", 0.0, 8.085708510208207),
        (True, "futures", "bybit", "isolated", 0.0, 11.9),
        (False, "futures", "bybit", "isolated", 0.0, 8.1),
    ],
)
def test_execute_entry(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    limit_order: Dict[str, Any],
    limit_order_open: Dict[str, Any],
    is_short: bool,
    trading_mode: str,
    exchange_name: str,
    margin_mode: Optional[str],
    liq_buffer: float,
    liq_price: Optional[float],
) -> None:
    """
    exchange_name = binance, is_short = true
        leverage = 5
        position = 0.2 * 5
        ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
        ((2 + 0.01) - ((-1) * 1 * 10)) / ((1 * 0.01) - ((-1) * 1)) = 11.89108910891089

    exchange_name = binance, is_short = false
        ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
        ((2 + 0.01) - (1 * 1 * 10)) / ((1 * 0.01) - (1 * 1)) = 8.070707070707071

    exchange_name = gate/okx, is_short = true
        (open_rate + (wallet_balance / position)) / (1 + (mm_ratio + taker_fee_rate))
        (10 + (2 / 1)) / (1 + (0.01 + 0.0006)) = 11.87413417771621

    exchange_name = gate/okx, is_short = false
        (open_rate - (wallet_balance / position)) / (1 - (mm_ratio + taker_fee_rate))
        (10 - (2 / 1)) / (1 - (0.01 + 0.0006)) = 8.085708510208207
    """
    # TODO: Split this test into multiple tests to improve readability
    # SETUP
    open_order = limit_order_open[entry_side(is_short)]
    order = limit_order[entry_side(is_short)]
    default_conf_usdt["trading_mode"] = trading_mode
    default_conf_usdt["liquidation_buffer"] = liq_buffer
    leverage = 1.0 if trading_mode == "spot" else 5.0
    default_conf_usdt["exchange"]["name"] = exchange_name
    if margin_mode:
        default_conf_usdt["margin_mode"] = margin_mode
    mocker.patch("freqtrade.exchange.gate.Gate.validate_ordertypes")
    patch_RPCManager(mocker)
    patch_exchange(mocker, exchange=exchange_name)
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    stake_amount = 2
    bid = 0.11
    enter_rate_mock = MagicMock(return_value=bid)
    enter_mm = MagicMock(return_value=open_order)
    mocker.patch.multiple(
        EXMS,
        get_rate=enter_rate_mock,
        fetch_ticker=MagicMock(return_value={"bid": 1.9, "ask": 2.2, "last": 1.9}),
        create_order=enter_mm,
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_max_pair_stake_amount=MagicMock(return_value=500000),
        get_fee=fee,
        get_funding_fees=MagicMock(return_value=0),
        name=exchange_name,
        get_maintenance_ratio_and_amt=MagicMock(return_value=(0.01, 0.01)),
        get_max_leverage=MagicMock(return_value=10),
    )
    mocker.patch.multiple(
        "freqtrade.exchange.okx.Okx",
        get_max_pair_stake_amount=MagicMock(return_value=500000),
    )
    pair = "ETH/USDT"

    assert not freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    assert enter_rate_mock.call_count == 1
    assert enter_mm.call_count == 0
    assert freqtrade.strategy.confirm_trade_entry.call_count == 1

    enter_rate_mock.reset_mock()

    open_order["id"] = "22"
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_entry(pair, stake_amount)
    assert enter_rate_mock.call_count == 2
    assert enter_mm.call_count == 1
    call_args = enter_mm.call_args_list[0][1]
    assert call_args["pair"] == pair
    assert call_args["rate"] == bid
    assert pytest.approx(call_args["amount"]) == round(stake_amount / bid * leverage, 8)
    enter_rate_mock.reset_mock()

    # Should create an open trade with an open order id
    # As the order is not fulfilled yet
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert not trade.has_open_orders
    assert trade.open_rate == 11.0
    assert trade.stake_amount == 22.0

    # Test calling with price
    open_order["id"] = "33"
    fix_price = 0.06
    assert freqtrade.execute_entry(pair, stake_amount, fix_price, is_short=is_short)
    # Make sure get_rate wasn't called again
    assert enter_rate_mock.call_count == 1

    assert enter_mm.call_count == 2
    call_args = enter_mm.call_args_list[1][1]
    assert call_args["pair"] == pair
    assert call_args["rate"] == fix_price
    assert pytest.approx(call_args["amount"]) == round(stake_amount / fix_price * leverage, 8)

    # In case of closed order
    order["status"] = "closed"
    order["average"] = 10
    order["cost"] = 300
    order["id"] = "444"

    mocker.patch(f"{EXMS}.create_order", MagicMock(return_value=order))
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[2]
    trade.is_short = is_short
    assert trade is not None
    assert not trade.has_open_orders
    assert trade.open_rate == 10
    assert trade.stake_amount == round(order["average"] * order["filled"] / leverage, 8)
    assert pytest.approx(trade.liquidation_price) == liq_price

    # In case of rejected or expired order and partially filled
    order["status"] = "expired"
    order["amount"] = 30.0
    order["filled"] = 20.0
    order["remaining"] = 10.00
    order["average"] = 0.5
    order["cost"] = 10.0
    order["id"] = "555"
    mocker.patch(f"{EXMS}.create_order", MagicMock(return_value=order))
    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.session.scalars(select(Trade)).all()[3]
    trade.is_short = is_short
    assert trade is not None
    assert not trade.has_open_orders
    assert trade.open_rate == 0.5
    assert trade.stake_amount == round(order["average"] * order["filled"] / leverage, 8)

    # Test with custom stake
    order["status"] = "open"
    order["id"] = "556"

    freqtrade.strategy.custom_stake_amount = lambda **kwargs: 150.0
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[4]
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 150

    # Exception case
    order["id"] = "557"
    freqtrade.strategy.custom_stake_amount = lambda **kwargs: 20 / 0
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[5]
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 2.0

    # In case of the order is rejected and not filled at all
    order["status"] = "rejected"
    order["amount"] = 30.0 * leverage
    order["filled"] = 0.0
    order["remaining"] = 30.0
    order["average"] = 0.5
    order["cost"] = 0.0
    order["id"] = "66"
    mocker.patch(f"{EXMS}.create_order", MagicMock(return_value=order))
    assert not freqtrade.execute_entry(pair, stake_amount)
    assert freqtrade.strategy.leverage.call_count == 0 if trading_mode == "spot" else 2

    # Fail to get price...
    mocker.patch(f"{EXMS}.get_rate", MagicMock(return_value=0.0))

    with pytest.raises(PricingError, match="Could not determine entry price."):
        freqtrade.execute_entry(pair, stake_amount, is_short=is_short)

    # In case of custom entry price
    mocker.patch(f"{EXMS}.get_rate", MagicMock(return_value=0.50))
    order["status"] = "open"
    order["id"] = "5566"
    freqtrade.strategy.custom_entry_price = lambda **kwargs: 0.508
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[6]
    trade.is_short = is_short
    assert trade is not None
    assert trade.open_rate_requested == 0.508

    # In case of custom entry price set to None

    order["status"] = "open"
    order["id"] = "5567"
    freqtrade.strategy.custom_entry_price = lambda **kwargs: None

    mocker.patch.multiple(
        EXMS,
        get_rate=MagicMock(return_value=10),
    )

    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[7]
    trade.is_short = is_short
    assert trade is not None
    assert trade.open_rate_requested == 10

    # In case of custom entry price not float type
    order["status"] = "open"
    order["id"] = "5568"
    freqtrade.strategy.custom_entry_price = lambda **kwargs: "string price"
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[8]
    # Trade(id=9, pair=ETH/USDT, amount=0.20000000, is_short=False,
    #   leverage=1.0, open_rate=10.00000000, open_since=...)
    # Trade(id=9, pair=ETH/USDT, amount=0.60000000, is_short=True,
    #   leverage=3.0, open_rate=10.00000000, open_since=...)
    trade.is_short = is_short
    assert trade is not None
    assert trade.open_rate_requested == 10

    # In case of too high stake amount

    order["status"] = "open"
    order["id"] = "55672"

    mocker.patch.multiple(
        EXMS,
        get_max_pair_stake_amount=MagicMock(return_value=500),
    )
    freqtrade.exchange.get_max_pair_stake_amount = MagicMock(return_value=500)

    assert freqtrade.execute_entry(pair, 2000, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[9]
    trade.is_short = is_short
    assert pytest.approx(trade.stake_amount) == 500

    order["id"] = "55673"

    freqtrade.strategy.leverage.reset_mock()
    assert freqtrade.execute_entry(pair, 200, leverage_=3)
    assert freqtrade.strategy.leverage.call_count == 0
    trade = Trade.session.scalars(select(Trade)).all()[10]
    assert trade.leverage == 1 if trading_mode == "spot" else 3


@pytest.mark.parametrize("is_short", [False, True])
def test_execute_entry_confirm_error(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    limit_order: Dict[str, Any],
    is_short: bool,
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={"bid": 1.9, "ask": 2.2, "last": 1.9}),
        create_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_rate=MagicMock(return_value=0.11),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    stake_amount = 2
    pair = "ETH/USDT"

    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=ValueError)
    assert freqtrade.execute_entry(pair, stake_amount)

    limit_order[entry_side(is_short)]["id"] = "222"
    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=Exception)
    assert freqtrade.execute_entry(pair, stake_amount)

    limit_order[entry_side(is_short)]["id"] = "2223"
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_entry(pair, stake_amount)

    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    assert not freqtrade.execute_entry(pair, stake_amount)


@pytest.mark.parametrize("is_short", [False, True])
def test_execute_entry_fully_canceled_on_create(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    limit_order_open: Dict[str, Any],
    is_short: bool,
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    mock_hce = mocker.spy(freqtrade, "handle_cancel_enter")
    order = limit_order_open[entry_side(is_short)]
    pair = "ETH/USDT"
    order["symbol"] = pair
    order["status"] = "canceled"
    order["filled"] = 0.0

    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={"bid": 1.9, "ask": 2.2, "last": 1.9}),
        create_order=MagicMock(return_value=order),
        get_rate=MagicMock(return_value=0.11),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    stake_amount = 2

    assert freqtrade.execute_entry(pair, stake_amount)
    assert mock_hce.call_count == 1
    # an order that immediately cancels completely should delete the order.
    trades = Trade.get_trades().all()
    assert len(trades) == 0


@pytest.mark.parametrize("is_short", [False, True])
def test_execute_entry_min_leverage(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    limit_order: Dict[str, Any],
    is_short: bool,
) -> None:
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={"bid": 1.9, "ask": 2.2, "last": 1.9}),
        create_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_rate=MagicMock(return_value=0.11),
        # Minimum stake-amount is ~5$
        get_maintenance_ratio_and_amt=MagicMock(return_value=(0.0, 0.0)),
        _fetch_and_calculate_funding_fees=MagicMock(return_value=0),
        get_fee=fee,
        get_max_leverage=MagicMock(return_value=5.0),
    )
    stake_amount = 2
    pair = "SOL/BUSD:BUSD"
    freqtrade.strategy.leverage = MagicMock(return_value=5.0)

    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.leverage == 5.0
    # assert trade.stake_amount == 2


@pytest.mark.parametrize(
    "buy_price_mult,ignore_strat_sl",
    [
        (0.79, False),  # Override stoploss
        (0.85, True),  # Override strategy stoploss
    ],
)
@pytest.mark.parametrize("is_short", [False, True])
def test_handle_cancelled_buy(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    caplog: Any,
    is_short: bool,
) -> None:
    """
    TODO: Add specific test function
    """
    # Placeholder for additional test functions with type annotations as needed
    pass


@pytest.mark.usefixtures("init_persistence")
def test_startup_backpopulate_precision(mocker: Any, default_conf_usdt: Dict[str, Any], fee: Any, caplog: Any) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades_usdt(fee)

    trades = Trade.get_trades().all()
    trades[-1].exchange = "some_other_exchange"
    for trade in trades:
        assert trade.price_precision is None
        assert trade.amount_precision is None
        assert trade.precision_mode is None

    freqtrade.startup_backpopulate_precision()
    trades = Trade.get_trades().all()
    for trade in trades:
        if trade.exchange == "some_other_exchange":
            assert trade.price_precision is None
            assert trade.amount_precision is None
            assert trade.precision_mode is None
        else:
            assert trade.price_precision is not None
            assert trade.amount_precision is not None
            assert trade.precision_mode is not None


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [False, True])
def test_reupdate_enter_order_fees(mocker: Any, default_conf_usdt: Dict[str, Any], fee: Any, caplog: Any, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(
        EXMS,
        fetch_order_or_stoploss_order=MagicMock(side_effect=[
            {"status": "open"},
            {"status": "open"},
            {"status": "open"},
            {"status": "open"},
            {"status": "open"},
        ]),
    )
    create_mock_trades(fee, is_short=is_short)
    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT
    freqtrade.handle_insufficient_funds(trades[3])
    # assert log_has_re(r"Trying to reupdate buy fees for .*", caplog)
    assert freqtrade.strategy.adjust_trade_position.call_count == 1

    assert mocker.spy(freqtrade, "update_trade_state").call_count == 1
    assert mocker.spy(freqtrade, "update_trade_state").call_args_list[0][0][0] == trades[3]
    assert mocker.spy(freqtrade, "update_trade_state").call_args_list[0][0][1] == "open"
    assert log_has_re(r"Trying to refind lost order for .*", caplog)


@pytest.mark.usefixtures("init_persistence")
def test_sync_wallet_dry_run(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    limit_buy_order_usdt_open: Dict[str, Any],
    caplog: Any,
) -> None:
    default_conf_usdt["dry_run"] = True
    # Initialize to 2 times stake amount
    default_conf_usdt["dry_run_wallet"] = 120.0
    default_conf_usdt["max_open_trades"] = 2
    default_conf_usdt["tradable_balance_ratio"] = 1.0
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )

    bot = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(bot)
    assert bot.wallets.get_free("USDT") == 120.0

    n = bot.enter_positions()
    assert n == 2
    trades = Trade.session.scalars(select(Trade)).all()
    assert len(trades) == 2

    bot.config["max_open_trades"] = 3
    n = bot.enter_positions()
    assert n == 0
    assert log_has_re(
        r"Unable to create trade for XRP/USDT: "
        r"Available balance \(0.0 USDT\) is lower than stake amount \(60.0 USDT\)",
        caplog,
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [False, True])
def test_manages_open_orders_entry_usercustom(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    limit_order: Dict[str, Any],
    is_short: bool,
    open_trade_usdt: Trade,
    caplog: Any,
) -> None:
    default_conf_usdt["unfilledtimeout"] = {"entry": 1400, "exit": 30}

    limit_sell_order_old = deepcopy(limit_order[exit_side(is_short)])
    limit_sell_order_old["id"] = open_trade_usdt.open_orders_ids[0]

    default_conf_usdt["dry_run"] = False

    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock(return_value=limit_sell_order_old)
    cancel_entry_order = deepcopy(limit_sell_order_old)
    cancel_entry_order["status"] = "canceled"
    cancel_order_wr_mock = MagicMock(return_value=cancel_entry_order)

    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        fetch_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order=MagicMock(return_value=limit_sell_order_old),
        cancel_order_with_result=cancel_order_wr_mock,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade_usdt.is_short = is_short
    open_trade_usdt.orders[0].side = "sell" if is_short else "buy"
    open_trade_usdt.orders[0].ft_order_side = "sell" if is_short else "buy"
    Trade.session.add(open_trade_usdt)
    Trade.commit()

    # Ensure default is to return empty (so not mocked yet)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0

    # Return false - trade remains open
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=False)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 1
    freqtrade.strategy.check_entry_timeout = MagicMock(side_effect=KeyError)

    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 1
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=True)

    # Trade should be closed since the function returns true
    freqtrade.manage_open_orders()
    assert cancel_order_wr_mock.call_count == 1
    assert rpc_mock.call_count == 2
    trades = Trade.session.scalars(
        select(Trade)
        .where(Order.ft_is_open.is_(True))
        .where(Order.ft_order_side != "stoploss")
        .where(Order.ft_trade_id == Trade.id)
    ).all()
    assert len(trades) == 0
    assert freqtrade.strategy.check_entry_timeout.call_count == 1


@pytest.mark.parametrize("is_short", [False, True])
def test_manage_open_orders_entry(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    limit_order: Dict[str, Any],
    is_short: bool,
    caplog: Any,
) -> None:
    """
    Test manage open orders entry
    """
    default_conf_usdt["dry_run"] = False
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades(fee, is_short=is_short)

    trade = Trade.session.scalars(select(Trade)).first()
    if is_short:
        trade.is_short = is_short

    mocker.patch.multiple(
        EXMS,
        fetch_order_or_stoploss_order=MagicMock(return_value={"status": "open"}),
    )
    assert freqtrade.handle_cancel_order(
        {"status": "open"}, {"side": "buy" if is_short else "sell"}
    ) is True
    assert freqtrade.strategy.check_entry_timeout.call_count == 1
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_open == False


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("factor,adjusts", [(0.99, True), (0.97, False)])
def test_apply_fee_conditional(
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    mocker: Any,
    caplog: Any,
    amount: float,
    fee_abs: float,
    wallet: float,
    amount_exp: Optional[float],
) -> None:
    # This is a placeholder for the test function with parameters
    pass


def test_update_real_amount_quote(
    default_conf_usdt: Dict[str, Any],
    trades_for_order: List[Dict[str, Any]],
    buy_order_fee: Dict[str, Any],
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    buy_order = deepcopy(buy_order_fee)
    buy_order["price"] = 10
    buy_order["cost"] = 300

    mocker.patch(f"{EXMS}.get_trades_for_order", return_value=trades_for_order)
    amount = float(sum(x["amount"] for x in trades_for_order))
    trade = Trade(
        pair="LTC/ETH",
        amount=amount,
        exchange="binance",
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    caplog.clear()
    order_obj = Order.parse_from_ccxt_object(buy_order_fee, "LTC/ETH", "buy")
    assert freqtrade.get_real_amount(trade, buy_order_fee, order_obj) == (amount * 0.001)
    assert log_has(
        "Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, is_short=False,"
        " leverage=1.0, open_rate=0.24544100, open_since=closed), fee=0.001.",
        caplog,
    )


def test_get_real_amount_quote_dust(
    default_conf_usdt: Dict[str, Any],
    trades_for_order: List[Dict[str, Any]],
    buy_order_fee: Dict[str, Any],
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    buy_order = deepcopy(buy_order_fee)
    buy_order["fee"] = {"cost": 0.004, "currency": "ETH"}

    mocker.patch(f"{EXMS}.get_trades_for_order", return_value=trades_for_order)
    amount = float(sum(x["amount"] for x in trades_for_order))
    trade = Trade(
        pair="LTC/ETH",
        amount=amount,
        exchange="binance",
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
    )
    limit_buy_order_usdt_open = {"amount": amount}
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    mocker.patch("freqtrade.wallets.Wallets.update", return_value=None)
    assert freqtrade.get_real_amount(trade, buy_order, buy_order_fee) is None
    assert log_has(
        "Applying fee on amount for Trade(id=None, pair=LTC/ETH, amount=8.00000000, "
        "is_short=False, leverage=1.0, open_rate=0.24544100, open_since=closed) failed: "
        "myTrade-dict empty found",
        caplog,
    )


def test_get_real_amount_no_trade(
    default_conf_usdt: Dict[str, Any],
    buy_order_fee: Dict[str, Any],
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    mocker.patch(f"{EXMS}.get_trades_for_order", return_value=[])
    amount = float(sum(x["amount"] for x in []))
    trade = Trade(
        pair="LTC/ETH",
        amount=amount,
        exchange="binance",
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    order_obj = Order.parse_from_ccxt_object(buy_order_fee, "LTC/ETH", "buy")
    # Amount does not change
    with pytest.raises(DependencyException, match=r"Half bought\? Amounts don't match"):
        freqtrade.get_real_amount(trade, buy_order_fee, order_obj)


def test_get_real_amount_invalid_order(
    default_conf_usdt: Dict[str, Any],
    trades_for_order: List[Dict[str, Any]],
    buy_order_fee: Dict[str, Any],
    fee: Any,
    mocker: Any,
) -> None:
    limit_buy_order_usdt = deepcopy(buy_order_fee)
    limit_buy_order_usdt["id"] = "mocked_order"

    mocker.patch(f"{EXMS}.get_trades_for_order", return_value=[])
    amount = float(sum(x["amount"] for x in trades_for_order))
    trade = Trade(
        pair="LTC/ETH",
        amount=amount,
        exchange="binance",
        open_rate=0.245441,
        fee_open=fee.return_value,
        fee_close=fee.return_value,
    )
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    order_obj = Order.parse_from_ccxt_object(buy_order_fee, "LTC/ETH", "buy")
    # Amount is reduced by "fee"
    assert freqtrade.get_real_amount(trade, limit_buy_order_usdt, order_obj) is None


@pytest.mark.parametrize(
    "fee_cost,fee_currency,fee_reduction_amount,expected_fee,expected_log_amount",
    [
        # basic, amount is reduced by fee
        (None, None, 0.001, 0.001, 7.992),
        # different fee currency on both trades, fee is average of both trade's fee
        (0.02, "BNB", 0.0005, 0.001518575, 7.996),
    ],
)
def test_get_real_amount_multi(
    default_conf_usdt: Dict[str, Any],
    trades_for_order2: List[Dict[str, Any]],
    buy_order_fee: Dict[str, Any],
    caplog: Any,
    fee: Any,
    mocker: Any,
    fee_cost: Optional[float],
    fee_currency: Optional[str],
    fee_reduction_amount: float,
    expected_fee: float,
    expected_log_amount: float,
) -> None:
    trades_for_order = deepcopy(trades_for_order2)
    if fee_cost:
        trades_for_order[0]["fee"]["cost"] = fee_cost
    if fee_currency:
        trades_for_order[0]["fee"]["currency"] = fee_currency

    mocker.patch(f"{EXMS}.get_trades_for_order", return_value=trades_for_order)
    amount = float(sum(x["amount"] for x in trades_for_order))
    default_conf_usdt["stake_currency"] = "ETH"

    trade = Trade(
        pair="LTC/USDT",
        amount=amount,
        exchange="binance",
        fee_open=fee.return_value,
        fee_close=fee.return_value,
        open_rate=0.245441,
    )
    limit_buy_order_usdt = {"amount": amount}
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch(f"{EXMS}.markets", PropertyMock(return_value={"BNB/USDT": {"symbol": "BNB/USDT"}}))
    mocker.patch(f"{EXMS}.get_conversion_rate", return_value=0.2)

    # Amount is reduced by "fee"
    expected_amount = amount * fee_reduction_amount
    order_obj = Order.parse_from_ccxt_object(buy_order_fee, "LTC/USDT", "buy")
    assert freqtrade.get_real_amount(trade, buy_order_fee, order_obj) == expected_amount
    assert log_has(
        (
            "Applying fee on amount for Trade(id=None, pair=LTC/USDT, amount=8.00000000, "
            "is_short=False, leverage=1.0, open_rate=0.24544100, open_since=closed), "
            f"fee={expected_amount}."
        ),
        caplog,
    )

    assert trade.fee_open == expected_fee
    assert trade.fee_close == expected_fee
    assert trade.fee_open_cost is not None
    assert trade.fee_open_currency is not None
    assert trade.fee_close_cost is None
    assert trade.fee_close_currency is None


def test_handle_cancel_exit_empty(mocker: Any, default_conf_usdt: Dict[str, Any], caplog: Any) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch(
        "freqtrade.freqtradebot.FreqtradeBot.handle_cancel_exit",
        return_value=False,
    )

    assert not freqtrade.handle_cancel_exit({"id": "order_id"}, {"side": "buy"}, MagicMock(), "reason")
    assert freqtrade.strategy.adjust_trade_position.call_count == 0


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.usefixtures("init_persistence")
def test_exit_positions_exception(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    limit_order: Dict[str, Any],
    caplog: Any,
    is_short: bool,
) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    order = limit_order[entry_side(is_short)]

    mocker.patch(f"{EXMS}.fetch_order", return_value=order)

    order_id = "123"
    trade = Trade(
        pair="ETH/USDT",
        fee_open=0.001,
        fee_close=0.001,
        open_rate=0.01,
        open_date=dt_now(),
        stake_amount=0.01,
        amount=11,
        exchange="binance",
        is_short=is_short,
        leverage=1,
    )
    trade.orders.append(
        Order(
            ft_order_side=entry_side(is_short),
            price=0.01,
            ft_pair=trade.pair,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id=order_id,
            ft_is_open=False,
            filled=11,
        )
    )
    Trade.session.add(trade)
    Trade.commit()
    freqtrade.wallets.update()
    trades = [trade]

    # Test raise of DependencyException exception
    mocker.patch(
        "freqtrade.freqtradebot.FreqtradeBot.handle_trade",
        side_effect=DependencyException(),
    )
    caplog.clear()
    n = freqtrade.exit_positions(trades)
    assert n == 0
    assert log_has("Unable to exit trade ETH/USDT: ", caplog)


def test_update_trade_state_orderexception(mocker: Any, default_conf_usdt: Dict[str, Any], caplog: Any) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch(f"{EXMS}.fetch_order", MagicMock(side_effect=InvalidOrderException))

    # TODO: should not be magicmock
    trade = MagicMock()
    open_order_id = "123"

    # Test raise of OperationalException exception
    grm_mock = mocker.patch("freqtrade.freqtradebot.FreqtradeBot.get_real_amount", MagicMock())
    freqtrade.update_trade_state(trade, open_order_id)
    assert grm_mock.call_count == 0
    assert log_has(f"Unable to fetch order {open_order_id}: ", caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_trade_roi(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    ticker_usdt_sell_up: Any,
    mocker: Any,
    is_short: bool,
    caplog: Any,
) -> None:
    open_order = ticker_usdt_sell_up[entry_side(is_short)]

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={"bid": 2.19, "ask": 2.2, "last": 2.19}),
        create_order=MagicMock(
            side_effect=[
                open_order,
                {"id": 1234553382},
            ]
        ),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)

    # Create some test data
    freqtrade.enter_positions()

    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade

    time.sleep(0.01)  # Race condition fix
    assert trade.is_open is True
    freqtrade.wallets.update()

    patch_get_signal(freqtrade)
    assert freqtrade.handle_trade(trade)
    assert log_has("ETH/USDT - Required profit reached. exit_type=ExitType.ROI", caplog)


@pytest.mark.parametrize("is_short", [False, True])
def test_handle_trade_use_exit_signal(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    ticker_usdt_sell_up: Any,
    mocker: Any,
    caplog: Any,
    is_short: bool,
) -> None:
    open_order = ticker_usdt_sell_up[exit_side(is_short)]

    # use_exit_signal is True buy default
    caplog.set_level(logging.DEBUG)
    patch_RPCManager(mocker)
    patch_exchange(mocker)

    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt_sell_up,
        create_order=MagicMock(return_value=open_order),
    )
    mocker.patch.multiple(
        EXMS,
        get_fee=MagicMock(return_value=fee),
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)

    freqtrade.enter_positions()

    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade

    # Mock sell signal received
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    assert freqtrade.handle_trade(trade)
    assert log_has("ETH/USDT - Sell signal received. exit_type=ExitType.EXIT_SIGNAL", caplog)


def test_close_trade(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    is_short: bool,
) -> None:
    """Placeholder function"""
    pass


@pytest.mark.parametrize(
    "trading_mode,calls,t1,t2",
    [
        ("spot", 0, "2021-09-01 00:00:00", "2021-09-01 08:00:00"),
        ("margin", 0, "2021-09-01 00:00:00", "2021-09-01 08:00:00"),
        ("futures", 15, "2021-09-01 00:01:02", "2021-09-01 08:00:01"),
        ("futures", 16, "2021-09-01 00:00:02", "2021-09-01 08:00:01"),
        ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:01"),
        ("futures", 16, "2021-09-01 00:00:02", "2021-09-01 08:00:02"),
        ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:02"),
        ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:03"),
        ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:04"),
        ("futures", 17, "2021-08-31 23:59:59", "2021-09-01 08:01:05"),
        ("futures", 17, "2021-08-31 23:59:59", "2021-09-01 08:01:06"),
        ("futures", 17, "2021-08-31 23:59:59", "2021-09-01 08:01:07"),
        ("futures", 17, "2021-08-31 23:59:58", "2021-09-01 08:01:07"),
    ],
)
@pytest.mark.parametrize("tzoffset", ["+00:00", "+01:00", "-02:00"])
def test_update_funding_fees_schedule(
    mocker: Any,
    default_conf: Dict[str, Any],
    trading_mode: str,
    calls: int,
    time_machine: Any,
    t1: str,
    t2: str,
    tzoffset: str,
) -> None:
    time_machine.move_to(f"{t1} {tzoffset}", tick=False)

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch("freqtrade.freqtradebot.FreqtradeBot.update_funding_fees", return_value=True)
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = "isolated"
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    time_machine.move_to(f"{t2} {tzoffset}", tick=False)
    # Check schedule jobs in debugging with freqtrade._schedule.jobs
    freqtrade._schedule.run_pending()

    assert freqtrade.update_funding_fees.call_count == calls


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize(
    "delta,is_high_delta",
    [
        (0.1, False),
        (100, True),
    ],
)
def test_order_book_depth_of_market(
    default_conf_usdt: Dict[str, Any],
    mocker: Any,
    order_book_l2: Dict[str, Any],
    exception_thrown: bool,
    ask: float,
    last: float,
    order_book_top: int,
    order_book: Optional[Dict[str, List[Any]]],
    is_short: bool,
) -> None:
    """
    test check_depth_of_market function
    """
    # Placeholder for the test function with parameters
    pass


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("trading_mode", ["spot", "futures"])
def test_execute_trade_exit(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    ticker_usdt_sell_up: Any,
    is_short: bool,
    caplog: Any,
) -> None:
    # Placeholder for the test function with parameters
    pass


@pytest.mark.parametrize(
    "is_short",
    [False, True],
)
@pytest.mark.parametrize(
    "trading_mode,calls,t1,t2",
    [
        ("spot", 0, "2021-09-01 00:00:00", "2021-09-01 08:00:00"),
        ("margin", 0, "2021-09-01 00:00:00", "2021-09-01 08:00:00"),
        ("futures", 15, "2021-09-01 00:01:02", "2021-09-01 08:00:01"),
        ("futures", 16, "2021-09-01 00:00:02", "2021-09-01 08:00:01"),
        ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:01"),
        ("futures", 16, "2021-09-01 00:00:02", "2021-09-01 08:00:02"),
        ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:02"),
        ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:03"),
        ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:04"),
        ("futures", 17, "2021-08-31 23:59:59", "2021-09-01 08:01:05"),
        ("futures", 17, "2021-08-31 23:59:59", "2021-09-01 08:01:06"),
        ("futures", 17, "2021-08-31 23:59:59", "2021-09-01 08:01:07"),
        ("futures", 17, "2021-08-31 23:59:58", "2021-09-01 08:01:07"),
    ],
)
@pytest.mark.parametrize("tzoffset", ["+00:00", "+01:00", "-02:00"])
def test_execute_trade_exit_market_order(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    mocker: Any,
    is_short: bool,
    caplog: Any,
) -> None:
    # Placeholder for the test function with parameters
    pass


def test_order_book_exit_pricing(
    default_conf_usdt: Dict[str, Any],
    mocker: Any,
    fee: Any,
    ticker_usdt_sell_down: Any,
    caplog: Any,
) -> None:
    # Placeholder for the test function with parameters
    pass


def test_check_depth_of_market(
    default_conf_usdt: Dict[str, Any],
    mocker: Any,
) -> None:
    # Placeholder for the test function with parameters
    pass


def test_handle_cancel_exit(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    limit_order_open: Dict[str, Any],
    is_short: bool,
    fee: Any,
    caplog: Any,
) -> None:
    # Placeholder for the test function with parameters
    pass


@pytest.mark.parametrize("is_short", [False, True])
def test_adjust_entry_replace_fail(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    is_short: bool,
) -> None:
    # Placeholder for the test function with parameters
    pass


def test_update_trade_state_withorderdict(
    default_conf_usdt: Dict[str, Any],
    trades_for_order: List[Dict[str, Any]],
    limit_order: Dict[str, Any],
    fee: Any,
    mocker: Any,
    initial_amount: float,
    has_rounding_fee: bool,
    is_short: bool,
    caplog: Any,
) -> None:
    # Placeholder for the test function with parameters
    pass


@pytest.mark.parametrize("is_short", [False, True])
def test_trailing_stop_loss(
    default_conf_usdt: Dict[str, Any],
    mocker: Any,
    fee: Any,
    is_short: bool,
    caplog: Any,
) -> None:
    # Placeholder for the test function with parameters
    pass


@pytest.mark.parametrize(
    "offset,trail_if_reached,second_sl,is_short",
    [
        (0.1, False, 2.0394, False),
        (0.011, False, 2.0394, False),
        (0.055, True, 1.8, False),
        (0, False, 2.1614, True),
        (0.011, False, 2.1614, True),
        (0.055, True, 2.42, True),
    ],
)
def test_trailing_stop_loss_positive(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    caplog: Any,
    offset: float,
    trail_if_reached: bool,
    second_sl: Optional[float],
    is_short: bool,
) -> None:
    # Placeholder for the test function with parameters
    pass


@pytest.mark.parametrize(
    "runmode",
    [RunMode.DRY_RUN, RunMode.LIVE],
)
def test_startup_state(mocker: Any, default_conf_usdt: Dict[str, Any], caplog: Any, runmode: RunMode) -> None:
    # Placeholder for the test function with parameters
    pass


@pytest.mark.parametrize("is_short", [False, True])
def test_manage_open_orders_partial(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    is_short: bool,
) -> None:
    # Placeholder for the test function with parameters
    pass


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [False, True])
def test_handle_cancel_exit_partial_fee(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    is_short: bool,
    caplog: Any,
) -> None:
    # Placeholder for the test function with parameters
    pass


@pytest.mark.parametrize("schedule_off", [False, True])
@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("trading_mode,calls,t1,t2", [
    ("spot", 0, "2021-09-01 00:00:00", "2021-09-01 08:00:00"),
    ("margin", 0, "2021-09-01 00:00:00", "2021-09-01 08:00:00"),
    ("futures", 15, "2021-09-01 00:01:02", "2021-09-01 08:00:01"),
    ("futures", 16, "2021-09-01 00:00:02", "2021-09-01 08:00:01"),
    ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:01"),
    ("futures", 16, "2021-09-01 00:00:02", "2021-09-01 08:00:02"),
    ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:02"),
    ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:03"),
    ("futures", 16, "2021-08-31 23:59:59", "2021-09-01 08:00:04"),
    ("futures", 17, "2021-08-31 23:59:59", "2021-09-01 08:01:05"),
    ("futures", 17, "2021-08-31 23:59:59", "2021-09-01 08:01:06"),
    ("futures", 17, "2021-08-31 23:59:59", "2021-09-01 08:01:07"),
    ("futures", 17, "2021-08-31 23:59:58", "2021-09-01 08:01:07"),
])
@pytest.mark.parametrize("tzoffset", ["+00:00", "+01:00", "-02:00"])
def test_update_funding_fees_schedule_parameterized(
    mocker: Any,
    default_conf: Dict[str, Any],
    is_short: bool,
    trading_mode: str,
    calls: int,
    time_machine: Any,
    t1: str,
    t2: str,
    tzoffset: str,
) -> None:
    time_machine.move_to(f"{t1} {tzoffset}", tick=False)

    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch("freqtrade.freqtradebot.FreqtradeBot.update_funding_fees", return_value=True)
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = "isolated"
    freqtrade = get_patched_freqtradebot(mocker, default_conf)

    time_machine.move_to(f"{t2} {tzoffset}", tick=False)
    # Check schedule jobs in debugging with freqtrade._schedule.jobs
    freqtrade._schedule.run_pending()

    assert freqtrade.update_funding_fees.call_count == calls


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("is_short", [False, True])
def test_handle_insufficient_funds(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    is_short: bool,
    caplog: Any,
) -> None:
    default_conf_usdt["position_adjustment_enable"] = True
    default_conf_usdt["max_entry_position_adjustment"] = 0
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=10000)
    default_conf_usdt.update(
        {
            "position_adjustment_enable": True,
            "dry_run": False,
            "stake_amount": 200.0,
            "dry_run_wallet": 1000.0,
        }
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    buy_rate_mock = MagicMock(return_value=10)
    mocker.patch.multiple(
        EXMS,
        get_rate=buy_rate_mock,
        fetch_ticker=MagicMock(return_value={"bid": 10, "ask": 12, "last": 11}),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_fee=fee,
    )
    pair = "ETH/USDT"

    # initial funding fees,
    freqtrade.execute_entry("ETH/USDT", 123, is_short=is_short)
    freqtrade.execute_entry("LTC/USDT", 2.0, is_short=is_short)
    freqtrade.execute_entry("XRP/USDT", 123, is_short=is_short)
    multiple = 1 if is_short else -1
    trades = Trade.get_open_trades()
    assert len(trades) == 3
    for trade in trades:
        assert pytest.approx(trade.funding_fees) == 0
    mocker.patch(f"{EXMS}.create_order", return_value=open_exit_order)
    time_machine.move_to("2021-09-01 08:00:00 +00:00")
    if False:  # schedule_off:
        for trade in trades:
            freqtrade.execute_trade_exit(
                trade=trade,
                # The values of the next 2 params are irrelevant for this test
                limit=ticker_usdt_sell_up()["bid"],
                exit_check=ExitCheckTuple(exit_type=ExitType.ROI),
            )
            assert trade.funding_fees == pytest.approx(
                sum(
                    trade.amount
                    * mark_prices[trade.pair].iloc[1:2]["open"]
                    * funding_rates[trade.pair].iloc[1:2]["open"]
                    * multiple
                )
            )

    # Placeholder for the actual test implementations
    pass
