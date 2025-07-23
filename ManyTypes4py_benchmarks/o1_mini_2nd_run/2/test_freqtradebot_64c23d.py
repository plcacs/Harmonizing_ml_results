import logging
import time
from copy import deepcopy
from datetime import timedelta
from unittest.mock import ANY, MagicMock, PropertyMock, patch
import pytest
from pandas import DataFrame
from sqlalchemy import select
from freqtrade.constants import CANCEL_REASON, UNLIMITED_STAKE_AMOUNT
from freqtrade.enums import CandleType, ExitCheckTuple, ExitType, RPCMessageType, RunMode, SignalDirection, State
from freqtrade.exceptions import DependencyException, ExchangeError, InsufficientFundsError, InvalidOrderException, OperationalException, PricingError, TemporaryError
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
from typing import Any, List, Tuple, Optional


def patch_RPCManager(mocker: Any) -> MagicMock:
    """
    This function mock RPC manager to avoid repeating this code in almost every tests
    :param mocker: mocker to patch RPCManager class
    :return: RPCManager.send_msg MagicMock to track if this method is called
    """
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    rpc_mock = mocker.patch('freqtrade.freqtradebot.RPCManager.send_msg', MagicMock())
    return rpc_mock


def test_freqtradebot_state(mocker: Any, default_conf_usdt: dict, markets: dict) -> None:
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    assert freqtrade.state is State.RUNNING
    default_conf_usdt.pop('initial_state')
    freqtrade = FreqtradeBot(default_conf_usdt)
    assert freqtrade.state is State.STOPPED


def test_process_stopped(mocker: Any, default_conf_usdt: dict) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    coo_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders')
    freqtrade.process_stopped()
    assert coo_mock.call_count == 0
    default_conf_usdt['cancel_open_orders_on_exit'] = True
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.process_stopped()
    assert coo_mock.call_count == 1


def test_process_calls_sendmsg(mocker: Any, default_conf_usdt: dict) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.process()
    assert freqtrade.rpc.process_msg_queue.call_count == 1


def test_bot_cleanup(mocker: Any, default_conf_usdt: dict, caplog: Any) -> None:
    mock_cleanup = mocker.patch('freqtrade.freqtradebot.Trade.commit')
    coo_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders')
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.cleanup()
    assert log_has('Cleaning up modules ...', caplog)
    assert mock_cleanup.call_count == 1
    assert coo_mock.call_count == 0
    freqtrade.config['cancel_open_orders_on_exit'] = True
    freqtrade.cleanup()
    assert coo_mock.call_count == 1


def test_bot_cleanup_db_errors(mocker: Any, default_conf_usdt: dict, caplog: Any) -> None:
    mocker.patch('freqtrade.freqtradebot.Trade.commit', side_effect=OperationalException())
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.check_for_open_trades', side_effect=OperationalException())
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.emc = MagicMock()
    freqtrade.emc.shutdown = MagicMock()
    freqtrade.cleanup()
    assert freqtrade.emc.shutdown.call_count == 1


@pytest.mark.parametrize('runmode', [RunMode.DRY_RUN, RunMode.LIVE])
def test_order_dict(default_conf_usdt: dict, mocker: Any, runmode: RunMode, caplog: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    conf['entry_pricing']['price_side'] = 'ask'
    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']
    caplog.clear()
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)


def test_get_trade_stake_amount(default_conf_usdt: dict, mocker: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(default_conf_usdt)
    result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
    assert result == default_conf_usdt['stake_amount']


@pytest.mark.parametrize('runmode', [RunMode.DRY_RUN, RunMode.LIVE])
def test_load_strategy_no_keys(default_conf_usdt: dict, mocker: Any, runmode: RunMode, caplog: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = deepcopy(default_conf_usdt)
    conf['runmode'] = runmode
    erm = mocker.patch('freqtrade.freqtradebot.ExchangeResolver.load_exchange')
    freqtrade = FreqtradeBot(conf)
    strategy_config = freqtrade.strategy.config
    assert id(strategy_config['exchange']) == id(conf['exchange'])
    assert strategy_config['exchange']['key'] == ''
    assert strategy_config['exchange']['secret'] == ''
    assert erm.call_count == 1
    ex_conf = erm.call_args_list[0][1]['exchange_config']
    assert id(ex_conf) != id(conf['exchange'])
    assert ex_conf['key'] != ''
    assert ex_conf['key'] == default_conf_usdt['exchange']['key']
    assert ex_conf['secret'] != ''
    assert ex_conf['secret'] == default_conf_usdt['exchange']['secret']


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    mocker: Any,
    fee: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_edge_called_in_process(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']


def test_edge_overrides_stake_amount(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['dry_run_wallet'] = 999.9
    freqtrade = FreqtradeBot(edge_conf)
    assert (
        freqtrade.wallets.get_trade_stake_amount('NEO/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.2
    )
    assert (
        freqtrade.wallets.get_trade_stake_amount('LTC/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.21
    )


@pytest.mark.parametrize(
    'buy_price_mult,ignore_strat_sl',
    [
        (0.79, False),
        (0.85, True),
    ],
)
def test_edge_overrides_stoploss(
    limit_order: dict,
    fee: Any,
    caplog: Any,
    mocker: Any,
    buy_price_mult: float,
    ignore_strat_sl: bool,
    edge_conf: dict,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')
    enter_price = limit_order['buy']['price']
    ticker_val = {'bid': enter_price, 'ask': enter_price, 'last': enter_price}
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value=ticker_val), get_fee=fee)
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    caplog.clear()
    ticker_val.update(
        {
            'bid': enter_price * buy_price_mult,
            'ask': enter_price * buy_price_mult,
            'last': enter_price * buy_price_mult,
        }
    )
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has_re('Exit for NEO/BTC detected. Reason: stop_loss.*', caplog)
        assert trade.exit_reason == ExitType.STOP_LOSS.value
        assert trade.sell_reason == ExitType.STOP_LOSS.value


def test_total_open_trades_stakes(mocker: Any, default_conf_usdt: dict, ticker_usdt: Any, fee: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order: dict,
    fee: Any,
    mocker: Any,
    is_short: bool,
    open_rate: float,
) -> None:
    send_msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    'stake_amount,create,amount_enough,max_open_trades',
    [
        (5.0, True, True, 99),
        (0.042, True, False, 99),
        (0, False, True, 99),
        (UNLIMITED_STAKE_AMOUNT, False, True, 0),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
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
    default_conf_usdt['max_open_trades'] = max_open_trades
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['stake_amount'] = stake_amount
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    if create:
        assert freqtrade.create_trade('ETH/USDT')
        if amount_enough:
            rate, amount = (enter_mock.call_args[1]['rate'], enter_mock.call_args[1]['amount'])
            assert rate * amount <= default_conf_usdt['stake_amount']
        else:
            assert log_has_re('Stake amount for pair .* is too small.*', caplog)
    else:
        assert not freqtrade.create_trade('ETH/USDT')
        if not max_open_trades:
            assert freqtrade.wallets.get_trade_stake_amount('ETH/USDT', default_conf_usdt['max_open_trades'], freqtrade.edge) == 0


def test_enter_positions_no_pairs_left(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
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
    mocker.patch('freqtrade.configuration.config_validation._validate_whitelist')
    default_conf_usdt['exchange']['pair_whitelist'] = whitelist
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    assert n == positions
    if positions:
        assert not log_has_re('No currency pair in active pair whitelist.*', caplog)
        n = freqtrade.enter_positions()
        assert n == 0
        assert log_has_re('No currency pair in active pair whitelist.*', caplog)
    else:
        assert n == 0
        assert log_has('Active pair whitelist is empty.', caplog)


@pytest.mark.usefixtures('init_persistence')
def test_enter_positions_global_pairlock(default_conf_usdt: dict, ticker_usdt: Any, limit_buy_order_usdt: dict, fee: Any, mocker: Any, caplog: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = 'Global pairlock active until.* Not creating new trades.'
    n = freqtrade.enter_positions()
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()
    PairLocks.lock_pair('*', dt_now() + timedelta(minutes=20), 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize('is_short', [False, True])
def test_handle_protections(
    mocker: Any,
    default_conf_usdt: dict,
    fee: Any,
    is_short: bool,
) -> None:
    default_conf_usdt['_strategy_protections'] = [
        {'method': 'CooldownPeriod', 'stop_duration': 60},
        {
            'method': 'StoplossGuard',
            'lookback_period_candles': 24,
            'trade_limit': 4,
            'stop_duration_candles': 4,
            'only_per_pair': False,
        },
    ]
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.protections._protection_handlers[1].global_stop = MagicMock(
        return_value=ProtectionReturn(True, dt_now() + timedelta(hours=1), 'asdf')
    )
    create_mock_trades(fee, is_short)
    freqtrade.handle_protections('ETC/BTC', '*')
    send_msg_mock = freqtrade.rpc.send_msg
    assert send_msg_mock.call_count == 2
    assert send_msg_mock.call_args_list[0][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER
    assert send_msg_mock.call_args_list[1][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER_GLOBAL


def test_create_trade_no_signal(default_conf_usdt: dict, fee: Any, mocker: Any) -> None:
    default_conf_usdt['dry_run'] = True
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, get_fee=fee)
    default_conf_usdt['stake_amount'] = 10
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)
    assert not freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'max_open',
    list(range(0, 5)),
)
@pytest.mark.parametrize(
    'tradable_balance_ratio,modifier',
    [
        (1.0, 1),
        (0.99, 0.8),
        (0.5, 0.5),
    ],
)
def test_create_trades_multiple_trades(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
    max_open: int,
    tradable_balance_ratio: float,
    modifier: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = max_open
    default_conf_usdt['tradable_balance_ratio'] = tradable_balance_ratio
    default_conf_usdt['dry_run_wallet'] = 60.0 * max_open
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
    assert n == max(int(max_open * modifier), 0)
    assert len(trades) == max(int(max_open * modifier), 0)


def test_create_trades_preopen(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 4
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.execute_entry('ETH/USDT', default_conf_usdt['stake_amount'])
    freqtrade.execute_entry('NEO/BTC', default_conf_usdt['stake_amount'])
    assert len(Trade.get_open_trades()) == 2
    limit_buy_order_usdt_open['id'] = '123444'
    assert freqtrade.create_trade('ETH/USDT')
    assert freqtrade.create_trade('NEO/BTC')
    trades = Trade.get_open_trades()
    assert len(trades) == 4


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_process_trade_creation(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order: dict,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    ticker_side = 'ask' if is_short else 'bid'
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=[limit_order_open[entry_side(is_short)], limit_order[entry_side(is_short)]]),
        fetch_order=MagicMock(side_effect=[limit_order[entry_side(is_short)], limit_order[entry_side(is_short)]]),
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
    assert pytest.approx(trade.stake_amount) == default_conf_usdt['stake_amount']
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    assert trade.open_rate == ticker_usdt.return_value[ticker_side]
    assert pytest.approx(trade.amount) == 0
    assert pytest.approx(trade.amount_requested) == 60 / ticker_usdt.return_value[ticker_side]
    assert log_has(
        f'{("Short" if is_short else "Long")} signal found: about create a new trade for ETH/USDT with stake_amount: 60.0 ...',
        caplog,
    )
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._check_and_execute_exit')
    freqtrade.process()
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    assert trade.open_rate == limit_order[entry_side(is_short)]['price']
    assert pytest.approx(trade.amount) == limit_order[entry_side(is_short)]['filled']


def test_process_exchange_failures(default_conf_usdt: dict, ticker_usdt: Any, mocker: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        reload_markets=MagicMock(),
        create_order=MagicMock(side_effect=TemporaryError),
    )
    sleep_mock = mocker.patch('time.sleep')
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)
    mocker.patch(f'{EXMS}.reload_markets', MagicMock(side_effect=TemporaryError))
    worker._process_running()
    assert sleep_mock.called is True


def test_process_operational_exception(default_conf_usdt: dict, ticker_usdt: Any, mocker: Any) -> None:
    msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=OperationalException),
    )
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)
    assert worker.freqtrade.state == State.RUNNING
    worker._process_running()
    assert worker.freqtrade.state == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]['status']


def test_process_trade_handling(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 2.19, 'ask': 2.2, 'last': 2.19}),
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
    freqtrade.process()
    assert len(trades) == 1


def test_process_trade_no_whitelist_pair(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
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
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        fetch_order=MagicMock(return_value=limit_buy_order_usdt),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    pair = 'BLK/BTC'
    assert pair not in default_conf_usdt['exchange']['pair_whitelist']
    Trade.session.add(
        Trade(
            pair=pair,
            stake_amount=0.001,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            is_open=True,
            amount=20,
            open_rate=0.01,
            exchange='binance',
        )
    )
    Trade.session.add(
        Trade(
            pair='ETH/USDT',
            stake_amount=0.001,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            is_open=True,
            amount=12,
            open_rate=0.001,
            exchange='binance',
        )
    )
    Trade.commit()
    assert pair not in freqtrade.active_pair_whitelist
    freqtrade.process()
    assert pair in freqtrade.active_pair_whitelist
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
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
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    default_conf_usdt['max_open_trades'] = max_open_trades
    default_conf_usdt['tradable_balance_ratio'] = tradable_balance_ratio
    default_conf_usdt['dry_run_wallet'] = 60.0 * max_open_trades
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
    assert n == max(int(max_open * modifier), 0)
    assert len(trades) == max(int(max_open * modifier), 0)


def test_handle_protections(
    mocker: Any,
    default_conf_usdt: dict,
    fee: Any,
    is_short: bool,
) -> None:
    default_conf_usdt['_strategy_protections'] = [
        {'method': 'CooldownPeriod', 'stop_duration': 60},
        {
            'method': 'StoplossGuard',
            'lookback_period_candles': 24,
            'trade_limit': 4,
            'stop_duration_candles': 4,
            'only_per_pair': False,
        },
    ]
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.protections._protection_handlers[1].global_stop = MagicMock(
        return_value=ProtectionReturn(True, dt_now() + timedelta(hours=1), 'asdf')
    )
    create_mock_trades(fee, is_short)
    freqtrade.handle_protections('ETC/BTC', '*')
    send_msg_mock = freqtrade.rpc.send_msg
    assert send_msg_mock.call_count == 2
    assert send_msg_mock.call_args_list[0][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER
    assert send_msg_mock.call_args_list[1][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER_GLOBAL


def test_create_trade_no_signal(default_conf_usdt: dict, fee: Any, mocker: Any) -> None:
    default_conf_usdt['dry_run'] = True
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, get_fee=fee)
    default_conf_usdt['stake_amount'] = 10
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)
    assert not freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_load_strategy_no_keys(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = deepcopy(default_conf_usdt)
    conf['runmode'] = runmode
    erm = mocker.patch('freqtrade.freqtradebot.ExchangeResolver.load_exchange')
    freqtrade = FreqtradeBot(conf)
    strategy_config = freqtrade.strategy.config
    assert id(strategy_config['exchange']) == id(conf['exchange'])
    assert strategy_config['exchange']['key'] == ''
    assert strategy_config['exchange']['secret'] == ''
    assert erm.call_count == 1
    ex_conf = erm.call_args_list[0][1]['exchange_config']
    assert id(ex_conf) != id(conf['exchange'])
    assert ex_conf['key'] != ''
    assert ex_conf['key'] == default_conf_usdt['exchange']['key']
    assert ex_conf['secret'] != ''
    assert ex_conf['secret'] == default_conf_usdt['exchange']['secret']


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_order_dict(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    conf['entry_pricing']['price_side'] = 'ask'
    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']
    caplog.clear()
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order: dict,
    is_short: bool,
    open_rate: float,
    fee: Any,
    mocker: Any,
) -> None:
    send_msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    'stake_amount,create,amount_enough,max_open_trades',
    [
        (5.0, True, True, 99),
        (0.042, True, False, 99),
        (0, False, True, 99),
        (UNLIMITED_STAKE_AMOUNT, False, True, 0),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
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
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    default_conf_usdt['max_open_trades'] = max_open_trades
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['stake_amount'] = stake_amount
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    if create:
        assert freqtrade.create_trade('ETH/USDT')
        if amount_enough:
            rate, amount = (mocker.call_args[1]['rate'], mocker.call_args[1]['amount'])
            assert rate * amount <= default_conf_usdt['stake_amount']
        else:
            assert log_has_re('Stake amount for pair .* is too small.*', caplog)
    else:
        assert not freqtrade.create_trade('ETH/USDT')
        if not max_open_trades:
            assert freqtrade.wallets.get_trade_stake_amount('ETH/USDT', default_conf_usdt['max_open_trades'], freqtrade.edge) == 0


@pytest.mark.parametrize(
    'whitelist,positions',
    [
        (['ETH/USDT'], 1),
        ([], 0),
    ],
)
def test_enter_positions_no_pairs_left(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
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
    mocker.patch('freqtrade.configuration.config_validation._validate_whitelist')
    default_conf_usdt['exchange']['pair_whitelist'] = whitelist
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    assert n == positions
    if positions:
        assert not log_has_re('No currency pair in active pair whitelist.*', caplog)
        n = freqtrade.enter_positions()
        assert n == 0
        assert log_has_re('No currency pair in active pair whitelist.*', caplog)
    else:
        assert n == 0
        assert log_has('Active pair whitelist is empty.', caplog)


@pytest.mark.usefixtures('init_persistence')
def test_enter_positions_global_pairlock(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = 'Global pairlock active until.* Not creating new trades.'
    n = freqtrade.enter_positions()
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()
    PairLocks.lock_pair('*', dt_now() + timedelta(minutes=20), 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_edge_called_in_process(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']


def test_edge_overrides_stake_amount(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['dry_run_wallet'] = 999.9
    freqtrade = FreqtradeBot(edge_conf)
    assert (
        freqtrade.wallets.get_trade_stake_amount('NEO/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.2
    )
    assert (
        freqtrade.wallets.get_trade_stake_amount('LTC/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.21
    )


@pytest.mark.parametrize(
    'buy_price_mult,ignore_strat_sl',
    [
        (0.79, False),
        (0.85, True),
    ],
)
def test_edge_overrides_stoploss(
    limit_order: dict,
    fee: Any,
    caplog: Any,
    mocker: Any,
    buy_price_mult: float,
    ignore_strat_sl: bool,
    edge_conf: dict,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')
    enter_price = limit_order['buy']['price']
    ticker_val = {'bid': enter_price, 'ask': enter_price, 'last': enter_price}
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value=ticker_val),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    caplog.clear()
    ticker_val.update(
        {
            'bid': enter_price * buy_price_mult,
            'ask': enter_price * buy_price_mult,
            'last': enter_price * buy_price_mult,
        }
    )
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has_re('Exit for NEO/BTC detected. Reason: stop_loss.*', caplog)
        assert trade.exit_reason == ExitType.STOP_LOSS.value
        assert trade.sell_reason == ExitType.STOP_LOSS.value


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    open_rate: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_order_dict(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    conf['entry_pricing']['price_side'] = 'ask'
    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']
    caplog.clear()
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    open_rate: float,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_load_strategy_no_keys(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = deepcopy(default_conf_usdt)
    conf['runmode'] = runmode
    erm = mocker.patch('freqtrade.freqtradebot.ExchangeResolver.load_exchange')
    freqtrade = FreqtradeBot(conf)
    strategy_config = freqtrade.strategy.config
    assert id(strategy_config['exchange']) == id(conf['exchange'])
    assert strategy_config['exchange']['key'] == ''
    assert strategy_config['exchange']['secret'] == ''
    assert erm.call_count == 1
    ex_conf = erm.call_args_list[0][1]['exchange_config']
    assert id(ex_conf) != id(conf['exchange'])
    assert ex_conf['key'] != ''
    assert ex_conf['key'] == default_conf_usdt['exchange']['key']
    assert ex_conf['secret'] != ''
    assert ex_conf['secret'] == default_conf_usdt['exchange']['secret']


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_order_dict(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    conf['entry_pricing']['price_side'] = 'ask'
    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']
    caplog.clear()
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    open_rate: float,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    'stake_amount,create,amount_enough,max_open_trades',
    [
        (5.0, True, True, 99),
        (0.042, True, False, 99),
        (0, False, True, 99),
        (UNLIMITED_STAKE_AMOUNT, False, True, 0),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    stake_amount: float,
    create: bool,
    amount_enough: bool,
    max_open_trades: int,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    default_conf_usdt['max_open_trades'] = max_open_trades
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['stake_amount'] = stake_amount
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    if create:
        assert freqtrade.create_trade('ETH/USDT')
        if amount_enough:
            rate, amount = (mocker.call_args[1]['rate'], mocker.call_args[1]['amount'])
            assert rate * amount <= default_conf_usdt['stake_amount']
        else:
            assert log_has_re('Stake amount for pair .* is too small.*', caplog)
    else:
        assert not freqtrade.create_trade('ETH/USDT')
        if not max_open_trades:
            assert freqtrade.wallets.get_trade_stake_amount('ETH/USDT', default_conf_usdt['max_open_trades'], freqtrade.edge) == 0


@pytest.mark.parametrize(
    'whitelist,positions',
    [
        (['ETH/USDT'], 1),
        ([], 0),
    ],
)
def test_enter_positions_no_pairs_left(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
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
    mocker.patch('freqtrade.configuration.config_validation._validate_whitelist')
    default_conf_usdt['exchange']['pair_whitelist'] = whitelist
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    assert n == positions
    if positions:
        assert not log_has_re('No currency pair in active pair whitelist.*', caplog)
        n = freqtrade.enter_positions()
        assert n == 0
        assert log_has_re('No currency pair in active pair whitelist.*', caplog)
    else:
        assert n == 0
        assert log_has('Active pair whitelist is empty.', caplog)


@pytest.mark.usefixtures('init_persistence')
def test_enter_positions_global_pairlock(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = 'Global pairlock active until.* Not creating new trades.'
    n = freqtrade.enter_positions()
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()
    PairLocks.lock_pair('*', dt_now() + timedelta(minutes=20), 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_edge_called_in_process(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']


def test_edge_overrides_stake_amount(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['dry_run_wallet'] = 999.9
    freqtrade = FreqtradeBot(edge_conf)
    assert (
        freqtrade.wallets.get_trade_stake_amount('NEO/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.2
    )
    assert (
        freqtrade.wallets.get_trade_stake_amount('LTC/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.21
    )


@pytest.mark.parametrize(
    'buy_price_mult,ignore_strat_sl',
    [
        (0.79, False),
        (0.85, True),
    ],
)
def test_edge_overrides_stoploss(
    limit_order: dict,
    fee: Any,
    caplog: Any,
    mocker: Any,
    buy_price_mult: float,
    ignore_strat_sl: bool,
    edge_conf: dict,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')
    enter_price = limit_order['buy']['price']
    ticker_val = {'bid': enter_price, 'ask': enter_price, 'last': enter_price}
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value=ticker_val),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    caplog.clear()
    ticker_val.update(
        {
            'bid': enter_price * buy_price_mult,
            'ask': enter_price * buy_price_mult,
            'last': enter_price * buy_price_mult,
        }
    )
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has_re('Exit for NEO/BTC detected. Reason: stop_loss.*', caplog)
        assert trade.exit_reason == ExitType.STOP_LOSS.value
        assert trade.sell_reason == ExitType.STOP_LOSS.value


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order: dict,
    is_short: bool,
    open_rate: float,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    open_rate: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


@pytest.mark.parametrize(
    'max_open',
    list(range(0, 5)),
)
@pytest.mark.parametrize(
    'tradable_balance_ratio,modifier',
    [
        (1.0, 1),
        (0.99, 0.8),
        (0.5, 0.5),
    ],
)
def test_create_trades_multiple_trades(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
    max_open: int,
    tradable_balance_ratio: float,
    modifier: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = max_open
    default_conf_usdt['tradable_balance_ratio'] = tradable_balance_ratio
    default_conf_usdt['dry_run_wallet'] = 60.0 * max_open
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
    assert n == max(int(max_open * modifier), 0)
    assert len(trades) == max(int(max_open * modifier), 0)


def test_create_trades_preopen(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 4
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.execute_entry('ETH/USDT', default_conf_usdt['stake_amount'])
    freqtrade.execute_entry('NEO/BTC', default_conf_usdt['stake_amount'])
    assert len(Trade.get_open_trades()) == 2
    limit_buy_order_usdt_open['id'] = '123444'
    assert freqtrade.create_trade('ETH/USDT')
    assert freqtrade.create_trade('NEO/BTC')
    trades = Trade.get_open_trades()
    assert len(trades) == 4


def test_process_trade_creation(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    is_short: bool,
    mocker: Any,
    caplog: Any,
    open_rate: float,
) -> None:
    ticker_side = 'ask' if is_short else 'bid'
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=[limit_order_open[entry_side(is_short)], limit_order[entry_side(is_short)]]),
        fetch_order=MagicMock(side_effect=[limit_order[entry_side(is_short)], limit_order[entry_side(is_short)]]),
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
    assert pytest.approx(trade.stake_amount) == default_conf_usdt['stake_amount']
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    assert trade.open_rate == ticker_usdt.return_value[ticker_side]
    assert pytest.approx(trade.amount) == 0
    assert pytest.approx(trade.amount_requested) == 60 / ticker_usdt.return_value[ticker_side]
    assert log_has(
        f'{("Short" if is_short else "Long")} signal found: about create a new trade for ETH/USDT with stake_amount: 60.0 ...',
        caplog,
    )
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._check_and_execute_exit')
    freqtrade.process()
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    assert trade.open_rate == limit_order[entry_side(is_short)]['price']
    assert pytest.approx(trade.amount) == limit_order[entry_side(is_short)]['filled']


def test_process_exchange_failures(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        reload_markets=MagicMock(),
        create_order=MagicMock(side_effect=TemporaryError),
    )
    sleep_mock = mocker.patch('time.sleep')
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)
    mocker.patch(f'{EXMS}.reload_markets', MagicMock(side_effect=TemporaryError))
    worker._process_running()
    assert sleep_mock.called is True


def test_process_operational_exception(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    mocker: Any,
) -> None:
    msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=OperationalException),
    )
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)
    assert worker.freqtrade.state == State.RUNNING
    worker._process_running()
    assert worker.freqtrade.state == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]['status']


def test_process_trade_handling(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 2.19, 'ask': 2.2, 'last': 2.19}),
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
    freqtrade.process()
    assert len(trades) == 1


def test_process_trade_no_whitelist_pair(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
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
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        fetch_order=MagicMock(return_value=limit_buy_order_usdt),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    pair = 'BLK/BTC'
    assert pair not in default_conf_usdt['exchange']['pair_whitelist']
    Trade.session.add(
        Trade(
            pair=pair,
            stake_amount=0.001,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            is_open=True,
            amount=20,
            open_rate=0.01,
            exchange='binance',
        )
    )
    Trade.session.add(
        Trade(
            pair='ETH/USDT',
            stake_amount=0.001,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            is_open=True,
            amount=12,
            open_rate=0.001,
            exchange='binance',
        )
    )
    Trade.commit()
    assert pair not in freqtrade.active_pair_whitelist
    freqtrade.process()
    assert pair in freqtrade.active_pair_whitelist
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))


@pytest.mark.usefixtures('init_persistence')
def test_enter_positions_global_pairlock(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = 'Global pairlock active until.* Not creating new trades.'
    n = freqtrade.enter_positions()
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()
    PairLocks.lock_pair('*', dt_now() + timedelta(minutes=20), 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_create_trade_no_signal(
    default_conf_usdt: dict,
    fee: Any,
    mocker: Any,
) -> None:
    default_conf_usdt['dry_run'] = True
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, get_fee=fee)
    default_conf_usdt['stake_amount'] = 10
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)
    assert not freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    'stake_amount,create,amount_enough,max_open_trades',
    [
        (5.0, True, True, 99),
        (0.042, True, False, 99),
        (0, False, True, 99),
        (UNLIMITED_STAKE_AMOUNT, False, True, 0),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    stake_amount: float,
    create: bool,
    amount_enough: bool,
    max_open_trades: int,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    default_conf_usdt['max_open_trades'] = max_open_trades
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['stake_amount'] = stake_amount
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    if create:
        assert freqtrade.create_trade('ETH/USDT')
        if amount_enough:
            rate, amount = (mocker.call_args[1]['rate'], mocker.call_args[1]['amount'])
            assert rate * amount <= default_conf_usdt['stake_amount']
        else:
            assert log_has_re('Stake amount for pair .* is too small.*', caplog)
    else:
        assert not freqtrade.create_trade('ETH/USDT')
        if not max_open_trades:
            assert freqtrade.wallets.get_trade_stake_amount('ETH/USDT', default_conf_usdt['max_open_trades'], freqtrade.edge) == 0


@pytest.mark.parametrize(
    'whitelist,positions',
    [
        (['ETH/USDT'], 1),
        ([], 0),
    ],
)
def test_enter_positions_no_pairs_left(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
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
    mocker.patch('freqtrade.configuration.config_validation._validate_whitelist')
    default_conf_usdt['exchange']['pair_whitelist'] = whitelist
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    assert n == positions
    if positions:
        assert not log_has_re('No currency pair in active pair whitelist.*', caplog)
        n = freqtrade.enter_positions()
        assert n == 0
        assert log_has_re('No currency pair in active pair whitelist.*', caplog)
    else:
        assert n == 0
        assert log_has('Active pair whitelist is empty.', caplog)


@pytest.mark.usefixtures('init_persistence')
def test_enter_positions_global_pairlock(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = 'Global pairlock active until.* Not creating new trades.'
    n = freqtrade.enter_positions()
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()
    PairLocks.lock_pair('*', dt_now() + timedelta(minutes=20), 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_edge_called_in_process(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']


def test_edge_overrides_stake_amount(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['dry_run_wallet'] = 999.9
    freqtrade = FreqtradeBot(edge_conf)
    assert (
        freqtrade.wallets.get_trade_stake_amount('NEO/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.2
    )
    assert (
        freqtrade.wallets.get_trade_stake_amount('LTC/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.21
    )


@pytest.mark.parametrize(
    'buy_price_mult,ignore_strat_sl',
    [
        (0.79, False),
        (0.85, True),
    ],
)
def test_edge_overrides_stoploss(
    limit_order: dict,
    fee: Any,
    caplog: Any,
    mocker: Any,
    buy_price_mult: float,
    ignore_strat_sl: bool,
    edge_conf: dict,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')
    enter_price = limit_order['buy']['price']
    ticker_val = {'bid': enter_price, 'ask': enter_price, 'last': enter_price}
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value=ticker_val),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    caplog.clear()
    ticker_val.update(
        {
            'bid': enter_price * buy_price_mult,
            'ask': enter_price * buy_price_mult,
            'last': enter_price * buy_price_mult,
        }
    )
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has_re('Exit for NEO/BTC detected. Reason: stop_loss.*', caplog)
        assert trade.exit_reason == ExitType.STOP_LOSS.value
        assert trade.sell_reason == ExitType.STOP_LOSS.value


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    open_rate: float,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    'stake_amount,create,amount_enough,max_open_trades',
    [
        (5.0, True, True, 99),
        (0.042, True, False, 99),
        (0, False, True, 99),
        (UNLIMITED_STAKE_AMOUNT, False, True, 0),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    stake_amount: float,
    create: bool,
    amount_enough: bool,
    max_open_trades: int,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    default_conf_usdt['max_open_trades'] = max_open_trades
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['stake_amount'] = stake_amount
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    if create:
        assert freqtrade.create_trade('ETH/USDT')
        if amount_enough:
            rate, amount = (mocker.call_args[1]['rate'], mocker.call_args[1]['amount'])
            assert rate * amount <= default_conf_usdt['stake_amount']
        else:
            assert log_has_re('Stake amount for pair .* is too small.*', caplog)
    else:
        assert not freqtrade.create_trade('ETH/USDT')
        if not max_open_trades:
            assert freqtrade.wallets.get_trade_stake_amount('ETH/USDT', max_open_trades, freqtrade.edge) == 0


@pytest.mark.parametrize(
    'whitelist,positions',
    [
        (['ETH/USDT'], 1),
        ([], 0),
    ],
)
def test_enter_positions_no_pairs_left(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
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
    mocker.patch('freqtrade.configuration.config_validation._validate_whitelist')
    default_conf_usdt['exchange']['pair_whitelist'] = whitelist
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    assert n == positions
    if positions:
        assert not log_has_re('No currency pair in active pair whitelist.*', caplog)
        n = freqtrade.enter_positions()
        assert n == 0
        assert log_has_re('No currency pair in active pair whitelist.*', caplog)
    else:
        assert n == 0
        assert log_has('Active pair whitelist is empty.', caplog)


@pytest.mark.usefixtures('init_persistence')
def test_enter_positions_global_pairlock(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = 'Global pairlock active until.* Not creating new trades.'
    n = freqtrade.enter_positions()
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()
    PairLocks.lock_pair('*', dt_now() + timedelta(minutes=20), 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_create_trade_no_signal(
    default_conf_usdt: dict,
    fee: Any,
    mocker: Any,
) -> None:
    default_conf_usdt['dry_run'] = True
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, get_fee=fee)
    default_conf_usdt['stake_amount'] = 10
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)
    assert not freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_load_strategy_no_keys(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = deepcopy(default_conf_usdt)
    conf['runmode'] = runmode
    erm = mocker.patch('freqtrade.freqtradebot.ExchangeResolver.load_exchange')
    freqtrade = FreqtradeBot(conf)
    strategy_config = freqtrade.strategy.config
    assert id(strategy_config['exchange']) == id(conf['exchange'])
    assert strategy_config['exchange']['key'] == ''
    assert strategy_config['exchange']['secret'] == ''
    assert erm.call_count == 1
    ex_conf = erm.call_args_list[0][1]['exchange_config']
    assert id(ex_conf) != id(conf['exchange'])
    assert ex_conf['key'] != ''
    assert ex_conf['key'] == default_conf_usdt['exchange']['key']
    assert ex_conf['secret'] != ''
    assert ex_conf['secret'] == default_conf_usdt['exchange']['secret']


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_order_dict(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    conf['entry_pricing']['price_side'] = 'ask'
    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']
    caplog.clear()
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    open_rate: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    open_rate: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_load_strategy_no_keys(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = deepcopy(default_conf_usdt)
    conf['runmode'] = runmode
    erm = mocker.patch('freqtrade.freqtradebot.ExchangeResolver.load_exchange')
    freqtrade = FreqtradeBot(conf)
    strategy_config = freqtrade.strategy.config
    assert id(strategy_config['exchange']) == id(conf['exchange'])
    assert strategy_config['exchange']['key'] == ''
    assert strategy_config['exchange']['secret'] == ''
    assert erm.call_count == 1
    ex_conf = erm.call_args_list[0][1]['exchange_config']
    assert id(ex_conf) != id(conf['exchange'])
    assert ex_conf['key'] != ''
    assert ex_conf['key'] == default_conf_usdt['exchange']['key']
    assert ex_conf['secret'] != ''
    assert ex_conf['secret'] == default_conf_usdt['exchange']['secret']


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_order_dict(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    conf['entry_pricing']['price_side'] = 'ask'
    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']
    caplog.clear()
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    open_rate: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    'stake_amount,create,amount_enough,max_open_trades',
    [
        (5.0, True, True, 99),
        (0.042, True, False, 99),
        (0, False, True, 99),
        (UNLIMITED_STAKE_AMOUNT, False, True, 0),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    stake_amount: float,
    create: bool,
    amount_enough: bool,
    max_open_trades: int,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    default_conf_usdt['max_open_trades'] = max_open_trades
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['stake_amount'] = stake_amount
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    if create:
        assert freqtrade.create_trade('ETH/USDT')
        if amount_enough:
            rate, amount = (mocker.call_args[1]['rate'], mocker.call_args[1]['amount'])
            assert rate * amount <= default_conf_usdt['stake_amount']
        else:
            assert log_has_re('Stake amount for pair .* is too small.*', caplog)
    else:
        assert not freqtrade.create_trade('ETH/USDT')
        if not max_open_trades:
            assert freqtrade.wallets.get_trade_stake_amount('ETH/USDT', max_open_trades, freqtrade.edge) == 0


@pytest.mark.parametrize(
    'whitelist,positions',
    [
        (['ETH/USDT'], 1),
        ([], 0),
    ],
)
def test_enter_positions_no_pairs_left(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
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
    mocker.patch('freqtrade.configuration.config_validation._validate_whitelist')
    default_conf_usdt['exchange']['pair_whitelist'] = whitelist
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    assert n == positions
    if positions:
        assert not log_has_re('No currency pair in active pair whitelist.*', caplog)
        n = freqtrade.enter_positions()
        assert n == 0
        assert log_has_re('No currency pair in active pair whitelist.*', caplog)
    else:
        assert n == 0
        assert log_has('Active pair whitelist is empty.', caplog)


@pytest.mark.usefixtures('init_persistence')
def test_enter_positions_global_pairlock(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = 'Global pairlock active until.* Not creating new trades.'
    n = freqtrade.enter_positions()
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()
    PairLocks.lock_pair('*', dt_now() + timedelta(minutes=20), 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_edge_called_in_process(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']


def test_edge_overrides_stake_amount(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['dry_run_wallet'] = 999.9
    freqtrade = FreqtradeBot(edge_conf)
    assert (
        freqtrade.wallets.get_trade_stake_amount('NEO/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.2
    )
    assert (
        freqtrade.wallets.get_trade_stake_amount('LTC/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.21
    )


@pytest.mark.parametrize(
    'buy_price_mult,ignore_strat_sl',
    [
        (0.79, False),
        (0.85, True),
    ],
)
def test_edge_overrides_stoploss(
    limit_order: dict,
    fee: Any,
    caplog: Any,
    mocker: Any,
    buy_price_mult: float,
    ignore_strat_sl: bool,
    edge_conf: dict,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')
    enter_price = limit_order['buy']['price']
    ticker_val = {'bid': enter_price, 'ask': enter_price, 'last': enter_price}
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value=ticker_val),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    caplog.clear()
    ticker_val.update(
        {
            'bid': enter_price * buy_price_mult,
            'ask': enter_price * buy_price_mult,
            'last': enter_price * buy_price_mult,
        }
    )
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has_re('Exit for NEO/BTC detected. Reason: stop_loss.*', caplog)
        assert trade.exit_reason == ExitType.STOP_LOSS.value
        assert trade.sell_reason == ExitType.STOP_LOSS.value


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    open_rate: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    open_rate: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


@pytest.mark.parametrize(
    'max_open',
    list(range(0, 5)),
)
@pytest.mark.parametrize(
    'tradable_balance_ratio,modifier',
    [
        (1.0, 1),
        (0.99, 0.8),
        (0.5, 0.5),
    ],
)
def test_create_trades_multiple_trades(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
    max_open: int,
    tradable_balance_ratio: float,
    modifier: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = max_open
    default_conf_usdt['tradable_balance_ratio'] = tradable_balance_ratio
    default_conf_usdt['dry_run_wallet'] = 60.0 * max_open
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
    assert n == max(int(max_open * modifier), 0)
    assert len(trades) == max(int(max_open * modifier), 0)


def test_create_trades_preopen(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 4
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_buy_order_usdt_open),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.execute_entry('ETH/USDT', default_conf_usdt['stake_amount'])
    freqtrade.execute_entry('NEO/BTC', default_conf_usdt['stake_amount'])
    assert len(Trade.get_open_trades()) == 2
    limit_buy_order_usdt_open['id'] = '123444'
    assert freqtrade.create_trade('ETH/USDT')
    assert freqtrade.create_trade('NEO/BTC')
    trades = Trade.get_open_trades()
    assert len(trades) == 4


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_process_trade_creation(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order: dict,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    caplog: Any,
    open_rate: float,
) -> None:
    ticker_side = 'ask' if is_short else 'bid'
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=[limit_order_open[entry_side(is_short)], limit_order[entry_side(is_short)]]),
        fetch_order=MagicMock(side_effect=[limit_order[entry_side(is_short)], limit_order[entry_side(is_short)]]),
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
    assert pytest.approx(trade.stake_amount) == default_conf_usdt['stake_amount']
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    assert trade.open_rate == ticker_usdt.return_value[ticker_side]
    assert pytest.approx(trade.amount) == 0
    assert pytest.approx(trade.amount_requested) == 60 / ticker_usdt.return_value[ticker_side]
    assert log_has(
        f'{("Short" if is_short else "Long")} signal found: about create a new trade for ETH/USDT with stake_amount: 60.0 ...',
        caplog,
    )
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._check_and_execute_exit')
    freqtrade.process()
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    trade = trades[0]
    assert trade is not None
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    assert trade.open_rate == limit_order[entry_side(is_short)]['price']
    assert pytest.approx(trade.amount) == limit_order[entry_side(is_short)]['filled']


def test_process_exchange_failures(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        reload_markets=MagicMock(),
        create_order=MagicMock(side_effect=TemporaryError),
    )
    sleep_mock = mocker.patch('time.sleep')
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)
    mocker.patch(f'{EXMS}.reload_markets', MagicMock(side_effect=TemporaryError))
    worker._process_running()
    assert sleep_mock.called is True


def test_process_operational_exception(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    mocker: Any,
) -> None:
    msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(side_effect=OperationalException),
    )
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)
    assert worker.freqtrade.state == State.RUNNING
    worker._process_running()
    assert worker.freqtrade.state == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]['status']


def test_process_trade_handling(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 2.19, 'ask': 2.2, 'last': 2.19}),
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
    freqtrade.process()
    assert len(trades) == 1


def test_process_trade_no_whitelist_pair(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
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
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        fetch_order=MagicMock(return_value=limit_buy_order_usdt),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    pair = 'BLK/BTC'
    assert pair not in default_conf_usdt['exchange']['pair_whitelist']
    Trade.session.add(
        Trade(
            pair=pair,
            stake_amount=0.001,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            is_open=True,
            amount=20,
            open_rate=0.01,
            exchange='binance',
        )
    )
    Trade.session.add(
        Trade(
            pair='ETH/USDT',
            stake_amount=0.001,
            fee_open=fee.return_value,
            fee_close=fee.return_value,
            is_open=True,
            amount=12,
            open_rate=0.001,
            exchange='binance',
        )
    )
    Trade.commit()
    assert pair not in freqtrade.active_pair_whitelist
    freqtrade.process()
    assert pair in freqtrade.active_pair_whitelist
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))


@pytest.mark.usefixtures('init_persistence')
def test_enter_positions_global_pairlock(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = 'Global pairlock active until.* Not creating new trades.'
    n = freqtrade.enter_positions()
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()
    PairLocks.lock_pair('*', dt_now() + timedelta(minutes=20), 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_handle_protections(
    mocker: Any,
    default_conf_usdt: dict,
    fee: Any,
    is_short: bool,
) -> None:
    default_conf_usdt['_strategy_protections'] = [
        {'method': 'CooldownPeriod', 'stop_duration': 60},
        {
            'method': 'StoplossGuard',
            'lookback_period_candles': 24,
            'trade_limit': 4,
            'stop_duration_candles': 4,
            'only_per_pair': False,
        },
    ]
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.protections._protection_handlers[1].global_stop = MagicMock(
        return_value=ProtectionReturn(True, dt_now() + timedelta(hours=1), 'asdf')
    )
    create_mock_trades(fee, is_short)
    freqtrade.handle_protections('ETC/BTC', '*')
    send_msg_mock = freqtrade.rpc.send_msg
    assert send_msg_mock.call_count == 2
    assert send_msg_mock.call_args_list[0][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER
    assert send_msg_mock.call_args_list[1][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER_GLOBAL


def test_create_trade_no_signal(
    default_conf_usdt: dict,
    fee: Any,
    mocker: Any,
) -> None:
    default_conf_usdt['dry_run'] = True
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, get_fee=fee)
    default_conf_usdt['stake_amount'] = 10
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)
    assert not freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_load_strategy_no_keys(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = deepcopy(default_conf_usdt)
    conf['runmode'] = runmode
    erm = mocker.patch('freqtrade.freqtradebot.ExchangeResolver.load_exchange')
    freqtrade = FreqtradeBot(conf)
    strategy_config = freqtrade.strategy.config
    assert id(strategy_config['exchange']) == id(conf['exchange'])
    assert strategy_config['exchange']['key'] == ''
    assert strategy_config['exchange']['secret'] == ''
    assert erm.call_count == 1
    ex_conf = erm.call_args_list[0][1]['exchange_config']
    assert id(ex_conf) != id(conf['exchange'])
    assert ex_conf['key'] != ''
    assert ex_conf['key'] == default_conf_usdt['exchange']['key']
    assert ex_conf['secret'] != ''
    assert ex_conf['secret'] == default_conf_usdt['exchange']['secret']


@pytest.mark.parametrize(
    'runmode',
    [
        RunMode.DRY_RUN,
        RunMode.LIVE,
    ],
)
def test_order_dict(
    default_conf_usdt: dict,
    mocker: Any,
    runmode: RunMode,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True,
    }
    conf['entry_pricing']['price_side'] = 'ask'
    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']
    caplog.clear()
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {
        'entry': 'market',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
    }
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    open_rate: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'is_short',
    [
        False,
        True,
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    fee: Any,
    mocker: Any,
    open_rate: float,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


@pytest.mark.parametrize(
    'whitelist,positions',
    [
        (['ETH/USDT'], 1),
        ([], 0),
    ],
)
def test_enter_positions_no_pairs_left(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt_open: dict,
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
    mocker.patch('freqtrade.configuration.config_validation._validate_whitelist')
    default_conf_usdt['exchange']['pair_whitelist'] = whitelist
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    assert n == positions
    if positions:
        assert not log_has_re('No currency pair in active pair whitelist.*', caplog)
        n = freqtrade.enter_positions()
        assert n == 0
        assert log_has_re('No currency pair in active pair whitelist.*', caplog)
    else:
        assert n == 0
        assert log_has('Active pair whitelist is empty.', caplog)


@pytest.mark.usefixtures('init_persistence')
def test_enter_positions_global_pairlock(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_buy_order_usdt: dict,
    fee: Any,
    mocker: Any,
    caplog: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    message = 'Global pairlock active until.* Not creating new trades.'
    n = freqtrade.enter_positions()
    assert n == 0
    assert not log_has_re(message, caplog)
    caplog.clear()
    PairLocks.lock_pair('*', dt_now() + timedelta(minutes=20), 'Just because', side='*')
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has_re(message, caplog)


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_edge_called_in_process(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']


def test_edge_overrides_stake_amount(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['dry_run_wallet'] = 999.9
    freqtrade = FreqtradeBot(edge_conf)
    assert (
        freqtrade.wallets.get_trade_stake_amount('NEO/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.2
    )
    assert (
        freqtrade.wallets.get_trade_stake_amount('LTC/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.21
    )


@pytest.mark.parametrize(
    'buy_price_mult,ignore_strat_sl',
    [
        (0.79, False),
        (0.85, True),
    ],
)
def test_edge_overrides_stoploss(
    limit_order: dict,
    fee: Any,
    caplog: Any,
    mocker: Any,
    buy_price_mult: float,
    ignore_strat_sl: bool,
    edge_conf: dict,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')
    enter_price = limit_order['buy']['price']
    ticker_val = {'bid': enter_price, 'ask': enter_price, 'last': enter_price}
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value=ticker_val),
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(edge_conf)
    freqtrade.active_pair_whitelist = ['NEO/BTC']
    patch_get_signal(freqtrade)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    caplog.clear()
    ticker_val.update(
        {
            'bid': enter_price * buy_price_mult,
            'ask': enter_price * buy_price_mult,
            'last': enter_price * buy_price_mult,
        }
    )
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has_re('Exit for NEO/BTC detected. Reason: stop_loss.*', caplog)
        assert trade.exit_reason == ExitType.STOP_LOSS.value
        assert trade.sell_reason == ExitType.STOP_LOSS.value


def test_total_open_trades_stakes(
    mocker: Any,
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
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


@pytest.mark.parametrize(
    'is_short,open_rate',
    [
        (False, 2.0),
        (True, 2.2),
    ],
)
def test_create_trade_minimal_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    limit_order_open: dict,
    is_short: bool,
    open_rate: float,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]),
        get_fee=fee,
    )
    whitelist = deepcopy(default_conf_usdt['exchange']['pair_whitelist'])
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    send_msg_mock = patch_RPCManager(mocker)
    send_msg_mock.reset_mock()
    freqtrade.create_trade('ETH/USDT')
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade is not None
    assert pytest.approx(trade.stake_amount) == 60.0
    assert trade.is_open
    assert trade.open_date is not None
    assert trade.exchange == 'binance'
    oobj = Order.parse_from_ccxt_object(limit_order_open[entry_side(is_short)], 'ADA/USDT', entry_side(is_short))
    trade.update_trade(oobj)
    assert send_msg_mock.call_count == 1
    entry_msg = send_msg_mock.call_args_list[0][0][0]
    assert entry_msg['type'] == RPCMessageType.ENTRY
    assert entry_msg['stake_amount'] == trade.stake_amount
    assert entry_msg['stake_currency'] == default_conf_usdt['stake_currency']
    assert entry_msg['pair'] == 'ETH/USDT'
    assert entry_msg['direction'] == ('Short' if is_short else 'Long')
    assert entry_msg['sub_trade'] is False
    assert trade.open_rate == open_rate
    assert trade.amount == 30.0
    assert whitelist == default_conf_usdt['exchange']['pair_whitelist']


def test_create_trade_no_stake_amount(
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
    )
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')


@pytest.mark.parametrize(
    'amend_last,wallet,max_open,lsamr,expected',
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
    default_conf_usdt: dict,
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    limit_buy_order_usdt_open: dict,
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
    default_conf_usdt['dry_run_wallet'] = wallet
    default_conf_usdt['amend_last_stake_amount'] = amend_last
    default_conf_usdt['last_stake_amount_min_ratio'] = lsamr
    freqtrade = FreqtradeBot(default_conf_usdt)
    for i in range(0, max_open):
        if expected[i] is not None:
            limit_buy_order_usdt_open['id'] = str(i)
            result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
            assert pytest.approx(result) == expected[i]
            freqtrade.execute_entry('ETH/USDT', result)
        else:
            with pytest.raises(DependencyException):
                freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)


def test_edge_called_in_process(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']


def test_edge_overrides_stake_amount(mocker: Any, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['dry_run_wallet'] = 999.9
    freqtrade = FreqtradeBot(edge_conf)
    assert (
        freqtrade.wallets.get_trade_stake_amount('NEO/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.2
    )
    assert (
        freqtrade.wallets.get_trade_stake_amount('LTC/BTC', 1, freqtrade.edge)
        == 999.9 * 0.5 * 0.01 / 0.21
    )
