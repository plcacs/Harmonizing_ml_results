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
from tests.conftest import EXMS, create_mock_trades, create_mock_trades_usdt, get_patched_freqtradebot, get_patched_worker, log_has, log_has_re, patch_edge, patch_exchange, patch_get_signal, patch_wallet, patch_whitelist
from tests.conftest_trades import MOCK_TRADE_COUNT, entry_side, exit_side, mock_order_2, mock_order_2_sell, mock_order_3, mock_order_3_sell, mock_order_4, mock_order_5_stoploss, mock_order_6_sell
from tests.conftest_trades_usdt import mock_trade_usdt_4

def patch_RPCManager(mocker: MagicMock) -> MagicMock:
    """
    This function mock RPC manager to avoid repeating this code in almost every tests
    :param mocker: mocker to patch RPCManager class
    :return: RPCManager.send_msg MagicMock to track if this method is called
    """
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    rpc_mock = mocker.patch('freqtrade.freqtradebot.RPCManager.send_msg', MagicMock())
    return rpc_mock

def test_freqtradebot_state(mocker: MagicMock, default_conf_usdt: dict, markets: dict) -> None:
    mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=markets))
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    assert freqtrade.state is State.RUNNING
    default_conf_usdt.pop('initial_state')
    freqtrade = FreqtradeBot(default_conf_usdt)
    assert freqtrade.state is State.STOPPED

def test_process_stopped(mocker: MagicMock, default_conf_usdt: dict) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    coo_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.cancel_all_open_orders')
    freqtrade.process_stopped()
    assert coo_mock.call_count == 0
    default_conf_usdt['cancel_open_orders_on_exit'] = True
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.process_stopped()
    assert coo_mock.call_count == 1

def test_process_calls_sendmsg(mocker: MagicMock, default_conf_usdt: dict) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.process()
    assert freqtrade.rpc.process_msg_queue.call_count == 1

def test_bot_cleanup(mocker: MagicMock, default_conf_usdt: dict, caplog: logging.LogRecord) -> None:
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

def test_bot_cleanup_db_errors(mocker: MagicMock, default_conf_usdt: dict, caplog: logging.LogRecord) -> None:
    mocker.patch('freqtrade.freqtradebot.Trade.commit', side_effect=OperationalException())
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.check_for_open_trades', side_effect=OperationalException())
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.emc = MagicMock()
    freqtrade.emc.shutdown = MagicMock()
    freqtrade.cleanup()
    assert freqtrade.emc.shutdown.call_count == 1

@pytest.mark.parametrize('runmode', [RunMode.DRY_RUN, RunMode.LIVE])
def test_order_dict(default_conf_usdt: dict, mocker: MagicMock, runmode: RunMode, caplog: logging.LogRecord) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {'entry': 'market', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': True}
    conf['entry_pricing']['price_side'] = 'ask'
    freqtrade = FreqtradeBot(conf)
    if runmode == RunMode.LIVE:
        assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)
    assert freqtrade.strategy.order_types['stoploss_on_exchange']
    caplog.clear()
    conf = default_conf_usdt.copy()
    conf['runmode'] = runmode
    conf['order_types'] = {'entry': 'market', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': False}
    freqtrade = FreqtradeBot(conf)
    assert not freqtrade.strategy.order_types['stoploss_on_exchange']
    assert not log_has_re('.*stoploss_on_exchange .* dry-run', caplog)

def test_get_trade_stake_amount(default_conf_usdt: dict, mocker: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(default_conf_usdt)
    result = freqtrade.wallets.get_trade_stake_amount('ETH/USDT', 1)
    assert result == default_conf_usdt['stake_amount']

@pytest.mark.parametrize('runmode', [RunMode.DRY_RUN, RunMode.LIVE])
def test_load_strategy_no_keys(default_conf_usdt: dict, mocker: MagicMock, runmode: RunMode, caplog: logging.LogRecord) -> None:
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

@pytest.mark.parametrize('amend_last,wallet,max_open,lsamr,expected', [(False, 120, 2, 0.5, [60, None]), (True, 120, 2, 0.5, [60, 58.8]), (False, 180, 3, 0.5, [60, 60, None]), (True, 180, 3, 0.5, [60, 60, 58.2]), (False, 122, 3, 0.5, [60, 60, None]), (True, 122, 3, 0.5, [60, 60, 0.0]), (True, 167, 3, 0.5, [60, 60, 45.33]), (True, 122, 3, 1, [60, 60, 0.0])])
def test_check_available_stake_amount(default_conf_usdt: dict, ticker_usdt: MagicMock, mocker: MagicMock, fee: MagicMock, limit_buy_order_usdt_open: dict, amend_last: bool, wallet: float, max_open: int, lsamr: float, expected: list) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
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

def test_edge_called_in_process(mocker: MagicMock, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_edge(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(edge_conf)
    patch_get_signal(freqtrade)
    freqtrade.process()
    assert freqtrade.active_pair_whitelist == ['NEO/BTC', 'LTC/BTC']

def test_edge_overrides_stake_amount(mocker: MagicMock, edge_conf: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['dry_run_wallet'] = 999.9
    freqtrade = FreqtradeBot(edge_conf)
    assert freqtrade.wallets.get_trade_stake_amount('NEO/BTC', 1, freqtrade.edge) == 999.9 * 0.5 * 0.01 / 0.2
    assert freqtrade.wallets.get_trade_stake_amount('LTC/BTC', 1, freqtrade.edge) == 999.9 * 0.5 * 0.01 / 0.21

@pytest.mark.parametrize('buy_price_mult,ignore_strat_sl', [(0.79, False), (0.85, True)])
def test_edge_overrides_stoploss(limit_order: dict, fee: MagicMock, caplog: logging.LogRecord, mocker: MagicMock, buy_price_mult: float, ignore_strat_sl: bool, edge_conf: dict) -> None:
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
    ticker_val.update({'bid': enter_price * buy_price_mult, 'ask': enter_price * buy_price_mult, 'last': enter_price * buy_price_mult})
    assert freqtrade.handle_trade(trade) is not ignore_strat_sl
    if not ignore_strat_sl:
        assert log_has_re('Exit for NEO/BTC detected. Reason: stop_loss.*', caplog)
        assert trade.exit_reason == ExitType.STOP_LOSS.value
        assert trade.sell_reason == ExitType.STOP_LOSS.value

def test_total_open_trades_stakes(mocker: MagicMock, default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 2
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value=limit_buy_order_usdt_open), get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=False))
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

@pytest.mark.parametrize('is_short,open_rate', [(False, 2.0), (True, 2.2)])
def test_create_trade(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_order: dict, limit_order_open: dict, fee: MagicMock, mocker: MagicMock, is_short: bool, open_rate: float) -> None:
    send_msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]), get_fee=fee)
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

def test_create_trade_no_stake_amount(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, mocker: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=default_conf_usdt['stake_amount'] * 0.5)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    with pytest.raises(DependencyException, match='.*stake amount.*'):
        freqtrade.create_trade('ETH/USDT')

@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.parametrize('stake_amount,create,amount_enough,max_open_trades', [(5.0, True, True, 99), (0.042, True, False, 99), (0, False, True, 99), (UNLIMITED_STAKE_AMOUNT, False, True, 0)])
def test_create_trade_minimal_amount(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_usdt_open: dict, fee: MagicMock, mocker: MagicMock, stake_amount: float, create: bool, amount_enough: bool, max_open_trades: int, caplog: logging.LogRecord, is_short: bool) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    enter_mock = MagicMock(return_value=limit_buy_order_usdt_open[entry_side(is_short)])
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=enter_mock, get_fee=fee)
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

@pytest.mark.parametrize('whitelist,positions', [(['ETH/USDT'], 1), ([], 0)])
def test_enter_positions_no_pairs_left(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_usdt_open: dict, fee: MagicMock, whitelist: list, positions: int, mocker: MagicMock, caplog: logging.LogRecord) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value=limit_buy_order_usdt_open), get_fee=fee)
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
def test_enter_positions_global_pairlock(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_usdt: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}), get_fee=fee)
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
def test_handle_protections(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, is_short: bool) -> None:
    default_conf_usdt['_strategy_protections'] = [{'method': 'CooldownPeriod', 'stop_duration': 60}, {'method': 'StoplossGuard', 'lookback_period_candles': 24, 'trade_limit': 4, 'stop_duration_candles': 4, 'only_per_pair': False}]
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.protections._protection_handlers[1].global_stop = MagicMock(return_value=ProtectionReturn(True, dt_now() + timedelta(hours=1), 'asdf'))
    create_mock_trades(fee, is_short)
    freqtrade.handle_protections('ETC/BTC', '*')
    send_msg_mock = freqtrade.rpc.send_msg
    assert send_msg_mock.call_count == 2
    assert send_msg_mock.call_args_list[0][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER
    assert send_msg_mock.call_args_list[1][0][0]['type'] == RPCMessageType.PROTECTION_TRIGGER_GLOBAL

def test_create_trade_no_signal(default_conf_usdt: dict, fee: MagicMock, mocker: MagicMock) -> None:
    default_conf_usdt['dry_run'] = True
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, get_fee=fee)
    default_conf_usdt['stake_amount'] = 10
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)
    assert not freqtrade.create_trade('ETH/USDT')

@pytest.mark.parametrize('max_open', range(0, 5))
@pytest.mark.parametrize('tradable_balance_ratio,modifier', [(1.0, 1), (0.99, 0.8), (0.5, 0.5)])
def test_create_trades_multiple_trades(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, mocker: MagicMock, limit_buy_order_usdt_open: dict, max_open: int, tradable_balance_ratio: float, modifier: float) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = max_open
    default_conf_usdt['tradable_balance_ratio'] = tradable_balance_ratio
    default_conf_usdt['dry_run_wallet'] = 60.0 * max_open
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value=limit_buy_order_usdt_open), get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    trades = Trade.get_open_trades()
    assert n == max(int(max_open * modifier), 0)
    assert len(trades) == max(int(max_open * modifier), 0)

def test_create_trades_preopen(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, mocker: MagicMock, limit_buy_order_usdt_open: dict, caplog: logging.LogRecord) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf_usdt['max_open_trades'] = 4
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value=limit_buy_order_usdt_open), get_fee=fee)
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

@pytest.mark.parametrize('is_short', [False, True])
def test_process_trade_creation(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_order_open: dict, limit_order: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    ticker_side = 'ask' if is_short else 'bid'
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value=limit_order_open[entry_side(is_short)]), fetch_order=MagicMock(return_value=limit_order[entry_side(is_short)]), get_fee=fee)
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
    assert log_has(f'{('Short' if is_short else 'Long')} signal found: about create a new trade for ETH/USDT with stake_amount: 60.0 ...', caplog)
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

def test_process_exchange_failures(default_conf_usdt: dict, ticker_usdt: MagicMock, mocker: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, reload_markets=MagicMock(), create_order=MagicMock(side_effect=TemporaryError))
    sleep_mock = mocker.patch('time.sleep')
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)
    mocker.patch(f'{EXMS}.reload_markets', MagicMock(side_effect=TemporaryError))
    worker._process_running()
    assert sleep_mock.called is True

def test_process_operational_exception(default_conf_usdt: dict, ticker_usdt: MagicMock, mocker: MagicMock) -> None:
    msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(side_effect=OperationalException))
    worker = Worker(args=None, config=default_conf_usdt)
    patch_get_signal(worker.freqtrade)
    assert worker.freqtrade.state == State.RUNNING
    worker._process_running()
    assert worker.freqtrade.state == State.STOPPED
    assert 'OperationalException' in msg_mock.call_args_list[-1][0][0]['status']

def test_process_trade_handling(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_usdt_open: dict, fee: MagicMock, mocker: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value=limit_buy_order_usdt_open), fetch_order=MagicMock(return_value=limit_buy_order_usdt_open), get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    trades = Trade.get_open_trades()
    assert not trades
    freqtrade.process()
    trades = Trade.get_open_trades()
    assert len(trades) == 1
    freqtrade.process()
    assert len(trades) == 1

def test_process_trade_no_whitelist_pair(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_usdt: dict, fee: MagicMock, mocker: MagicMock) -> None:
    """Test process with trade not in pair list"""
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value={'id': limit_buy_order_usdt['id']}), fetch_order=MagicMock(return_value=limit_buy_order_usdt), get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    pair = 'BLK/BTC'
    assert pair not in default_conf_usdt['exchange']['pair_whitelist']
    Trade.session.add(Trade(pair=pair, stake_amount=0.001, fee_open=fee.return_value, fee_close=fee.return_value, is_open=True, amount=20, open_rate=0.01, exchange='binance'))
    Trade.session.add(Trade(pair='ETH/USDT', stake_amount=0.001, fee_open=fee.return_value, fee_close=fee.return_value, is_open=True, amount=12, open_rate=0.001, exchange='binance'))
    Trade.commit()
    assert pair not in freqtrade.active_pair_whitelist
    freqtrade.process()
    assert pair in freqtrade.active_pair_whitelist
    assert len(freqtrade.active_pair_whitelist) == len(set(freqtrade.active_pair_whitelist))

def test_process_informative_pairs_added(default_conf_usdt: dict, ticker_usdt: MagicMock, mocker: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    refresh_mock = MagicMock()
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(side_effect=TemporaryError), refresh_latest_ohlcv=refresh_mock)
    inf_pairs = MagicMock(return_value=[('BTC/ETH', '1m', CandleType.SPOT), ('ETH/USDT', '1h', CandleType.SPOT)])
    mocker.patch.multiple('freqtrade.strategy.interface.IStrategy', get_exit_signal=MagicMock(return_value=(False, False)), get_entry_signal=MagicMock(return_value=(None, None)))
    mocker.patch('time.sleep', return_value=None)
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.informative_pairs = inf_pairs
    freqtrade.process()
    assert inf_pairs.call_count == 1
    assert refresh_mock.call_count == 1
    assert ('BTC/ETH', '1m', CandleType.SPOT) in refresh_mock.call_args[0][0]
    assert ('ETH/USDT', '1h', CandleType.SPOT) in refresh_mock.call_args[0][0]
    assert ('ETH/USDT', default_conf_usdt['timeframe'], CandleType.SPOT) in refresh_mock.call_args[0][0]

@pytest.mark.parametrize('is_short,trading_mode,exchange_name,margin_mode,liq_buffer,liq_price', [(False, 'spot', 'binance', None, 0.0, None), (True, 'spot', 'binance', None, 0.0, None), (False, 'spot', 'gate', None, 0.0, None), (True, 'spot', 'gate', None, 0.0, None), (False, 'spot', 'okx', None, 0.0, None), (True, 'spot', 'okx', None, 0.0, None), (True, 'futures', 'binance', 'isolated', 0.0, 11.88151815181518), (False, 'futures', 'binance', 'isolated', 0.0, 8.080471380471382), (True, 'futures', 'gate', 'isolated', 0.0, 11.87413417771621), (False, 'futures', 'gate', 'isolated', 0.0, 8.085708510208207), (True, 'futures', 'binance', 'isolated', 0.05, 11.7874422442244), (False, 'futures', 'binance', 'isolated', 0.05, 8.17644781144781), (True, 'futures', 'gate', 'isolated', 0.05, 11.7804274688304), (False, 'futures', 'gate', 'isolated', 0.05, 8.181423084697796), (True, 'futures', 'okx', 'isolated', 0.0, 11.87413417771621), (False, 'futures', 'okx', 'isolated', 0.0, 8.085708510208207), (True, 'futures', 'bybit', 'isolated', 0.0, 11.9), (False, 'futures', 'bybit', 'isolated', 0.0, 8.1)])
def test_execute_entry(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, limit_order: dict, limit_order_open: dict, is_short: bool, trading_mode: str, exchange_name: str, margin_mode: str, liq_buffer: float, liq_price: float) -> None:
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
    open_order = limit_order_open[exit_side(is_short)]
    order = limit_order[exit_side(is_short)]
    default_conf_usdt['trading_mode'] = trading_mode
    default_conf_usdt['liquidation_buffer'] = liq_buffer
    leverage = 1.0 if trading_mode == 'spot' else 5.0
    default_conf_usdt['exchange']['name'] = exchange_name
    if margin_mode:
        default_conf_usdt['margin_mode'] = margin_mode
    mocker.patch('freqtrade.exchange.gate.Gate.validate_ordertypes')
    patch_RPCManager(mocker)
    patch_exchange(mocker, exchange=exchange_name)
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    freqtrade.strategy.leverage = MagicMock(return_value=leverage)
    stake_amount = 2
    bid = 0.11
    enter_rate_mock = MagicMock(return_value=bid)
    enter_mm = MagicMock(return_value=open_order)
    mocker.patch.multiple(EXMS, get_rate=enter_rate_mock, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=enter_mm, get_min_pair_stake_amount=MagicMock(return_value=1), get_fee=fee, get_maintenance_ratio_and_amt=MagicMock(return_value=(0.01, 0.01)), get_max_leverage=MagicMock(return_value=10))
    mocker.patch.multiple('freqtrade.exchange.okx.Okx', get_max_pair_stake_amount=MagicMock(return_value=500000))
    pair = 'ETH/USDT'
    assert not freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    assert enter_rate_mock.call_count == 1
    assert enter_mm.call_count == 0
    assert freqtrade.strategy.confirm_trade_entry.call_count == 1
    enter_rate_mock.reset_mock()
    open_order['id'] = '22'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_entry(pair, stake_amount)
    assert enter_rate_mock.call_count == 2
    assert enter_mm.call_count == 1
    call_args = enter_mm.call_args_list[0][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == bid
    assert pytest.approx(call_args['amount']) == round(stake_amount / bid * leverage, 8)
    enter_rate_mock.reset_mock()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade
    assert trade.is_open is True
    assert trade.has_open_orders
    assert '22' in trade.open_orders_ids
    open_order['id'] = '33'
    fix_price = 0.06
    assert freqtrade.execute_entry(pair, stake_amount, fix_price, is_short=is_short)
    assert enter_rate_mock.call_count == 1
    assert enter_mm.call_count == 2
    call_args = enter_mm.call_args_list[1][1]
    assert call_args['pair'] == pair
    assert call_args['rate'] == fix_price
    assert pytest.approx(call_args['amount']) == round(stake_amount / fix_price * leverage, 8)
    order['status'] = 'closed'
    order['average'] = 10
    order['cost'] = 300
    order['id'] = '444'
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=order))
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[2]
    trade.is_short = is_short
    assert trade
    assert not trade.has_open_orders
    assert trade.open_rate == 10
    assert trade.stake_amount == round(order['average'] * order['filled'] / leverage, 8)
    assert pytest.approx(trade.liquidation_price) == liq_price
    order['status'] = 'expired'
    order['amount'] = 30.0
    order['filled'] = 20.0
    order['remaining'] = 10.0
    order['average'] = 0.5
    order['cost'] = 10.0
    order['id'] = '555'
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=order))
    assert freqtrade.execute_entry(pair, stake_amount)
    trade = Trade.session.scalars(select(Trade)).all()[3]
    trade.is_short = is_short
    assert trade
    assert not trade.has_open_orders
    assert trade.open_rate == 0.5
    assert trade.stake_amount == round(order['average'] * order['filled'] / leverage, 8)
    order['status'] = 'open'
    order['id'] = '556'
    freqtrade.strategy.custom_stake_amount = lambda **kwargs: 150.0
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[4]
    trade.is_short = is_short
    assert trade
    assert pytest.approx(trade.stake_amount) == 150
    order['id'] = '557'
    freqtrade.strategy.custom_stake_amount = lambda **kwargs: 20 / 0
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[5]
    trade.is_short = is_short
    assert trade
    assert pytest.approx(trade.stake_amount) == 2.0
    order['status'] = 'rejected'
    order['amount'] = 30.0 * leverage
    order['filled'] = 0.0
    order['remaining'] = 30.0
    order['average'] = 0.5
    order['cost'] = 0.0
    order['id'] = '66'
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=order))
    assert not freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    assert freqtrade.strategy.leverage.call_count == 0 if trading_mode == 'spot' else 2
    mocker.patch(f'{EXMS}.get_rate', MagicMock(return_value=0.0))
    with pytest.raises(PricingError, match='Could not determine entry price.'):
        freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    mocker.patch(f'{EXMS}.get_rate', return_value=0.5)
    order['status'] = 'open'
    order['id'] = '5566'
    freqtrade.strategy.custom_entry_price = lambda **kwargs: 0.508
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[6]
    trade.is_short = is_short
    assert trade
    assert trade.open_rate_requested == 0.508
    order['status'] = 'open'
    order['id'] = '5567'
    freqtrade.strategy.custom_entry_price = lambda **kwargs: None
    mocker.patch.multiple(EXMS, get_rate=MagicMock(return_value=10))
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[7]
    trade.is_short = is_short
    assert trade
    assert trade.open_rate_requested == 10
    order['status'] = 'open'
    order['id'] = '5568'
    freqtrade.strategy.custom_entry_price = lambda **kwargs: 'string price'
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[8]
    trade.is_short = is_short
    assert trade
    assert trade.open_rate_requested == 10
    order['status'] = 'open'
    order['id'] = '55672'
    mocker.patch.multiple(EXMS, get_max_pair_stake_amount=MagicMock(return_value=500))
    freqtrade.exchange.get_max_pair_stake_amount = MagicMock(return_value=500)
    assert freqtrade.execute_entry(pair, 2000, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).all()[9]
    trade.is_short = is_short
    assert pytest.approx(trade.stake_amount) == 500
    order['id'] = '55673'
    freqtrade.strategy.leverage.reset_mock()
    assert freqtrade.execute_entry(pair, 200, leverage_=3, is_short=is_short)
    assert freqtrade.strategy.leverage.call_count == 0
    trade = Trade.session.scalars(select(Trade)).all()[10]
    assert trade.leverage == 1 if trading_mode == 'spot' else 3

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_entry_confirm_error(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, limit_order: dict, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(return_value=limit_order[entry_side(is_short)]), get_rate=MagicMock(return_value=0.11), get_min_pair_stake_amount=MagicMock(return_value=1), get_fee=fee)
    stake_amount = 2
    pair = 'ETH/USDT'
    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=ValueError)
    assert freqtrade.execute_entry(pair, stake_amount)
    limit_order[entry_side(is_short)]['id'] = '222'
    freqtrade.strategy.confirm_trade_entry = MagicMock(side_effect=Exception)
    assert freqtrade.execute_entry(pair, stake_amount)
    limit_order[entry_side(is_short)]['id'] = '2223'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    assert freqtrade.execute_entry(pair, stake_amount)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=False)
    assert not freqtrade.execute_entry(pair, stake_amount)

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_entry_fully_canceled_on_create(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, limit_order_open: dict, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_hce = mocker.spy(freqtrade, 'handle_cancel_enter')
    order = limit_order_open[entry_side(is_short)]
    pair = 'ETH/USDT'
    order['symbol'] = pair
    order['status'] = 'canceled'
    order['filled'] = 0.0
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(return_value=order), get_rate=MagicMock(return_value=0.11), get_min_pair_stake_amount=MagicMock(return_value=1), get_fee=fee)
    stake_amount = 2
    assert freqtrade.execute_entry(pair, stake_amount)
    assert mock_hce.call_count == 1
    trades = Trade.get_trades().all()
    assert len(trades) == 0

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_entry_min_leverage(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, limit_order: dict, is_short: bool) -> None:
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(return_value=limit_order[entry_side(is_short)]), get_rate=MagicMock(return_value=0.11), get_maintenance_ratio_and_amt=MagicMock(return_value=(0.0, 0.0)), _fetch_and_calculate_funding_fees=MagicMock(return_value=0), get_fee=fee, get_max_leverage=MagicMock(return_value=5.0))
    stake_amount = 2
    pair = 'SOL/BUSD:BUSD'
    freqtrade.strategy.leverage = MagicMock(return_value=5.0)
    assert freqtrade.execute_entry(pair, stake_amount, is_short=is_short)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.leverage == 5.0

@pytest.mark.parametrize('return_value,side_effect,log_message', [(False, None, 'Found no enter signals for whitelisted currencies. Trying again...'), (None, DependencyException, 'Unable to create trade for ETH/USDT: ')])
def test_enter_positions(mocker: MagicMock, default_conf_usdt: dict, return_value: bool, side_effect: Exception, log_message: str, caplog: logging.LogRecord) -> None:
    caplog.set_level(logging.DEBUG)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_ct = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trade', MagicMock(return_value=return_value, side_effect=side_effect))
    n = freqtrade.enter_positions()
    assert n == 0
    assert log_has(log_message, caplog)
    assert mock_ct.call_count == len(default_conf_usdt['exchange']['pair_whitelist'])

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_exit_positions(mocker: MagicMock, default_conf_usdt: dict, limit_order: dict, is_short: bool, caplog: logging.LogRecord) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch(f'{EXMS}.fetch_order', return_value=limit_order[entry_side(is_short)])
    mocker.patch(f'{EXMS}.get_trades_for_order', return_value=[])
    order_id = '123'
    trade = Trade(pair='ETH/USDT', fee_open=0.001, fee_close=0.001, open_rate=0.01, open_date=dt_now(), stake_amount=0.01, amount=11, exchange='binance', is_short=is_short, leverage=1)
    trade.orders.append(Order(ft_order_side=entry_side(is_short), price=0.01, ft_pair=trade.pair, ft_amount=trade.amount, ft_price=trade.open_rate, order_id=order_id))
    Trade.session.add(trade)
    Trade.commit()
    trades = [trade]
    freqtrade.wallets.update()
    n = freqtrade.exit_positions(trades)
    assert n == 1
    assert not log_has_re('Applying fee to amount for Trade .*', caplog)
    gra = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=0.0)
    n = freqtrade.exit_positions(trades)
    assert n == 1
    assert gra.call_count == 0

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_exit_positions_exception(mocker: MagicMock, default_conf_usdt: dict, limit_order: dict, caplog: logging.LogRecord, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    order = limit_order[entry_side(is_short)]
    mocker.patch(f'{EXMS}.fetch_order', return_value=order)
    order_id = '123'
    trade = Trade(pair='ETH/USDT', fee_open=0.001, fee_close=0.001, open_rate=0.01, open_date=dt_now(), stake_amount=0.01, amount=11, exchange='binance', is_short=is_short, leverage=1)
    trade.orders.append(Order(ft_order_side=entry_side(is_short), price=0.01, ft_pair=trade.pair, ft_amount=trade.amount, ft_price=trade.open_rate, order_id=order_id, ft_is_open=False, filled=11))
    Trade.session.add(trade)
    Trade.commit()
    freqtrade.wallets.update()
    trades = [trade]
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', side_effect=DependencyException())
    caplog.clear()
    n = freqtrade.exit_positions(trades)
    assert n == 0
    assert log_has('Unable to exit trade ETH/USDT: ', caplog)

@pytest.mark.parametrize('is_short', [False, True])
def test_update_trade_state(mocker: MagicMock, default_conf_usdt: dict, limit_order: dict, is_short: bool, caplog: logging.LogRecord) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    order = limit_order[entry_side(is_short)]
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._notify_enter')
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(side_effect=ValueError))
    order_id = order['id']
    trade = Trade(fee_open=0.001, fee_close=0.001, open_rate=0.01, open_date=dt_now(), amount=11, exchange='binance', is_short=is_short, leverage=1)
    trade.orders.append(Order(ft_order_side=entry_side(is_short), price=0.01, order_id=order_id))
    freqtrade.strategy.order_filled = MagicMock(return_value=None)
    assert not freqtrade.update_trade_state(trade, None)
    assert log_has_re('Orderid for trade .* is empty.', caplog)
    caplog.clear()
    freqtrade.update_trade_state(trade, order_id)
    assert not log_has_re('Applying fee to .*', caplog)
    caplog.clear()
    assert not trade.has_open_orders
    assert trade.amount == order['amount']
    assert freqtrade.strategy.order_filled.call_count == 1
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', return_value=0.01)
    assert trade.amount == 30.0
    freqtrade.update_trade_state(trade, order_id)
    assert trade.amount == 29.99
    assert not trade.has_open_orders
    trade.is_open = True
    freqtrade.update_trade_state(trade, order_id)
    assert log_has_re('Found open order for.*', caplog)
    limit_buy_order_usdt_new = deepcopy(limit_order)
    limit_buy_order_usdt_new['filled'] = 0.0
    limit_buy_order_usdt_new['status'] = 'canceled'
    freqtrade.strategy.order_filled = MagicMock(return_value=None)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', side_effect=ValueError)
    mocker.patch(f'{EXMS}.fetch_order', return_value=limit_buy_order_usdt_new)
    res = freqtrade.update_trade_state(trade, order_id)
    assert res is True
    assert freqtrade.strategy.order_filled.call_count == 0

@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.parametrize('initial_amount,has_rounding_fee', [(30.0 + 1e-14, True), (8.0, False)])
def test_update_trade_state_withorderdict(default_conf_usdt: dict, trades_for_order: list, limit_order: dict, fee: MagicMock, mocker: MagicMock, initial_amount: float, has_rounding_fee: bool, is_short: bool, caplog: logging.LogRecord) -> None:
    order = limit_order[entry_side(is_short)]
    trades_for_order[0]['amount'] = initial_amount
    order_id = 'oid_123456'
    order['id'] = order_id
    mocker.patch(f'{EXMS}.get_trades_for_order', return_value=trades_for_order)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._notify_enter')
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(side_effect=ValueError))
    patch_exchange(mocker)
    amount = sum((x['amount'] for x in trades_for_order))
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    caplog.clear()
    trade = Trade(pair='LTC/USDT', amount=amount, exchange='binance', open_rate=0.245441, open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value)
    trade.orders.append(Order(ft_order_side=entry_side(is_short), ft_pair=trade.pair, ft_is_open=True, order_id=order_id))
    log_text = 'Applying fee on amount for .*'
    freqtrade.update_trade_state(trade, order_id, order)
    assert trade.amount != amount
    if has_rounding_fee:
        assert pytest.approx(trade.amount) == 29.992
        assert log_has_re(log_text, caplog)
    else:
        assert pytest.approx(trade.amount) == order['amount']
        assert not log_has_re(log_text, caplog)

@pytest.mark.parametrize('is_short', [False, True])
def test_update_trade_state_exception(mocker: MagicMock, default_conf_usdt: dict, is_short: bool, limit_order: dict, caplog: logging.LogRecord) -> None:
    order = limit_order[entry_side(is_short)]
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch(f'{EXMS}.fetch_order', return_value=order)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot._notify_enter')
    trade = MagicMock()
    trade.amount = 123
    open_order_id = '123'
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', side_effect=DependencyException())
    freqtrade.update_trade_state(trade, open_order_id)
    assert log_has('Could not update trade amount: ', caplog)

def test_update_trade_state_orderexception(mocker: MagicMock, default_conf_usdt: dict, caplog: logging.LogRecord) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(side_effect=InvalidOrderException))
    trade = MagicMock()
    open_order_id = '123'
    grm_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', MagicMock())
    freqtrade.update_trade_state(trade, open_order_id)
    assert grm_mock.call_count == 0
    assert log_has(f'Unable to fetch order {open_order_id}: ', caplog)

@pytest.mark.parametrize('is_short', [False, True])
def test_update_trade_state_sell(default_conf_usdt: dict, trades_for_order: list, limit_buy_order_usdt_open: dict, limit_buy_order_usdt: dict, fee: MagicMock, mocker: MagicMock, is_short: bool) -> None:
    buy_order = limit_order[entry_side(is_short)]
    open_order = limit_order_open[exit_side(is_short)]
    l_order = limit_order[exit_side(is_short)]
    mocker.patch(f'{EXMS}.get_trades_for_order', return_value=trades_for_order)
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(side_effect=ValueError))
    wallet_mock = MagicMock()
    mocker.patch('freqtrade.wallets.Wallets.update', wallet_mock)
    patch_exchange(mocker)
    amount = sum((x['amount'] for x in trades_for_order))
    trade = Trade(pair='LTC/ETH', amount=amount, exchange='binance', open_rate=0.245441, open_date=dt_now(), fee_open=fee.return_value, fee_close=fee.return_value, is_open=True, interest_rate=0.0005, leverage=1, is_short=is_short)
    order = Order.parse_from_ccxt_object(buy_order, 'LTC/ETH', entry_side(is_short))
    trade.orders.append(order)
    order = Order.parse_from_ccxt_object(open_order, 'LTC/ETH', exit_side(is_short))
    trade.orders.append(order)
    assert order.status == 'open'
    freqtrade.update_trade_state(trade, trade.open_orders_ids[-1], l_order)
    assert trade.amount == l_order['amount']
    assert wallet_mock.call_count == 1
    assert not trade.is_open
    assert order.status == 'closed'

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_trade(default_conf_usdt: dict, limit_order_open: dict, limit_order: dict, fee: MagicMock, mocker: MagicMock, is_short: bool) -> None:
    open_order = limit_order_open[exit_side(is_short)]
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 2.19, 'ask': 2.2, 'last': 2.19}), create_order=MagicMock(side_effect=[enter_order, open_order]), get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade
    time.sleep(0.01)
    assert trade.is_open is True
    freqtrade.wallets.update()
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)
    assert freqtrade.handle_trade(trade) is True
    assert trade.open_orders_ids[-1] == exit_order['id']
    trade.orders[-1].ft_is_open = False
    trade.orders[-1].status = 'closed'
    trade.orders[-1].filled = trade.orders[-1].remaining
    trade.orders[-1].remaining = 0.0
    trade.update_trade(trade.orders[-1])
    assert trade.close_rate == (2.0 if is_short else 2.2)
    assert pytest.approx(trade.close_profit) == 5.685
    assert trade.close_date is not None
    assert trade.exit_reason == 'sell_signal1'

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_overlapping_signals(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_order_open: dict, fee: MagicMock, mocker: MagicMock, is_short: bool) -> None:
    open_order = limit_order_open[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(side_effect=[open_order, {'id': 1234553382}]), get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, enter_short=True, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=True, exit_long=True)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trades = Trade.session.scalars(select(Trade)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trades = Trade.session.scalars(select(Trade)).all()
    for trade in trades:
        trade.is_short = is_short
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True
    patch_get_signal(freqtrade, enter_long=False)
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.session.scalars(select(Trade)).all()
    for trade in trades:
        trade.is_short = is_short
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, enter_short=True, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=True, exit_long=True)
    assert freqtrade.handle_trade(trades[0]) is False
    trades = Trade.session.scalars(select(Trade)).all()
    for trade in trades:
        trade.is_short = is_short
    nb_trades = len(trades)
    assert nb_trades == 1
    assert trades[0].is_open is True
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    trades = Trade.session.scalars(select(Trade)).all()
    for trade in trades:
        trade.is_short = is_short
    assert freqtrade.handle_trade(trades[0]) is True

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_trade_roi(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_order_open: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    open_order = limit_order_open[exit_side(is_short)]
    caplog.set_level(logging.DEBUG)
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(side_effect=[open_order, {'id': 1234553382, 'amount': open_order['amount']}]), get_fee=fee)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    caplog.clear()
    patch_get_signal(freqtrade)
    assert freqtrade.handle_trade(trade)
    assert log_has('ETH/USDT - Required profit reached. exit_type=ExitType.ROI', caplog)

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_trade_use_exit_signal(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_order_open: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    enter_open_order = limit_order_open[exit_side(is_short)]
    exit_open_order = limit_order_open[entry_side(is_short)]
    caplog.set_level(logging.DEBUG)
    patch_RPCManager(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(side_effect=[enter_open_order, exit_open_order]), get_fee=fee)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    patch_get_signal(freqtrade, enter_long=False, exit_long=False)
    assert not freqtrade.handle_trade(trade)
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    assert freqtrade.handle_trade(trade)
    assert log_has('ETH/USDT - Sell signal received. exit_type=ExitType.EXIT_SIGNAL', caplog)

@pytest.mark.parametrize('is_short', [False, True])
def test_close_trade(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_order_open: dict, limit_order: dict, fee: MagicMock, mocker: MagicMock, is_short: bool) -> None:
    open_order = limit_order_open[exit_side(is_short)]
    enter_order = limit_order[exit_side(is_short)]
    exit_order = limit_order[entry_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value=open_order), get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade
    oobj = Order.parse_from_ccxt_object(enter_order, enter_order['symbol'], trade.entry_side)
    trade.update_trade(oobj)
    oobj = Order.parse_from_ccxt_object(exit_order, exit_order['symbol'], trade.exit_side)
    trade.update_trade(oobj)
    assert trade.is_open is False
    with pytest.raises(DependencyException, match='.*closed trade.*'):
        freqtrade.handle_trade(trade)

def test_bot_loop_start_called_once(mocker: MagicMock, default_conf_usdt: dict, caplog: logging.LogRecord) -> None:
    ftbot = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.create_trade')
    patch_get_signal(ftbot)
    ftbot.strategy.bot_loop_start = MagicMock(side_effect=ValueError)
    ftbot.strategy.analyze = MagicMock()
    ftbot.process()
    assert log_has_re('Strategy caused the following exception.*', caplog)
    assert ftbot.strategy.bot_loop_start.call_count == 1
    assert ftbot.strategy.analyze.call_count == 1

@pytest.mark.parametrize('is_short', [False, True])
def test_manage_open_orders_entry_usercustom(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_old: dict, open_trade: Trade, limit_sell_order_old: dict, fee: MagicMock, mocker: MagicMock, is_short: bool) -> None:
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    old_order['id'] = open_trade.open_orders_ids[0]
    default_conf_usdt['unfilledtimeout'] = {'entry': 1400, 'exit': 30}
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock(return_value=old_order)
    cancel_enter_order = deepcopy(old_order)
    cancel_enter_order['status'] = 'canceled'
    cancel_order_wr_mock = MagicMock(return_value=cancel_enter_order)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=old_order), cancel_order=cancel_order_mock, cancel_order_with_result=cancel_order_wr_mock, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade.is_short = is_short
    open_trade.orders[0].side = 'sell' if is_short else 'buy'
    open_trade.orders[0].ft_order_side = 'sell' if is_short else 'buy'
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=False)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    trades = Trade.session.scalars(select(Trade).where(Order.ft_is_open.is_(True)).where(Order.ft_order_side != 'stoploss').where(Order.ft_trade_id == Trade.id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 1
    freqtrade.strategy.check_entry_timeout = MagicMock(side_effect=KeyError)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    trades = Trade.session.scalars(select(Trade).where(Order.ft_is_open.is_(True)).where(Order.ft_order_side != 'stoploss').where(Order.ft_trade_id == Trade.id)).all()
    nb_trades = len(trades)
    assert nb_trades == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 1
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=True)
    freqtrade.manage_open_orders()
    assert cancel_order_wr_mock.call_count == 1
    assert rpc_mock.call_count == 2
    trades = Trade.session.scalars(select(Trade).where(Order.ft_is_open.is_(True)).where(Order.ft_order_side != 'stoploss').where(Order.ft_trade_id == Trade.id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    assert freqtrade.strategy.check_entry_timeout.call_count == 1

@pytest.mark.parametrize('is_short', [False, True])
def test_manage_open_orders_entry(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_old: dict, open_trade: Trade, limit_sell_order_old: dict, fee: MagicMock, mocker: MagicMock, is_short: bool) -> None:
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    rpc_mock = patch_RPCManager(mocker)
    order = Order.parse_from_ccxt_object(old_order, 'mocked', 'buy' if is_short else 'sell')
    open_trade.orders[0] = order
    limit_entry_cancel = deepcopy(old_order)
    limit_entry_cancel['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_entry_cancel)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=old_order), cancel_order_with_result=cancel_order_mock, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade.is_short = is_short
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=False)
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=1234)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 2
    trades = Trade.session.scalars(select(Trade).where(Order.ft_is_open.is_(True)).where(Order.ft_order_side != 'stoploss').where(Order.ft_trade_id == Trade.id)).all()
    nb_trades = len(trades)
    assert nb_trades == 0
    assert freqtrade.strategy.check_entry_timeout.call_count == 0
    assert freqtrade.strategy.adjust_entry_price.call_count == 0

@pytest.mark.parametrize('is_short', [False, True])
def test_adjust_entry_cancel(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_old: dict, open_trade: Trade, limit_sell_order_old: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    old_order['id'] = open_trade.open_orders[0].order_id
    limit_entry_cancel = deepcopy(old_order)
    limit_entry_cancel['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_entry_cancel)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=old_order), cancel_order_with_result=cancel_order_mock, get_fee=fee)
    open_trade.is_short = is_short
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.strategy.ft_check_timed_out = MagicMock(return_value=False)
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=None)
    freqtrade.manage_open_orders()
    trades = Trade.session.scalars(select(Trade).where(Order.ft_trade_id == Trade.id)).all()
    assert len(trades) == 0
    assert len(Order.session.scalars(select(Order)).all()) == 0
    assert log_has_re(f'{("Sell" if is_short else "Buy")} order user requested order cancel*', caplog)
    assert log_has_re(f'{("Sell" if is_short else "Buy")} order fully cancelled.*', caplog)
    assert freqtrade.strategy.adjust_entry_price.call_count == 1

@pytest.mark.parametrize('is_short', [False, True])
def test_adjust_entry_replace_fail(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_old: dict, open_trade: Trade, limit_sell_order_old: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    old_order['id'] = open_trade.open_orders[0].order_id
    limit_entry_cancel = deepcopy(old_order)
    limit_entry_cancel['status'] = 'open'
    cancel_order_mock = MagicMock(return_value=limit_entry_cancel)
    fetch_order_mock = MagicMock(return_value=old_order)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=fetch_order_mock, cancel_order_with_result=cancel_order_mock, get_fee=fee)
    mocker.patch('freqtrade.freqtradebot.sleep')
    open_trade.is_short = is_short
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.strategy.ft_check_timed_out = MagicMock(return_value=False)
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=12234)
    freqtrade.manage_open_orders()
    trades = Trade.session.scalars(select(Trade).where(Order.ft_trade_id == Trade.id)).all()
    assert len(trades) == 0
    assert len(Order.session.scalars(select(Order)).all()) == 0
    assert fetch_order_mock.call_count == 4
    assert log_has_re('Could not fully cancel order.*, therefore not replacing\\.', caplog)
    assert freqtrade.strategy.adjust_entry_price.call_count == 1

@pytest.mark.parametrize('is_short', [False, True])
def test_adjust_entry_replace_fail_create_order(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_old: dict, open_trade: Trade, limit_sell_order_old: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    old_order['id'] = open_trade.open_orders[0].order_id
    limit_entry_cancel = deepcopy(old_order)
    limit_entry_cancel['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_entry_cancel)
    fetch_order_mock = MagicMock(return_value=old_order)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=fetch_order_mock, cancel_order_with_result=cancel_order_mock, get_fee=fee)
    mocker.patch('freqtrade.freqtradebot.sleep')
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.execute_entry', side_effect=DependencyException())
    open_trade.is_short = is_short
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.strategy.ft_check_timed_out = MagicMock(return_value=False)
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=12234)
    freqtrade.manage_open_orders()
    trades = Trade.session.scalars(select(Trade).where(Trade.is_open.is_(True))).all()
    assert len(trades) == 0
    assert len(Order.session.scalars(select(Order)).all()) == 0
    assert fetch_order_mock.call_count == 1
    assert log_has_re('Could not replace order for.*', caplog)

@pytest.mark.parametrize('is_short', [False, True])
def test_adjust_entry_maintain_replace(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_old: dict, open_trade: Trade, limit_sell_order_old: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    old_order['id'] = open_trade.open_orders_ids[0]
    limit_entry_cancel = deepcopy(old_order)
    limit_entry_cancel['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_entry_cancel)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=old_order), cancel_order_with_result=cancel_order_mock, get_trades_for_order=MagicMock(return_value=trades_for_order), get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=False))
    open_trade.is_short = is_short
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.strategy.ft_check_timed_out = MagicMock(return_value=False)
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=old_order['price'])
    freqtrade.manage_open_orders()
    trades = Trade.session.scalars(select(Trade).where(Order.ft_is_open.is_(True)).where(Order.ft_trade_id == Trade.id)).all()
    assert len(trades) == 1
    assert len(Order.get_open_orders()) == 1
    assert freqtrade.strategy.adjust_entry_price.call_count == 1
    freqtrade.get_valid_enter_price_and_stake = MagicMock(return_value={100, 10, 1})
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=1234)
    freqtrade.manage_open_orders()
    assert freqtrade.strategy.adjust_entry_price.call_count == 1
    trades = Trade.session.scalars(select(Trade).where(Order.ft_is_open.is_(True)).where(Order.ft_trade_id == Trade.id)).all()
    assert len(trades) == 1
    nb_all_orders = len(Order.session.scalars(select(Order)).all())
    assert nb_all_orders == 2
    assert log_has_re(f'{("Sell" if is_short else "Buy")} order cancelled to be replaced*', caplog)
    assert freqtrade.strategy.adjust_entry_price.call_count == 1

@pytest.mark.parametrize('is_short', [False, True])
def test_check_handle_cancelled_buy(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_old: dict, open_trade: Trade, limit_sell_order_old: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    """Handle Buy order cancelled on exchange"""
    old_order = limit_sell_order_old if is_short else limit_buy_order_old
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    old_order.update({'status': 'canceled', 'filled': 0.0})
    old_order['side'] = 'buy' if is_short else 'sell'
    old_order['id'] = open_trade.open_orders[0].order_id
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=old_order), cancel_order=cancel_order_mock, get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade.is_short = is_short
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 2
    trades = Trade.session.scalars(select(Trade).where(Order.ft_is_open.is_(True)).where(Order.ft_trade_id == Trade.id)).all()
    assert len(trades) == 0
    exit_name = 'Buy' if is_short else 'Sell'
    assert log_has_re(f'{exit_name} order cancelled on exchange for Trade.*', caplog)

@pytest.mark.parametrize('is_short', [False, True])
def test_manage_open_orders_buy_exception(default_conf_usdt: dict, ticker_usdt: MagicMock, open_trade: Trade, is_short: bool, fee: MagicMock, mocker: MagicMock) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(side_effect=ExchangeError('Oh snap')), cancel_order=cancel_order_mock)
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade.is_short = is_short
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    assert len(open_trade.open_orders) == 1

@pytest.mark.parametrize('is_short', [False, True])
def test_manage_open_orders_exit_usercustom(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_sell_order_old: dict, mocker: MagicMock, is_short: bool, open_trade_usdt: Trade, caplog: logging.LogRecord) -> None:
    default_conf_usdt['unfilledtimeout'] = {'entry': 1440, 'exit': 1440, 'exit_timeout_count': 1}
    limit_sell_order_old['amount'] = open_trade_usdt.amount
    limit_sell_order_old['remaining'] = open_trade_usdt.amount
    if is_short:
        limit_sell_order_old['side'] = 'buy'
        open_trade_usdt.is_short = is_short
    open_exit_order = Order.parse_from_ccxt_object(limit_sell_order_old, 'mocked', 'buy' if is_short else 'sell')
    open_trade_usdt.orders[-1] = open_exit_order
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=0.0)
    et_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.execute_trade_exit')
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=limit_sell_order_old), cancel_order=cancel_order_mock)
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade_usdt.open_date = dt_now() - timedelta(hours=5)
    open_trade_usdt.close_date = dt_now() - timedelta(minutes=601)
    open_trade_usdt.close_profit_abs = 0.001
    Trade.session.add(open_trade_usdt)
    Trade.commit()
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    freqtrade.strategy.check_exit_timeout = MagicMock(return_value=False)
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=False)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    assert freqtrade.strategy.check_exit_timeout.call_count == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 0
    freqtrade.strategy.check_exit_timeout = MagicMock(side_effect=KeyError)
    freqtrade.strategy.check_entry_timeout = MagicMock(side_effect=KeyError)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 1
    assert freqtrade.strategy.check_exit_timeout.call_count == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 0
    freqtrade.strategy.check_exit_timeout = MagicMock(return_value=True)
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=True)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 2
    assert freqtrade.strategy.check_exit_timeout.call_count == 1
    assert freqtrade.strategy.check_entry_timeout.call_count == 0
    caplog.clear()
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_exit', return_value=False)
    with patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_exit'):
        freqtrade.manage_open_orders()
        assert et_mock.call_count == 0
    freqtrade.manage_open_orders()
    assert log_has_re('Emergency exiting trade.*', caplog)
    assert et_mock.call_count == 1

@pytest.mark.parametrize('is_short', [False, True])
def test_manage_open_orders_exit(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_sell_order_old: dict, mocker: MagicMock, is_short: bool, open_trade_usdt: Trade) -> None:
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    limit_sell_order_old['id'] = '123456789_exit'
    limit_sell_order_old['side'] = 'buy' if is_short else 'sell'
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=limit_sell_order_old), cancel_order=cancel_order_mock, get_min_pair_stake_amount=MagicMock(return_value=0))
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade_usdt.open_date = dt_now() - timedelta(hours=5)
    open_trade_usdt.close_date = dt_now() - timedelta(minutes=601)
    open_trade_usdt.close_profit_abs = 0.001
    open_trade_usdt.is_short = is_short
    Trade.session.add(open_trade_usdt)
    Trade.commit()
    freqtrade.strategy.check_exit_timeout = MagicMock(return_value=False)
    freqtrade.strategy.check_entry_timeout = MagicMock(return_value=False)
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 2
    assert open_trade_usdt.is_open is True
    assert freqtrade.strategy.check_exit_timeout.call_count == 0
    assert freqtrade.strategy.check_entry_timeout.call_count == 0

@pytest.mark.parametrize('is_short', [False, True])
def test_check_handle_cancelled_exit(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_sell_order_old: dict, open_trade_usdt: Trade, is_short: bool, mocker: MagicMock, caplog: logging.LogRecord) -> None:
    """Handle sell order cancelled on exchange"""
    rpc_mock = patch_RPCManager(mocker)
    cancel_order_mock = MagicMock()
    limit_sell_order_old.update({'status': 'canceled', 'filled': 0.0})
    limit_sell_order_old['side'] = 'buy' if is_short else 'sell'
    limit_sell_order_old['id'] = open_trade_usdt.open_orders[0].order_id
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=limit_sell_order_old), cancel_order_with_result=cancel_order_mock)
    freqtrade = FreqtradeBot(default_conf_usdt)
    open_trade_usdt.open_date = dt_now() - timedelta(hours=5)
    open_trade_usdt.close_date = dt_now() - timedelta(minutes=601)
    open_trade_usdt.is_short = is_short
    Trade.session.add(open_trade_usdt)
    Trade.commit()
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 0
    assert rpc_mock.call_count == 2
    assert open_trade_usdt.is_open is True
    exit_name = 'Buy' if is_short else 'Sell'
    assert log_has_re(f'{exit_name} order cancelled on exchange for Trade.*', caplog)

@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.parametrize('leverage', [1, 3, 5, 10])
def test_manage_open_orders_partial(default_conf_usdt: dict, ticker_usdt: MagicMock, limit_buy_order_old_partial: dict, is_short: bool, leverage: int, open_trade: Trade, mocker: MagicMock) -> None:
    rpc_mock = patch_RPCManager(mocker)
    open_trade.is_short = is_short
    open_trade.leverage = leverage
    open_trade.orders[0].ft_order_side = 'sell' if is_short else 'buy'
    limit_buy_order_old_partial['id'] = open_trade.orders[0].order_id
    limit_buy_order_old_partial['side'] = 'sell' if is_short else 'buy'
    limit_buy_canceled = deepcopy(limit_buy_order_old_partial)
    limit_buy_canceled['status'] = 'canceled'
    cancel_order_mock = MagicMock(return_value=limit_buy_canceled)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=limit_buy_order_old_partial), cancel_order_with_result=cancel_order_mock)
    freqtrade = FreqtradeBot(default_conf_usdt)
    prior_stake = open_trade.stake_amount
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.manage_open_orders()
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 3
    trades = Trade.session.scalars(select(Trade)).all()
    assert len(trades) == 1
    assert trades[0].amount == 23.0
    assert trades[0].stake_amount == open_trade.open_rate * trades[0].amount / leverage
    assert trades[0].stake_amount != prior_stake
    assert not trades[0].has_open_orders

@pytest.mark.parametrize('is_short', [False, True])
def test_manage_open_orders_partial_fee(default_conf_usdt: dict, ticker_usdt: MagicMock, open_trade: Trade, caplog: logging.LogRecord, fee: MagicMock, limit_buy_order_old_partial: dict, trades_for_order: list, limit_buy_order_old_partial_canceled: dict, mocker: MagicMock) -> None:
    open_trade.is_short = is_short
    open_trade.orders[0].ft_order_side = 'sell' if is_short else 'buy'
    rpc_mock = patch_RPCManager(mocker)
    limit_buy_order_old_partial['id'] = open_trade.orders[0].order_id
    limit_buy_order_old_partial_canceled['id'] = open_trade.open_orders_ids[0]
    limit_buy_order_old_partial['side'] = 'sell' if is_short else 'buy'
    limit_buy_order_old_partial_canceled['side'] = 'sell' if is_short else 'buy'
    cancel_order_mock = MagicMock(return_value=limit_buy_order_old_partial_canceled)
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=0))
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=limit_buy_order_old_partial), cancel_order_with_result=cancel_order_mock, get_trades_for_order=MagicMock(return_value=trades_for_order))
    freqtrade = FreqtradeBot(default_conf_usdt)
    assert open_trade.amount == limit_buy_order_old_partial['amount']
    open_trade.fee_open = fee()
    open_trade.fee_close = fee()
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.manage_open_orders()
    assert log_has_re('Applying fee on amount for Trade.*', caplog)
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 3
    trades = Trade.session.scalars(select(Trade).where(Order.ft_trade_id == Trade.id)).all()
    assert len(trades) == 1
    assert trades[0].amount == limit_buy_order_old_partial['amount'] - limit_buy_order_old_partial['remaining'] - 0.023
    assert not trades[0].has_open_orders
    assert trades[0].fee_updated(open_trade.entry_side)
    assert pytest.approx(trades[0].fee_open) == 0.001

@pytest.mark.parametrize('is_short', [False, True])
def test_manage_open_orders_partial_except(default_conf_usdt: dict, ticker_usdt: MagicMock, open_trade: Trade, caplog: logging.LogRecord, fee: MagicMock, limit_buy_order_old_partial: dict, trades_for_order: list, limit_buy_order_old_partial_canceled: dict, mocker: MagicMock) -> None:
    open_trade.is_short = is_short
    open_trade.orders[0].ft_order_side = 'sell' if is_short else 'buy'
    rpc_mock = patch_RPCManager(mocker)
    limit_buy_order_old_partial_canceled['id'] = open_trade.open_orders_ids[0]
    limit_buy_order_old_partial['id'] = open_trade.open_orders_ids[0]
    if is_short:
        limit_buy_order_old_partial['side'] = 'sell'
    cancel_order_mock = MagicMock(return_value=limit_buy_order_old_partial_canceled)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(return_value=limit_buy_order_old_partial), cancel_order_with_result=cancel_order_mock, get_trades_for_order=MagicMock(return_value=trades_for_order))
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.get_real_amount', MagicMock(side_effect=DependencyException))
    freqtrade = FreqtradeBot(default_conf_usdt)
    assert open_trade.amount == limit_buy_order_old_partial['amount']
    open_trade.fee_open = fee()
    open_trade.fee_close = fee()
    Trade.session.add(open_trade)
    Trade.commit()
    freqtrade.manage_open_orders()
    assert log_has_re('Could not update trade amount: .*', caplog)
    assert cancel_order_mock.call_count == 1
    assert rpc_mock.call_count == 3
    trades = Trade.session.scalars(select(Trade)).all()
    assert len(trades) == 1
    assert trades[0].amount == limit_buy_order_old_partial['amount'] - limit_buy_order_old_partial['remaining']
    assert not trades[0].has_open_orders
    assert trades[0].fee_open == fee()

def test_manage_open_orders_exception(default_conf_usdt: dict, ticker_usdt: MagicMock, open_trade_usdt: Trade, mocker: MagicMock, caplog: logging.LogRecord) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, fetch_order=MagicMock(side_effect=ExchangeError('Oh snap')), cancel_order=cancel_order_mock)
    freqtrade = FreqtradeBot(default_conf_usdt)
    Trade.session.add(open_trade_usdt)
    Trade.commit()
    caplog.clear()
    freqtrade.manage_open_orders()
    assert log_has_re(f'Cannot query order for Trade\\(id=1, pair=ADA/USDT, amount=30.00000000, is_short=False, leverage=1.0, open_rate=2.00000000, open_since={open_trade_usdt.open_date.strftime("%Y-%m-%d %H:%M:%S")}\\) due to Traceback \\(most recent call last\\):\\n*', caplog)

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_cancel_enter(mocker: MagicMock, caplog: logging.LogRecord, default_conf_usdt: dict, limit_order: dict, is_short: bool, fee: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    l_order = deepcopy(limit_order[entry_side(is_short)])
    cancel_entry_order = deepcopy(limit_order[entry_side(is_short)])
    cancel_entry_order['status'] = 'canceled'
    del cancel_entry_order['filled']
    cancel_order_mock = MagicMock(return_value=cancel_entry_order)
    mocker.patch(f'{EXMS}.cancel_order_with_result', cancel_order_mock)
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade._notify_enter_cancel = MagicMock()
    trade = mock_trade_usdt_4(fee, is_short)
    Trade.session.add(trade)
    Trade.commit()
    l_order['filled'] = 0.0
    l_order['status'] = 'open'
    reason = CANCEL_REASON['TIMEOUT']
    assert freqtrade.handle_cancel_enter(trade, l_order, trade.open_orders[0], reason)
    assert cancel_order_mock.call_count == 1
    cancel_order_mock.reset_mock()
    caplog.clear()
    l_order['filled'] = 0.01
    assert not freqtrade.handle_cancel_enter(trade, l_order, trade.open_orders[0], reason)
    assert cancel_order_mock.call_count == 0
    assert log_has_re('Order .* for .* not cancelled, as the filled amount.* unexitable.*', caplog)
    caplog.clear()
    cancel_order_mock.reset_mock()
    l_order['filled'] = 2
    assert not freqtrade.handle_cancel_enter(trade, l_order, trade.open_orders[0], reason)
    assert cancel_order_mock.call_count == 1
    cancel_entry_order['status'] = 'open'
    cancel_order_mock = MagicMock(return_value=cancel_entry_order)
    mocker.patch(f'{EXMS}.cancel_order_with_result', cancel_order_mock)
    assert not freqtrade.handle_cancel_enter(trade, l_order, trade.open_orders[0], reason)
    assert log_has_re('Order .* for .* not cancelled.', caplog)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=None)
    assert not freqtrade.handle_cancel_enter(trade, limit_order[entry_side(is_short)], trade.open_orders[0], reason)
    cbo = limit_order[entry_side(is_short)]
    mocker.patch('freqtrade.freqtradebot.sleep')
    cbo['status'] = 'open'
    co_mock = mocker.patch(f'{EXMS}.cancel_order_with_result', return_value=cbo)
    fo_mock = mocker.patch(f'{EXMS}.fetch_order', return_value=cbo)
    assert not freqtrade.handle_cancel_enter(trade, cbo, trade.open_orders[0], reason, replacing=True)
    assert co_mock.call_count == 1
    assert fo_mock.call_count == 3

@pytest.mark.parametrize('is_short', [True, False])
@pytest.mark.parametrize('leverage', [1, 5])
@pytest.mark.parametrize('amount', [2, 50])
def test_handle_cancel_enter_limit(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, is_short: bool, leverage: int, amount: int) -> None:
    send_msg_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    cancel_order_mock = MagicMock()
    mocker.patch.multiple(EXMS, cancel_order=cancel_order_mock)
    entry_price = 0.245441
    mocker.patch(f'{EXMS}.get_rate', return_value=entry_price)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=0.2)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_order_fee')
    freqtrade = FreqtradeBot(default_conf_usdt)
    trade = Trade(pair='LTC/USDT', amount=amount * leverage, exchange='binance', open_rate=entry_price, open_date=dt_now() - timedelta(days=2), fee_open=fee.return_value, fee_close=fee.return_value, close_rate=0.555, close_date=dt_now(), exit_reason='sell_reason_whatever', stake_amount=entry_price * amount, leverage=leverage, is_short=is_short)
    trade.orders = [Order(ft_order_side=entry_side(is_short), ft_pair=trade.pair, ft_is_open=False, order_id='buy_123456', status='closed', symbol=trade.pair, order_type='market', side=entry_side(is_short), price=trade.open_rate, average=trade.open_rate, filled=trade.amount, remaining=0, cost=trade.open_rate * trade.amount, order_date=trade.open_date, order_filled_date=trade.open_date), Order(ft_order_side=exit_side(is_short), ft_pair=trade.pair, ft_is_open=True, order_id='sell_123456', status='open', symbol=trade.pair, order_type='limit', side=exit_side(is_short), price=trade.open_rate, average=trade.open_rate, filled=0.0, remaining=trade.amount, cost=trade.open_rate * trade.amount, order_date=trade.open_date, order_filled_date=trade.open_date)]
    order = {'id': 'sell_123456', 'remaining': 1, 'amount': 1, 'status': 'open'}
    reason = CANCEL_REASON['TIMEOUT']
    order_obj = trade.open_orders[-1]
    send_msg_mock.reset_mock()
    assert freqtrade.handle_cancel_enter(trade, order, order_obj, reason)
    assert cancel_order_mock.call_count == 1
    assert send_msg_mock.call_count == 1
    assert trade.close_rate is None
    assert trade.exit_reason is None
    assert not trade.has_open_orders
    send_msg_mock.reset_mock()
    order['amount'] = amount * leverage
    order['filled'] = amount * 0.99 * leverage
    assert not freqtrade.handle_cancel_enter(trade, order, order_obj, reason)
    assert cancel_order_mock.call_count == 1
    assert send_msg_mock.call_args_list[0][0][0]['reason'] == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    assert not freqtrade.handle_cancel_enter(trade, order, order_obj, reason)
    assert send_msg_mock.call_args_list[0][0][0]['reason'] == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    assert trade.exit_order_status == CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN']
    assert send_msg_mock.call_count == 1
    send_msg_mock.reset_mock()
    order['filled'] = amount * 0.5 * leverage
    assert freqtrade.handle_cancel_enter(trade, order, order_obj, reason)
    assert send_msg_mock.call_count == 1
    assert send_msg_mock.call_args_list[0][0][0]['reason'] == CANCEL_REASON['PARTIALLY_FILLED']

def test_handle_cancel_enter_cancel_exception(mocker: MagicMock, default_conf_usdt: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=0.0)
    mocker.patch(f'{EXMS}.cancel_order_with_result', side_effect=InvalidOrderException())
    freqtrade = FreqtradeBot(default_conf_usdt)
    trade = MagicMock()
    order_obj = MagicMock()
    order_obj.order_id = '125'
    reason = CANCEL_REASON['TIMEOUT']
    order = {'remaining': 1, 'id': '125', 'amount': 1, 'status': 'open'}
    assert not freqtrade.handle_cancel_enter(trade, order, order_obj, reason)

@pytest.mark.parametrize('is_short, open_rate, amt', [(False, 2.0, 30.0), (True, 2.02, 29.70297029)])
def test_execute_trade_exit_up(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, ticker_usdt_sell_up: MagicMock, mocker: MagicMock, ticker_usdt_sell_down: MagicMock, is_short: bool, open_rate: float, amt: float) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee, _dry_is_price_crossed=MagicMock(side_effect=[True, False]))
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=False)
    freqtrade.enter_positions()
    rpc_mock.reset_mock()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_short == is_short
    assert trade
    assert freqtrade.strategy.confirm_trade_exit.call_count == 0
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt_sell_down if is_short else ticker_usdt_sell_up)
    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_down()['ask'] if is_short else ticker_usdt_sell_up()['bid'], exit_check=ExitCheckTuple(exit_type=ExitType.ROI))
    assert rpc_mock.call_count == 0
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1
    assert id(freqtrade.strategy.confirm_trade_exit.call_args_list[0][1]['trade']) != id(trade)
    assert freqtrade.strategy.confirm_trade_exit.call_args_list[0][1]['trade'].id == trade.id
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=True)
    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_down()['ask'] if is_short else ticker_usdt_sell_up()['bid'], exit_check=ExitCheckTuple(exit_type=ExitType.ROI))
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1
    assert rpc_mock.call_count == 1
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'ETH/USDT', 'direction': 'Short' if trade.is_short else 'Long', 'leverage': 1.0, 'gain': 'profit', 'limit': 2.0 if is_short else 2.2, 'order_rate': 2.0 if is_short else 2.2, 'amount': pytest.approx(amt), 'order_type': 'limit', 'buy_tag': None, 'enter_tag': None, 'open_rate': open_rate, 'current_rate': 2.01 if is_short else 2.3, 'profit_amount': 0.29554455 if is_short else 5.685, 'profit_ratio': 0.00493809 if is_short else 0.09451372, 'stake_currency': 'USDT', 'quote_currency': 'USDT', 'base_currency': 'ETH', 'fiat_currency': 'USD', 'exit_reason': ExitType.ROI.value, 'open_date': ANY, 'close_date': ANY, 'close_rate': ANY, 'sub_trade': False, 'cumulative_profit': 0.0, 'stake_amount': pytest.approx(60), 'is_final_exit': False, 'final_profit_ratio': None} == last_msg

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_down(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, ticker_usdt_sell_down: MagicMock, mocker: MagicMock, is_short: bool) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee, _dry_is_price_crossed=MagicMock(side_effect=[True, False]))
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt_sell_up if is_short else ticker_usdt_sell_down)
    freqtrade.execute_trade_exit(trade=trade, limit=(ticker_usdt_sell_up if is_short else ticker_usdt_sell_down)()['bid'], exit_check=ExitCheckTuple(exit_type=ExitType.STOP_LOSS))
    assert rpc_mock.call_count == 3
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'ETH/USDT', 'direction': 'Short' if trade.is_short else 'Long', 'leverage': 1.0, 'gain': 'loss', 'limit': 2.2 if is_short else 2.01, 'order_rate': 2.2 if is_short else 2.01, 'amount': pytest.approx(29.70297029) if is_short else 30.0, 'order_type': 'limit', 'buy_tag': None, 'enter_tag': None, 'open_rate': 2.02 if is_short else 2.0, 'current_rate': 2.2 if is_short else 2.0, 'profit_amount': -5.65990099 if is_short else -0.00075, 'profit_ratio': -0.0945681 if is_short else -1.247e-05, 'stake_currency': 'USDT', 'quote_currency': 'USDT', 'base_currency': 'ETH', 'fiat_currency': 'USD', 'exit_reason': ExitType.STOP_LOSS.value, 'open_date': ANY, 'close_date': ANY, 'close_rate': ANY, 'sub_trade': False, 'cumulative_profit': 0.0, 'stake_amount': pytest.approx(60), 'is_final_exit': False, 'final_profit_ratio': None} == last_msg

@pytest.mark.parametrize('is_short,amount,open_rate,current_rate,limit,profit_amount,profit_ratio,profit_or_loss', [(False, 30, 2.0, 2.3, 2.25, 7.18125, 0.11938903, 'profit'), (True, 29.70297029, 2.02, 2.2, 2.25, -7.14876237, -0.11944465, 'loss')])
def test_execute_trade_exit_custom_exit_price(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, ticker_usdt_sell_up: MagicMock, is_short: bool, amount: float, open_rate: float, current_rate: float, limit: float, profit_amount: float, profit_ratio: float, profit_or_loss: str, mocker: MagicMock) -> None:
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee, _dry_is_price_crossed=MagicMock(side_effect=[True, False]))
    config = deepcopy(default_conf_usdt)
    config['custom_price_max_distance_ratio'] = 0.1
    patch_whitelist(mocker, config)
    freqtrade = FreqtradeBot(config)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=False)
    freqtrade.enter_positions()
    rpc_mock.reset_mock()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade
    assert freqtrade.strategy.confirm_trade_exit.call_count == 0
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt_sell_up)
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=True)
    freqtrade.strategy.custom_exit_price = lambda **kwargs: 2.25
    freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_up()['ask' if is_short else 'bid'], exit_check=ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL, exit_reason='foo'))
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1
    assert rpc_mock.call_count == 1
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {'trade_id': 1, 'type': RPCMessageType.EXIT, 'exchange': 'Binance', 'pair': 'ETH/USDT', 'direction': 'Short' if trade.is_short else 'Long', 'leverage': 1.0, 'gain': profit_or_loss, 'limit': limit, 'order_rate': limit, 'amount': pytest.approx(amount), 'order_type': 'limit', 'buy_tag': None, 'enter_tag': None, 'open_rate': open_rate, 'current_rate': current_rate, 'profit_amount': pytest.approx(profit_amount), 'profit_ratio': profit_ratio, 'stake_currency': 'USDT', 'quote_currency': 'USDT', 'base_currency': 'ETH', 'fiat_currency': 'USD', 'exit_reason': 'foo', 'open_date': ANY, 'close_date': ANY, 'close_rate': ANY, 'sub_trade': False, 'cumulative_profit': 0.0, 'stake_amount': pytest.approx(60), 'is_final_exit': False, 'final_profit_ratio': None} == last_msg

@pytest.mark.parametrize('is_short,amount,current_rate,limit,profit_amount,profit_ratio,profit_or_loss', [(False, 30, 2.3, 2.2, 5.685, 0.09451372, 'profit'), (True, 29.70297029, 2.2, 2.3, -8.63762376, -0.1443212, 'loss')])
def test_execute_trade_exit_market_order(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, is_short: bool, current_rate: float, amount: float, caplog: logging.LogRecord, limit: float, profit_amount: float, profit_ratio: float, profit_or_loss: str, ticker_usdt_sell_up: MagicMock, mocker: MagicMock) -> None:
    """
    amount
        long: 60 / 2.0 = 30
        short: 60 / 2.02 = 29.70297029
    open_value
        long: (30 * 2.0) + (30 * 2.0 * 0.0025) = 60.15
        short: (29.702970297029704 * 2.02) - (29.702970297029704 * 2.02 * 0.0025) = 59.85
    close_value
        long: (30 * 2.2) - (30 * 2.2 * 0.0025) = 65.835
        short: (29.702970297029704 * 2.3) + (29.702970297029704 * 2.3 * 0.0025) = 68.48762376237624
    profit
        long: 65.835 - 60.15 = 5.684999999999995
        short: 59.85 - 68.48762376237624 = -8.637623762376244
    profit_ratio
        long: (65.835/60.15) - 1 = 0.0945137157107232
        short: 1 - (68.48762376237624/59.85) = -0.1443211990371971
    """
    open_rate = ticker_usdt.return_value['ask' if is_short else 'bid']
    rpc_mock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=True))
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.config['order_types']['exit'] = 'market'
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt_sell_up, _dry_is_price_crossed=MagicMock(return_value=False))
    assert freqtrade.handle_trade(trade) is True
    assert not trade.is_open
    assert pytest.approx(trade.close_profit) == profit_ratio
    assert rpc_mock.call_count == 4
    last_msg = rpc_mock.call_args_list[-2][0][0]
    assert {'type': RPCMessageType.EXIT, 'trade_id': 1, 'exchange': 'Binance', 'pair': 'ETH/USDT', 'direction': 'Short' if trade.is_short else 'Long', 'leverage': 1.0, 'gain': profit_or_loss, 'limit': limit, 'order_rate': limit, 'amount': pytest.approx(amount), 'order_type': 'market', 'buy_tag': None, 'enter_tag': None, 'open_rate': open_rate, 'current_rate': current_rate, 'profit_amount': pytest.approx(profit_amount), 'profit_ratio': profit_ratio, 'stake_currency': 'USDT', 'quote_currency': 'USDT', 'base_currency': 'ETH', 'fiat_currency': 'USD', 'exit_reason': ExitType.ROI.value, 'open_date': ANY, 'close_date': ANY, 'close_rate': ANY, 'sub_trade': False, 'cumulative_profit': 0.0, 'stake_amount': pytest.approx(60), 'is_final_exit': False, 'final_profit_ratio': None} == last_msg

@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_insufficient_funds_error(default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, is_short: bool, ticker_usdt_sell_up: MagicMock, mocker: MagicMock) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_insuf = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_insufficient_funds')
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee, create_order=MagicMock(side_effect=[{'id': 1234553382}, InsufficientFundsError()]))
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt_sell_up)
    sell_reason = ExitCheckTuple(exit_type=ExitType.ROI)
    assert not freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_up()['ask' if is_short else 'bid'], exit_check=sell_reason)
    assert mock_insuf.call_count == 1

@pytest.mark.parametrize('profit_only,bid,ask,handle_first,handle_second,exit_type,is_short', [(True, 2.18, 2.2, False, True, ExitType.EXIT_SIGNAL.value, False), (True, 2.18, 2.2, False, True, ExitType.EXIT_SIGNAL.value, True), (False, 3.19, 3.2, True, False, ExitType.EXIT_SIGNAL.value, False), (False, 3.19, 3.2, True, False, ExitType.EXIT_SIGNAL.value, True), (True, 0.21, 0.22, False, False, None, False), (True, 2.41, 2.42, False, False, None, True), (False, 0.1, 0.22, True, False, ExitType.EXIT_SIGNAL.value, False), (False, 0.1, 0.22, True, False, ExitType.EXIT_SIGNAL.value, True)])
def test_exit_profit_only(default_conf_usdt: dict, limit_order: dict, limit_order_open: dict, is_short: bool, fee: MagicMock, mocker: MagicMock, profit_only: bool, bid: float, ask: float, handle_first: bool, handle_second: bool, exit_type: int) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    eside = entry_side(is_short)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': bid, 'ask': ask, 'last': bid}), create_order=MagicMock(side_effect=[limit_order[eside], {'id': 1234553382}]), get_fee=fee)
    default_conf_usdt.update({'use_exit_signal': True, 'exit_profit_only': profit_only, 'exit_profit_offset': 0.1})
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.custom_exit = MagicMock(return_value=None)
    if exit_type == ExitType.EXIT_SIGNAL.value:
        freqtrade.strategy.min_roi_reached = MagicMock(return_value=False)
    else:
        freqtrade.strategy.ft_stoploss_reached = MagicMock(return_value=ExitCheckTuple(exit_type=ExitType.NONE))
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_short == is_short
    assert freqtrade.handle_trade(trade) is False
    mocker.patch(f'{EXMS}.fetch_ticker', MagicMock(return_value={'bid': 2.0 * bid, 'ask': 2.0 * bid, 'last': 2.0 * bid}))
    assert freqtrade.handle_trade(trade) is False
    caplog.clear()
    mocker.patch(f'{EXMS}.fetch_ticker', MagicMock(return_value={'bid': 2.0 * ask, 'ask': 2.0 * ask, 'last': 2.0 * ask}))
    caplog.set_level(logging.DEBUG)
    assert freqtrade.handle_trade(trade) is True
    stop_multi = 1.1 if is_short else 0.9
    assert log_has(f'ETH/USDT - HIT STOP: current price at {2.0 * ask:6f}, stoploss is {2.0 * bid * stop_multi:6f}, initial stoploss was at {2.0 * stop_multi:6f}, trade opened at 2.000000', caplog)
    assert trade.exit_reason == ExitType.TRAILING_STOP_LOSS.value

@pytest.mark.parametrize('is_short', [False, True])
def test_disable_ignore_roi_if_entry_signal(default_conf_usdt: dict, limit_order: dict, limit_order_open: dict, is_short: bool, fee: MagicMock, mocker: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    eside = entry_side(is_short)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 2.0, 'ask': 2.0, 'last': 2.0}), create_order=MagicMock(side_effect=[limit_order_open[eside], {'id': 1234553382}, {'id': 1234553383}]), get_fee=fee, _dry_is_price_crossed=MagicMock(return_value=False))
    default_conf_usdt['exit_pricing'] = {'ignore_roi_if_entry_signal': False}
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.min_roi_reached = MagicMock(return_value=True)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    oobj = Order.parse_from_ccxt_object(limit_order[eside], limit_order[eside]['symbol'], eside)
    trade.update_trade(oobj)
    patch_get_signal(freqtrade, enter_long=not is_short, enter_short=is_short, exit_short=is_short)
    assert freqtrade.handle_trade(trade) is True
    patch_get_signal(freqtrade)
    assert freqtrade.handle_trade(trade) is True
    assert trade.exit_reason == ExitType.ROI.value

def test_get_real_amount_quote(default_conf_usdt: dict, trades_for_order: list, buy_order_fee: dict, fee: MagicMock, mocker: MagicMock) -> None:
    limit_buy_order_usdt = deepcopy(buy_order_fee)
    limit_buy_order_usdt['fee'] = {'cost': 0.004}
    mocker.patch(f'{EXMS}.get_trades_for_order', return_value=trades_for_order)
    amount = float(sum((x['amount'] for x in trades_for_order)))
    trade = Trade(pair='LTC/ETH', amount=amount, exchange='binance', fee_open=fee.return_value, fee_close=fee.return_value, open_rate=0.245441)
    limit_buy_order_usdt['amount'] = amount
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    order_obj = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/ETH', 'buy')
    res = freqtrade.get_real_amount(trade, limit_buy_order_usdt, order_obj)
    assert res is None
    assert trade.fee_open_currency is None
    assert trade.fee_open_cost is None
    message = 'Not updating buy-fee - rate: None, POINT.'
    assert log_has(message, caplog)
    caplog.clear()
    freqtrade.config['exchange']['unknown_fee_rate'] = 1
    res = freqtrade.get_real_amount(trade, limit_buy_order_usdt, order_obj)
    assert res is None
    assert trade.fee_open_currency == 'POINT'
    assert pytest.approx(trade.fee_open_cost) == 0.3046651026
    assert trade.fee_open == 0.002
    assert trade.fee_open != fee.return_value
    assert not log_has(message, caplog)

@pytest.mark.parametrize('amount,fee_abs,wallet,amount_exp', [(8.0, 0.0, 10, None), (8.0, 0.0, 0, None), (8.0, 0.1, 0, 0.1), (8.0, 0.1, 10, None), (8.0, 0.1, 8.0, None), (8.0, 0.1, 7.9, 0.1)])
def test_apply_fee_conditional(default_conf_usdt: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, amount: float, fee_abs: float, wallet: float, amount_exp: float) -> None:
    walletmock = mocker.patch('freqtrade.wallets.Wallets.update')
    mocker.patch('freqtrade.wallets.Wallets.get_free', return_value=wallet)
    trade = Trade(pair='LTC/ETH', amount=amount, exchange='binance', open_rate=0.245441, fee_open=fee.return_value, fee_close=fee.return_value)
    order = Order(ft_order_side='buy', order_id='100', ft_pair=trade.pair, ft_is_open=True)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    walletmock.reset_mock()
    assert freqtrade.apply_fee_conditional(trade, 'LTC', amount, fee_abs, order) == amount_exp
    assert walletmock.call_count == 1
    if fee_abs != 0 and amount_exp is None:
        assert log_has_re('Fee amount.*Eating.*dust\\.', caplog)

@pytest.mark.parametrize('amount,fee_abs,wallet,amount_exp', [(8.0, 0.0, 16, None), (8.0, 0.0, 0, None), (8.0, 0.1, 8, 0.1), (8.0, 0.1, 20, None), (8.0, 0.1, 16.0, None), (8.0, 0.1, 7.9, 0.1), (8.0, 0.1, 12, 0.1), (8.0, 0.1, 15.9, 0.1)])
def test_apply_fee_conditional_multibuy(default_conf_usdt: dict, fee: MagicMock, mocker: MagicMock, caplog: logging.LogRecord, amount: float, fee_abs: float, wallet: float, amount_exp: float) -> None:
    walletmock = mocker.patch('freqtrade.wallets.Wallets.update')
    mocker.patch('freqtrade.wallets.Wallets.get_free', return_value=wallet)
    trade = Trade(pair='LTC/ETH', amount=amount, exchange='binance', open_rate=0.245441, fee_open=fee.return_value, fee_close=fee.return_value)
    order = Order(ft_order_side='buy', order_id='10', ft_pair=trade.pair, ft_is_open=False, filled=amount, status='closed')
    trade.orders.append(order)
    order1 = Order(ft_order_side='buy', order_id='100', ft_pair=trade.pair, ft_is_open=True)
    trade.orders.append(order1)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    walletmock.reset_mock()
    assert freqtrade.apply_fee_conditional(trade, 'LTC', amount, fee_abs, order1) == amount_exp
    assert walletmock.call_count == 1
    if fee_abs != 0 and amount_exp is None:
        assert log_has_re('Fee amount.*Eating.*dust\\.', caplog)

@pytest.mark.parametrize('exception_thrown,ask,last,order_book_top,order_book', [(False, 0.045, 0.046, 2, None), (True, 0.042, 0.046, 1, {'bids': [[]], 'asks': [[]]})])
def test_order_book_entry_pricing1(mocker: MagicMock, default_conf_usdt: dict, order_book_l2: dict, exception_thrown: bool, ask: float, last: float, order_book_top: int, order_book: dict) -> None:
    """
    test if function get_rate will return the order book price instead of the ask rate
    """
    patch_exchange(mocker)
    ticker_usdt_mock = MagicMock(return_value={'ask': ask, 'last': last})
    mocker.patch.multiple(EXMS, fetch_l2_order_book=MagicMock(return_value=order_book) if order_book else order_book_l2, fetch_ticker=ticker_usdt_mock)
    default_conf_usdt['exchange']['name'] = 'binance'
    default_conf_usdt['entry_pricing']['use_order_book'] = True
    default_conf_usdt['entry_pricing']['order_book_top'] = order_book_top
    default_conf_usdt['entry_pricing']['price_last_balance'] = 0
    default_conf_usdt['telegram']['enabled'] = False
    freqtrade = FreqtradeBot(default_conf_usdt)
    if exception_thrown:
        with pytest.raises(PricingError):
            freqtrade.exchange.get_rate('ETH/USDT', side='entry', is_short=False, refresh=True)
        assert log_has_re('ETH/USDT - Entry Price at location 1 from orderbook could not be determined.', caplog)
    else:
        assert freqtrade.exchange.get_rate('ETH/USDT', side='entry', is_short=False, refresh=True) == 0.043935
        assert ticker_usdt_mock.call_count == 0

def test_check_depth_of_market(default_conf_usdt: dict, mocker: MagicMock, order_book_l2: dict) -> None:
    """
    test check depth of market
    """
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_l2_order_book=order_book_l2)
    default_conf_usdt['telegram']['enabled'] = False
    default_conf_usdt['exchange']['name'] = 'binance'
    default_conf_usdt['entry_pricing']['check_depth_of_market']['enabled'] = True
    default_conf_usdt['entry_pricing']['check_depth_of_market']['bids_to_ask_delta'] = 100
    freqtrade = FreqtradeBot(default_conf_usdt)
    conf = default_conf_usdt['entry_pricing']['check_depth_of_market']
    assert freqtrade._check_depth_of_market('ETH/BTC', conf, side=SignalDirection.LONG) is False

@pytest.mark.parametrize('is_short', [False, True])
def test_order_book_exit_pricing(default_conf_usdt: dict, limit_buy_order_usdt_open: dict, limit_buy_order_usdt: dict, fee: MagicMock, is_short: bool, limit_sell_order_usdt_open: dict, mocker: MagicMock, order_book_l2: dict, caplog: logging.LogRecord) -> None:
    """
    test order book ask strategy
    """
    mocker.patch(f'{EXMS}.fetch_l2_order_book', order_book_l2)
    default_conf_usdt['exchange']['name'] = 'binance'
    default_conf_usdt['exit_pricing']['use_order_book'] = True
    default_conf_usdt['exit_pricing']['order_book_top'] = 1
    default_conf_usdt['telegram']['enabled'] = False
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(side_effect=[limit_buy_order_usdt_open, limit_sell_order_usdt_open]), get_fee=fee)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    time.sleep(0.01)
    oobj = Order.parse_from_ccxt_object(limit_buy_order_usdt, limit_buy_order_usdt['symbol'], 'buy')
    trade.update_trade(oobj)
    freqtrade.wallets.update()
    assert trade.is_open is True
    if is_short:
        patch_get_signal(freqtrade, enter_long=False, exit_short=True)
    else:
        patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    assert freqtrade.handle_trade(trade) is True
    assert trade.close_rate_requested == order_book_l2.return_value['asks'][0][0]
    mocker.patch(f'{EXMS}.fetch_l2_order_book', return_value={'bids': [[]], 'asks': [[]]})
    with pytest.raises(PricingError):
        freqtrade.handle_trade(trade)
    assert log_has_re('ETH/USDT - Exit Price at location 1 from orderbook could not be determined\\..*', caplog)

def test_startup_state(default_conf_usdt: dict, mocker: MagicMock) -> None:
    default_conf_usdt['pairlist'] = {'method': 'VolumePairList', 'config': {'number_assets': 20}}
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    worker = get_patched_worker(mocker, default_conf_usdt)
    assert worker.freqtrade.state is State.RUNNING

def test_startup_trade_reinit(default_conf_usdt: dict, edge_conf: dict, mocker: MagicMock) -> None:
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    reinit_mock = MagicMock()
    mocker.patch('freqtrade.persistence.Trade.stoploss_reinitialization', reinit_mock)
    ftbot = get_patched_freqtradebot(mocker, default_conf_usdt)
    ftbot.startup()
    assert reinit_mock.call_count == 1
    reinit_mock.reset_mock()
    ftbot = get_patched_freqtradebot(mocker, edge_conf)
    ftbot.startup()
    assert reinit_mock.call_count == 0

@pytest.mark.usefixtures('init_persistence')
def test_sync_wallet_dry_run(mocker: MagicMock, default_conf_usdt: dict, ticker_usdt: MagicMock, fee: MagicMock, limit_buy_order_usdt_open: dict, caplog: logging.LogRecord) -> None:
    default_conf_usdt['dry_run'] = True
    default_conf_usdt['dry_run_wallet'] = 120.0
    default_conf_usdt['max_open_trades'] = 2
    default_conf_usdt['tradable_balance_ratio'] = 1.0
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, create_order=MagicMock(return_value=limit_buy_order_usdt_open), get_fee=fee)
    bot = get_patched_freqtradebot(mocker, default_conf_usdt)
    patch_get_signal(bot)
    assert bot.wallets.get_free('USDT') == 120.0
    n = bot.enter_positions()
    assert n == 2
    trades = Trade.session.scalars(select(Trade)).all()
    assert len(trades) == 2
    bot.config['max_open_trades'] = 3
    n = bot.enter_positions()
    assert n == 0
    assert log_has_re('Unable to create trade for XRP/USDT: Available balance \\(0.0 USDT\\) is lower than stake amount \\(60.0 USDT\\)', caplog)

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_cancel_all_open_orders(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, limit_order: dict, limit_order_open: dict, is_short: bool) -> None:
    default_conf_usdt['cancel_open_orders_on_exit'] = True
    mocker.patch(f'{EXMS}.fetch_order', side_effect=[ExchangeError(), limit_order[exit_side(is_short)], limit_order_open[entry_side(is_short)], limit_order_open[exit_side(is_short)]])
    buy_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_enter')
    sell_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_exit')
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades(fee, is_short=is_short)
    trades = Trade.session.scalars(select(Trade)).all()
    assert len(trades) == MOCK_TRADE_COUNT
    freqtrade.cancel_all_open_orders()
    assert buy_mock.call_count == 2
    assert sell_mock.call_count == 2

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_check_for_open_trades(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.check_for_open_trades()
    assert freqtrade.rpc.send_msg.call_count == 0
    create_mock_trades(fee, is_short)
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    freqtrade.check_for_open_trades()
    assert freqtrade.rpc.send_msg.call_count == 1
    assert 'Handle these trades manually' in freqtrade.rpc.send_msg.call_args[0][0]['status']

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_startup_update_open_orders(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades(fee, is_short=is_short)
    freqtrade.startup_update_open_orders()
    assert not log_has_re('Error updating Order .*', caplog)
    caplog.clear()
    freqtrade.config['dry_run'] = False
    freqtrade.startup_update_open_orders()
    assert len(Order.get_open_orders()) == 4
    matching_buy_order = mock_order_4(is_short=is_short)
    matching_buy_order.update({'status': 'closed'})
    mocker.patch(f'{EXMS}.fetch_order', return_value=matching_buy_order)
    freqtrade.startup_update_open_orders()
    assert len(Order.get_open_orders()) == 3
    caplog.clear()
    mocker.patch(f'{EXMS}.fetch_order', side_effect=ExchangeError)
    freqtrade.startup_update_open_orders()
    assert log_has_re('Error updating Order .*', caplog)
    mocker.patch(f'{EXMS}.fetch_order', side_effect=InvalidOrderException)
    hto_mock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_cancel_order')
    freqtrade.startup_update_open_orders()
    assert log_has_re('Order is older than \\d days.*', caplog)
    assert hto_mock.call_count == 3
    assert hto_mock.call_args_list[0][0][0]['status'] == 'canceled'
    assert hto_mock.call_args_list[1][0][0]['status'] == 'canceled'

@pytest.mark.usefixtures('init_persistence')
def test_startup_backpopulate_precision(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, caplog: logging.LogRecord) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades_usdt(fee)
    trades = Trade.get_trades().all()
    for trade in trades:
        assert trade.price_precision is None
        assert trade.amount_precision is None
        assert trade.precision_mode is None
    freqtrade.startup_backpopulate_precision()
    trades = Trade.get_trades().all()
    for trade in trades:
        if trade.exchange == 'some_other_exchange':
            assert trade.price_precision is None
            assert trade.amount_precision is None
            assert trade.precision_mode is None
        else:
            assert trade.price_precision is not None
            assert trade.amount_precision is not None
            assert trade.precision_mode is not None

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_update_trades_without_assigned_fees(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)

    def patch_with_fee(order):
        order.update({'fee': {'cost': 0.1, 'rate': 0.01, 'currency': order['symbol'].split('/')[0]}})
        return order
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', side_effect=[patch_with_fee(mock_order_2_sell(is_short=is_short)), patch_with_fee(mock_order_3_sell(is_short=is_short)), patch_with_fee(mock_order_2(is_short=is_short)), patch_with_fee(mock_order_3(is_short=is_short)), patch_with_fee(mock_order_4(is_short=is_short))])
    create_mock_trades(fee, is_short=is_short)
    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT
    for trade in trades:
        trade.is_short = is_short
        assert trade.fee_open_cost is None
        assert trade.fee_open_currency is None
        assert trade.fee_close_cost is None
        assert trade.fee_close_currency is None
    freqtrade.update_trades_without_assigned_fees()
    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT
    for trade in trades:
        assert trade.fee_open_cost is None
        assert trade.fee_open_currency is None
        assert trade.fee_close_cost is None
        assert trade.fee_close_currency is None
    freqtrade.config['dry_run'] = False
    freqtrade.update_trades_without_assigned_fees()
    trades = Trade.get_trades().all()
    assert len(trades) == MOCK_TRADE_COUNT
    for trade in trades:
        if trade.is_open:
            if trade.select_order(entry_side(is_short), False):
                assert trade.fee_open_cost is not None
                assert trade.fee_open_currency is not None
            else:
                assert trade.fee_open_cost is None
                assert trade.fee_open_currency is None
        else:
            assert trade.fee_close_cost is not None
            assert trade.fee_close_currency is not None

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_reupdate_enter_order_fees(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, caplog: logging.LogRecord, is_short: bool) -> None:
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_uts = mocker.spy(freqtrade, 'update_trade_state')
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', return_value={'status': 'open'})
    create_mock_trades(fee, is_short)
    trades = Trade.get_trades().all()
    freqtrade.handle_insufficient_funds(trades[3])
    assert mock_uts.call_count == 1
    assert mock_uts.call_args_list[0][0][0] == trades[3]
    assert mock_uts.call_args_list[0][0][1] == mock_order_4(is_short)['id']
    assert log_has_re('Trying to refind lost order for .*', caplog)
    mock_uts.reset_mock()
    caplog.clear()
    trade = Trade(pair='XRP/ETH', stake_amount=60.0, fee_open=fee.return_value, fee_close=fee.return_value, open_date=dt_now(), is_open=True, amount=30, open_rate=2.0, exchange='binance', is_short=is_short)
    Trade.session.add(trade)
    freqtrade.handle_insufficient_funds(trade)
    assert mock_uts.call_count == 0

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_handle_insufficient_funds(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock, is_short: bool, caplog: logging.LogRecord) -> None:
    caplog.set_level(logging.DEBUG)
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_uts = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.update_trade_state')
    mock_fo = mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', return_value={'status': 'open'})

    def reset_open_orders(trade):
        trade.is_short = is_short
    create_mock_trades(fee, is_short=is_short)
    trades = Trade.get_trades().all()
    caplog.clear()
    trade = trades[1]
    reset_open_orders(trade)
    assert not trade.has_open_orders
    assert trade.has_open_sl_orders is False
    freqtrade.handle_insufficient_funds(trade)
    order = trade.orders[0]
    assert log_has_re('Order Order(.*order_id=' + order.order_id + '.*) is no longer open.', caplog)
    assert mock_fo.call_count == 0
    assert mock_uts.call_count == 0
    assert not trade.has_open_orders
    assert trade.has_open_sl_orders is False
    caplog.clear()
    mock_fo.reset_mock()
    trade = trades[3]
    reset_open_orders(trade)
    assert trade.has_open_sl_orders is False
    freqtrade.handle_insufficient_funds(trade)
    order = mock_order_4(is_short=is_short)
    assert log_has_re('Trying to refind Order\\(.*', caplog)
    assert mock_fo.call_count == 1
    assert mock_uts.call_count == 1
    assert trade.has_open_orders is True
    assert trade.has_open_sl_orders is False
    caplog.clear()
    mock_fo.reset_mock()
    trade = trades[4]
    reset_open_orders(trade)
    assert not trade.has_open_orders
    assert trade.has_open_sl_orders
    freqtrade.handle_insufficient_funds(trade)
    order = mock_order_5_stoploss(is_short=is_short)
    assert log_has_re('Trying to refind Order\\(.*', caplog)
    assert mock_fo.call_count == 1
    assert mock_uts.call_count == 2
    assert not trade.has_open_orders
    assert trade.has_open_sl_orders is True
    caplog.clear()
    mock_fo.reset_mock()
    mock_uts.reset_mock()
    trade = trades[5]
    reset_open_orders(trade)
    assert trade.has_open_sl_orders is False
    freqtrade.handle_insufficient_funds(trade)
    order = mock_order_6_sell(is_short=is_short)
    assert log_has_re('Trying to refind Order\\(.*', caplog)
    assert mock_fo.call_count == 1
    assert mock_uts.call_count == 1
    assert trade.open_orders_ids[0] == order['id']
    assert trade.has_open_sl_orders is False
    caplog.clear()
    mock_fo = mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', side_effect=ExchangeError())
    order = mock_order_5_stoploss(is_short=is_short)
    freqtrade.handle_insufficient_funds(trades[4])
    assert log_has(f'Error updating {order["id"]}.', caplog)

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_handle_onexchange_order(mocker: MagicMock, default_conf_usdt: dict, limit_order: dict, is_short: bool, caplog: logging.LogRecord) -> None:
    default_conf_usdt['dry_run'] = False
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_uts = mocker.spy(freqtrade, 'update_trade_state')
    entry_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    mock_fo = mocker.patch(f'{EXMS}.fetch_orders', return_value=[entry_order, exit_order])
    trade = Trade(pair='ETH/USDT', fee_open=0.001, fee_close=0.001, open_rate=entry_order['price'], open_date=dt_now(), stake_amount=entry_order['cost'], amount=entry_order['amount'], exchange='binance', is_short=is_short, leverage=1)
    trade.orders.append(Order.parse_from_ccxt_object(entry_order, 'ADA/USDT', entry_side(is_short)))
    Trade.session.add(trade)
    freqtrade.handle_onexchange_order(trade)
    assert log_has_re('Found previously unknown order .*', caplog)
    assert mock_uts.call_count == 2
    assert mock_fo.call_count == 1
    trade = Trade.session.scalars(select(Trade)).first()
    assert len(trade.orders) == 2
    assert trade.is_open is False
    assert trade.exit_reason == ExitType.SOLD_ON_EXCHANGE.value

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.parametrize('factor,adjusts', [(0.99, True), (0.97, False)])
def test_handle_onexchange_order_changed_amount(mocker: MagicMock, default_conf_usdt: dict, limit_order: dict, is_short: bool, caplog: logging.LogRecord, factor: float, adjusts: bool) -> None:
    default_conf_usdt['dry_run'] = False
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_uts = mocker.spy(freqtrade, 'update_trade_state')
    entry_order = limit_order[entry_side(is_short)]
    mock_fo = mocker.patch(f'{EXMS}.fetch_orders', return_value=[entry_order])
    trade = Trade(pair='ETH/USDT', fee_open=0.001, base_currency='ETH', fee_close=0.001, open_rate=entry_order['price'], open_date=dt_now(), stake_amount=entry_order['cost'], amount=entry_order['amount'], exchange='binance', is_short=is_short, leverage=1)
    freqtrade.wallets = MagicMock()
    freqtrade.wallets.get_owned = MagicMock(return_value=entry_order['amount'] * factor)
    trade.orders.append(Order.parse_from_ccxt_object(entry_order, 'ADA/USDT', entry_side(is_short)))
    Trade.session.add(trade)
    freqtrade.handle_onexchange_order(trade)
    assert mock_uts.call_count == 1
    assert mock_fo.call_count == 1
    trade = Trade.session.scalars(select(Trade)).first()
    assert log_has_re('.*has a total of .* but the Wallet shows.*', caplog)
    if adjusts:
        assert trade.amount == entry_order['amount'] * factor
        assert log_has_re('.*Adjusting trade amount to.*', caplog)
    else:
        assert log_has_re('.*Refusing to adjust as the difference.*', caplog)
        assert trade.amount == entry_order['amount']
    assert len(trade.orders) == 1
    assert trade.is_open is True

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_handle_onexchange_order_exit(mocker: MagicMock, default_conf_usdt: dict, limit_order: dict, is_short: bool, caplog: logging.LogRecord) -> None:
    default_conf_usdt['dry_run'] = False
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    entry_order = limit_order[entry_side(is_short)]
    add_entry_order = deepcopy(entry_order)
    add_entry_order.update({'id': '_partial_entry_id', 'amount': add_entry_order['amount'] / 1.5, 'cost': add_entry_order['cost'] / 1.5, 'filled': add_entry_order['filled'] / 1.5})
    exit_order_part = deepcopy(limit_order[exit_side(is_short)])
    exit_order_part.update({'id': 'some_random_partial_id', 'amount': exit_order_part['amount'] / 2, 'cost': exit_order_part['cost'] / 2, 'filled': exit_order_part['filled'] / 2})
    exit_order = limit_order[exit_side(is_short)]
    mock_fo = mocker.patch(f'{EXMS}.fetch_orders', return_value=[entry_order, exit_order_part, exit_order, add_entry_order])
    trade = Trade(pair='ETH/USDT', fee_open=0.001, fee_close=0.001, open_rate=entry_order['price'], open_date=dt_now(), stake_amount=entry_order['cost'], amount=entry_order['amount'], exchange='binance', is_short=is_short, leverage=1, is_open=True)
    trade.orders = [Order.parse_from_ccxt_object(entry_order, 'ADA/USDT', entry_side(is_short)), Order.parse_from_ccxt_object(exit_order_part, 'ADA/USDT', exit_side(is_short)), Order.parse_from_ccxt_object(add_entry_order, 'ADA/USDT', entry_side(is_short)), Order.parse_from_ccxt_object(exit_order, 'ADA/USDT', exit_side(is_short))]
    trade.recalc_trade_from_orders()
    Trade.session.add(trade)
    Trade.commit()
    freqtrade.handle_onexchange_order(trade)
    assert mock_uts.call_count == 4
    assert mock_fo.call_count == 1
    trade = Trade.session.scalars(select(Trade)).first()
    assert len(trade.orders) == 4
    assert trade.is_open is True
    assert trade.exit_reason is None
    assert trade.amount == 5.0

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_handle_onexchange_order_fully_canceled_enter(mocker: MagicMock, default_conf_usdt: dict, limit_order: dict, is_short: bool, caplog: logging.LogRecord) -> None:
    default_conf_usdt['dry_run'] = False
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    entry_order = limit_order[entry_side(is_short)]
    entry_order['status'] = 'canceled'
    entry_order['filled'] = 0.0
    mock_fo = mocker.patch(f'{EXMS}.fetch_orders', return_value=[entry_order])
    mocker.patch(f'{EXMS}.get_rate', return_value=entry_order['price'])
    trade = Trade(pair='ETH/USDT', fee_open=0.001, fee_close=0.001, open_rate=entry_order['price'], open_date=dt_now(), stake_amount=entry_order['cost'], amount=entry_order['amount'], exchange='binance', is_short=is_short, leverage=1)
    trade.orders.append(Order.parse_from_ccxt_object(entry_order, 'ADA/USDT', entry_side(is_short)))
    Trade.session.add(trade)
    assert freqtrade.handle_onexchange_order(trade) is True
    assert log_has_re('Trade only had fully canceled entry orders\\. .*', caplog)
    assert mock_fo.call_count == 1
    trades = Trade.get_trades().all()
    assert len(trades) == 0

def test_get_valid_price(mocker: MagicMock, default_conf_usdt: dict) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.config['custom_price_max_distance_ratio'] = 0.02
    custom_price_string = '10'
    custom_price_badstring = '10abc'
    custom_price_float = 10.0
    custom_price_int = 10
    custom_price_over_max_alwd = 11.0
    custom_price_under_min_alwd = 9.0
    proposed_price = 10.1
    valid_price_from_string = freqtrade.get_valid_price(custom_price_string, proposed_price)
    valid_price_from_badstring = freqtrade.get_valid_price(custom_price_badstring, proposed_price)
    valid_price_from_int = freqtrade.get_valid_price(custom_price_int, proposed_price)
    valid_price_from_float = freqtrade.get_valid_price(custom_price_float, proposed_price)
    valid_price_at_max_alwd = freqtrade.get_valid_price(custom_price_over_max_alwd, proposed_price)
    valid_price_at_min_alwd = freqtrade.get_valid_price(custom_price_under_min_alwd, proposed_price)
    assert isinstance(valid_price_from_string, float)
    assert isinstance(valid_price_from_badstring, float)
    assert isinstance(valid_price_from_int, float)
    assert isinstance(valid_price_from_float, float)
    assert valid_price_from_string == custom_price_float
    assert valid_price_from_badstring == proposed_price
    assert valid_price_from_int == custom_price_int
    assert valid_price_at_max_alwd < custom_price_over_max_alwd
    assert valid_price_at_max_alwd > proposed_price
    assert valid_price_at_min_alwd > custom_price_under_min_alwd
    assert valid_price_at_min_alwd < proposed_price

@pytest.mark.parametrize('trading_mode,calls,t1,t2', [('spot', 0, '2021-09-01 00:00:00', '2021-09-01 08:00:00'), ('margin', 0, '2021-09-01 00:00:00', '2021-09-01 08:00:00'), ('futures', 15, '2021-09-01 00:01:02', '2021-09-01 08:00:01'), ('futures', 16, '2021-09-01 00:00:02', '2021-09-01 08:00:01'), ('futures', 16, '2021-08-31 23:59:59', '2021-09-01 08:00:01'), ('futures', 16, '2021-08-31 23:59:59', '2021-09-01 08:00:02'), ('futures', 16, '2021-08-31 23:59:59', '2021-09-01 08:00:03'), ('futures', 16, '2021-08-31 23:59:59', '2021-09-01 08:00:04'), ('futures', 17, '2021-08-31 23:59:59', '2021-09-01 08:01:05'), ('futures', 17, '2021-08-31 23:59:59', '2021-09-01 08:01:06'), ('futures', 17, '2021-08-31 23:59:59', '2021-09-01 08:01:07'), ('futures', 17, '2021-08-31 23:59:58', '2021-09-01 08:01:07')])
@pytest.mark.parametrize('tzoffset', ['+00:00', '+01:00', '-02:00'])
def test_update_funding_fees_schedule(mocker: MagicMock, default_conf: dict, trading_mode: str, calls: int, time_machine: TimeMachine, t1: str, t2: str, tzoffset: str) -> None:
    time_machine.move_to(f'{t1} {tzoffset}', tick=False)
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.update_funding_fees', return_value=True)
    default_conf['trading_mode'] = trading_mode
    default_conf['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    time_machine.move_to(f'{t2} {tzoffset}', tick=False)
    freqtrade._schedule.run_pending()
    assert freqtrade.update_funding_fees.call_count == calls

@pytest.mark.parametrize('schedule_off', [False, True])
@pytest.mark.parametrize('is_short', [True, False])
def test_update_funding_fees(mocker: MagicMock, default_conf: dict, time_machine: TimeMachine, fee: MagicMock, ticker_usdt_sell_up: MagicMock, is_short: bool, limit_order_open: dict, schedule_off: bool) -> None:
    """
    nominal_value = mark_price * size
    funding_fee = nominal_value * funding_rate
    size = 123
    "LTC/USDT"
        time: 0, mark: 3.3, fundRate: 0.00032583, nominal_value: 405.9, fundFee: 0.132254397
        time: 8, mark: 3.2, fundRate: 0.00024472, nominal_value: 393.6, fundFee: 0.096321792
    "ETH/USDT"
        time: 0, mark: 2.4, fundRate: 0.0001, nominal_value: 295.2, fundFee: 0.02952
        time: 8, mark: 2.5, fundRate: 0.0001, nominal_value: 307.5, fundFee: 0.03075
    "ETC/USDT"
        time: 0, mark: 4.3, fundRate: 0.00031077, nominal_value: 528.9, fundFee: 0.164366253
        time: 8, mark: 4.1, fundRate: 0.00022655, nominal_value: 504.3, fundFee: 0.114249165
    "XRP/USDT"
        time: 0, mark: 1.2, fundRate: 0.00049426, nominal_value: 147.6, fundFee: 0.072952776
        time: 8, mark: 1.2, fundRate: 0.00032715, nominal_value: 147.6, fundFee: 0.04828734
    """
    time_machine.move_to('2021-09-01 00:00:16 +00:00')
    open_order = limit_order_open[exit_side(is_short)]
    open_exit_order = limit_order_open[entry_side(is_short)]
    bid = 0.11
    enter_rate_mock = MagicMock(return_value=bid)
    open_order.update({'status': 'closed', 'filled': open_order['amount'], 'remaining': 0})
    enter_mm = MagicMock(return_value=open_order)
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    date_midnight = dt_utc(2021, 9, 1)
    date_eight = dt_utc(2021, 9, 1, 8)
    date_sixteen = dt_utc(2021, 9, 1, 16)
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    funding_rates = {'LTC/USDT': DataFrame([[date_midnight, 0.00032583, 0, 0, 0, 0], [date_eight, 0.00024472, 0, 0, 0, 0], [date_sixteen, 0.00024472, 0, 0, 0, 0]], columns=columns), 'ETH/USDT': DataFrame([[date_midnight, 0.0001, 0, 0, 0, 0], [date_eight, 0.0001, 0, 0, 0, 0], [date_sixteen, 0.0001, 0, 0, 0, 0]], columns=columns), 'XRP/USDT': DataFrame([[date_midnight, 0.00049426, 0, 0, 0, 0], [date_eight, 0.00032715, 0, 0, 0, 0], [date_sixteen, 0.00032715, 0, 0, 0, 0]], columns=columns)}
    mark_prices = {'LTC/USDT': DataFrame([[date_midnight, 3.3, 0, 0, 0, 0], [date_eight, 3.2, 0, 0, 0, 0], [date_sixteen, 3.2, 0, 0, 0, 0]], columns=columns), 'ETH/USDT': DataFrame([[date_midnight, 2.4, 0, 0, 0, 0], [date_eight, 2.5, 0, 0, 0, 0], [date_sixteen, 2.5, 0, 0, 0, 0]], columns=columns), 'XRP/USDT': DataFrame([[date_midnight, 1.2, 0, 0, 0, 0], [date_eight, 1.2, 0, 0, 0, 0], [date_sixteen, 1.2, 0, 0, 0, 0]], columns=columns)}

    def refresh_latest_ohlcv_mock(pairlist, **kwargs):
        ret = {}
        for p, tf, ct in pairlist:
            if ct == CandleType.MARK:
                ret[p, tf, ct] = mark_prices[p]
            else:
                ret[p, tf, ct] = funding_rates[p]
        return ret
    mocker.patch(f'{EXMS}.refresh_latest_ohlcv', side_effect=refresh_latest_ohlcv_mock)
    mocker.patch.multiple(EXMS, get_rate=enter_rate_mock, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=enter_mm, get_min_pair_stake_amount=MagicMock(return_value=1), get_fee=fee, get_maintenance_ratio_and_amt=MagicMock(return_value=(0.01, 0.01)))
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.execute_entry('ETH/USDT', 123, is_short=is_short)
    freqtrade.execute_entry('LTC/USDT', 2.0, is_short=is_short)
    freqtrade.execute_entry('XRP/USDT', 123, is_short=is_short)
    multiple = 1 if is_short else -1
    trades = Trade.get_open_trades()
    assert len(trades) == 3
    for trade in trades:
        assert pytest.approx(trade.funding_fees) == 0
    mocker.patch(f'{EXMS}.create_order', return_value=open_exit_order)
    time_machine.move_to('2021-09-01 08:00:00 +00:00')
    if schedule_off:
        for trade in trades:
            freqtrade.execute_trade_exit(trade=trade, limit=ticker_usdt_sell_up()['bid'], exit_check=ExitCheckTuple(exit_type=ExitType.ROI))
            assert trade.funding_fees == pytest.approx(sum(trade.amount * mark_prices[trade.pair].iloc[1:2]['open'] * funding_rates[trade.pair].iloc[1:2]['open'] * multiple))
    else:
        freqtrade._schedule.run_pending()
    for trade in trades:
        assert trade.funding_fees == pytest.approx(sum(trade.amount * mark_prices[trade.pair].iloc[1:2]['open'] * funding_rates[trade.pair].iloc[1:2]['open'] * multiple))

def test_update_funding_fees_error(mocker: MagicMock, default_conf: dict, caplog: logging.LogRecord) -> None:
    mocker.patch(f'{EXMS}.get_funding_fees', side_effect=ExchangeError())
    default_conf['trading_mode'] = 'futures'
    default_conf['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.update_funding_fees()
    log_has('Could not update funding fees for open trades.', caplog)

def test_position_adjust(mocker: MagicMock, default_conf_usdt: dict, fee: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=10000)
    default_conf_usdt.update({'position_adjustment_enable': True, 'dry_run': False, 'stake_amount': 10.0, 'dry_run_wallet': 1000.0})
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    bid = 11
    stake_amount = 10
    buy_rate_mock = MagicMock(return_value=bid)
    mocker.patch.multiple(EXMS, get_rate=buy_rate_mock, fetch_ticker=MagicMock(return_value={'bid': 10, 'ask': 12, 'last': 11}), get_min_pair_stake_amount=MagicMock(return_value=1), get_fee=fee)
    pair = 'ETH/USDT'
    closed_successful_buy_order = {'pair': pair, 'ft_pair': pair, 'ft_order_side': 'buy', 'side': 'buy', 'type': 'limit', 'status': 'closed', 'price': bid, 'average': bid, 'cost': bid * stake_amount, 'amount': stake_amount, 'filled': stake_amount, 'ft_is_open': False, 'id': '650', 'order_id': '650'}
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=closed_successful_buy_order))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=closed_successful_buy_order))
    assert freqtrade.execute_entry(pair, stake_amount)
    orders = Order.session.scalars(select(Order)).all()
    assert orders
    assert len(orders) == 1
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert trade.is_open is True
    assert not trade.has_open_orders
    assert trade.open_rate == 11
    assert trade.stake_amount == 110
    freqtrade.update_trades_without_assigned_fees()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert trade.is_open is True
    assert not trade.has_open_orders
    assert trade.open_rate == 11
    assert trade.stake_amount == 110
    assert not trade.fee_updated('buy')
    freqtrade.manage_open_orders()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert trade.is_open is True
    assert not trade.has_open_orders
    assert trade.open_rate == 11
    assert trade.stake_amount == 110
    assert not trade.fee_updated('buy')
    open_dca_order_1 = {'ft_pair': pair, 'ft_order_side': 'buy', 'side': 'buy', 'type': 'limit', 'status': None, 'price': 9, 'amount': 12, 'cost': 108, 'ft_is_open': True, 'id': '651', 'order_id': '651'}
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=open_dca_order_1))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=open_dca_order_1))
    assert freqtrade.execute_entry(pair, stake_amount, trade=trade)
    orders = Order.session.scalars(select(Order)).all()
    assert orders
    assert len(orders) == 2
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert '651' in trade.open_orders_ids
    assert trade.open_rate == 11
    assert trade.amount == 10
    assert trade.stake_amount == 110
    assert not trade.fee_updated('buy')
    trades = Trade.get_open_trades_without_assigned_fees()
    assert len(trades) == 1
    assert trade.is_open
    assert not trade.fee_updated('buy')
    order = trade.select_order('buy', False)
    assert order.order_id == '650'

    def make_sure_its_651(*args, **kwargs):
        if args[0] == '650':
            return closed_successful_buy_order
        if args[0] == '651':
            return open_dca_order_1
        return None
    fetch_order_mm = MagicMock(side_effect=make_sure_its_651)
    mocker.patch(f'{EXMS}.create_order', fetch_order_mm)
    mocker.patch(f'{EXMS}.fetch_order', fetch_order_mm)
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', fetch_order_mm)
    freqtrade.update_trades_without_assigned_fees()
    orders = Order.session.scalars(select(Order)).all()
    assert orders
    assert len(orders) == 2
    trades = Trade.get_open_trades_without_assigned_fees()
    assert len(trades) == 1
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert '651' in trade.open_orders_ids
    assert trade.open_rate == 11
    assert trade.amount == 10
    assert trade.stake_amount == 110
    assert not trade.fee_updated('buy')
    order = trade.select_order('buy', False)
    assert order.order_id == '650'
    closed_dca_order_1 = {'ft_pair': pair, 'ft_order_side': 'buy', 'side': 'buy', 'type': 'limit', 'status': 'closed', 'price': 9, 'average': 9, 'amount': 12, 'filled': 12, 'cost': 108, 'ft_is_open': False, 'id': '651', 'order_id': '651', 'datetime': dt_now().isoformat()}
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=closed_dca_order_1))
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(return_value=closed_dca_order_1))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=closed_dca_order_1))
    freqtrade.manage_open_orders()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert not trade.has_open_orders
    assert pytest.approx(trade.open_rate) == 9.90909090909
    assert trade.amount == 22
    assert pytest.approx(trade.stake_amount) == 218
    orders = Order.session.scalars(select(Order)).all()
    assert orders
    assert len(orders) == 2
    order = trade.select_order('buy', False)
    assert order.order_id == '651'
    trades = Trade.get_open_trades_without_assigned_fees()
    assert len(trades) == 1
    closed_dca_order_2 = {'ft_pair': pair, 'status': 'closed', 'ft_order_side': 'buy', 'side': 'buy', 'type': 'limit', 'price': 7, 'average': 7, 'amount': 15, 'filled': 15, 'cost': 105, 'ft_is_open': False, 'id': '652', 'order_id': '652'}
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=closed_dca_order_2))
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(return_value=closed_dca_order_2))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=closed_dca_order_2))
    assert freqtrade.execute_entry(pair, stake_amount, trade=trade)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert not trade.has_open_orders
    assert pytest.approx(trade.open_rate) == 8.729729729729
    assert trade.amount == 37
    assert trade.stake_amount == 323
    orders = Order.session.scalars(select(Order)).all()
    assert orders
    assert len(orders) == 3
    order = trade.select_order('buy', False)
    assert order.order_id == '652'
    closed_sell_dca_order_1 = {'ft_pair': pair, 'status': 'closed', 'ft_order_side': 'sell', 'side': 'sell', 'type': 'limit', 'price': 8, 'average': 8, 'amount': 15, 'filled': 15, 'cost': 120, 'ft_is_open': False, 'id': '653', 'order_id': '653'}
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=closed_sell_dca_order_1))
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(return_value=closed_sell_dca_order_1))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=closed_sell_dca_order_1))
    assert freqtrade.execute_trade_exit(trade=trade, limit=8, exit_check=ExitCheckTuple(exit_type=ExitType.PARTIAL_EXIT), sub_trade_amt=15)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert not trade.has_open_orders
    assert trade.is_open
    assert trade.amount == 22
    assert trade.stake_amount == 192.05405405405406
    assert pytest.approx(trade.open_rate) == 8.729729729729
    orders = Order.session.scalars(select(Order)).all()
    assert orders
    assert len(orders) == 4
    order = trade.select_order('sell', False)
    assert order.order_id == '653'
