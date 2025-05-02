import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock
import pytest
from sqlalchemy import select
from freqtrade.enums import ExitCheckTuple, ExitType, TradingMode
from freqtrade.persistence import Trade
from freqtrade.persistence.models import Order
from freqtrade.rpc.rpc import RPC
from tests.conftest import EXMS, get_patched_freqtradebot, log_has_re, patch_get_signal

def test_may_execute_exit_stoploss_on_exchange_multi(
    default_conf: Dict[str, Any],
    ticker: Dict[str, Any],
    fee: float,
    mocker: MagicMock
) -> None:
    """
    Tests workflow of selling stoploss_on_exchange.
    Sells
    * first trade as stoploss
    * 2nd trade is kept
    * 3rd trade is sold via sell-signal
    """
    default_conf['max_open_trades'] = 3
    default_conf['exchange']['name'] = 'binance'
    stoploss: Dict[str, Any] = {'id': 123, 'info': {}}
    stoploss_order_open: Dict[str, Any] = {'id': '123', 'timestamp': 1542707426845, 'datetime': '2018-11-20T09:50:26.845Z', 'lastTradeTimestamp': None, 'symbol': 'BTC/USDT', 'type': 'stop_loss_limit', 'side': 'sell', 'price': 1.08801, 'amount': 91.07468123, 'cost': 0.0, 'average': 0.0, 'filled': 0.0, 'remaining': 0.0, 'status': 'open', 'fee': None, 'trades': None}
    stoploss_order_closed: Dict[str, Any] = stoploss_order_open.copy()
    stoploss_order_closed['status'] = 'closed'
    stoploss_order_closed['filled'] = stoploss_order_closed['amount']
    stop_orders: List[Dict[str, Any]] = [stoploss_order_closed, stoploss_order_open.copy(), stoploss_order_open.copy()]
    stoploss_order_mock: MagicMock = MagicMock(side_effect=stop_orders)
    should_sell_mock: MagicMock = MagicMock(side_effect=[[], [ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL)]])
    cancel_order_mock: MagicMock = MagicMock()
    mocker.patch.multiple(EXMS, create_stoploss=stoploss, fetch_ticker=ticker, get_fee=fee, amount_to_precision=lambda s, x, y: y, price_to_precision=lambda s, x, y: y, fetch_stoploss_order=stoploss_order_mock, cancel_stoploss_order_with_result=cancel_order_mock)
    mocker.patch.multiple('freqtrade.freqtradebot.FreqtradeBot', create_stoploss_order=MagicMock(return_value=True), _notify_exit=MagicMock())
    mocker.patch('freqtrade.strategy.interface.IStrategy.should_exit', should_sell_mock)
    wallets_mock: MagicMock = mocker.patch('freqtrade.wallets.Wallets.update')
    mocker.patch('freqtrade.wallets.Wallets.get_free', return_value=1000)
    mocker.patch('freqtrade.wallets.Wallets.check_exit_amount', return_value=True)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    freqtrade.strategy.order_types['exit'] = 'market'
    freqtrade.strategy.confirm_trade_entry = MagicMock(return_value=True)
    freqtrade.strategy.confirm_trade_exit = MagicMock(return_value=True)
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()
    assert freqtrade.strategy.confirm_trade_entry.call_count == 3
    freqtrade.strategy.confirm_trade_entry.reset_mock()
    assert freqtrade.strategy.confirm_trade_exit.call_count == 0
    wallets_mock.reset_mock()
    trades = Trade.session.scalars(select(Trade)).all()
    for idx, trade in enumerate(trades):
        stop_order = stop_orders[idx]
        stop_order['id'] = f'stop{idx}'
        oobj = Order.parse_from_ccxt_object(stop_order, trade.pair, 'stoploss')
        oobj.ft_is_open = True
        trade.orders.append(oobj)
        assert len(trade.open_sl_orders) == 1
    n = freqtrade.exit_positions(trades)
    assert n == 2
    assert should_sell_mock.call_count == 2
    assert freqtrade.strategy.confirm_trade_entry.call_count == 0
    assert freqtrade.strategy.confirm_trade_exit.call_count == 1
    freqtrade.strategy.confirm_trade_exit.reset_mock()
    assert cancel_order_mock.call_count == 1
    assert stoploss_order_mock.call_count == 3
    assert wallets_mock.call_count == 4
    trade = trades[0]
    assert trade.exit_reason == ExitType.STOPLOSS_ON_EXCHANGE.value
    assert not trade.is_open
    trade = trades[1]
    assert not trade.exit_reason
    assert trade.is_open
    trade = trades[2]
    assert trade.exit_reason == ExitType.EXIT_SIGNAL.value
    assert not trade.is_open

@pytest.mark.parametrize('balance_ratio,result1', [(1, 200), (0.99, 198)])
def test_forcebuy_last_unlimited(
    default_conf: Dict[str, Any],
    ticker: Dict[str, Any],
    fee: float,
    mocker: MagicMock,
    balance_ratio: float,
    result1: float
) -> None:
    """
    Tests workflow unlimited stake-amount
    Buy 4 trades, forcebuy a 5th trade
    Sell one trade, calculated stake amount should now be lower than before since
    one trade was sold at a loss.
    """
    default_conf['max_open_trades'] = 5
    default_conf['force_entry_enable'] = True
    default_conf['stake_amount'] = 'unlimited'
    default_conf['tradable_balance_ratio'] = balance_ratio
    default_conf['dry_run_wallet'] = 1000
    default_conf['exchange']['name'] = 'binance'
    default_conf['telegram']['enabled'] = True
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee, amount_to_precision=lambda s, x, y: y, price_to_precision=lambda s, x, y: y)
    mocker.patch.multiple('freqtrade.freqtradebot.FreqtradeBot', create_stoploss_order=MagicMock(return_value=True), _notify_exit=MagicMock())
    should_sell_mock: MagicMock = MagicMock(side_effect=[[], [ExitCheckTuple(exit_type=ExitType.EXIT_SIGNAL)], [], [], []])
    mocker.patch('freqtrade.strategy.interface.IStrategy.should_exit', should_sell_mock)
    freqtrade = get_patched_freqtradebot(mocker, default_conf)
    rpc = RPC(freqtrade)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    freqtrade.strategy.order_types['exit'] = 'market'
    patch_get_signal(freqtrade)
    n = freqtrade.enter_positions()
    assert n == 4
    trades = Trade.session.scalars(select(Trade)).all()
    assert len(trades) == 4
    assert freqtrade.wallets.get_trade_stake_amount('XRP/BTC', 5) == result1
    rpc._rpc_force_entry('TKN/BTC', None)
    trades = Trade.session.scalars(select(Trade)).all()
    assert len(trades) == 5
    for trade in trades:
        assert pytest.approx(trade.stake_amount) == result1
    trades = Trade.get_open_trades()
    assert len(trades) == 5
    bals = freqtrade.wallets.get_all_balances()
    n = freqtrade.exit_positions(trades)
    assert n == 1
    trades = Trade.get_open_trades()
    assert len(trades) == 4
    assert freqtrade.wallets.get_trade_stake_amount('XRP/BTC', 5) < result1
    bals2 = freqtrade.wallets.get_all_balances()
    assert bals != bals2
    assert len(bals) == 6
    assert len(bals2) == 5
    assert 'LTC' in bals
    assert 'LTC' not in bals2

def test_dca_buying(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Dict[str, Any],
    fee: float,
    mocker: MagicMock
) -> None:
    default_conf_usdt['position_adjustment_enable'] = True
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee)
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()
    assert len(Trade.get_trades().all()) == 1
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60
    assert trade.open_rate == 2.0
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60
    ticker_usdt_modif = ticker_usdt.return_value
    ticker_usdt_modif['bid'] = ticker_usdt_modif['bid'] * 0.995
    mocker.patch(f'{EXMS}.fetch_ticker', return_value=ticker_usdt_modif)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    for o in trade.orders:
        assert o.status == 'closed'
    assert pytest.approx(trade.stake_amount) == 120
    assert trade.open_rate < 2.0
    assert trade.open_rate > 2.0 * 0.995
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert pytest.approx(trade.stake_amount) == 120
    assert trade.orders[0].amount == 30
    assert pytest.approx(trade.orders[1].amount) == 60 / ticker_usdt_modif['bid']
    assert pytest.approx(trade.amount) == trade.orders[0].amount + trade.orders[1].amount
    assert trade.nr_of_successful_buys == 2
    assert trade.nr_of_successful_entries == 2
    patch_get_signal(freqtrade, enter_long=False, exit_long=True)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert trade.is_open is False
    assert trade.orders[0].amount == 30
    assert trade.orders[0].side == 'buy'
    assert pytest.approx(trade.orders[1].amount) == 60 / ticker_usdt_modif['bid']
    assert trade.orders[-1].side == 'sell'
    assert trade.orders[2].amount == trade.amount
    assert trade.nr_of_successful_buys == 2
    assert trade.nr_of_successful_entries == 2

def test_dca_short(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Dict[str, Any],
    fee: float,
    mocker: MagicMock
) -> None:
    default_conf_usdt['position_adjustment_enable'] = True
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee, amount_to_precision=lambda s, x, y: round(y, 4), price_to_precision=lambda s, x, y: y)
    patch_get_signal(freqtrade, enter_long=False, enter_short=True)
    freqtrade.enter_positions()
    assert len(Trade.get_trades().all()) == 1
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60
    assert trade.open_rate == 2.02
    assert trade.orders[0].amount == trade.amount
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert pytest.approx(trade.stake_amount) == 60
    ticker_usdt_modif = ticker_usdt.return_value
    ticker_usdt_modif['ask'] = ticker_usdt_modif['ask'] * 1.004
    mocker.patch(f'{EXMS}.fetch_ticker', return_value=ticker_usdt_modif)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    for o in trade.orders:
        assert o.status == 'closed'
    assert pytest.approx(trade.stake_amount) == 120
    assert trade.open_rate >= 2.02
    assert trade.open_rate < 2.02 * 1.015
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert pytest.approx(trade.stake_amount) == 120
    assert trade.orders[1].amount == round(60 / ticker_usdt_modif['ask'], 4)
    assert trade.amount == trade.orders[0].amount + trade.orders[1].amount
    assert trade.nr_of_successful_entries == 2
    patch_get_signal(freqtrade, enter_long=False, exit_short=True)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert trade.is_open is False
    assert trade.orders[0].side == 'sell'
    assert trade.orders[1].amount == round(60 / ticker_usdt_modif['ask'], 4)
    assert trade.orders[-1].side == 'buy'
    assert trade.orders[2].amount == trade.amount
    assert trade.nr_of_successful_entries == 2
    assert trade.nr_of_successful_exits == 1

@pytest.mark.parametrize('leverage', [1, 2])
def test_dca_order_adjust(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Dict[str, Any],
    leverage: int,
    fee: float,
    mocker: MagicMock
) -> None:
    default_conf_usdt['position_adjustment_enable'] = True
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    freqtrade = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee, amount_to_precision=lambda s, x, y: y, price_to_precision=lambda s, x, y: y)
    mocker.patch(f'{EXMS}._dry_is_price_crossed', return_value=False)
    mocker.patch(f'{EXMS}.get_max_leverage', return_value=10)
    mocker.patch(f'{EXMS}.get_funding_fees', return_value=0)
    mocker.patch(f'{EXMS}.get_maintenance_ratio_and_amt', return_value=(0, 0))
    patch_get_signal(freqtrade)
    freqtrade.strategy.custom_entry_price = lambda **kwargs: ticker_usdt['ask'] * 0.96
    freqtrade.strategy.leverage = MagicMock(return_value=leverage)
    freqtrade.strategy.minimal_roi = {0: 0.2}
    freqtrade.enter_positions()
    assert len(Trade.get_trades().all()) == 1
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert trade.has_open_orders
    assert pytest.approx(trade.stake_amount) == 60
    assert trade.open_rate == 1.96
    assert trade.stop_loss_pct == -0.1
    assert pytest.approx(trade.stop_loss) == trade.open_rate * (1 - 0.1 / leverage)
    assert pytest.approx(trade.initial_stop_loss) == trade.open_rate * (1 - 0.1 / leverage)
    assert trade.initial_stop_loss_pct == -0.1
    assert trade.leverage == leverage
    assert trade.stake_amount == 60
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 1
    assert trade.has_open_orders
    assert pytest.approx(trade.stake_amount) == 60
    freqtrade.strategy.adjust_entry_price = MagicMock(return_value=1.99)
    freqtrade.process()
    trade = Trade.get_trades().first()
    assert len(trade.orders) == 2
    assert trade.has_open_orders
    assert trade.open_rate == 1.96
    assert trade.stop_loss_pct == -0.1
    assert pytest.approx(trade.stop_loss) == trade.open_rate * (1 - 0.1 / leverage)
    assert pytest.approx(trade.initial_stop_loss) == trade.open_rate * (1 - 0.1 / leverage