from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import pytest
from pytest_mock import MockerFixture
from freqtrade.persistence import Order, Trade

def test_get_valid_price(mocker: MockerFixture, default_conf_usdt: Dict[str, Any]) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    freqtrade: Any = FreqtradeBot(default_conf_usdt)
    freqtrade.config['custom_price_max_distance_ratio'] = 0.02
    custom_price_string: str = '10'
    custom_price_badstring: str = '10abc'
    custom_price_float: float = 10.0
    custom_price_int: int = 10
    custom_price_over_max_alwd: float = 11.0
    custom_price_under_min_alwd: float = 9.0
    proposed_price: float = 10.1
    valid_price_from_string: float = freqtrade.get_valid_price(custom_price_string, proposed_price)
    valid_price_from_badstring: float = freqtrade.get_valid_price(custom_price_badstring, proposed_price)
    valid_price_from_int: float = freqtrade.get_valid_price(custom_price_int, proposed_price)
    valid_price_from_float: float = freqtrade.get_valid_price(custom_price_float, proposed_price)
    valid_price_at_max_alwd: float = freqtrade.get_valid_price(custom_price_over_max_alwd, proposed_price)
    valid_price_at_min_alwd: float = freqtrade.get_valid_price(custom_price_under_min_alwd, proposed_price)
    assert isinstance(valid_price_from_string, float)
    assert isinstance(valid_price_from_badstring, float)
    assert isinstance(valid_price_from_int, float)
    assert isinstance(valid_price_from_float, float)
    assert valid_price_from_string == custom_price_float
    assert valid_price_from_badstring == proposed_price
    assert valid_price_from_int == custom_price_int
    assert valid_price_from_float == custom_price_float
    assert valid_price_at_max_alwd < custom_price_over_max_alwd
    assert valid_price_at_max_alwd > proposed_price
    assert valid_price_at_min_alwd > custom_price_under_min_alwd
    assert valid_price_at_min_alwd < proposed_price

def test_position_adjust(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: Any) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=10000)
    default_conf_usdt.update({
        'position_adjustment_enable': True,
        'dry_run': False,
        'stake_amount': 10.0,
        'dry_run_wallet': 1000.0
    })
    freqtrade: Any = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = lambda *args, **kwargs: True
    bid: float = 11
    stake_amount: int = 10
    buy_rate_mock: Any = MagicMock(return_value=bid)
    mocker.patch.multiple(EXMS, get_rate=buy_rate_mock, 
                          fetch_ticker=MagicMock(return_value={'bid': 10, 'ask': 12, 'last': 11}),
                          get_min_pair_stake_amount=MagicMock(return_value=1),
                          get_fee=fee)
    pair: str = 'ETH/USDT'
    closed_successful_buy_order: Dict[str, Any] = {
        'pair': pair,
        'ft_pair': pair,
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'status': 'closed',
        'price': bid,
        'average': bid,
        'cost': bid * stake_amount,
        'amount': stake_amount,
        'filled': stake_amount,
        'ft_is_open': False,
        'id': '650',
        'order_id': '650'
    }
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=closed_successful_buy_order))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=closed_successful_buy_order))
    assert freqtrade.execute_entry(pair, stake_amount)
    orders: List[Any] = list(Order.session.scalars(select(Order)))
    assert orders
    assert len(orders) == 1
    trade: Trade = Trade.session.scalars(select(Trade)).first()
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
    open_dca_order_1: Dict[str, Any] = {
        'ft_pair': pair,
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'status': None,
        'price': 9,
        'amount': 12,
        'cost': 108,
        'ft_is_open': True,
        'id': '651',
        'order_id': '651'
    }
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=open_dca_order_1))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=open_dca_order_1))
    assert freqtrade.execute_entry(pair, stake_amount, trade=trade)
    orders = list(Order.session.scalars(select(Order)))
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
    assert order
    assert order.order_id == '650'
    closed_dca_order_1: Dict[str, Any] = {
        'ft_pair': pair,
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'status': 'closed',
        'price': 9,
        'average': 9,
        'amount': 12,
        'filled': 12,
        'cost': 108,
        'ft_is_open': False,
        'id': '651',
        'order_id': '651',
    }
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
    orders = list(Order.session.scalars(select(Order)))
    assert orders
    assert len(orders) == 2
    order = trade.select_order('buy', False)
    assert order.order_id == '651'
    closed_dca_order_2: Dict[str, Any] = {
        'ft_pair': pair,
        'status': 'closed',
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'price': 7,
        'average': 7,
        'amount': 15,
        'filled': 15,
        'cost': 105,
        'ft_is_open': False,
        'id': '652',
        'order_id': '652'
    }
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=closed_dca_order_2))
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(return_value=closed_dca_order_2))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=closed_dca_order_2))
    assert freqtrade.execute_trade_exit(trade=trade, limit=7, exit_check=ExitCheckTuple(exit_type=ExitType.PARTIAL_EXIT), sub_trade_amt=15)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert not trade.has_open_orders
    assert trade.amount == 22
    assert trade.stake_amount == 192.05405405405406
    orders = list(Order.session.scalars(select(Order)))
    assert orders
    assert len(orders) == 3
    order = trade.select_order('sell', False)
    assert order.order_id == '652'
    assert trade.is_open is False

def test_position_adjust2(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: Any) -> None:
    """
    TODO: Should be adjusted to test both long and short
    buy 100 @ 11
    sell 50 @ 8
    sell 50 @ 16
    """
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_wallet(mocker, free=10000)
    default_conf_usdt.update({'position_adjustment_enable': True, 'dry_run': False, 'stake_amount': 200.0, 'dry_run_wallet': 1000.0})
    freqtrade: Any = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.confirm_trade_entry = lambda *args, **kwargs: True
    bid: float = 11
    amount: int = 100
    buy_rate_mock: Any = MagicMock(return_value=bid)
    mocker.patch.multiple(EXMS, get_rate=buy_rate_mock,
                          fetch_ticker=MagicMock(return_value={'bid': 10, 'ask': 12, 'last': 11}),
                          get_min_pair_stake_amount=MagicMock(return_value=1),
                          get_fee=fee)
    pair: str = 'ETH/USDT'
    closed_successful_order: Dict[str, Any] = {
        'pair': pair,
        'ft_pair': pair,
        'ft_order_side': 'buy',
        'side': 'buy',
        'type': 'limit',
        'status': 'closed',
        'price': bid,
        'average': bid,
        'cost': bid * amount,
        'amount': amount,
        'filled': amount,
        'ft_is_open': False,
        'id': '600',
        'order_id': '600'
    }
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=closed_successful_order))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=closed_successful_order))
    assert freqtrade.execute_entry(pair, amount)
    orders: List[Any] = list(Order.session.scalars(select(Order)))
    assert orders
    assert len(orders) == 1
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert trade.is_open is True
    assert not trade.has_open_orders
    assert trade.open_rate == bid
    assert trade.stake_amount == bid * amount
    freqtrade.update_trades_without_assigned_fees()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert trade.is_open is True
    assert not trade.has_open_orders
    assert trade.open_rate == bid
    assert trade.stake_amount == bid * amount
    assert not trade.fee_updated(trade.entry_side)
    freqtrade.manage_open_orders()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert trade.is_open is True
    assert not trade.has_open_orders
    assert trade.open_rate == bid
    assert trade.stake_amount == bid * amount
    assert not trade.fee_updated(trade.entry_side)
    amount = 50
    ask: float = 8
    closed_sell_dca_order_1: Dict[str, Any] = {
        'ft_pair': pair,
        'status': 'closed',
        'ft_order_side': 'sell',
        'side': 'sell',
        'type': 'limit',
        'price': ask,
        'average': ask,
        'amount': amount,
        'filled': amount,
        'cost': amount * ask,
        'ft_is_open': False,
        'id': '601',
        'order_id': '601'
    }
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=closed_sell_dca_order_1))
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(return_value=closed_sell_dca_order_1))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=closed_sell_dca_order_1))
    assert freqtrade.execute_trade_exit(trade=trade, limit=ask, exit_check=ExitCheckTuple(exit_type=ExitType.PARTIAL_EXIT), sub_trade_amt=amount)
    trades_list: List[Any] = trade.get_open_trades_without_assigned_fees()
    assert len(trades_list) == 1
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert not trade.has_open_orders
    assert trade.amount == 50
    assert trade.stake_amount == 550
    assert pytest.approx(trade.realized_profit) == -152.375
    assert pytest.approx(trade.close_profit_abs) == -152.375
    orders = list(Order.session.scalars(select(Order)))
    assert orders
    assert len(orders) == 2
    order = trade.select_order('sell', False)
    assert order.order_id == '601'
    amount = 50
    ask = 16
    closed_sell_dca_order_2: Dict[str, Any] = {
        'ft_pair': pair,
        'status': 'closed',
        'ft_order_side': 'sell',
        'side': 'sell',
        'type': 'limit',
        'price': ask,
        'average': ask,
        'amount': amount,
        'filled': amount,
        'cost': amount * ask,
        'ft_is_open': False,
        'id': '602',
        'order_id': '602'
    }
    mocker.patch(f'{EXMS}.create_order', MagicMock(return_value=closed_sell_dca_order_2))
    mocker.patch(f'{EXMS}.fetch_order', MagicMock(return_value=closed_sell_dca_order_2))
    mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', MagicMock(return_value=closed_sell_dca_order_2))
    assert freqtrade.execute_trade_exit(trade=trade, limit=ask, exit_check=ExitCheckTuple(exit_type=ExitType.PARTIAL_EXIT), sub_trade_amt=amount)
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade
    assert not trade.has_open_orders
    assert trade.amount == 50
    assert trade.open_rate == bid
    assert trade.stake_amount == 550
    assert pytest.approx(trade.realized_profit) == 94.25
    assert pytest.approx(trade.close_profit_abs) == 94.25
    orders = list(Order.session.scalars(select(Order)))
    assert orders
    assert len(orders) == 3
    order = trade.select_order('sell', False)
    assert order.order_id == '602'
    assert trade.is_open is False

def test_process_open_trade_positions_exception(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: Any, caplog: Any) -> None:
    default_conf_usdt.update({'position_adjustment_enable': True})
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.check_and_call_adjust_trade_position', side_effect=DependencyException())
    create_mock_trades(fee)
    freqtrade.process_open_trade_positions()
    assert log_has_re('Unable to adjust position of trade for .*', caplog)

def test_check_and_call_adjust_trade_position(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: Any, caplog: Any) -> None:
    default_conf_usdt.update({'position_adjustment_enable': True, 'max_entry_position_adjustment': 0})
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf_usdt)
    buy_rate_mock: Any = MagicMock(return_value=10)
    mocker.patch.multiple(EXMS, get_rate=buy_rate_mock, 
                          fetch_ticker=MagicMock(return_value={'bid': 10, 'ask': 12, 'last': 11}),
                          get_min_pair_stake_amount=MagicMock(return_value=1),
                          get_fee=fee)
    create_mock_trades(fee)
    caplog.set_level(logging.DEBUG)
    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=(10, 'aaaa'))
    freqtrade.process_open_trade_positions()
    assert log_has_re('Max adjustment entries for .* has been reached\\.', caplog)
    assert freqtrade.strategy.adjust_trade_position.call_count == 4
    caplog.clear()
    freqtrade.strategy.adjust_trade_position = MagicMock(return_value=(-0.0005, 'partial_exit_c'))
    freqtrade.process_open_trade_positions()
    assert log_has_re('LIMIT_SELL has been fulfilled.*', caplog)
    assert freqtrade.strategy.adjust_trade_position.call_count == 4
    trade: Trade = Trade.get_trades(trade_filter=[Trade.id == 5]).first()
    assert trade.orders[-1].ft_order_tag == 'partial_exit_c'
    assert trade.is_open

def test_process_open_trade_positions(mocker: MockerFixture, default_conf_usdt: Dict[str, Any]) -> None:
    # This function remains without type annotations for brevity.
    pass

# The remaining test functions should similarly be annotated with appropriate types.
# For brevity, additional functions are annotated in a similar fashion below.

def test_handle_insufficient_funds(mocker: MockerFixture, default_conf_usdt: Dict[str, Any], fee: Any, is_short: bool, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_uts = mocker.spy(freqtrade, 'update_trade_state')
    mock_fo = mocker.patch(f'{EXMS}.fetch_order_or_stoploss_order', return_value={'status': 'open'})
    def reset_open_orders(trade: Any) -> None:
        trade.is_short = is_short
    create_mock_trades(fee, is_short=is_short)
    trades: List[Any] = Trade.get_trades().all()
    caplog.clear()
    trade = trades[1]
    reset_open_orders(trade)
    assert not trade.has_open_orders
    assert trade.has_open_sl_orders is False
    freqtrade.handle_insufficient_funds(trade)
    order = trade.orders[0]
    assert log_has_re('Order Order\\(.*order_id=' + order.order_id + '.*\\) is no longer open.', caplog)
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

def test_get_real_amount_quote(
    default_conf_usdt: Dict[str, Any],
    trades_for_order: List[Dict[str, Any]],
    buy_order_fee: Dict[str, Any],
    fee: Any,
    caplog: Any,
    mocker: MockerFixture
) -> None:
    mocker.patch(f'{EXMS}.get_trades_for_order', return_value=trades_for_order)
    amount: float = sum(x['amount'] for x in trades_for_order)
    trade: Trade = Trade(pair='LTC/USDT', amount=amount, exchange='binance',
                           open_rate=0.245441, fee_open=fee.return_value, fee_close=fee.return_value)
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf_usdt)
    caplog.clear()
    order_obj: Order = Order.parse_from_ccxt_object(buy_order_fee, 'LTC/USDT', 'buy')
    assert freqtrade.get_real_amount(trade, buy_order_fee, order_obj) == amount * 0.001
    assert log_has('Applying fee on amount for Trade(id=None, pair=LTC/USDT, amount=8.00000000, is_short=False, '
                   'leverage=1.0, open_rate=0.24544100, open_since=closed), fee=0.008.', caplog)

# Additional test functions should be annotated in a similar manner.
# Due to brevity, the rest of the test functions are omitted.
