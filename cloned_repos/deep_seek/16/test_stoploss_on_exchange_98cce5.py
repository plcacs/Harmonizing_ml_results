from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import ANY, MagicMock
import pytest
from sqlalchemy import select
from freqtrade.enums import ExitCheckTuple, ExitType, RPCMessageType
from freqtrade.exceptions import ExchangeError, InsufficientFundsError, InvalidOrderException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.models import PairLock
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import EXMS, get_patched_freqtradebot, log_has, log_has_re, patch_edge, patch_exchange, patch_get_signal, patch_whitelist
from tests.conftest_trades import entry_side, exit_side
from tests.freqtradebot.test_freqtradebot import patch_RPCManager

@pytest.mark.parametrize('is_short', [False, True])
def test_add_stoploss_on_exchange(mocker: MagicMock, default_conf_usdt: Dict[str, Any], limit_order: Dict[str, Any], is_short: bool, fee: MagicMock) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(return_value=limit_order[entry_side(is_short)]), get_fee=fee)
    order = limit_order[entry_side(is_short)]
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch(f'{EXMS}.fetch_order', return_value=order)
    mocker.patch(f'{EXMS}.get_trades_for_order', return_value=[])
    stoploss = MagicMock(return_value={'id': 13434334})
    mocker.patch(f'{EXMS}.create_stoploss', stoploss)
    freqtrade = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    trades = [trade]
    freqtrade.exit_positions(trades)
    assert trade.has_open_sl_orders is True
    assert stoploss.call_count == 1
    assert trade.is_open is True

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange(mocker: MagicMock, default_conf_usdt: Dict[str, Any], fee: MagicMock, caplog: pytest.LogCaptureFixture, is_short: bool, limit_order: Dict[str, Any]) -> None:
    stop_order_dict: Dict[str, Any] = {'id': '13434334'}
    stoploss = MagicMock(return_value=stop_order_dict)
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(side_effect=[enter_order, exit_order]), get_fee=fee, create_stoploss=stoploss)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_short == is_short
    assert trade.is_open
    assert trade.has_open_sl_orders is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 1
    assert trade.open_sl_orders[-1].order_id == '13434334'
    trade.is_open = True
    hanging_stoploss_order = MagicMock(return_value={'id': '13434334', 'status': 'open'})
    mocker.patch(f'{EXMS}.fetch_stoploss_order', hanging_stoploss_order)
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    hanging_stoploss_order.assert_called_once_with('13434334', trade.pair)
    assert len(trade.open_sl_orders) == 1
    assert trade.open_sl_orders[-1].order_id == '13434334'
    caplog.clear()
    trade.is_open = True
    canceled_stoploss_order = MagicMock(return_value={'id': '13434334', 'status': 'canceled'})
    mocker.patch(f'{EXMS}.fetch_stoploss_order', canceled_stoploss_order)
    stoploss.reset_mock()
    amount_before = trade.amount
    stop_order_dict.update({'id': '103_1'})
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 1
    assert len(trade.open_sl_orders) == 1
    assert trade.open_sl_orders[-1].order_id == '103_1'
    assert trade.amount == amount_before
    caplog.clear()
    stop_order_dict.update({'id': '103_1'})
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    stoploss_order_hit = MagicMock(return_value={'id': '103_1', 'status': 'closed', 'type': 'stop_loss_limit', 'price': 3, 'average': 2, 'filled': enter_order['amount'], 'remaining': 0, 'amount': enter_order['amount']})
    mocker.patch(f'{EXMS}.fetch_stoploss_order', stoploss_order_hit)
    freqtrade.strategy.order_filled = MagicMock(return_value=None)
    assert freqtrade.handle_stoploss_on_exchange(trade) is True
    assert log_has_re('STOP_LOSS_LIMIT is hit for Trade\\(id=1, .*\\)\\.', caplog)
    assert len(trade.open_sl_orders) == 0
    assert trade.is_open is False
    assert freqtrade.strategy.order_filled.call_count == 1
    caplog.clear()
    mocker.patch(f'{EXMS}.create_stoploss', side_effect=ExchangeError())
    trade.is_open = True
    freqtrade.handle_stoploss_on_exchange(trade)
    assert log_has('Unable to place a stoploss order on exchange.', caplog)
    assert len(trade.open_sl_orders) == 0
    stop_order_dict.update({'id': '105'})
    stoploss.reset_mock()
    mocker.patch(f'{EXMS}.fetch_stoploss_order', side_effect=InvalidOrderException())
    mocker.patch(f'{EXMS}.create_stoploss', stoploss)
    freqtrade.handle_stoploss_on_exchange(trade)
    assert len(trade.open_sl_orders) == 1
    assert stoploss.call_count == 1
    trade.is_open = False
    trade.open_sl_orders[-1].ft_is_open = False
    stoploss.reset_mock()
    mocker.patch(f'{EXMS}.fetch_order')
    mocker.patch(f'{EXMS}.create_stoploss', stoploss)
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.has_open_sl_orders is False
    assert stoploss.call_count == 0

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_emergency(mocker: MagicMock, default_conf_usdt: Dict[str, Any], fee: MagicMock, is_short: bool, limit_order: Dict[str, Any]) -> None:
    stop_order_dict: Dict[str, Any] = {'id': '13434334'}
    stoploss = MagicMock(return_value=stop_order_dict)
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(side_effect=[enter_order, exit_order]), get_fee=fee, create_stoploss=stoploss)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_short == is_short
    assert trade.is_open
    assert trade.has_open_sl_orders is False
    stoploss_order_cancelled = MagicMock(side_effect=[{'id': '107', 'status': 'canceled', 'type': 'stop_loss_limit', 'price': 3, 'average': 2, 'amount': enter_order['amount'], 'filled': 0, 'remaining': enter_order['amount'], 'info': {'stopPrice': 22}}])
    trade.stoploss_last_update = dt_now() - timedelta(hours=1)
    trade.stop_loss = 24
    trade.exit_reason = None
    trade.orders.append(Order(ft_order_side='stoploss', ft_pair=trade.pair, ft_is_open=True, ft_amount=trade.amount, ft_price=trade.stop_loss, order_id='107', status='open'))
    freqtrade.config['trailing_stop'] = True
    stoploss = MagicMock(side_effect=InvalidOrderException())
    assert trade.has_open_sl_orders is True
    Trade.commit()
    mocker.patch(f'{EXMS}.cancel_stoploss_order_with_result', side_effect=InvalidOrderException())
    mocker.patch(f'{EXMS}.fetch_stoploss_order', stoploss_order_cancelled)
    mocker.patch(f'{EXMS}.create_stoploss', stoploss)
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.has_open_sl_orders is False
    assert trade.is_open is False
    assert trade.exit_reason == str(ExitType.EMERGENCY_EXIT)

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_partial(mocker: MagicMock, default_conf_usdt: Dict[str, Any], fee: MagicMock, is_short: bool, limit_order: Dict[str, Any]) -> None:
    stop_order_dict: Dict[str, Any] = {'id': '101', 'status': 'open'}
    stoploss = MagicMock(return_value=stop_order_dict)
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(side_effect=[enter_order, exit_order]), get_fee=fee, create_stoploss=stoploss)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 1
    assert trade.has_open_sl_orders is True
    assert trade.open_sl_orders[-1].order_id == '101'
    assert trade.amount == 30
    stop_order_dict.update({'id': '102'})
    stoploss_order_hit = MagicMock(return_value={'id': '101', 'status': 'canceled', 'type': 'stop_loss_limit', 'price': 3, 'average': 2, 'filled': trade.amount / 2, 'remaining': trade.amount / 2, 'amount': enter_order['amount']})
    mocker.patch(f'{EXMS}.fetch_stoploss_order', stoploss_order_hit)
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.amount == 15
    assert trade.open_sl_orders[-1].order_id == '102'

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_partial_cancel_here(mocker: MagicMock, default_conf_usdt: Dict[str, Any], fee: MagicMock, is_short: bool, limit_order: Dict[str, Any], caplog: pytest.LogCaptureFixture, time_machine: Any) -> None:
    stop_order_dict: Dict[str, Any] = {'id': '101', 'status': 'open'}
    time_machine.move_to(dt_now())
    default_conf_usdt['trailing_stop'] = True
    stoploss = MagicMock(return_value=stop_order_dict)
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(side_effect=[enter_order, exit_order]), get_fee=fee, create_stoploss=stoploss)
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 1
    assert trade.has_open_sl_orders is True
    assert trade.open_sl_orders[-1].order_id == '101'
    assert trade.amount == 30
    stop_order_dict.update({'id': '102'})
    stoploss_order_hit = MagicMock(return_value={'id': '101', 'status': 'open', 'type': 'stop_loss_limit', 'price': 3, 'average': 2, 'filled': 0, 'remaining': trade.amount, 'amount': enter_order['amount']})
    stoploss_order_cancel = MagicMock(return_value={'id': '101', 'status': 'canceled', 'type': 'stop_loss_limit', 'price': 3, 'average': 2, 'filled': trade.amount / 2, 'remaining': trade.amount / 2, 'amount': enter_order['amount']})
    mocker.patch(f'{EXMS}.fetch_stoploss_order', stoploss_order_hit)
    mocker.patch(f'{EXMS}.cancel_stoploss_order_with_result', stoploss_order_cancel)
    time_machine.shift(timedelta(minutes=15))
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert log_has_re('Cancelling current stoploss on exchange.*', caplog)
    assert trade.has_open_sl_orders is True
    assert trade.open_sl_orders[-1].order_id == '102'
    assert trade.amount == 15

@pytest.mark.parametrize('is_short', [False, True])
def test_handle_sle_cancel_cant_recreate(mocker: MagicMock, default_conf_usdt: Dict[str, Any], fee: MagicMock, caplog: pytest.LogCaptureFixture, is_short: bool, limit_order: Dict[str, Any]) -> None:
    enter_order = limit_order[entry_side(is_short)]
    exit_order = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}), create_order=MagicMock(side_effect=[enter_order, exit_order]), get_fee=fee)
    mocker.patch.multiple(EXMS, fetch_stoploss_order=MagicMock(return_value={'status': 'canceled', 'id': '100'}), create_stoploss=MagicMock(side_effect=ExchangeError()))
    freqtrade = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_short == is_short
    trade.is_open = True
    trade.orders.append(Order(ft_order_side='stoploss', ft_pair=trade.pair, ft_is_open=True, ft_amount=trade.amount, ft_price=trade.stop_loss, order_id='100', status='open'))
    assert trade
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert log_has_re('All Stoploss orders are cancelled, but unable to recreate one\\.', caplog)
    assert