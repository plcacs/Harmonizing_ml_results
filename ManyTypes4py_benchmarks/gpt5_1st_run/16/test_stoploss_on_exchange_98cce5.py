from copy import deepcopy
from datetime import timedelta
from unittest.mock import ANY, MagicMock
from typing import Any, Dict, List, Sequence, Tuple, Union

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
def test_add_stoploss_on_exchange(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    limit_order: Dict[str, Dict[str, Any]],
    is_short: bool,
    fee: Any,
) -> None:
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=MagicMock(return_value=limit_order[entry_side(is_short)]),
        get_fee=fee,
    )
    order: Dict[str, Any] = limit_order[entry_side(is_short)]
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_trade', MagicMock(return_value=True))
    mocker.patch(f'{EXMS}.fetch_order', return_value=order)
    mocker.patch(f'{EXMS}.get_trades_for_order', return_value=[])
    stoploss: MagicMock = MagicMock(return_value={'id': 13434334})
    mocker.patch(f'{EXMS}.create_stoploss', stoploss)
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    trades: List[Trade] = [trade]
    freqtrade.exit_positions(trades)
    assert trade.has_open_sl_orders is True
    assert stoploss.call_count == 1
    assert trade.is_open is True


@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    caplog: Any,
    is_short: bool,
    limit_order: Dict[str, Dict[str, Any]],
) -> None:
    stop_order_dict: Dict[str, Any] = {'id': '13434334'}
    stoploss: MagicMock = MagicMock(return_value=stop_order_dict)
    enter_order: Dict[str, Any] = limit_order[entry_side(is_short)]
    exit_order: Dict[str, Any] = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=MagicMock(side_effect=[enter_order, exit_order]),
        get_fee=fee,
        create_stoploss=stoploss,
    )
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_short == is_short
    assert trade.is_open
    assert trade.has_open_sl_orders is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 1
    assert trade.open_sl_orders[-1].order_id == '13434334'
    trade.is_open = True
    hanging_stoploss_order: MagicMock = MagicMock(return_value={'id': '13434334', 'status': 'open'})
    mocker.patch(f'{EXMS}.fetch_stoploss_order', hanging_stoploss_order)
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    hanging_stoploss_order.assert_called_once_with('13434334', trade.pair)
    assert len(trade.open_sl_orders) == 1
    assert trade.open_sl_orders[-1].order_id == '13434334'
    caplog.clear()
    trade.is_open = True
    canceled_stoploss_order: MagicMock = MagicMock(return_value={'id': '13434334', 'status': 'canceled'})
    mocker.patch(f'{EXMS}.fetch_stoploss_order', canceled_stoploss_order)
    stoploss.reset_mock()
    amount_before: float = trade.amount
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
    stoploss_order_hit: MagicMock = MagicMock(
        return_value={
            'id': '103_1',
            'status': 'closed',
            'type': 'stop_loss_limit',
            'price': 3,
            'average': 2,
            'filled': enter_order['amount'],
            'remaining': 0,
            'amount': enter_order['amount'],
        }
    )
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
def test_handle_stoploss_on_exchange_emergency(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    is_short: bool,
    limit_order: Dict[str, Dict[str, Any]],
) -> None:
    stop_order_dict: Dict[str, Any] = {'id': '13434334'}
    stoploss: MagicMock = MagicMock(return_value=stop_order_dict)
    enter_order: Dict[str, Any] = limit_order[entry_side(is_short)]
    exit_order: Dict[str, Any] = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=MagicMock(side_effect=[enter_order, exit_order]),
        get_fee=fee,
        create_stoploss=stoploss,
    )
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_short == is_short
    assert trade.is_open
    assert trade.has_open_sl_orders is False
    stoploss_order_cancelled: MagicMock = MagicMock(
        side_effect=[
            {
                'id': '107',
                'status': 'canceled',
                'type': 'stop_loss_limit',
                'price': 3,
                'average': 2,
                'amount': enter_order['amount'],
                'filled': 0,
                'remaining': enter_order['amount'],
                'info': {'stopPrice': 22},
            }
        ]
    )
    trade.stoploss_last_update = dt_now() - timedelta(hours=1)
    trade.stop_loss = 24
    trade.exit_reason = None
    trade.orders.append(
        Order(
            ft_order_side='stoploss',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.stop_loss,
            order_id='107',
            status='open',
        )
    )
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
def test_handle_stoploss_on_exchange_partial(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    is_short: bool,
    limit_order: Dict[str, Dict[str, Any]],
) -> None:
    stop_order_dict: Dict[str, Any] = {'id': '101', 'status': 'open'}
    stoploss: MagicMock = MagicMock(return_value=stop_order_dict)
    enter_order: Dict[str, Any] = limit_order[entry_side(is_short)]
    exit_order: Dict[str, Any] = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=MagicMock(side_effect=[enter_order, exit_order]),
        get_fee=fee,
        create_stoploss=stoploss,
    )
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 1
    assert trade.has_open_sl_orders is True
    assert trade.open_sl_orders[-1].order_id == '101'
    assert trade.amount == 30
    stop_order_dict.update({'id': '102'})
    stoploss_order_hit: MagicMock = MagicMock(
        return_value={
            'id': '101',
            'status': 'canceled',
            'type': 'stop_loss_limit',
            'price': 3,
            'average': 2,
            'filled': trade.amount / 2,
            'remaining': trade.amount / 2,
            'amount': enter_order['amount'],
        }
    )
    mocker.patch(f'{EXMS}.fetch_stoploss_order', stoploss_order_hit)
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.amount == 15
    assert trade.open_sl_orders[-1].order_id == '102'


@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_partial_cancel_here(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    is_short: bool,
    limit_order: Dict[str, Dict[str, Any]],
    caplog: Any,
    time_machine: Any,
) -> None:
    stop_order_dict: Dict[str, Any] = {'id': '101', 'status': 'open'}
    time_machine.move_to(dt_now())
    default_conf_usdt['trailing_stop'] = True
    stoploss: MagicMock = MagicMock(return_value=stop_order_dict)
    enter_order: Dict[str, Any] = limit_order[entry_side(is_short)]
    exit_order: Dict[str, Any] = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=MagicMock(side_effect=[enter_order, exit_order]),
        get_fee=fee,
        create_stoploss=stoploss,
    )
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert stoploss.call_count == 1
    assert trade.has_open_sl_orders is True
    assert trade.open_sl_orders[-1].order_id == '101'
    assert trade.amount == 30
    stop_order_dict.update({'id': '102'})
    stoploss_order_hit: MagicMock = MagicMock(
        return_value={
            'id': '101',
            'status': 'open',
            'type': 'stop_loss_limit',
            'price': 3,
            'average': 2,
            'filled': 0,
            'remaining': trade.amount,
            'amount': enter_order['amount'],
        }
    )
    stoploss_order_cancel: MagicMock = MagicMock(
        return_value={
            'id': '101',
            'status': 'canceled',
            'type': 'stop_loss_limit',
            'price': 3,
            'average': 2,
            'filled': trade.amount / 2,
            'remaining': trade.amount / 2,
            'amount': enter_order['amount'],
        }
    )
    mocker.patch(f'{EXMS}.fetch_stoploss_order', stoploss_order_hit)
    mocker.patch(f'{EXMS}.cancel_stoploss_order_with_result', stoploss_order_cancel)
    time_machine.shift(timedelta(minutes=15))
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert log_has_re('Cancelling current stoploss on exchange.*', caplog)
    assert trade.has_open_sl_orders is True
    assert trade.open_sl_orders[-1].order_id == '102'
    assert trade.amount == 15


@pytest.mark.parametrize('is_short', [False, True])
def test_handle_sle_cancel_cant_recreate(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    caplog: Any,
    is_short: bool,
    limit_order: Dict[str, Dict[str, Any]],
) -> None:
    enter_order: Dict[str, Any] = limit_order[entry_side(is_short)]
    exit_order: Dict[str, Any] = limit_order[exit_side(is_short)]
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=MagicMock(side_effect=[enter_order, exit_order]),
        get_fee=fee,
    )
    mocker.patch.multiple(
        EXMS,
        fetch_stoploss_order=MagicMock(return_value={'status': 'canceled', 'id': '100'}),
        create_stoploss=MagicMock(side_effect=ExchangeError()),
    )
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_short == is_short
    trade.is_open = True
    trade.orders.append(
        Order(
            ft_order_side='stoploss',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.stop_loss,
            order_id='100',
            status='open',
        )
    )
    assert trade
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert log_has_re('All Stoploss orders are cancelled, but unable to recreate one\\.', caplog)
    assert trade.has_open_sl_orders is False
    assert trade.is_open is True


@pytest.mark.parametrize('is_short', [False, True])
def test_create_stoploss_order_invalid_order(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    caplog: Any,
    fee: Any,
    is_short: bool,
    limit_order: Dict[str, Dict[str, Any]],
) -> None:
    open_order: Dict[str, Any] = limit_order[entry_side(is_short)]
    order: Dict[str, Any] = limit_order[exit_side(is_short)]
    rpc_mock: MagicMock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    create_order_mock: MagicMock = MagicMock(side_effect=[open_order, order])
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=create_order_mock,
        get_fee=fee,
    )
    mocker.patch.multiple(
        EXMS,
        fetch_order=MagicMock(return_value={'status': 'canceled'}),
        create_stoploss=MagicMock(side_effect=InvalidOrderException()),
    )
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    caplog.clear()
    rpc_mock.reset_mock()
    freqtrade.create_stoploss_order(trade, 200)
    assert trade.has_open_sl_orders is False
    assert trade.exit_reason == ExitType.EMERGENCY_EXIT.value
    assert log_has('Unable to place a stoploss order on exchange. ', caplog)
    assert log_has('Exiting the trade forcefully', caplog)
    assert create_order_mock.call_count == 2
    assert create_order_mock.call_args[1]['ordertype'] == 'market'
    assert create_order_mock.call_args[1]['pair'] == trade.pair
    assert create_order_mock.call_args[1]['amount'] == trade.amount
    assert rpc_mock.call_count == 2
    assert rpc_mock.call_args_list[0][0][0]['exit_reason'] == ExitType.EMERGENCY_EXIT.value
    assert rpc_mock.call_args_list[0][0][0]['order_type'] == 'market'
    assert rpc_mock.call_args_list[0][0][0]['type'] == 'exit'
    assert rpc_mock.call_args_list[1][0][0]['type'] == 'exit_fill'


@pytest.mark.parametrize('is_short', [False, True])
def test_create_stoploss_order_insufficient_funds(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    caplog: Any,
    fee: Any,
    limit_order: Dict[str, Dict[str, Any]],
    is_short: bool,
) -> None:
    exit_order: Any = limit_order[exit_side(is_short)]['id']
    freqtrade: FreqtradeBot = get_patched_freqtradebot(mocker, default_conf_usdt)
    mock_insuf: MagicMock = mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_insufficient_funds')
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=MagicMock(side_effect=[limit_order[entry_side(is_short)], exit_order]),
        get_fee=fee,
        fetch_order=MagicMock(return_value={'status': 'canceled'}),
    )
    mocker.patch.multiple(EXMS, create_stoploss=MagicMock(side_effect=InsufficientFundsError()))
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    caplog.clear()
    freqtrade.create_stoploss_order(trade, 200)
    assert trade.has_open_sl_orders is False
    assert mock_insuf.call_count == 1
    mock_insuf.reset_mock()
    freqtrade.create_stoploss_order(trade, 200)
    assert trade.has_open_sl_orders is False
    assert mock_insuf.call_count == 1


@pytest.mark.parametrize(
    'is_short,bid,ask,stop_price,hang_price',
    [
        (False, [4.38, 4.16], [4.4, 4.17], ['2.0805', 4.4 * 0.95], 3),
        (True, [1.09, 1.21], [1.1, 1.22], ['2.321', 1.09 * 1.05], 1.5),
    ],
)
@pytest.mark.usefixtures('init_persistence')
def test_handle_stoploss_on_exchange_trailing(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    is_short: bool,
    bid: Sequence[float],
    ask: Sequence[float],
    limit_order: Dict[str, Dict[str, Any]],
    stop_price: Sequence[Union[str, float]],
    hang_price: float,
    time_machine: Any,
) -> None:
    enter_order: Dict[str, Any] = limit_order[entry_side(is_short)]
    exit_order: Dict[str, Any] = limit_order[exit_side(is_short)]
    stoploss: MagicMock = MagicMock(return_value={'id': '13434334', 'status': 'open'})
    start_dt = dt_now()
    time_machine.move_to(start_dt, tick=False)
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 2.19, 'ask': 2.2, 'last': 2.19}),
        create_order=MagicMock(side_effect=[enter_order, exit_order]),
        get_fee=fee,
    )
    mocker.patch.multiple(EXMS, create_stoploss=stoploss, stoploss_adjust=MagicMock(return_value=True))
    default_conf_usdt['trailing_stop'] = True
    default_conf_usdt['minimal_roi']['0'] = 999999999
    freqtrade: FreqtradeBot = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    freqtrade.strategy.stoploss = 0.05 if is_short else -0.05
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    assert trade.has_open_sl_orders is False
    trade.stoploss_last_update = dt_now() - timedelta(minutes=20)
    trade.orders.append(
        Order(
            ft_order_side='stoploss',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.stop_loss,
            order_id='100',
            order_date=dt_now() - timedelta(minutes=20),
        )
    )
    stoploss_order_hanging: Dict[str, Any] = {
        'id': '100',
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': hang_price,
        'average': 2,
        'fee': {},
        'amount': 0,
        'info': {'stopPrice': stop_price[0]},
    }
    stoploss_order_cancel: Dict[str, Any] = deepcopy(stoploss_order_hanging)
    stoploss_order_cancel['status'] = 'canceled'
    mocker.patch(f'{EXMS}.fetch_stoploss_order', return_value=stoploss_order_hanging)
    mocker.patch(f'{EXMS}.cancel_stoploss_order', return_value=stoploss_order_cancel)
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert len(trade.open_sl_orders) == 1
    assert trade.open_sl_orders[-1].order_id == '13434334'
    mocker.patch(
        f'{EXMS}.fetch_ticker',
        MagicMock(return_value={'bid': bid[0], 'ask': ask[0], 'last': bid[0]}),
    )
    cancel_order_mock: MagicMock = MagicMock(
        return_value={'id': '13434334', 'status': 'canceled', 'fee': {}, 'amount': trade.amount}
    )
    stoploss_order_mock: MagicMock = MagicMock(return_value={'id': 'so1', 'status': 'open'})
    mocker.patch(f'{EXMS}.fetch_stoploss_order')
    mocker.patch(f'{EXMS}.cancel_stoploss_order', cancel_order_mock)
    mocker.patch(f'{EXMS}.create_stoploss', stoploss_order_mock)
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert len(trade.open_sl_orders) == 1
    cancel_order_mock.assert_not_called()
    stoploss_order_mock.assert_not_called()
    time_machine.move_to(start_dt + timedelta(minutes=10), tick=False)
    assert freqtrade.handle_trade(trade) is False
    assert trade.stop_loss == stop_price[1]
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    cancel_order_mock.assert_called_once_with('13434334', 'ETH/USDT')
    stoploss_order_mock.assert_called_once_with(
        amount=30,
        pair='ETH/USDT',
        order_types=freqtrade.strategy.order_types,
        stop_price=stop_price[1],
        side=exit_side(is_short),
        leverage=1.0,
    )
    mocker.patch(
        f'{EXMS}.fetch_ticker',
        MagicMock(return_value={'bid': bid[1], 'ask': ask[1], 'last': bid[1]}),
    )
    mocker.patch(f'{EXMS}.cancel_stoploss_order_with_result', return_value={'id': 'so1', 'status': 'canceled'})
    assert len(trade.open_sl_orders) == 1
    assert trade.open_sl_orders[-1].order_id == 'so1'
    assert freqtrade.handle_trade(trade) is True
    assert trade.is_open is False
    assert trade.has_open_sl_orders is False


@pytest.mark.parametrize('is_short', [False, True])
def test_handle_stoploss_on_exchange_trailing_error(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    caplog: Any,
    limit_order: Dict[str, Dict[str, Any]],
    is_short: bool,
    time_machine: Any,
) -> None:
    time_machine.move_to(dt_now() - timedelta(minutes=601))
    enter_order: Dict[str, Any] = limit_order[entry_side(is_short)]
    exit_order: Dict[str, Any] = limit_order[exit_side(is_short)]
    stoploss: MagicMock = MagicMock(return_value={'id': '13434334', 'status': 'open'})
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=MagicMock(side_effect=[{'id': enter_order['id']}, {'id': exit_order['id']}]),
        get_fee=fee,
        create_stoploss=stoploss,
        stoploss_adjust=MagicMock(return_value=True),
    )
    default_conf_usdt['trailing_stop'] = True
    freqtrade: FreqtradeBot = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    freqtrade.strategy.stoploss = 0.05 if is_short else -0.05
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    trade.stop_loss = 0.2
    stoploss_order_hanging: Dict[str, Any] = {
        'id': 'abcd',
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'info': {'stopPrice': '0.1'},
    }
    trade.orders.append(
        Order(
            ft_order_side='stoploss',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=3,
            order_id='abcd',
            order_date=dt_now(),
        )
    )
    mocker.patch(f'{EXMS}.cancel_stoploss_order', side_effect=InvalidOrderException())
    mocker.patch(f'{EXMS}.fetch_stoploss_order', return_value=stoploss_order_hanging)
    time_machine.shift(timedelta(minutes=50))
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert log_has_re('Could not cancel stoploss order abcd for pair ETH/USDT.*', caplog)
    assert stoploss.call_count == 1
    assert len(trade.open_sl_orders) == 2
    caplog.clear()
    cancel_mock: MagicMock = mocker.patch(f'{EXMS}.cancel_stoploss_order')
    mocker.patch(f'{EXMS}.create_stoploss', side_effect=ExchangeError())
    time_machine.shift(timedelta(minutes=50))
    freqtrade.handle_trailing_stoploss_on_exchange(trade, stoploss_order_hanging)
    assert cancel_mock.call_count == 2
    assert log_has_re('Could not create trailing stoploss order for pair ETH/USDT\\..*', caplog)


def test_stoploss_on_exchange_price_rounding(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    open_trade_usdt: Trade,
) -> None:
    patch_RPCManager(mocker)
    mocker.patch.multiple(EXMS, get_fee=fee)
    price_mock: MagicMock = MagicMock(side_effect=lambda p, s, **kwargs: int(s))
    stoploss_mock: MagicMock = MagicMock(return_value={'id': '13434334'})
    adjust_mock: MagicMock = MagicMock(return_value=False)
    mocker.patch.multiple(
        EXMS,
        create_stoploss=stoploss_mock,
        stoploss_adjust=adjust_mock,
        price_to_precision=price_mock,
    )
    freqtrade: FreqtradeBot = get_patched_freqtradebot(mocker, default_conf_usdt)
    open_trade_usdt.stop_loss = 222.55
    freqtrade.handle_trailing_stoploss_on_exchange(open_trade_usdt, {})
    assert price_mock.call_count == 1
    assert adjust_mock.call_count == 1
    assert adjust_mock.call_args_list[0][0][0] == 222


@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_handle_stoploss_on_exchange_custom_stop(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    fee: Any,
    is_short: bool,
    limit_order: Dict[str, Dict[str, Any]],
) -> None:
    enter_order: Dict[str, Any] = limit_order[entry_side(is_short)]
    exit_order: Dict[str, Any] = limit_order[exit_side(is_short)]
    stoploss: MagicMock = MagicMock(return_value={'id': 13434334, 'status': 'open'})
    patch_RPCManager(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 1.9, 'ask': 2.2, 'last': 1.9}),
        create_order=MagicMock(side_effect=[enter_order, exit_order]),
        get_fee=fee,
        is_cancel_order_result_suitable=MagicMock(return_value=True),
    )
    mocker.patch.multiple(EXMS, create_stoploss=stoploss, stoploss_adjust=MagicMock(return_value=True))
    default_conf_usdt['use_custom_stoploss'] = True
    default_conf_usdt['minimal_roi']['0'] = 999999999
    freqtrade: FreqtradeBot = get_patched_freqtradebot(mocker, default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    freqtrade.strategy.custom_stoploss = lambda *args, **kwargs: -0.04
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 60
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    trade.is_open = True
    trade.orders.append(
        Order(
            ft_order_side='stoploss',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.stop_loss,
            order_date=dt_now() - timedelta(minutes=601),
            order_id='100',
        )
    )
    Trade.commit()
    slo: Dict[str, Any] = {
        'id': '100',
        'status': 'open',
        'type': 'stop_loss_limit',
        'price': 3,
        'average': 2,
        'info': {'stopPrice': '2.0805'},
    }
    slo_canceled: Dict[str, Any] = deepcopy(slo)
    slo_canceled.update({'status': 'canceled'})

    def fetch_stoploss_order_mock(order_id: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        x = deepcopy(slo)
        x['id'] = order_id
        return x

    mocker.patch(f'{EXMS}.fetch_stoploss_order', MagicMock(fetch_stoploss_order_mock))
    mocker.patch(f'{EXMS}.cancel_stoploss_order', return_value=slo_canceled)
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    mocker.patch(
        f'{EXMS}.fetch_ticker',
        MagicMock(
            return_value={
                'bid': 4.38 if not is_short else 1.9 / 2,
                'ask': 4.4 if not is_short else 2.2 / 2,
                'last': 4.38 if not is_short else 1.9 / 2,
            }
        ),
    )
    cancel_order_mock: MagicMock = MagicMock()
    stoploss_order_mock: MagicMock = MagicMock(return_value={'id': 'so1', 'status': 'open'})
    mocker.patch(f'{EXMS}.cancel_stoploss_order', cancel_order_mock)
    mocker.patch(f'{EXMS}.create_stoploss', stoploss_order_mock)
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    cancel_order_mock.assert_not_called()
    stoploss_order_mock.assert_not_called()
    assert freqtrade.handle_trade(trade) is False
    assert trade.stop_loss == 4.4 * 0.96 if not is_short else 1.1
    assert trade.stop_loss_pct == -0.04 if not is_short else 0.04
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0
    cancel_order_mock.assert_not_called()
    stoploss_order_mock.assert_not_called()
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    cancel_order_mock.assert_called_once_with('13434334', 'ETH/USDT')
    stoploss_order_mock.assert_called_once_with(
        amount=pytest.approx(trade.amount),
        pair='ETH/USDT',
        order_types=freqtrade.strategy.order_types,
        stop_price=4.4 * 0.96 if not is_short else 0.95 * 1.04,
        side=exit_side(is_short),
        leverage=1.0,
    )
    mocker.patch(f'{EXMS}.fetch_ticker', MagicMock(return_value={'bid': 4.17, 'ask': 4.19, 'last': 4.17}))
    assert freqtrade.handle_trade(trade) is True


def test_tsl_on_exchange_compatible_with_edge(
    mocker: Any,
    edge_conf: Dict[str, Any],
    fee: Any,
    limit_order: Dict[str, Dict[str, Any]],
) -> None:
    enter_order: Dict[str, Any] = limit_order['buy']
    exit_order: Dict[str, Any] = limit_order['sell']
    enter_order['average'] = 2.19
    stoploss: MagicMock = MagicMock(return_value={'id': '13434334', 'status': 'open'})
    patch_RPCManager(mocker)
    patch_exchange(mocker)
    patch_edge(mocker)
    edge_conf['max_open_trades'] = float('inf')
    edge_conf['dry_run_wallet'] = 999.9
    edge_conf['exchange']['name'] = 'binance'
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=MagicMock(return_value={'bid': 2.19, 'ask': 2.2, 'last': 2.19}),
        create_order=MagicMock(side_effect=[enter_order, exit_order]),
        get_fee=fee,
        create_stoploss=stoploss,
    )
    edge_conf['trailing_stop'] = True
    edge_conf['trailing_stop_positive'] = 0.01
    edge_conf['trailing_stop_positive_offset'] = 0.011
    edge_conf['minimal_roi']['0'] = 999999999
    freqtrade: FreqtradeBot = FreqtradeBot(edge_conf)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    freqtrade.strategy.stoploss = -0.02
    freqtrade.strategy.order_types['stoploss_on_exchange_interval'] = 0
    patch_get_signal(freqtrade)
    freqtrade.active_pair_whitelist = freqtrade.edge.adjust(freqtrade.active_pair_whitelist)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_open = True
    trade.stoploss_last_update = dt_now()
    trade.orders.append(
        Order(
            ft_order_side='stoploss',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.stop_loss,
            order_id='100',
        )
    )
    stoploss_order_hanging: MagicMock = MagicMock(
        return_value={
            'id': '100',
            'status': 'open',
            'type': 'stop_loss_limit',
            'price': 3,
            'average': 2,
            'stopPrice': '2.178',
        }
    )
    mocker.patch(f'{EXMS}.fetch_stoploss_order', stoploss_order_hanging)
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert pytest.approx(trade.stop_loss) == 1.76
    cancel_order_mock: MagicMock = MagicMock()
    stoploss_order_mock: MagicMock = MagicMock()
    mocker.patch(f'{EXMS}.cancel_stoploss_order', cancel_order_mock)
    mocker.patch(f'{EXMS}.create_stoploss', stoploss_order_mock)
    mocker.patch(
        f'{EXMS}.fetch_ticker',
        MagicMock(return_value={'bid': 2.19 * 0.95, 'ask': 2.2 * 0.95, 'last': 2.19 * 0.95}),
    )
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert pytest.approx(trade.stop_loss) == 1.76
    cancel_order_mock.assert_not_called()
    mocker.patch(f'{EXMS}.fetch_ticker', MagicMock(return_value={'bid': 4.38, 'ask': 4.4, 'last': 4.38}))
    assert freqtrade.handle_trade(trade) is False
    assert freqtrade.handle_stoploss_on_exchange(trade) is False
    assert trade.stop_loss == 4.4 * 0.99
    cancel_order_mock.assert_called_once_with('100', 'NEO/BTC')
    stoploss_order_mock.assert_called_once_with(
        amount=30,
        pair='NEO/BTC',
        order_types=freqtrade.strategy.order_types,
        stop_price=4.4 * 0.99,
        side='sell',
        leverage=1.0,
    )


@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_down_stoploss_on_exchange_dry_run(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    is_short: bool,
    ticker_usdt_sell_down: Any,
    ticker_usdt_sell_up: Any,
    mocker: Any,
) -> None:
    rpc_mock: MagicMock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        _dry_is_price_crossed=MagicMock(side_effect=[True, False]),
    )
    patch_whitelist(mocker, default_conf_usdt)
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    assert trade.is_short == is_short
    assert trade
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt_sell_up if is_short else ticker_usdt_sell_down)
    default_conf_usdt['dry_run'] = True
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    trade.stop_loss = 2.0 * 1.01 if is_short else 2.0 * 0.99
    freqtrade.execute_trade_exit(
        trade=trade,
        limit=trade.stop_loss,
        exit_check=ExitCheckTuple(exit_type=ExitType.STOP_LOSS),
    )
    last_msg = rpc_mock.call_args_list[-1][0][0]
    assert {
        'type': RPCMessageType.EXIT,
        'trade_id': 1,
        'exchange': 'Binance',
        'pair': 'ETH/USDT',
        'direction': 'Short' if trade.is_short else 'Long',
        'leverage': 1.0,
        'gain': 'loss',
        'limit': 2.02 if is_short else 1.98,
        'order_rate': 2.02 if is_short else 1.98,
        'amount': pytest.approx(29.70297029 if is_short else 30.0),
        'order_type': 'limit',
        'buy_tag': None,
        'enter_tag': None,
        'open_rate': 2.02 if is_short else 2.0,
        'current_rate': 2.2 if is_short else 2.0,
        'profit_amount': -0.3 if is_short else -0.8985,
        'profit_ratio': -0.00501253 if is_short else -0.01493766,
        'stake_currency': 'USDT',
        'quote_currency': 'USDT',
        'fiat_currency': 'USD',
        'base_currency': 'ETH',
        'exit_reason': ExitType.STOP_LOSS.value,
        'open_date': ANY,
        'close_date': ANY,
        'close_rate': ANY,
        'sub_trade': False,
        'cumulative_profit': 0.0,
        'stake_amount': pytest.approx(60),
        'is_final_exit': False,
        'final_profit_ratio': ANY,
    } == last_msg


def test_execute_trade_exit_sloe_cancel_exception(
    mocker: Any,
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    caplog: Any,
) -> None:
    freqtrade: FreqtradeBot = get_patched_freqtradebot(mocker, default_conf_usdt)
    mocker.patch(f'{EXMS}.cancel_stoploss_order', side_effect=InvalidOrderException())
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=300))
    create_order_mock: MagicMock = MagicMock(side_effect=[{'id': '12345554'}, {'id': '12345555'}])
    patch_exchange(mocker)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt, get_fee=fee, create_order=create_order_mock)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    PairLock.session = MagicMock()
    freqtrade.config['dry_run'] = False
    trade.orders.append(
        Order(
            ft_order_side='stoploss',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.stop_loss,
            order_id='abcd',
            status='open',
        )
    )
    freqtrade.execute_trade_exit(
        trade=trade,
        limit=1234,
        exit_check=ExitCheckTuple(exit_type=ExitType.STOP_LOSS),
    )
    assert create_order_mock.call_count == 2
    assert log_has('Could not cancel stoploss order abcd for pair ETH/USDT', caplog)


@pytest.mark.parametrize('is_short', [False, True])
def test_execute_trade_exit_with_stoploss_on_exchange(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    ticker_usdt_sell_up: Any,
    is_short: bool,
    mocker: Any,
) -> None:
    default_conf_usdt['exchange']['name'] = 'binance'
    rpc_mock: MagicMock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    stoploss: MagicMock = MagicMock(return_value={'id': 123, 'status': 'open', 'info': {'foo': 'bar'}})
    mocker.patch('freqtrade.freqtradebot.FreqtradeBot.handle_order_fee')
    cancel_order: MagicMock = MagicMock(return_value=True)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
        create_stoploss=stoploss,
        cancel_stoploss_order=cancel_order,
        _dry_is_price_crossed=MagicMock(side_effect=[True, False]),
    )
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade, enter_short=is_short, enter_long=not is_short)
    freqtrade.enter_positions()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade
    trades: List[Trade] = [trade]
    freqtrade.manage_open_orders()
    freqtrade.exit_positions(trades)
    mocker.patch.multiple(EXMS, fetch_ticker=ticker_usdt_sell_up)
    freqtrade.execute_trade_exit(
        trade=trade,
        limit=ticker_usdt_sell_up()['ask' if is_short else 'bid'],
        exit_check=ExitCheckTuple(exit_type=ExitType.STOP_LOSS),
    )
    trade = Trade.session.scalars(select(Trade)).first()
    trade.is_short = is_short
    assert trade
    assert cancel_order.call_count == 1
    assert rpc_mock.call_count == 4


@pytest.mark.parametrize('is_short', [False, True])
def test_may_execute_trade_exit_after_stoploss_on_exchange_hit(
    default_conf_usdt: Dict[str, Any],
    ticker_usdt: Any,
    fee: Any,
    mocker: Any,
    is_short: bool,
) -> None:
    default_conf_usdt['exchange']['name'] = 'binance'
    rpc_mock: MagicMock = patch_RPCManager(mocker)
    patch_exchange(mocker)
    mocker.patch.multiple(
        EXMS,
        fetch_ticker=ticker_usdt,
        get_fee=fee,
        amount_to_precision=lambda s, x, y: y,
        price_to_precision=lambda s, x, y: y,
        _dry_is_price_crossed=MagicMock(side_effect=[False, True]),
    )
    stoploss: MagicMock = MagicMock(return_value={'id': 123, 'info': {'foo': 'bar'}})
    mocker.patch(f'{EXMS}.create_stoploss', stoploss)
    freqtrade: FreqtradeBot = FreqtradeBot(default_conf_usdt)
    freqtrade.strategy.order_types['stoploss_on_exchange'] = True
    patch_get_signal(freqtrade, enter_long=not is_short, enter_short=is_short)
    freqtrade.enter_positions()
    freqtrade.manage_open_orders()
    trade: Trade = Trade.session.scalars(select(Trade)).first()
    trades: List[Trade] = [trade]
    assert trade.has_open_sl_orders is False
    freqtrade.exit_positions(trades)
    assert trade
    assert trade.has_open_sl_orders is True
    assert not trade.has_open_orders
    stoploss_executed: MagicMock = MagicMock(
        return_value={
            'id': '123',
            'timestamp': 1542707426845,
            'datetime': '2018-11-20T09:50:26.845Z',
            'lastTradeTimestamp': None,
            'symbol': 'BTC/USDT',
            'type': 'stop_loss_limit',
            'side': 'buy' if is_short else 'sell',
            'price': 1.08801,
            'amount': trade.amount,
            'cost': 1.08801 * trade.amount,
            'average': 1.08801,
            'filled': trade.amount,
            'remaining': 0.0,
            'status': 'closed',
            'fee': None,
            'trades': None,
        }
    )
    mocker.patch(f'{EXMS}.fetch_stoploss_order', stoploss_executed)
    freqtrade.exit_positions(trades)
    assert trade.has_open_sl_orders is False
    assert trade.is_open is False
    assert trade.exit_reason == ExitType.STOPLOSS_ON_EXCHANGE.value
    assert rpc_mock.call_count == 4
    assert rpc_mock.call_args_list[1][0][0]['type'] == RPCMessageType.ENTRY
    assert rpc_mock.call_args_list[1][0][0]['amount'] > 20
    assert rpc_mock.call_args_list[2][0][0]['type'] == RPCMessageType.ENTRY_FILL
    assert rpc_mock.call_args_list[3][0][0]['type'] == RPCMessageType.EXIT_FILL