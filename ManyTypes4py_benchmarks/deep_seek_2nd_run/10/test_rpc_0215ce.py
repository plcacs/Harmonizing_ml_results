from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from unittest.mock import ANY, MagicMock, PropertyMock

import pytest
from numpy import isnan
from sqlalchemy import select
from freqtrade.edge import PairInfo
from freqtrade.enums import SignalDirection, State, TradingMode
from freqtrade.exceptions import ExchangeError, InvalidOrderException, TemporaryError
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.key_value_store import set_startup_time
from freqtrade.rpc import RPC, RPCException
from freqtrade.rpc.fiat_convert import CryptoToFiatConverter
from tests.conftest import EXMS, create_mock_trades, create_mock_trades_usdt, get_patched_freqtradebot, patch_get_signal

def test_rpc_trade_status(default_conf: Dict[str, Any], ticker: MagicMock, fee: MagicMock, mocker: MagicMock) -> None:
    gen_response: Dict[str, Any] = {'trade_id': 1, 'pair': 'ETH/BTC', 'base_currency': 'ETH', 'quote_currency': 'BTC', 'open_date': ANY, 'open_timestamp': ANY, 'open_fill_date': ANY, 'open_fill_timestamp': ANY, 'is_open': ANY, 'fee_open': ANY, 'fee_open_cost': ANY, 'fee_open_currency': ANY, 'fee_close': fee.return_value, 'fee_close_cost': ANY, 'fee_close_currency': ANY, 'open_rate_requested': ANY, 'open_trade_value': 0.0010025, 'close_rate_requested': ANY, 'exit_reason': ANY, 'exit_order_status': ANY, 'min_rate': ANY, 'max_rate': ANY, 'strategy': ANY, 'enter_tag': ANY, 'timeframe': 5, 'close_date': None, 'close_timestamp': None, 'open_rate': 1.098e-05, 'close_rate': None, 'current_rate': 1.099e-05, 'amount': 91.07468123, 'amount_requested': 91.07468124, 'stake_amount': 0.001, 'max_stake_amount': None, 'trade_duration': None, 'trade_duration_s': None, 'close_profit': None, 'close_profit_pct': None, 'close_profit_abs': None, 'profit_ratio': -0.00408133, 'profit_pct': -0.41, 'profit_abs': -4.09e-06, 'profit_fiat': ANY, 'stop_loss_abs': 9.89e-06, 'stop_loss_pct': -10.0, 'stop_loss_ratio': -0.1, 'stoploss_last_update': ANY, 'stoploss_last_update_timestamp': ANY, 'initial_stop_loss_abs': 9.89e-06, 'initial_stop_loss_pct': -10.0, 'initial_stop_loss_ratio': -0.1, 'stoploss_current_dist': pytest.approx(-1.0999999e-06), 'stoploss_current_dist_ratio': -0.10009099, 'stoploss_current_dist_pct': -10.01, 'stoploss_entry_dist': -0.00010402, 'stoploss_entry_dist_ratio': -0.10376381, 'open_orders': '', 'realized_profit': 0.0, 'realized_profit_ratio': None, 'total_profit_abs': -4.09e-06, 'total_profit_fiat': ANY, 'total_profit_ratio': None, 'exchange': 'binance', 'leverage': 1.0, 'interest_rate': 0.0, 'liquidation_price': None, 'is_short': False, 'funding_fees': 0.0, 'trading_mode': TradingMode.SPOT, 'amount_precision': 8.0, 'price_precision': 8.0, 'precision_mode': 2, 'precision_mode_price': 2, 'contract_size': 1, 'has_open_orders': False, 'nr_of_successful_entries': ANY, 'orders': [{'amount': 91.07468123, 'average': 1.098e-05, 'safe_price': 1.098e-05, 'cost': 0.0009999999999054, 'filled': 91.07468123, 'ft_order_side': 'buy', 'order_date': ANY, 'order_timestamp': ANY, 'order_filled_date': ANY, 'order_filled_timestamp': ANY, 'order_type': 'limit', 'price': 1.098e-05, 'is_open': False, 'pair': 'ETH/BTC', 'order_id': ANY, 'remaining': ANY, 'status': ANY, 'ft_is_entry': True, 'ft_fee_base': None, 'funding_fee': ANY, 'ft_order_tag': None}]}
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee, _dry_is_price_crossed=MagicMock(side_effect=[False, True]))
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match='.*no active trade*'):
        rpc._rpc_trade_status()
    freqtradebot.enter_positions()
    results = rpc._rpc_trade_status()
    response_unfilled = deepcopy(gen_response)
    response_unfilled.update({'amount': 0.0, 'open_trade_value': 0.0, 'stoploss_entry_dist': 0.0, 'stoploss_entry_dist_ratio': 0.0, 'profit_ratio': 0.0, 'profit_pct': 0.0, 'profit_abs': 0.0, 'total_profit_abs': 0.0, 'open_orders': '(limit buy rem=91.07468123)', 'has_open_orders': True})
    response_unfilled['orders'][0].update({'is_open': True, 'filled': 0.0, 'remaining': 91.07468123})
    assert results[0] == response_unfilled
    trade = Trade.get_open_trades()[0]
    trade.orders[0].remaining = None
    Trade.commit()
    results = rpc._rpc_trade_status()
    response_unfilled['orders'][0].update({'remaining': None})
    assert results[0] == response_unfilled
    trade = Trade.get_open_trades()[0]
    trade.orders[0].remaining = trade.amount
    Trade.commit()
    freqtradebot.manage_open_orders()
    trades = Trade.get_open_trades()
    freqtradebot.exit_positions(trades)
    results = rpc._rpc_trade_status()
    response = deepcopy(gen_response)
    response.update({'max_stake_amount': 0.001, 'total_profit_ratio': pytest.approx(-0.00409153), 'has_open_orders': False})
    assert results[0] == response
    mocker.patch(f'{EXMS}.get_rate', MagicMock(side_effect=ExchangeError("Pair 'ETH/BTC' not available")))
    results = rpc._rpc_trade_status()
    assert isnan(results[0]['profit_ratio'])
    assert isnan(results[0]['current_rate'])
    response_norate = deepcopy(gen_response)
    response_norate.update({'stoploss_current_dist': ANY, 'stoploss_current_dist_ratio': ANY, 'stoploss_current_dist_pct': ANY, 'max_stake_amount': 0.001, 'profit_ratio': ANY, 'profit_pct': ANY, 'profit_abs': ANY, 'total_profit_abs': ANY, 'total_profit_ratio': ANY, 'current_rate': ANY})
    assert results[0] == response_norate

def test_rpc_status_table(default_conf: Dict[str, Any], ticker: MagicMock, fee: MagicMock, mocker: MagicMock) -> None:
    mocker.patch.multiple('freqtrade.rpc.fiat_convert.FtCoinGeckoApi', get_price=MagicMock(return_value={'bitcoin': {'usd': 15000.0}}))
    mocker.patch('freqtrade.rpc.rpc.CryptoToFiatConverter._find_price', return_value=15000.0)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee)
    del default_conf['fiat_display_currency']
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    patch_get_signal(freqtradebot)
    rpc = RPC(freqtradebot)
    freqtradebot.state = State.RUNNING
    with pytest.raises(RPCException, match='.*no active trade*'):
        rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    mocker.patch(f'{EXMS}._dry_is_price_crossed', return_value=False)
    freqtradebot.enter_positions()
    result, headers, fiat_profit_sum, total_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert 'Since' in headers
    assert 'Pair' in headers
    assert 'now' == result[0][2]
    assert 'ETH/BTC' in result[0][1]
    assert '0.00% (0.00)' == result[0][3]
    assert '0.00' == f'{fiat_profit_sum:.2f}'
    assert '0.00' == f'{total_sum:.2f}'
    mocker.patch(f'{EXMS}._dry_is_price_crossed', return_value=True)
    freqtradebot.process()
    result, headers, fiat_profit_sum, total_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert 'Since' in headers
    assert 'Pair' in headers
    assert 'now' == result[0][2]
    assert 'ETH/BTC' in result[0][1]
    assert '-0.41% (-0.00)' == result[0][3]
    assert '-0.00' == f'{fiat_profit_sum:.2f}'
    rpc._config['fiat_display_currency'] = 'USD'
    rpc._fiat_converter = CryptoToFiatConverter({})
    result, headers, fiat_profit_sum, total_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert 'Since' in headers
    assert 'Pair' in headers
    assert len(result[0]) == 4
    assert 'now' == result[0][2]
    assert 'ETH/BTC' in result[0][1]
    assert '-0.41% (-0.06)' == result[0][3]
    assert '-0.06' == f'{fiat_profit_sum:.2f}'
    assert '-0.06' == f'{total_sum:.2f}'
    rpc._config['position_adjustment_enable'] = True
    rpc._config['max_entry_position_adjustment'] = 3
    result, headers, fiat_profit_sum, total_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert '# Entries' in headers
    assert len(result[0]) == 5
    assert result[0][4] == '1/4'
    mocker.patch(f'{EXMS}.get_rate', MagicMock(side_effect=ExchangeError("Pair 'ETH/BTC' not available")))
    result, headers, fiat_profit_sum, total_sum = rpc._rpc_status_table(default_conf['stake_currency'], 'USD')
    assert 'now' == result[0][2]
    assert 'ETH/BTC' in result[0][1]
    assert 'nan%' == result[0][3]
    assert isnan(fiat_profit_sum)

def test__rpc_timeunit_profit(default_conf_usdt: Dict[str, Any], ticker: MagicMock, fee: MagicMock, markets: Dict[str, Any], mocker: MagicMock, time_machine: Any) -> None:
    time_machine.move_to('2023-09-05 10:00:00 +00:00', tick=False)
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(EXMS, fetch_ticker=ticker, get_fee=fee, markets=PropertyMock(return_value=markets))
    freqtradebot = get_patched_freqtradebot(mocker, default_conf_usdt)
    create_mock_trades_usdt(fee)
    stake_currency = default_conf_usdt['stake_currency']
    fiat_display_currency = default_conf_usdt['fiat_display_currency']
    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter({})
    days = rpc._rpc_timeunit_profit(7, stake_currency, fiat_display_currency)
    assert len(days['data']) == 7
    assert days['stake_currency'] == default_conf_usdt['stake_currency']
    assert days['fiat_display_currency'] == default_conf_usdt['fiat_display_currency']
    for day in days['data']:
        assert day['abs_profit'] in (0.0, pytest.approx(6.83), pytest.approx(-4.09))
        assert day['rel_profit'] in (0.0, pytest.approx(0.00642902), pytest.approx(-0.00383512))
        assert day['trade_count'] in (0, 1, 2)
        assert day['starting_balance'] in (pytest.approx(1062.37), pytest.approx(1066.46))
        assert day['fiat_value'] in (0.0,)
    assert str(days['data'][0]['date']) == str(datetime.now(timezone.utc).date())
    with pytest.raises(RPCException, match='.*must be an integer greater than 0*'):
        rpc._rpc_timeunit_profit(0, stake_currency, fiat_display_currency)

@pytest.mark.parametrize('is_short', [True, False])
def test_rpc_trade_history(mocker: MagicMock, default_conf: Dict[str, Any], markets: Dict[str, Any], fee: MagicMock, is_short: bool) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=markets))
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    create_mock_trades(fee, is_short)
    rpc = RPC(freqtradebot)
    rpc._fiat_converter = CryptoToFiatConverter({})
    trades = rpc._rpc_trade_history(2)
    assert len(trades['trades']) == 2
    assert trades['trades_count'] == 2
    assert isinstance(trades['trades'][0], dict)
    assert isinstance(trades['trades'][1], dict)
    trades = rpc._rpc_trade_history(0)
    assert len(trades['trades']) == 2
    assert trades['trades_count'] == 2
    assert trades['trades'][-1]['pair'] == 'ETC/BTC'
    assert trades['trades'][0]['pair'] == 'XRP/BTC'

@pytest.mark.parametrize('is_short', [True, False])
def test_rpc_delete_trade(mocker: MagicMock, default_conf: Dict[str, Any], fee: MagicMock, markets: Dict[str, Any], caplog: Any, is_short: bool) -> None:
    mocker.patch('freqtrade.rpc.telegram.Telegram', MagicMock())
    stoploss_mock = MagicMock()
    cancel_mock = MagicMock()
    mocker.patch.multiple(EXMS, markets=PropertyMock(return_value=markets), cancel_order=cancel_mock, cancel_stoploss_order=stoploss_mock)
    freqtradebot = get_patched_freqtradebot(mocker, default_conf)
    freqtradebot.strategy.order_types['stoploss_on_exchange'] = True
    create_mock_trades(fee, is_short)
    rpc = RPC(freqtradebot)
    with pytest.raises(RPCException, match='invalid argument'):
        rpc._rpc_delete('200')
    trades = Trade.session.scalars(select(Trade)).all()
    trades[2].orders.append(Order(ft_order_side='stoploss', ft_pair=trades[2].pair, ft_is_open=True, ft_amount=trades[2].amount, ft_price=trades[2].stop_loss, order_id='102', status='open'))
    assert len(trades) > 2
    res = rpc._rpc_delete('1')
    assert isinstance(res, dict)
    assert res['result'] == 'success'
    assert res['trade_id'] == '1'
    assert res['cancel_order_count'] == 1
    assert cancel_mock.call_count == 1
    assert stoploss_mock.call_count == 0
    cancel_mock.reset_mock()
    stoploss_mock.reset_mock()
    res = rpc._rpc_delete('5')
    assert isinstance(res, dict)
    assert stoploss_mock.call_count == 1
    assert res['cancel_order_count'] == 1
    stoploss_mock = mocker.patch(f'{EXMS}.cancel_stoploss_order', side_effect=InvalidOrderException)
    res = rpc._rpc_delete('