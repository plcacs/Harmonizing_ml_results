from typing import Any, Optional, List, Union, Dict
import random
from datetime import datetime, timedelta, timezone
import pytest
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import PairLocks, Trade
from freqtrade.persistence.trade_model import Order
from freqtrade.plugins.protectionmanager import ProtectionManager
from tests.conftest import get_patched_freqtradebot, log_has_re

AVAILABLE_PROTECTIONS: List[str] = ['CooldownPeriod', 'LowProfitPairs', 'MaxDrawdown', 'StoplossGuard']

def generate_mock_trade(pair: str, fee: Any, is_open: bool, exit_reason: Union[ExitType, str] = ExitType.EXIT_SIGNAL,
                        min_ago_open: Optional[int] = None, min_ago_close: Optional[int] = None,
                        profit_rate: float = 0.9, is_short: bool = False) -> Trade:
    open_rate: float = random.random()
    trade: Trade = Trade(
        pair=pair,
        stake_amount=0.01,
        fee_open=fee,
        fee_close=fee,
        open_date=datetime.now(timezone.utc) - timedelta(minutes=min_ago_open or 200),
        close_date=datetime.now(timezone.utc) - timedelta(minutes=min_ago_close or 30),
        open_rate=open_rate,
        is_open=is_open,
        amount=0.01 / open_rate,
        exchange='binance',
        is_short=is_short,
        leverage=1
    )
    trade.orders.append(Order(
        ft_order_side=trade.entry_side,
        order_id=f'{pair}-{trade.entry_side}-{trade.open_date}',
        ft_is_open=False,
        ft_pair=pair,
        ft_amount=trade.amount,
        ft_price=trade.open_rate,
        amount=trade.amount,
        filled=trade.amount,
        remaining=0,
        price=open_rate,
        average=open_rate,
        status='closed',
        order_type='market',
        side=trade.entry_side
    ))
    if not is_open:
        close_price: float = open_rate * (2 - profit_rate if is_short else profit_rate)
        trade.orders.append(Order(
            ft_order_side=trade.exit_side,
            order_id=f'{pair}-{trade.exit_side}-{trade.close_date}',
            ft_is_open=False,
            ft_pair=pair,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            amount=trade.amount,
            filled=trade.amount,
            remaining=0,
            price=close_price,
            average=close_price,
            status='closed',
            order_type='market',
            side=trade.exit_side
        ))
    trade.recalc_open_trade_value()
    if not is_open:
        trade.close(close_price)
        trade.exit_reason = exit_reason
    Trade.session.add(trade)
    Trade.commit()
    return trade

def test_protectionmanager(mocker: Any, default_conf: Dict[str, Any]) -> None:
    default_conf['_strategy_protections'] = [{'method': protection} for protection in AVAILABLE_PROTECTIONS]
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf)
    for handler in freqtrade.protections._protection_handlers:
        assert handler.name in AVAILABLE_PROTECTIONS
        if not handler.has_global_stop:
            assert handler.global_stop(datetime.now(timezone.utc), '*') is None
        if not handler.has_local_stop:
            assert handler.stop_per_pair('XRP/BTC', datetime.now(timezone.utc), '*') is None

@pytest.mark.parametrize('protconf,expected', [
    ([], None),
    ([{'method': 'StoplossGuard', 'lookback_period': 2000, 'stop_duration_candles': 10}], None),
    ([{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'stop_duration': 10}], None),
    ([{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'lookback_period': 2000, 'stop_duration': 10}], 'Protections must specify either `lookback_period`.*'),
    ([{'method': 'StoplossGuard', 'lookback_period': 20, 'stop_duration': 10, 'stop_duration_candles': 10}], 'Protections must specify either `stop_duration`.*'),
    ([{'method': 'StoplossGuard', 'lookback_period': 20, 'stop_duration': 10, 'unlock_at': '20:02'}], 'Protections must specify either `unlock_at`, `stop_duration` or.*'),
    ([{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'unlock_at': '20:02'}], None),
    ([{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'unlock_at': '55:102'}], 'Invalid date format for unlock_at: 55:102.')
])
def test_validate_protections(protconf: List[Dict[str, Any]], expected: Optional[str]) -> None:
    if expected:
        with pytest.raises(OperationalException, match=expected):
            ProtectionManager.validate_protections(protconf)
    else:
        ProtectionManager.validate_protections(protconf)

@pytest.mark.parametrize('timeframe,expected_lookback,expected_stop,protconf', [
    ('1m', 20, 10, [{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'stop_duration': 10}]),
    ('5m', 100, 15, [{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'stop_duration': 15}]),
    ('1h', 1200, 40, [{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'stop_duration': 40}]),
    ('1d', 1440, 5, [{'method': 'StoplossGuard', 'lookback_period_candles': 1, 'stop_duration': 5}]),
    ('1m', 20, 5, [{'method': 'StoplossGuard', 'lookback_period': 20, 'stop_duration_candles': 5}]),
    ('5m', 15, 25, [{'method': 'StoplossGuard', 'lookback_period': 15, 'stop_duration_candles': 5}]),
    ('1h', 50, 600, [{'method': 'StoplossGuard', 'lookback_period': 50, 'stop_duration_candles': 10}]),
    ('1h', 60, 540, [{'method': 'StoplossGuard', 'lookback_period_candles': 1, 'stop_duration_candles': 9}]),
    ('1m', 20, '01:00', [{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'unlock_at': '01:00'}]),
    ('5m', 100, '02:00', [{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'unlock_at': '02:00'}]),
    ('1h', 1200, '03:00', [{'method': 'StoplossGuard', 'lookback_period_candles': 20, 'unlock_at': '03:00'}]),
    ('1d', 1440, '04:00', [{'method': 'StoplossGuard', 'lookback_period_candles': 1, 'unlock_at': '04:00'}])
])
def test_protections_init(default_conf: Dict[str, Any], timeframe: str, expected_lookback: int,
                            expected_stop: Union[int, str], protconf: List[Dict[str, Any]]) -> None:
    """
    Test the initialization of protections with different configurations, including unlock_at.
    """
    default_conf['timeframe'] = timeframe
    man: ProtectionManager = ProtectionManager(default_conf, protconf)
    assert len(man._protection_handlers) == len(protconf)
    assert man._protection_handlers[0]._lookback_period == expected_lookback
    if isinstance(expected_stop, int):
        assert man._protection_handlers[0]._stop_duration == expected_stop
    else:
        assert man._protection_handlers[0]._unlock_at == expected_stop

@pytest.mark.parametrize('is_short', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_stoploss_guard(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any, is_short: bool) -> None:
    default_conf['_strategy_protections'] = [{'method': 'StoplossGuard', 'lookback_period': 60, 'stop_duration': 40, 'trade_limit': 3}]
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf)
    message: str = 'Trading stopped due to .*'
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value, min_ago_open=200, min_ago_close=30, is_short=is_short)
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()
    generate_mock_trade('BCH/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value, min_ago_open=250, min_ago_close=100, is_short=is_short)
    generate_mock_trade('ETH/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value, min_ago_open=240, min_ago_close=30, is_short=is_short)
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()
    generate_mock_trade('LTC/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value, min_ago_open=180, min_ago_close=30, is_short=is_short)
    assert freqtrade.protections.global_stop()
    assert log_has_re(message, caplog)
    assert PairLocks.is_global_lock()
    end_time: datetime = PairLocks.get_pair_longest_lock('*').lock_end_time + timedelta(minutes=5)
    freqtrade.protections.global_stop(end_time)
    assert not PairLocks.is_global_lock(end_time)

@pytest.mark.parametrize('only_per_pair', [False, True])
@pytest.mark.parametrize('only_per_side', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_stoploss_guard_perpair(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any,
                                only_per_pair: bool, only_per_side: bool) -> None:
    default_conf['_strategy_protections'] = [{'method': 'StoplossGuard', 'lookback_period': 60, 'trade_limit': 2, 'stop_duration': 60, 'only_per_pair': only_per_pair, 'only_per_side': only_per_side}]
    check_side: str = 'long' if only_per_side else '*'
    is_short: bool = False
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf)
    message: str = 'Trading stopped due to .*'
    pair: str = 'XRP/BTC'
    assert not freqtrade.protections.stop_per_pair(pair)
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()
    generate_mock_trade(pair, fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=200, min_ago_close=30, profit_rate=0.9, is_short=is_short)
    assert not freqtrade.protections.stop_per_pair(pair)
    assert not freqtrade.protections.global_stop()
    assert not log_has_re(message, caplog)
    caplog.clear()
    generate_mock_trade(pair, fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=250, min_ago_close=100, profit_rate=0.9, is_short=is_short)
    generate_mock_trade('ETH/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=240, min_ago_close=30, profit_rate=0.9, is_short=is_short)
    assert not freqtrade.protections.stop_per_pair(pair)
    assert freqtrade.protections.global_stop() != only_per_pair
    if not only_per_pair:
        assert log_has_re(message, caplog)
    else:
        assert not log_has_re(message, caplog)
    caplog.clear()
    generate_mock_trade(pair, fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=150, min_ago_close=25, profit_rate=0.9, is_short=not is_short)
    freqtrade.protections.stop_per_pair(pair)
    assert freqtrade.protections.global_stop() != only_per_pair
    assert PairLocks.is_pair_locked(pair, side=check_side) != (only_per_side and only_per_pair)
    assert PairLocks.is_global_lock(side=check_side) != only_per_pair
    if only_per_side:
        assert not PairLocks.is_pair_locked(pair, side='*')
        assert not PairLocks.is_global_lock(side='*')
    caplog.clear()
    generate_mock_trade(pair, fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=180, min_ago_close=31, profit_rate=0.9, is_short=is_short)
    freqtrade.protections.stop_per_pair(pair)
    assert freqtrade.protections.global_stop() != only_per_pair
    assert PairLocks.is_pair_locked(pair, side=check_side)
    assert PairLocks.is_global_lock(side=check_side) != only_per_pair
    if only_per_side:
        assert not PairLocks.is_pair_locked(pair, side='*')
        assert not PairLocks.is_global_lock(side='*')

@pytest.mark.usefixtures('init_persistence')
def test_CooldownPeriod(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any) -> None:
    default_conf['_strategy_protections'] = [{'method': 'CooldownPeriod', 'stop_duration': 60}]
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf)
    message: str = 'Trading stopped due to .*'
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    assert not log_has_re(message, caplog)
    caplog.clear()
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=200, min_ago_close=30)
    assert not freqtrade.protections.global_stop()
    assert freqtrade.protections.stop_per_pair('XRP/BTC')
    assert PairLocks.is_pair_locked('XRP/BTC')
    assert not PairLocks.is_global_lock()
    generate_mock_trade('ETH/BTC', fee.return_value, False, exit_reason=ExitType.ROI.value,
                        min_ago_open=205, min_ago_close=35)
    assert not freqtrade.protections.global_stop()
    assert not PairLocks.is_pair_locked('ETH/BTC')
    assert freqtrade.protections.stop_per_pair('ETH/BTC')
    assert PairLocks.is_pair_locked('ETH/BTC')
    assert not PairLocks.is_global_lock()

@pytest.mark.usefixtures('init_persistence')
def test_CooldownPeriod_unlock_at(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any, time_machine: Any) -> None:
    default_conf['_strategy_protections'] = [{'method': 'CooldownPeriod', 'unlock_at': '05:00'}]
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf)
    message: str = 'Trading stopped due to .*'
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    assert not log_has_re(message, caplog)
    caplog.clear()
    start_dt: datetime = datetime(2024, 5, 2, 0, 30, 0, tzinfo=timezone.utc)
    time_machine.move_to(start_dt, tick=False)
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=20, min_ago_close=10)
    assert not freqtrade.protections.global_stop()
    assert freqtrade.protections.stop_per_pair('XRP/BTC')
    assert PairLocks.is_pair_locked('XRP/BTC')
    assert not PairLocks.is_global_lock()
    time_machine.move_to(start_dt + timedelta(hours=4), tick=False)
    assert PairLocks.is_pair_locked('XRP/BTC')
    assert not PairLocks.is_global_lock()
    time_machine.move_to(start_dt + timedelta(hours=5), tick=False)
    assert not PairLocks.is_pair_locked('XRP/BTC')
    assert not PairLocks.is_global_lock()
    start_dt = datetime(2024, 5, 2, 22, 0, 0, tzinfo=timezone.utc)
    time_machine.move_to(start_dt, tick=False)
    generate_mock_trade('ETH/BTC', fee.return_value, False, exit_reason=ExitType.ROI.value,
                        min_ago_open=20, min_ago_close=10)
    assert not freqtrade.protections.global_stop()
    assert not PairLocks.is_pair_locked('ETH/BTC')
    assert freqtrade.protections.stop_per_pair('ETH/BTC')
    assert PairLocks.is_pair_locked('ETH/BTC')
    assert not PairLocks.is_global_lock()
    time_machine.move_to(start_dt + timedelta(hours=1), tick=False)
    assert PairLocks.is_pair_locked('ETH/BTC')
    assert not PairLocks.is_global_lock()
    time_machine.move_to(start_dt + timedelta(hours=6, minutes=59), tick=False)
    assert PairLocks.is_pair_locked('ETH/BTC')
    assert not PairLocks.is_global_lock()
    time_machine.move_to(start_dt + timedelta(hours=7, minutes=1), tick=False)
    assert PairLocks.is_pair_locked('ETH/BTC')
    assert not PairLocks.is_global_lock()
    time_machine.move_to(start_dt + timedelta(hours=7, minutes=5), tick=False)
    assert not PairLocks.is_pair_locked('ETH/BTC')
    assert not PairLocks.is_global_lock()

@pytest.mark.parametrize('only_per_side', [False, True])
@pytest.mark.usefixtures('init_persistence')
def test_LowProfitPairs(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any, only_per_side: bool) -> None:
    default_conf['_strategy_protections'] = [{'method': 'LowProfitPairs', 'lookback_period': 400, 'stop_duration': 60, 'trade_limit': 2, 'required_profit': 0.0, 'only_per_side': only_per_side}]
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf)
    message: str = 'Trading stopped due to .*'
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    assert not log_has_re(message, caplog)
    caplog.clear()
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=800, min_ago_close=450, profit_rate=0.9)
    Trade.commit()
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    assert not PairLocks.is_pair_locked('XRP/BTC')
    assert not PairLocks.is_global_lock()
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=200, min_ago_close=120, profit_rate=0.9)
    Trade.commit()
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    assert not PairLocks.is_pair_locked('XRP/BTC')
    assert not PairLocks.is_global_lock()
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.ROI.value,
                        min_ago_open=20, min_ago_close=10, profit_rate=1.15, is_short=True)
    Trade.commit()
    assert freqtrade.protections.stop_per_pair('XRP/BTC') != only_per_side
    assert not PairLocks.is_pair_locked('XRP/BTC', side='*')
    assert PairLocks.is_pair_locked('XRP/BTC', side='long') == only_per_side
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=110, min_ago_close=21, profit_rate=0.8)
    Trade.commit()
    assert freqtrade.protections.global_stop() != only_per_side
    assert freqtrade.protections.stop_per_pair('XRP/BTC') != only_per_side
    assert PairLocks.is_pair_locked('XRP/BTC', side='long')
    assert PairLocks.is_pair_locked('XRP/BTC', side='*') != only_per_side
    assert not PairLocks.is_global_lock()
    Trade.commit()

@pytest.mark.usefixtures('init_persistence')
def test_MaxDrawdown(mocker: Any, default_conf: Dict[str, Any], fee: Any, caplog: Any) -> None:
    default_conf['_strategy_protections'] = [{'method': 'MaxDrawdown', 'lookback_period': 1000, 'stop_duration': 60, 'trade_limit': 3, 'max_allowed_drawdown': 0.15}]
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf)
    message: str = 'Trading stopped due to Max.*'
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    caplog.clear()
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=1000, min_ago_close=900, profit_rate=1.1)
    generate_mock_trade('ETH/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=1000, min_ago_close=900, profit_rate=1.1)
    generate_mock_trade('NEO/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=1000, min_ago_close=900, profit_rate=1.1)
    Trade.commit()
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=500, min_ago_close=400, profit_rate=0.9)
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    assert not PairLocks.is_pair_locked('XRP/BTC')
    assert not PairLocks.is_global_lock()
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.STOP_LOSS.value,
                        min_ago_open=1200, min_ago_close=1100, profit_rate=0.5)
    Trade.commit()
    assert not freqtrade.protections.global_stop()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    assert not PairLocks.is_pair_locked('XRP/BTC')
    assert not PairLocks.is_global_lock()
    assert not log_has_re(message, caplog)
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.ROI.value,
                        min_ago_open=320, min_ago_close=410, profit_rate=1.5)
    Trade.commit()
    assert not freqtrade.protections.global_stop()
    assert not PairLocks.is_global_lock()
    caplog.clear()
    generate_mock_trade('XRP/BTC', fee.return_value, False, exit_reason=ExitType.ROI.value,
                        min_ago_open=20, min_ago_close=10, profit_rate=0.8)
    Trade.commit()
    assert not freqtrade.protections.stop_per_pair('XRP/BTC')
    assert not PairLocks.is_pair_locked('XRP/BTC')
    assert freqtrade.protections.global_stop()
    assert PairLocks.is_global_lock()
    assert log_has_re(message, caplog)

@pytest.mark.parametrize('protectionconf,desc_expected,exception_expected', [
    (
        {'method': 'StoplossGuard', 'lookback_period': 60, 'trade_limit': 2, 'stop_duration': 60},
        "[{'StoplossGuard': 'StoplossGuard - Frequent Stoploss Guard, 2 stoplosses with profit < 0.00% within 60 minutes.'}]",
        None
    ),
    (
        {'method': 'CooldownPeriod', 'stop_duration': 60},
        "[{'CooldownPeriod': 'CooldownPeriod - Cooldown period for 60 minutes.'}]",
        None
    ),
    (
        {'method': 'LowProfitPairs', 'lookback_period': 60, 'stop_duration': 60},
        "[{'LowProfitPairs': 'LowProfitPairs - Low Profit Protection, locks pairs with profit < 0.0 within 60 minutes.'}]",
        None
    ),
    (
        {'method': 'MaxDrawdown', 'lookback_period': 60, 'stop_duration': 60},
        "[{'MaxDrawdown': 'MaxDrawdown - Max drawdown protection, stop trading if drawdown is > 0.0 within 60 minutes.'}]",
        None
    ),
    (
        {'method': 'StoplossGuard', 'lookback_period_candles': 12, 'trade_limit': 2, 'required_profit': -0.05, 'stop_duration': 60},
        "[{'StoplossGuard': 'StoplossGuard - Frequent Stoploss Guard, 2 stoplosses with profit < -5.00% within 12 candles.'}]",
        None
    ),
    (
        {'method': 'CooldownPeriod', 'stop_duration_candles': 5},
        "[{'CooldownPeriod': 'CooldownPeriod - Cooldown period for 5 candles.'}]",
        None
    ),
    (
        {'method': 'LowProfitPairs', 'lookback_period_candles': 11, 'stop_duration': 60},
        "[{'LowProfitPairs': 'LowProfitPairs - Low Profit Protection, locks pairs with profit < 0.0 within 11 candles.'}]",
        None
    ),
    (
        {'method': 'MaxDrawdown', 'lookback_period_candles': 20, 'stop_duration': 60},
        "[{'MaxDrawdown': 'MaxDrawdown - Max drawdown protection, stop trading if drawdown is > 0.0 within 20 candles.'}]",
        None
    ),
    (
        {'method': 'CooldownPeriod', 'unlock_at': '01:00'},
        "[{'CooldownPeriod': 'CooldownPeriod - Cooldown period until 01:00.'}]",
        None
    ),
    (
        {'method': 'StoplossGuard', 'lookback_period_candles': 12, 'trade_limit': 2, 'required_profit': -0.05, 'unlock_at': '01:00'},
        "[{'StoplossGuard': 'StoplossGuard - Frequent Stoploss Guard, 2 stoplosses with profit < -5.00% within 12 candles.'}]",
        None
    ),
    (
        {'method': 'LowProfitPairs', 'lookback_period_candles': 11, 'unlock_at': '03:00'},
        "[{'LowProfitPairs': 'LowProfitPairs - Low Profit Protection, locks pairs with profit < 0.0 within 11 candles.'}]",
        None
    ),
    (
        {'method': 'MaxDrawdown', 'lookback_period_candles': 20, 'unlock_at': '04:00'},
        "[{'MaxDrawdown': 'MaxDrawdown - Max drawdown protection, stop trading if drawdown is > 0.0 within 20 candles.'}]",
        None
    )
])
def test_protection_manager_desc(mocker: Any, default_conf: Dict[str, Any],
                                 protectionconf: Dict[str, Any],
                                 desc_expected: str,
                                 exception_expected: Optional[str]) -> None:
    default_conf['_strategy_protections'] = [protectionconf]
    freqtrade: Any = get_patched_freqtradebot(mocker, default_conf)
    short_desc: str = str(freqtrade.protections.short_desc())
    assert short_desc == desc_expected
