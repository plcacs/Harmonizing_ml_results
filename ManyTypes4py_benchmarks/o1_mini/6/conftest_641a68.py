import json
import logging
import platform
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from xdist.scheduler.loadscope import LoadScopeScheduling
from freqtrade import constants
from freqtrade.commands import Arguments
from freqtrade.data.converter import ohlcv_to_dataframe, trades_list_to_df
from freqtrade.edge import PairInfo
from freqtrade.enums import CandleType, MarginMode, RunMode, SignalDirection, TradingMode
from freqtrade.exchange import Exchange, timeframe_to_minutes, timeframe_to_seconds
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import LocalTrade, Order, Trade, init_db
from freqtrade.resolvers import ExchangeResolver
from freqtrade.util import dt_now, dt_ts
from freqtrade.worker import Worker
from tests.conftest_trades import (
    leverage_trade,
    mock_trade_1,
    mock_trade_2,
    mock_trade_3,
    mock_trade_4,
    mock_trade_5,
    mock_trade_6,
    short_trade,
)
from tests.conftest_trades_usdt import (
    mock_trade_usdt_1,
    mock_trade_usdt_2,
    mock_trade_usdt_3,
    mock_trade_usdt_4,
    mock_trade_usdt_5,
    mock_trade_usdt_6,
    mock_trade_usdt_7,
)

logging.getLogger('').setLevel(logging.INFO)
np.seterr(all='raise')
CURRENT_TEST_STRATEGY: str = 'StrategyTestV3'
TRADE_SIDES: Tuple[str, ...] = ('long', 'short')
EXMS: str = 'freqtrade.exchange.exchange.Exchange'


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        '--longrun',
        action='store_true',
        dest='longrun',
        default=False,
        help='Enable long-run tests (ccxt compat)',
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        'markers', 'longrun: mark test that is running slowly and should not be run regularly'
    )
    if not config.option.longrun:
        config.option.markexpr = 'not longrun'


class FixtureScheduler(LoadScopeScheduling):

    def _split_scope(self, nodeid: str) -> str:
        if 'exchange_online' in nodeid:
            try:
                exchange_id = nodeid.split('[')[1].split('-')[0].rstrip(']')
                return exchange_id
            except Exception as e:
                print(e)
                pass
        return nodeid


def pytest_xdist_make_scheduler(config: Any, log: Any) -> LoadScopeScheduling:
    return FixtureScheduler(config, log)


def log_has(line: str, logs: Any) -> bool:
    """Check if line is found on some caplog's message."""
    return any((line == message for message in logs.messages))


def log_has_when(line: str, logs: Any, when: str) -> bool:
    """Check if line is found in caplog's messages during a specified stage"""
    return any((line == message.message for message in logs.get_records(when)))


def log_has_re(line: str, logs: Any) -> bool:
    """Check if line matches some caplog's message."""
    return any((re.match(line, message) for message in logs.messages))


def num_log_has(line: str, logs: Any) -> int:
    """Check how many times line is found in caplog's messages."""
    return sum((line == message for message in logs.messages))


def num_log_has_re(line: str, logs: Any) -> int:
    """Check how many times line matches caplog's messages."""
    return sum((bool(re.match(line, message)) for message in logs.messages))


def get_args(args: List[str]) -> Any:
    return Arguments(args).get_parsed_arg()


def generate_trades_history(
    n_rows: int,
    start_date: Optional[datetime] = None,
    days: int = 5
) -> pd.DataFrame:
    np.random.seed(42)
    if not start_date:
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = start_date + timedelta(days=days)
    _start_timestamp: float = start_date.timestamp()
    _end_timestamp: float = pd.to_datetime(end_date).timestamp()
    random_timestamps_in_seconds: np.ndarray = np.random.uniform(
        _start_timestamp, _end_timestamp, n_rows
    )
    timestamp: pd.DatetimeIndex = pd.to_datetime(random_timestamps_in_seconds, unit='s')
    trade_id: List[str] = [
        f'a{np.random.randint(1000000, 10000000 - 1)}cd{np.random.randint(100, 999)}' for _ in range(n_rows)
    ]
    side: np.ndarray = np.random.choice(['buy', 'sell'], n_rows)
    initial_price: float = 0.019626
    price_changes: np.ndarray = np.random.normal(0, initial_price * 0.05, n_rows)
    price: np.ndarray = np.cumsum(np.concatenate(([initial_price], price_changes)))[:n_rows]
    amount: np.ndarray = np.random.uniform(0.011, 20, n_rows)
    cost: np.ndarray = price * amount
    df: pd.DataFrame = pd.DataFrame({
        'timestamp': timestamp,
        'id': trade_id,
        'type': None,
        'side': side,
        'price': price,
        'amount': amount,
        'cost': cost
    })
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    assert list(df.columns) == constants.DEFAULT_TRADES_COLUMNS + ['date']
    return df


def generate_test_data(
    timeframe: str,
    size: int,
    start: str = '2020-07-05',
    random_seed: int = 42
) -> pd.DataFrame:
    np.random.seed(random_seed)
    base: np.ndarray = np.random.normal(20, 2, size=size)
    if timeframe == '1y':
        date: pd.DatetimeIndex = pd.date_range(start, periods=size, freq='1YS', tz='UTC')
    elif timeframe == '1M':
        date = pd.date_range(start, periods=size, freq='1MS', tz='UTC')
    elif timeframe == '3M':
        date = pd.date_range(start, periods=size, freq='3MS', tz='UTC')
    elif timeframe in ('1w', '7d'):
        date = pd.date_range(start, periods=size, freq='1W-MON', tz='UTC')
    else:
        tf_mins: int = timeframe_to_minutes(timeframe)
        if tf_mins >= 1:
            date = pd.date_range(start, periods=size, freq=f'{tf_mins}min', tz='UTC')
        else:
            tf_secs: int = timeframe_to_seconds(timeframe)
            date = pd.date_range(start, periods=size, freq=f'{tf_secs}s', tz='UTC')
    df: pd.DataFrame = pd.DataFrame({
        'date': date,
        'open': base,
        'high': base + np.random.normal(2, 1, size=size),
        'low': base - np.random.normal(2, 1, size=size),
        'close': base + np.random.normal(0, 1, size=size),
        'volume': np.random.normal(200, size=size)
    })
    df = df.dropna()
    return df


def generate_test_data_raw(
    timeframe: str,
    size: int,
    start: str = '2020-07-05',
    random_seed: int = 42
) -> List[List[Union[float, int]]]:
    """Generates data in the ohlcv format used by ccxt"""
    df: pd.DataFrame = generate_test_data(timeframe, size, start, random_seed)
    df['date'] = df.loc[:, 'date'].astype(np.int64) // 1000 // 1000
    return [list(x) for x in zip(*(df[x].values.tolist() for x in df.columns), strict=False)]


def get_mock_coro(
    return_value: Any = None,
    side_effect: Optional[Union[Exception, List[Any], Callable[..., Any]]] = None
) -> Mock:
    
    async def mock_coro(*args: Any, **kwargs: Any) -> Any:
        if side_effect:
            if isinstance(side_effect, list):
                effect = side_effect.pop(0)
            else:
                effect = side_effect
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(*args, **kwargs)
            return effect
        else:
            return return_value
    return Mock(wraps=mock_coro)


def patched_configuration_load_config_file(mocker: Any, config: Dict[str, Any]) -> None:
    mocker.patch(
        'freqtrade.configuration.load_config.load_config_file',
        lambda *args: config
    )


def patch_exchange(
    mocker: Any,
    api_mock: Optional[Any] = None,
    exchange: str = 'binance',
    mock_markets: Union[bool, Dict[str, Any]] = True,
    mock_supported_modes: bool = True
) -> None:
    mocker.patch(f'{EXMS}.validate_config', MagicMock())
    mocker.patch(f'{EXMS}.validate_timeframes', MagicMock())
    mocker.patch(f'{EXMS}.id', PropertyMock(return_value=exchange))
    mocker.patch(f'{EXMS}.name', PropertyMock(return_value=exchange.title()))
    mocker.patch(f'{EXMS}.precisionMode', PropertyMock(return_value=2))
    mocker.patch(f'{EXMS}.precision_mode_price', PropertyMock(return_value=2))
    mocker.patch('freqtrade.exchange.bybit.Bybit.cache_leverage_tiers')
    if mock_markets:
        mocker.patch(f'{EXMS}._load_async_markets', return_value={})
        if isinstance(mock_markets, bool):
            mock_markets_data: Dict[str, Any] = get_markets()
        else:
            mock_markets_data = mock_markets
        mocker.patch(f'{EXMS}.markets', PropertyMock(return_value=mock_markets_data))
    if mock_supported_modes:
        mocker.patch(
            f'freqtrade.exchange.{exchange}.{exchange.capitalize()}._supported_trading_mode_margin_pairs',
            PropertyMock(return_value=[
                (TradingMode.MARGIN, MarginMode.CROSS),
                (TradingMode.MARGIN, MarginMode.ISOLATED),
                (TradingMode.FUTURES, MarginMode.CROSS),
                (TradingMode.FUTURES, MarginMode.ISOLATED)
            ])
        )
    if api_mock:
        mocker.patch(f'{EXMS}._init_ccxt', return_value=api_mock)
    else:
        mocker.patch(f'{EXMS}.get_fee', return_value=0.0025)
        mocker.patch(f'{EXMS}._init_ccxt', MagicMock())
        mocker.patch(f'{EXMS}.timeframes', PropertyMock(return_value=['5m', '15m', '1h', '1d']))


def get_patched_exchange(
    mocker: Any,
    config: Dict[str, Any],
    api_mock: Optional[Any] = None,
    exchange: str = 'binance',
    mock_markets: Union[bool, Dict[str, Any]] = True,
    mock_supported_modes: bool = True
) -> Exchange:
    patch_exchange(mocker, api_mock, exchange, mock_markets, mock_supported_modes)
    config['exchange']['name'] = exchange
    try:
        exchange_instance: Exchange = ExchangeResolver.load_exchange(config, load_leverage_tiers=True)
    except ImportError:
        exchange_instance = Exchange(config)
    return exchange_instance


def patch_wallet(mocker: Any, free: float = 999.9) -> None:
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=free))


def patch_whitelist(mocker: Any, conf: Dict[str, Any]) -> None:
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot._refresh_active_whitelist',
        MagicMock(return_value=conf['exchange']['pair_whitelist'])
    )


def patch_edge(mocker: Any) -> None:
    mocker.patch(
        'freqtrade.edge.Edge._cached_pairs',
        mocker.PropertyMock(return_value={
            'NEO/BTC': PairInfo(-0.2, 0.66, 3.71, 0.5, 1.71, 10, 25),
            'LTC/BTC': PairInfo(-0.21, 0.66, 3.71, 0.5, 1.71, 11, 20)
        })
    )
    mocker.patch('freqtrade.edge.Edge.calculate', MagicMock(return_value=True))


def patch_freqtradebot(mocker: Any, config: Dict[str, Any]) -> None:
    """
    This function patch _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: None
    """
    mocker.patch('freqtrade.freqtradebot.RPCManager', MagicMock())
    patch_exchange(mocker)
    mocker.patch('freqtrade.freqtradebot.RPCManager._init', MagicMock())
    mocker.patch('freqtrade.freqtradebot.RPCManager.send_msg', MagicMock())
    patch_whitelist(mocker, config)
    mocker.patch('freqtrade.freqtradebot.ExternalMessageConsumer', MagicMock())
    mocker.patch('freqtrade.configuration.config_validation._validate_consumers', MagicMock())


def get_patched_freqtradebot(mocker: Any, config: Dict[str, Any]) -> FreqtradeBot:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: FreqtradeBot
    """
    patch_freqtradebot(mocker, config)
    return FreqtradeBot(config)


def get_patched_worker(mocker: Any, config: Dict[str, Any]) -> Worker:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: Worker
    """
    patch_freqtradebot(mocker, config)
    return Worker(args=None, config=config)


def patch_get_signal(
    freqtrade: Any,
    enter_long: bool = True,
    exit_long: bool = False,
    enter_short: bool = False,
    exit_short: bool = False,
    enter_tag: Optional[Any] = None,
    exit_tag: Optional[Any] = None
) -> None:
    """
    :param mocker: mocker to patch IStrategy class
    :return: None
    """

    def patched_get_entry_signal(*args: Any, **kwargs: Any) -> Tuple[Optional[SignalDirection], Optional[Any]]:
        direction: Optional[SignalDirection] = None
        if enter_long and not any([exit_long, enter_short]):
            direction = SignalDirection.LONG
        if enter_short and not any([exit_short, enter_long]):
            direction = SignalDirection.SHORT
        return (direction, enter_tag)

    freqtrade.strategy.get_entry_signal = patched_get_entry_signal

    def patched_get_exit_signal(
        pair: str,
        timeframe: str,
        dataframe: pd.DataFrame,
        is_short: bool
    ) -> Tuple[bool, bool, Optional[Any]]:
        if is_short:
            return (enter_short, exit_short, exit_tag)
        else:
            return (enter_long, exit_long, exit_tag)

    freqtrade.strategy.get_exit_signal = patched_get_exit_signal
    freqtrade.exchange.refresh_latest_ohlcv = lambda p: None


def create_mock_trades(
    fee: Any,
    is_short: Optional[bool] = False,
    use_db: bool = True
) -> None:
    """
    Create some fake trades ...
    :param is_short: Optional bool, None creates a mix of long and short trades.
    """
    def add_trade(trade: Trade) -> None:
        if use_db:
            Trade.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)

    is_short1: Optional[bool] = is_short if is_short is not None else True
    is_short2: Optional[bool] = is_short if is_short is not None else False
    trade1: Trade = mock_trade_1(fee, is_short1)
    add_trade(trade1)
    trade2: Trade = mock_trade_2(fee, is_short1)
    add_trade(trade2)
    trade3: Trade = mock_trade_3(fee, is_short2)
    add_trade(trade3)
    trade4: Trade = mock_trade_4(fee, is_short2)
    add_trade(trade4)
    trade5: Trade = mock_trade_5(fee, is_short2)
    add_trade(trade5)
    trade6: Trade = mock_trade_6(fee, is_short1)
    add_trade(trade6)
    if use_db:
        Trade.commit()


def create_mock_trades_with_leverage(fee: Any, use_db: bool = True) -> None:
    """
    Create some fake trades ...
    """
    if use_db:
        Trade.session.rollback()

    def add_trade(trade: Trade) -> None:
        if use_db:
            Trade.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)

    trade1: Trade = mock_trade_1(fee, False)
    add_trade(trade1)
    trade2: Trade = mock_trade_2(fee, False)
    add_trade(trade2)
    trade3: Trade = mock_trade_3(fee, False)
    add_trade(trade3)
    trade4: Trade = mock_trade_4(fee, False)
    add_trade(trade4)
    trade5: Trade = mock_trade_5(fee, False)
    add_trade(trade5)
    trade6: Trade = mock_trade_6(fee, False)
    add_trade(trade6)
    trade7: Trade = short_trade(fee)
    add_trade(trade7)
    trade8: Trade = leverage_trade(fee)
    add_trade(trade8)
    if use_db:
        Trade.session.flush()


def create_mock_trades_usdt(
    fee: Any,
    is_short: Optional[bool] = False,
    use_db: bool = True
) -> None:
    """
    Create some fake trades ...
    """
    def add_trade(trade: Trade) -> None:
        if use_db:
            Trade.session.add(trade)
        else:
            LocalTrade.add_bt_trade(trade)

    is_short1: Optional[bool] = is_short if is_short is not None else True
    is_short2: Optional[bool] = is_short if is_short is not None else False
    trade1: Trade = mock_trade_usdt_1(fee, is_short1)
    add_trade(trade1)
    trade2: Trade = mock_trade_usdt_2(fee, is_short1)
    add_trade(trade2)
    trade3: Trade = mock_trade_usdt_3(fee, is_short1)
    add_trade(trade3)
    trade4: Trade = mock_trade_usdt_4(fee, is_short2)
    add_trade(trade4)
    trade5: Trade = mock_trade_usdt_5(fee, is_short2)
    add_trade(trade5)
    trade6: Trade = mock_trade_usdt_6(fee, is_short1)
    add_trade(trade6)
    trade7: Trade = mock_trade_usdt_7(fee, is_short1)
    add_trade(trade7)
    if use_db:
        Trade.commit()


@pytest.fixture(autouse=True)
def patch_gc(mocker: Any) -> None:
    mocker.patch('freqtrade.main.gc_set_threshold')


def is_arm() -> bool:
    machine: str = platform.machine()
    return 'arm' in machine or 'aarch64' in machine


def is_mac() -> bool:
    system: str = platform.system()
    return 'Darwin' in system


@pytest.fixture(autouse=True)
def patch_torch_initlogs(mocker: Any) -> None:
    if is_mac():
        import sys
        import types
        module_name: str = 'torch'
        mocked_module = types.ModuleType(module_name)
        sys.modules[module_name] = mocked_module
    else:
        mocker.patch('torch._logging._init_logs')


@pytest.fixture(autouse=True)
def user_dir(mocker: Any, tmp_path: Path) -> Path:
    user_dir: Path = tmp_path / 'user_data'
    mocker.patch(
        'freqtrade.configuration.configuration.create_userdata_dir',
        return_value=user_dir
    )
    return user_dir


@pytest.fixture(autouse=True)
def patch_coingecko(mocker: Any) -> None:
    """
    Mocker to coingecko to speed up tests
    :param mocker: mocker to patch coingecko class
    :return: None
    """
    tickermock: MagicMock = MagicMock(
        return_value={'bitcoin': {'usd': 12345.0}, 'ethereum': {'usd': 12345.0}}
    )
    listmock: MagicMock = MagicMock(
        return_value=[
            {'id': 'bitcoin', 'name': 'Bitcoin', 'symbol': 'btc', 'website_slug': 'bitcoin'},
            {'id': 'ethereum', 'name': 'Ethereum', 'symbol': 'eth', 'website_slug': 'ethereum'}
        ]
    )
    mocker.patch.multiple(
        'freqtrade.rpc.fiat_convert.FtCoinGeckoApi',
        get_price=tickermock,
        get_coins_list=listmock
    )


@pytest.fixture(scope='function')
def init_persistence(default_conf: Dict[str, Any]) -> None:
    init_db(default_conf['db_url'])


@pytest.fixture(scope='function')
def default_conf(testdatadir: Path) -> Dict[str, Any]:
    return get_default_conf(testdatadir)


@pytest.fixture(scope='function')
def default_conf_usdt(testdatadir: Path) -> Dict[str, Any]:
    return get_default_conf_usdt(testdatadir)


def get_default_conf(testdatadir: Path) -> Dict[str, Any]:
    """Returns validated configuration suitable for most tests"""
    configuration: Dict[str, Any] = {
        'max_open_trades': 1,
        'stake_currency': 'BTC',
        'stake_amount': 0.001,
        'fiat_display_currency': 'USD',
        'timeframe': '5m',
        'dry_run': True,
        'cancel_open_orders_on_exit': False,
        'minimal_roi': {'40': 0.0, '30': 0.01, '20': 0.02, '0': 0.04},
        'dry_run_wallet': 1000,
        'stoploss': -0.1,
        'unfilledtimeout': {'entry': 10, 'exit': 30},
        'entry_pricing': {
            'price_last_balance': 0.0,
            'use_order_book': False,
            'order_book_top': 1,
            'check_depth_of_market': {
                'enabled': False,
                'bids_to_ask_delta': 1
            }
        },
        'exit_pricing': {
            'use_order_book': False,
            'order_book_top': 1
        },
        'exchange': {
            'name': 'binance',
            'key': 'key',
            'enable_ws': False,
            'secret': 'secret',
            'pair_whitelist': ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC'],
            'pair_blacklist': ['DOGE/BTC', 'HOT/BTC']
        },
        'pairlists': [{'method': 'StaticPairList'}],
        'telegram': {
            'enabled': False,
            'token': 'token',
            'chat_id': '1235',
            'notification_settings': {}
        },
        'datadir': Path(testdatadir),
        'initial_state': 'running',
        'db_url': 'sqlite://',
        'user_data_dir': Path('user_data'),
        'verbosity': 3,
        'strategy_path': str(Path(__file__).parent / 'strategy' / 'strats'),
        'strategy': CURRENT_TEST_STRATEGY,
        'disableparamexport': True,
        'internals': {},
        'export': 'none',
        'dataformat_ohlcv': 'feather',
        'dataformat_trades': 'feather',
        'runmode': 'dry_run',
        'trading_mode': 'spot',
        'margin_mode': '',
        'candle_type_def': CandleType.SPOT
    }
    return configuration


def get_default_conf_usdt(testdatadir: Path) -> Dict[str, Any]:
    configuration: Dict[str, Any] = get_default_conf(testdatadir)
    configuration.update({
        'stake_amount': 60.0,
        'stake_currency': 'USDT',
        'exchange': {
            'name': 'binance',
            'enabled': True,
            'key': 'key',
            'enable_ws': False,
            'secret': 'secret',
            'pair_whitelist': ['ETH/USDT', 'LTC/USDT', 'XRP/USDT', 'NEO/USDT', 'TKN/USDT'],
            'pair_blacklist': ['DOGE/USDT', 'HOT/USDT']
        }
    })
    return configuration


@pytest.fixture
def fee() -> MagicMock:
    return MagicMock(return_value=0.0025)


@pytest.fixture
def ticker() -> MagicMock:
    return MagicMock(return_value={'bid': 1.098e-05, 'ask': 1.099e-05, 'last': 1.098e-05})


@pytest.fixture
def ticker_sell_up() -> MagicMock:
    return MagicMock(return_value={'bid': 1.172e-05, 'ask': 1.173e-05, 'last': 1.172e-05})


@pytest.fixture
def ticker_sell_down() -> MagicMock:
    return MagicMock(return_value={'bid': 1.044e-05, 'ask': 1.043e-05, 'last': 1.044e-05})


@pytest.fixture
def ticker_usdt() -> MagicMock:
    return MagicMock(return_value={'bid': 2.0, 'ask': 2.02, 'last': 2.0})


@pytest.fixture
def ticker_usdt_sell_up() -> MagicMock:
    return MagicMock(return_value={'bid': 2.2, 'ask': 2.3, 'last': 2.2})


@pytest.fixture
def ticker_usdt_sell_down() -> MagicMock:
    return MagicMock(return_value={'bid': 2.01, 'ask': 2.0, 'last': 2.01})


@pytest.fixture
def markets() -> Dict[str, Any]:
    return get_markets()


def get_markets() -> Dict[str, Any]:
    return {
        'ETH/BTC': {
            'id': 'ethbtc',
            'symbol': 'ETH/BTC',
            'base': 'ETH',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {'price': 8, 'amount': 8, 'cost': 8},
            'lot': 1e-08,
            'contractSize': None,
            'limits': {
                'amount': {'min': 0.01, 'max': 100000000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000},
                'leverage': {'min': 1.0, 'max': 2.0}
            }
        },
        'TKN/BTC': {
            'id': 'tknbtc',
            'symbol': 'TKN/BTC',
            'base': 'TKN',
            'quote': 'BTC',
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {'price': 8, 'amount': 8, 'cost': 8},
            'lot': 1e-08,
            'contractSize': None,
            'limits': {
                'amount': {'min': 0.01, 'max': 100000000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000},
                'leverage': {'min': 1.0, 'max': 5.0}
            }
        },
        'BLK/BTC': {
            'id': 'blkbtc',
            'symbol': 'BLK/BTC',
            'base': 'BLK',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {'price': 8, 'amount': 8, 'cost': 8},
            'lot': 1e-08,
            'contractSize': None,
            'limits': {
                'amount': {'min': 0.01, 'max': 1000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000},
                'leverage': {'min': 1.0, 'max': 3.0}
            }
        },
        'LTC/BTC': {
            'id': 'ltcbtc',
            'symbol': 'LTC/BTC',
            'base': 'LTC',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {'price': 8, 'amount': 8, 'cost': 8},
            'lot': 1e-08,
            'contractSize': None,
            'limits': {
                'amount': {'min': 0.01, 'max': 100000000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000},
                'leverage': {'min': None, 'max': None}
            },
            'info': {}
        },
        'XRP/BTC': {
            'id': 'xrpbtc',
            'symbol': 'XRP/BTC',
            'base': 'XRP',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {'price': 8, 'amount': 8, 'cost': 8},
            'lot': 1e-08,
            'contractSize': None,
            'limits': {
                'amount': {'min': 0.01, 'max': 100000000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000},
                'leverage': {'min': None, 'max': None}
            },
            'info': {}
        },
        'NEO/BTC': {
            'id': 'neobtc',
            'symbol': 'NEO/BTC',
            'base': 'NEO',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'precision': {'price': 8, 'amount': 8, 'cost': 8},
            'lot': 1e-08,
            'contractSize': None,
            'limits': {
                'amount': {'min': 0.01, 'max': 100000000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000},
                'leverage': {'min': None, 'max': None}
            },
            'info': {}
        },
        'BTT/BTC': {
            'id': 'BTTBTC',
            'symbol': 'BTT/BTC',
            'base': 'BTT',
            'quote': 'BTC',
            'active': False,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'contractSize': None,
            'precision': {
                'base': 8,
                'quote': 8,
                'amount': 0,
                'price': 8
            },
            'limits': {
                'amount': {'min': 1.0, 'max': 90000000.0},
                'price': {'min': None, 'max': None},
                'cost': {'min': 0.0001, 'max': None},
                'leverage': {'min': None, 'max': None}
            },
            'info': {}
        },
        'ETC/BTC': {
            'id': 'ETCBTC',
            'symbol': 'ETC/BTC',
            'base': 'ETC',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'contractSize': None,
            'precision': {'base': 8, 'quote': 8, 'amount': 2, 'price': 7},
            'limits': {
                'amount': {'min': 0.01, 'max': 90000000.0},
                'price': {'min': 1e-07, 'max': 1000.0},
                'cost': {'min': 0.0001, 'max': 9000000.0},
                'leverage': {'min': None, 'max': None}
            },
            'info': {}
        },
        'ETH/USDT': {
            'id': 'USDT-ETH',
            'symbol': 'ETH/USDT',
            'base': 'ETH',
            'quote': 'USDT',
            'settle': None,
            'baseId': 'ETH',
            'quoteId': 'USDT',
            'settleId': None,
            'type': 'spot',
            'spot': True,
            'margin': True,
            'swap': True,
            'future': True,
            'option': False,
            'active': True,
            'contract': None,
            'linear': None,
            'inverse': False,
            'taker': 0.0006,
            'maker': 0.0002,
            'contractSize': None,
            'expiry': None,
            'expiryDateTime': None,
            'strike': None,
            'optionType': None,
            'precision': {'amount': 8, 'price': 8},
            'limits': {
                'leverage': {'min': 1, 'max': 100},
                'amount': {'min': 0.02214286, 'max': None},
                'price': {'min': 1e-08, 'max': None},
                'cost': {'min': None, 'max': None}
            },
            'info': {'maintenance_rate': '0.005'}
        },
        'BTC/USDT': {
            'id': 'USDT-BTC',
            'symbol': 'BTC/USDT',
            'base': 'BTC',
            'quote': 'USDT',
            'settle': None,
            'baseId': 'BTC',
            'quoteId': 'USDT',
            'settleId': None,
            'type': 'spot',
            'spot': True,
            'margin': True,
            'swap': False,
            'future': False,
            'option': False,
            'active': True,
            'contract': None,
            'linear': None,
            'inverse': False,
            'taker': 0.0006,
            'maker': 0.0002,
            'contractSize': None,
            'expiry': None,
            'expiryDateTime': None,
            'strike': None,
            'optionType': None,
            'precision': {'amount': 4, 'price': 4},
            'limits': {
                'leverage': {'min': 1, 'max': 100},
                'amount': {'min': 0.000221, 'max': None},
                'price': {'min': 0.01, 'max': None},
                'cost': {'min': None, 'max': None}
            },
            'info': {'maintenance_rate': '0.005'}
        },
        'LTC/USDT': {
            'id': 'USDT-LTC',
            'symbol': 'LTC/USDT',
            'base': 'LTC',
            'quote': 'USDT',
            'active': False,
            'spot': True,
            'future': True,
            'swap': True,
            'margin': True,
            'linear': None,
            'inverse': False,
            'type': 'spot',
            'contractSize': None,
            'taker': 0.0006,
            'maker': 0.0002,
            'precision': {'amount': 8, 'price': 8},
            'limits': {
                'leverage': {'min': None, 'max': None},
                'amount': {'min': 0.06646786, 'max': None},
                'price': {'min': 1e-08, 'max': None},
                'leverage': {'min': None, 'max': None},
                'cost': {'min': None, 'max': None}
            },
            'info': {}
        },
        'XRP/USDT': {
            'id': 'xrpusdt',
            'symbol': 'XRP/USDT',
            'base': 'XRP',
            'quote': 'USDT',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'taker': 0.0006,
            'maker': 0.0002,
            'precision': {'price': 8, 'amount': 8, 'cost': 8},
            'lot': 1e-08,
            'contractSize': None,
            'limits': {
                'leverage': {'min': 1, 'max': 100},
                'amount': {'min': 0.01, 'max': 1000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000}
            },
            'info': {}
        },
        'NEO/USDT': {
            'id': 'neousdt',
            'symbol': 'NEO/USDT',
            'base': 'NEO',
            'quote': 'USDT',
            'settle': '',
            'baseId': 'NEO',
            'quoteId': 'USDT',
            'settleId': '',
            'type': 'swap',
            'spot': True,
            'margin': True,
            'swap': False,
            'futures': False,
            'option': False,
            'active': True,
            'contract': False,
            'linear': None,
            'inverse': None,
            'taker': 0.0006,
            'maker': 0.0002,
            'contractSize': None,
            'expiry': None,
            'expiryDatetime': None,
            'strike': None,
            'optionType': None,
            'tierBased': None,
            'percentage': None,
            'lot': 1e-08,
            'precision': {'price': 8, 'amount': 8, 'cost': 8},
            'limits': {
                'leverage': {'min': 1, 'max': 10},
                'amount': {'min': 0.01, 'max': 1000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000}
            },
            'info': {}
        },
        'TKN/USDT': {
            'id': 'tknusdt',
            'symbol': 'TKN/USDT',
            'base': 'TKN',
            'quote': 'USDT',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'contractSize': None,
            'taker': 0.0006,
            'maker': 0.0002,
            'precision': {'price': 8, 'amount': 8, 'cost': 8},
            'lot': 1e-08,
            'limits': {
                'leverage': {'min': 1, 'max': 100},
                'amount': {'min': 0.01, 'max': 100000000000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000}
            },
            'info': {}
        },
        'LTC/USD': {
            'id': 'USD-LTC',
            'symbol': 'LTC/USD',
            'base': 'LTC',
            'quote': 'USD',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'contractSize': None,
            'precision': {'amount': 8, 'price': 8},
            'limits': {
                'amount': {'min': 0.06646786, 'max': None},
                'price': {'min': 1e-08, 'max': None},
                'leverage': {'min': None, 'max': None},
                'cost': {'min': None, 'max': None}
            },
            'info': {}
        },
        'XLTCUSDT': {
            'id': 'xLTCUSDT',
            'symbol': 'XLTCUSDT',
            'base': 'LTC',
            'quote': 'USDT',
            'active': True,
            'spot': False,
            'type': 'swap',
            'contractSize': 0.01,
            'swap': False,
            'linear': False,
            'taker': 0.0006,
            'maker': 0.0002,
            'precision': {'amount': 8, 'price': 8},
            'limits': {
                'leverage': {'min': None, 'max': None},
                'amount': {'min': 0.06646786, 'max': None},
                'price': {'min': 1e-08, 'max': None},
                'cost': {'min': None, 'max': None}
            },
            'info': {}
        },
        'LTC/ETH': {
            'id': 'LTCETH',
            'symbol': 'LTC/ETH',
            'base': 'LTC',
            'quote': 'ETH',
            'active': True,
            'spot': True,
            'swap': False,
            'linear': None,
            'type': 'spot',
            'contractSize': None,
            'precision': {'base': 8, 'quote': 8, 'amount': 3, 'price': 5},
            'limits': {
                'leverage': {'min': None, 'max': None},
                'amount': {'min': 0.001, 'max': 10000000.0},
                'price': {'min': 1e-05, 'max': 1000.0},
                'cost': {'min': 0.01, 'max': None}
            },
            'info': {}
        },
        'ETH/USDT:USDT': {
            'id': 'ETH_USDT',
            'symbol': 'ETH/USDT:USDT',
            'base': 'ETH',
            'quote': 'USDT',
            'settle': 'USDT',
            'baseId': 'ETH',
            'quoteId': 'USDT',
            'settleId': 'USDT',
            'type': 'swap',
            'spot': False,
            'margin': False,
            'swap': True,
            'future': True,
            'option': False,
            'contract': True,
            'linear': True,
            'inverse': False,
            'tierBased': False,
            'percentage': True,
            'taker': 0.0006,
            'maker': 0.0002,
            'contractSize': 10,
            'expiry': None,
            'expiryDatetime': None,
            'strike': None,
            'optionType': None,
            'limits': {
                'leverage': {'min': 1, 'max': 100},
                'amount': {'min': 1, 'max': 300000},
                'price': {'min': None, 'max': None},
                'cost': {'min': None, 'max': None}
            },
            'precision': {'price': 0.05, 'amount': 1},
            'info': {}
        },
        'ADA/USDT:USDT': {
            'id': 'ADA_USDT',
            'symbol': 'ADA/USDT:USDT',
            'base': 'ADA',
            'quote': 'USDT',
            'settle': 'USDT',
            'baseId': 'ADA',
            'quoteId': 'USDT',
            'settleId': 'usdt',
            'type': 'swap',
            'spot': False,
            'margin': False,
            'swap': True,
            'future': True,
            'option': False,
            'contract': True,
            'linear': True,
            'inverse': False,
            'tierBased': True,
            'percentage': True,
            'taker': 7.5e-06,
            'maker': -2.5e-06,
            'feeSide': 'get',
            'tiers': {
                'maker': [
                    [0, 0.002], [1.5, 0.00185], [3, 0.00175], [6, 0.00165],
                    [12.5, 0.00155], [25, 0.00145], [75, 0.00135],
                    [200, 0.00125], [500, 0.00115], [1250, 0.00105],
                    [2500, 0.00095], [3000, 0.00085], [6000, 0.00075],
                    [11000, 0.00065], [20000, 0.00055], [40000, 0.00055],
                    [75000, 0.00055]
                ],
                'taker': [
                    [0, 0.002], [1.5, 0.00195], [3, 0.00185], [6, 0.00175],
                    [12.5, 0.00165], [25, 0.00155], [75, 0.00145],
                    [200, 0.00135], [500, 0.00125], [1250, 0.00115],
                    [2500, 0.00105], [3000, 0.00095], [6000, 0.00085],
                    [11000, 0.00075], [20000, 0.00065], [40000, 0.00065],
                    [75000, 0.00065]
                ]
            },
            'contractSize': 0.01,
            'info': {}
        },
        'SOL/BUSD:BUSD': {
            'id': 'SOLBUSD',
            'symbol': 'SOL/BUSD',
            'base': 'SOL',
            'quote': 'BUSD',
            'settle': 'BUSD',
            'baseId': 'SOL',
            'quoteId': 'BUSD',
            'settleId': 'BUSD',
            'type': 'future',
            'spot': False,
            'margin': False,
            'swap': False,
            'future': True,
            'delivery': False,
            'option': False,
            'active': True,
            'contract': True,
            'linear': True,
            'inverse': False,
            'contractSize': 1,
            'expiry': None,
            'expiryDatetime': None,
            'strike': None,
            'optionType': None,
            'limits': {
                'leverage': {'min': None, 'max': None},
                'amount': {'min': 1, 'max': 1000000},
                'price': {'min': 0.04, 'max': 100000},
                'cost': {'min': 5, 'max': None},
                'market': {'min': 1, 'max': 1500}
            },
            'precision': {'amount': 0, 'price': 2, 'base': 8, 'quote': 8},
            'tierBased': False,
            'percentage': True,
            'taker': 0.0004,
            'maker': 0.0002,
            'feeSide': 'get',
            'info': {
                'symbol': 'SOLBUSD',
                'pair': 'SOLBUSD',
                'contractType': 'PERPETUAL',
                'deliveryDate': '4133404800000',
                'onboardDate': '1630566000000',
                'status': 'TRADING',
                'maintMarginPercent': '2.5000',
                'requiredMarginPercent': '5.0000',
                'baseAsset': 'SOL',
                'quoteAsset': 'BUSD',
                'marginAsset': 'BUSD',
                'pricePrecision': '4',
                'quantityPrecision': '0',
                'baseAssetPrecision': '8',
                'quotePrecision': '8',
                'underlyingType': 'COIN',
                'underlyingSubType': [],
                'settlePlan': '0',
                'triggerProtect': '0.0500',
                'liquidationFee': '0.005000',
                'marketTakeBound': '0.05',
                'filters': [
                    {'minPrice': '0.0400', 'maxPrice': '100000', 'filterType': 'PRICE_FILTER', 'tickSize': '0.0100'},
                    {'stepSize': '1', 'filterType': 'LOT_SIZE', 'maxQty': '1000000', 'minQty': '1'},
                    {'stepSize': '1', 'filterType': 'MARKET_LOT_SIZE', 'maxQty': '1500', 'minQty': '1'},
                    {'limit': '200', 'filterType': 'MAX_NUM_ORDERS'},
                    {'limit': '10', 'filterType': 'MAX_NUM_ALGO_ORDERS'},
                    {'notional': '5', 'filterType': 'MIN_NOTIONAL'},
                    {'multiplierDown': '0.9500', 'multiplierUp': '1.0500', 'multiplierDecimal': '4', 'filterType': 'PERCENT_PRICE'}
                ],
                'orderTypes': [
                    'LIMIT', 'MARKET', 'STOP', 'STOP_MARKET',
                    'TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET'
                ],
                'timeInForce': ['GTC', 'IOC', 'FOK', 'GTX']
            }
        }
    }


@pytest.fixture
def markets_static(markets: Dict[str, Any]) -> Dict[str, Any]:
    static_markets: List[str] = [
        'BLK/BTC', 'BTT/BTC', 'ETH/BTC',
        'ETH/USDT', 'LTC/BTC', 'LTC/ETH',
        'LTC/USD', 'LTC/USDT', 'NEO/BTC',
        'TKN/BTC', 'XLTCUSDT', 'XRP/BTC',
        'ADA/USDT:USDT', 'ETH/USDT:USDT'
    ]
    return {m: markets[m] for m in static_markets}


@pytest.fixture
def shitcoinmarkets(markets_static: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fixture with shitcoin markets - used to test filters in pairlists
    """
    shitmarkets: Dict[str, Any] = deepcopy(markets_static)
    shitmarkets.update({
        'HOT/BTC': {
            'id': 'HOTBTC',
            'symbol': 'HOT/BTC',
            'base': 'HOT',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'type': 'spot',
            'precision': {'base': 8, 'quote': 8, 'amount': 0, 'price': 8},
            'limits': {
                'amount': {'min': 1.0, 'max': 90000000.0},
                'price': {'min': 1e-08, 'max': 1000.0},
                'cost': {'min': 0.001, 'max': None}
            },
            'info': {}
        },
        'FUEL/BTC': {
            'id': 'FUELBTC',
            'symbol': 'FUEL/BTC',
            'base': 'FUEL',
            'quote': 'BTC',
            'active': True,
            'spot': True,
            'type': 'spot',
            'precision': {'base': 8, 'quote': 8, 'amount': 0, 'price': 8},
            'limits': {
                'amount': {'min': 1.0, 'max': 90000000.0},
                'price': {'min': 1e-08, 'max': 1000.0},
                'cost': {'min': 0.001, 'max': None}
            },
            'info': {}
        },
        'NANO/USDT': {
            'percentage': True,
            'tierBased': False,
            'taker': 0.001,
            'maker': 0.001,
            'precision': {'base': 8, 'quote': 8, 'amount': 2, 'price': 4},
            'limits': {
                'leverage': {'min': None, 'max': None},
                'amount': {'min': None, 'max': None},
                'price': {'min': None, 'max': None},
                'cost': {'min': None, 'max': None}
            },
            'id': 'NANOUSDT',
            'symbol': 'NANO/USDT',
            'base': 'NANO',
            'quote': 'USDT',
            'baseId': 'NANO',
            'quoteId': 'USDT',
            'settleId': 'USDT',
            'type': 'spot',
            'spot': True,
            'future': False,
            'active': True
        },
        'ADAHALF/USDT': {
            'percentage': True,
            'tierBased': False,
            'taker': 0.001,
            'maker': 0.001,
            'precision': {'base': 8, 'quote': 8, 'amount': 2, 'price': 4},
            'limits': {
                'leverage': {'min': None, 'max': None},
                'amount': {'min': None, 'max': None},
                'price': {'min': None, 'max': None},
                'cost': {'min': None, 'max': None}
            },
            'id': 'ADAHALFUSDT',
            'symbol': 'ADAHALF/USDT',
            'base': 'ADAHALF',
            'quote': 'USDT',
            'baseId': 'ADAHALF',
            'quoteId': 'USDT',
            'settleId': 'USDT',
            'type': 'spot',
            'spot': True,
            'future': False,
            'active': True
        },
        'ADADOUBLE/USDT': {
            'percentage': True,
            'tierBased': False,
            'taker': 0.001,
            'maker': 0.001,
            'precision': {'base': 8, 'quote': 8, 'amount': 2, 'price': 4},
            'limits': {
                'leverage': {'min': None, 'max': None},
                'amount': {'min': None, 'max': None},
                'price': {'min': None, 'max': None},
                'cost': {'min': None, 'max': None}
            },
            'id': 'ADADOUBLEUSDT',
            'symbol': 'ADADOUBLE/USDT',
            'base': 'ADADOUBLE',
            'quote': 'USDT',
            'baseId': 'ADADOUBLE',
            'quoteId': 'USDT',
            'settleId': 'USDT',
            'type': 'spot',
            'spot': True,
            'future': False,
            'active': True
        }
    })
    return shitmarkets


@pytest.fixture
def markets_empty() -> MagicMock:
    return MagicMock(return_value=[])


@pytest.fixture(scope='function')
def limit_buy_order_open() -> Dict[str, Any]:
    return {
        'id': 'mocked_limit_buy',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'timestamp': dt_ts(),
        'datetime': dt_now().isoformat(),
        'price': 1.099e-05,
        'average': 1.099e-05,
        'amount': 90.99181073,
        'filled': 0.0,
        'cost': 0.0009999,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_buy_order_old() -> Dict[str, Any]:
    return {
        'id': 'mocked_limit_buy_old',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
        'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
        'price': 1.099e-05,
        'amount': 90.99181073,
        'filled': 0.0,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_sell_order_old() -> Dict[str, Any]:
    return {
        'id': 'mocked_limit_sell_old',
        'type': 'limit',
        'side': 'sell',
        'symbol': 'ETH/BTC',
        'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
        'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
        'price': 1.099e-05,
        'amount': 90.99181073,
        'filled': 0.0,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_buy_order_old_partial() -> Dict[str, Any]:
    return {
        'id': 'mocked_limit_buy_old_partial',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'ETH/BTC',
        'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
        'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
        'price': 1.099e-05,
        'amount': 90.99181073,
        'filled': 23.0,
        'cost': 90.99181073 * 23.0,
        'remaining': 67.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_buy_order_old_partial_canceled(limit_buy_order_old_partial: Dict[str, Any]) -> Dict[str, Any]:
    res: Dict[str, Any] = deepcopy(limit_buy_order_old_partial)
    res['status'] = 'canceled'
    res['fee'] = {'cost': 0.023, 'currency': 'ETH'}
    return res


@pytest.fixture(scope='function')
def limit_buy_order_canceled_empty(request: Any) -> Dict[str, Any]:
    exchange_name: str = request.param
    if exchange_name == 'kraken':
        return {
            'info': {},
            'id': 'AZNPFF-4AC4N-7MKTAT',
            'clientOrderId': None,
            'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
            'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
            'lastTradeTimestamp': None,
            'status': 'canceled',
            'symbol': 'LTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'price': 34.3225,
            'cost': 0.0,
            'amount': 0.55,
            'filled': 0.0,
            'average': 0.0,
            'remaining': 0.55,
            'fee': {'cost': 0.0, 'rate': None, 'currency': 'USDT'},
            'trades': []
        }
    elif exchange_name == 'binance':
        return {
            'info': {},
            'id': '1234512345',
            'clientOrderId': 'alb1234123',
            'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
            'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
            'lastTradeTimestamp': None,
            'symbol': 'LTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'price': 0.016804,
            'amount': 0.55,
            'cost': 0.0,
            'average': None,
            'filled': 0.0,
            'remaining': 0.55,
            'status': 'canceled',
            'fee': None,
            'trades': None
        }
    else:
        return {
            'info': {},
            'id': '1234512345',
            'clientOrderId': 'alb1234123',
            'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
            'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
            'lastTradeTimestamp': None,
            'symbol': 'LTC/USDT',
            'type': 'limit',
            'side': 'buy',
            'price': 0.016804,
            'amount': 0.55,
            'cost': 0.0,
            'average': None,
            'filled': 0.0,
            'remaining': 0.55,
            'status': 'canceled',
            'fee': None,
            'trades': None
        }


@pytest.fixture
def limit_sell_order_open() -> Dict[str, Any]:
    return {
        'id': 'mocked_limit_sell',
        'type': 'limit',
        'side': 'sell',
        'symbol': 'mocked',
        'datetime': dt_now().isoformat(),
        'timestamp': dt_ts(),
        'price': 1.173e-05,
        'amount': 90.99181073,
        'filled': 0.0,
        'remaining': 90.99181073,
        'status': 'open'
    }


@pytest.fixture
def limit_sell_order(limit_sell_order_open: Dict[str, Any]) -> Dict[str, Any]:
    order: Dict[str, Any] = deepcopy(limit_sell_order_open)
    order['remaining'] = 0.0
    order['filled'] = order['amount']
    order['status'] = 'closed'
    return order


@pytest.fixture
def order_book_l2() -> MagicMock:
    return MagicMock(return_value={
        'bids': [
            [0.043936, 10.442], [0.043935, 31.865], [0.043933, 11.212],
            [0.043928, 0.088], [0.043925, 10.0], [0.043921, 10.0],
            [0.04392, 37.64], [0.043899, 0.066], [0.043885, 0.676],
            [0.04387, 22.758]
        ],
        'asks': [
            [0.043949, 0.346], [0.04395, 0.608], [0.043951, 3.948],
            [0.043954, 0.288], [0.043958, 9.277], [0.043995, 1.566],
            [0.044, 0.588], [0.044002, 0.992], [0.044003, 0.095],
            [0.04402, 37.64]
        ],
        'timestamp': None,
        'datetime': None,
        'nonce': 288004540
    })


@pytest.fixture
def order_book_l2_usd() -> MagicMock:
    return MagicMock(return_value={
        'symbol': 'LTC/USDT',
        'bids': [
            [25.563, 49.269], [25.562, 83.0], [25.56, 106.0],
            [25.559, 15.381], [25.558, 29.299], [25.557, 34.624],
            [25.556, 10.0], [25.555, 14.684], [25.554, 45.91],
            [25.553, 50.0]
        ],
        'asks': [
            [25.566, 14.27], [25.567, 48.484], [25.568, 92.349],
            [25.572, 31.48], [25.573, 23.0], [25.574, 20.0],
            [25.575, 89.606], [25.576, 262.016], [25.577, 178.557],
            [25.578, 78.614]
        ],
        'timestamp': None,
        'datetime': None,
        'nonce': 2372149736
    })


@pytest.fixture
def ohlcv_history_list() -> List[List[Union[int, float, None]]]:
    return [
        [1511686200000, 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 0.0877869],
        [1511686500000, 8.88e-05, 8.942e-05, 8.88e-05, 8.893e-05, 0.05874751],
        [1511686800000, 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05, 0.7039405]
    ]


@pytest.fixture
def ohlcv_history(ohlcv_history_list: List[List[Union[int, float, None]]]) -> pd.DataFrame:
    df: pd.DataFrame = ohlcv_to_dataframe(
        ohlcv_history_list,
        '5m',
        pair='UNITTEST/BTC',
        fill_missing=True,
        drop_incomplete=False
    )
    return df


@pytest.fixture
def tickers() -> MagicMock:
    return MagicMock(return_value={
        'ETH/BTC': {
            'symbol': 'ETH/BTC',
            'timestamp': 1522014806207,
            'datetime': '2018-03-25T21:53:26.207Z',
            'high': 0.061697,
            'low': 0.060531,
            'bid': 0.061588,
            'bidVolume': 3.321,
            'ask': 0.061655,
            'askVolume': 0.212,
            'vwap': 0.06105296,
            'open': 0.060809,
            'close': 0.060761,
            'first': None,
            'last': 0.061588,
            'change': 1.281,
            'percentage': None,
            'average': None,
            'baseVolume': 111649.001,
            'quoteVolume': 6816.50176926,
            'info': {}
        },
        'TKN/BTC': {
            'symbol': 'TKN/BTC',
            'timestamp': 1522014806169,
            'datetime': '2018-03-25T21:53:26.169Z',
            'high': 0.01885,
            'low': 0.018497,
            'bid': 0.018799,
            'bidVolume': 8.38,
            'ask': 0.018802,
            'askVolume': 15.0,
            'vwap': 0.01869197,
            'open': 0.018585,
            'close': 0.018573,
            'last': 0.018799,
            'baseVolume': 81058.66,
            'quoteVolume': 2247.48374509
        },
        'BLK/BTC': {
            'symbol': 'BLK/BTC',
            'timestamp': 1522014806072,
            'datetime': '2018-03-25T21:53:26.072Z',
            'high': 0.007745,
            'low': 0.007512,
            'bid': 0.007729,
            'bidVolume': 0.01,
            'ask': 0.007743,
            'askVolume': 21.37,
            'vwap': 0.00761466,
            'open': 0.007653,
            'close': 0.007652,
            'first': None,
            'last': 0.007743,
            'change': 1.176,
            'percentage': None,
            'average': None,
            'baseVolume': 295152.26,
            'quoteVolume': 1515.14631229,
            'info': {}
        },
        'LTC/BTC': {
            'symbol': 'LTC/BTC',
            'timestamp': 1523787258992,
            'datetime': '2018-04-15T10:14:19.992Z',
            'high': 0.015978,
            'low': 0.0157,
            'bid': 0.015954,
            'bidVolume': 12.83,
            'ask': 0.015957,
            'askVolume': 0.49,
            'vwap': 0.01581636,
            'open': 0.015823,
            'close': 0.01582,
            'first': None,
            'last': 0.015951,
            'change': 0.809,
            'percentage': None,
            'average': None,
            'baseVolume': 88620.68,
            'quoteVolume': 1401.65697943,
            'info': {}
        },
        'BTT/BTC': {
            'symbol': 'BTT/BTC',
            'timestamp': 1550936557206,
            'datetime': '2019-02-23T15:42:37.206Z',
            'high': 2.6e-07,
            'low': 2.4e-07,
            'bid': 2.4e-07,
            'bidVolume': 2446894197.0,
            'ask': 2.5e-07,
            'askVolume': 2447913837.0,
            'vwap': 2.5e-07,
            'open': 2.6e-07,
            'close': 2.4e-07,
            'last': 2.4e-07,
            'previousClose': 2.6e-07,
            'change': -2e-08,
            'percentage': -7.692,
            'average': None,
            'baseVolume': 4886464537.0,
            'quoteVolume': 1215.14489611,
            'info': {}
        },
        'HOT/BTC': {
            'symbol': 'HOT/BTC',
            'timestamp': 1572273518661,
            'datetime': '2019-10-28T14:38:38.661Z',
            'high': 1.1e-07,
            'low': 9e-08,
            'bid': 1e-07,
            'bidVolume': 1476027288.0,
            'ask': 1.1e-07,
            'askVolume': 820153831.0,
            'vwap': 1e-07,
            'open': 9e-08,
            'close': 1.1e-07,
            'last': 1.1e-07,
            'previousClose': 9e-08,
            'change': 2e-08,
            'percentage': 22.222,
            'average': None,
            'baseVolume': 1442290324.0,
            'quoteVolume': 143.78311994,
            'info': {}
        },
        'FUEL/BTC': {
            'symbol': 'FUEL/BTC',
            'timestamp': 1572340250771,
            'datetime': '2019-10-29T09:10:50.771Z',
            'high': 4e-07,
            'low': 3.5e-07,
            'bid': 3.6e-07,
            'bidVolume': 8932318.0,
            'ask': 3.7e-07,
            'askVolume': 10140774.0,
            'vwap': 3.7e-07,
            'open': 3.9e-07,
            'close': 3.7e-07,
            'last': 3.7e-07,
            'previousClose': 3.8e-07,
            'change': -2e-08,
            'percentage': -5.128,
            'average': None,
            'baseVolume': 168927742.0,
            'quoteVolume': 62.68220262,
            'info': {}
        },
        'BTC/USDT': {
            'symbol': 'BTC/USDT',
            'timestamp': 1573758371399,
            'datetime': '2019-11-14T19:06:11.399Z',
            'high': 8800.0,
            'low': 8582.6,
            'bid': 8648.16,
            'bidVolume': 0.238771,
            'ask': 8648.72,
            'askVolume': 0.016253,
            'vwap': 8683.13647806,
            'open': 8759.7,
            'close': 8648.72,
            'last': 8648.72,
            'previousClose': 8759.67,
            'change': -110.98,
            'percentage': -1.267,
            'average': None,
            'baseVolume': 35025.943355,
            'quoteVolume': 304135046.4242901,
            'info': {}
        },
        'ETH/USDT': {
            'symbol': 'ETH/USDT',
            'timestamp': 1522014804118,
            'datetime': '2018-03-25T21:53:24.118Z',
            'high': 530.88,
            'low': 512.0,
            'bid': 529.73,
            'bidVolume': 0.2,
            'ask': 530.21,
            'askVolume': 0.2464,
            'vwap': 521.02438405,
            'open': 527.27,
            'close': 528.42,
            'first': None,
            'last': 530.21,
            'change': 0.558,
            'percentage': 2.349,
            'average': None,
            'baseVolume': 72300.0659,
            'quoteVolume': 37670097.3022171,
            'info': {}
        },
        'TKN/USDT': {
            'symbol': 'TKN/USDT',
            'timestamp': 1522014806198,
            'datetime': '2018-03-25T21:53:26.198Z',
            'high': 8718.0,
            'low': 8365.77,
            'bid': 8603.64,
            'bidVolume': 0.15846,
            'ask': 8603.67,
            'askVolume': 0.069147,
            'vwap': 8536.35621697,
            'open': 8680.0,
            'close': 8680.0,
            'first': None,
            'last': 8603.67,
            'change': -0.879,
            'percentage': -8.95,
            'average': None,
            'baseVolume': 30414.604298,
            'quoteVolume': 259629896.48584127,
            'info': {}
        },
        'BLK/USDT': {
            'symbol': 'BLK/USDT',
            'timestamp': 1522014806145,
            'datetime': '2018-03-25T21:53:26.145Z',
            'high': 66.95,
            'low': 63.38,
            'bid': 66.473,
            'bidVolume': 4.968,
            'ask': 66.54,
            'askVolume': 2.704,
            'vwap': 65.0526901,
            'open': 66.43,
            'close': 66.383,
            'first': None,
            'last': 66.5,
            'change': 0.105,
            'percentage': None,
            'average': None,
            'baseVolume': 294106.204,
            'quoteVolume': 19132399.743954,
            'info': {}
        },
        'LTC/USDT': {
            'symbol': 'LTC/USDT',
            'timestamp': 1523787257812,
            'datetime': '2018-04-15T10:14:18.812Z',
            'high': 129.94,
            'low': 124.0,
            'bid': 129.28,
            'bidVolume': 0.03201,
            'ask': 129.52,
            'askVolume': 0.14529,
            'vwap': 126.92838682,
            'open': 127.0,
            'close': 127.1,
            'first': None,
            'last': 129.28,
            'change': 1.795,
            'percentage': -2.5,
            'average': None,
            'baseVolume': 59698.79897,
            'quoteVolume': 29132399.743954,
            'info': {}
        },
        'XRP/BTC': {
            'symbol': 'XRP/BTC',
            'timestamp': 1573758257534,
            'datetime': '2019-11-14T19:04:17.534Z',
            'high': 3.126e-05,
            'low': 3.061e-05,
            'bid': 3.093e-05,
            'bidVolume': 27901.0,
            'ask': 3.095e-05,
            'askVolume': 10551.0,
            'vwap': 3.091e-05,
            'open': 3.119e-05,
            'close': 3.094e-05,
            'last': 3.094e-05,
            'previousClose': 3.117e-05,
            'change': -2.5e-07,
            'percentage': -0.802,
            'average': None,
            'baseVolume': 37334921.0,
            'quoteVolume': 1154.19266394,
            'info': {}
        },
        'NANO/USDT': {
            'symbol': 'NANO/USDT',
            'timestamp': 1580469388244,
            'datetime': '2020-01-31T11:16:28.244Z',
            'high': 0.7519,
            'low': 0.7154,
            'bid': 0.7305,
            'bidVolume': 300.3,
            'ask': 0.7342,
            'askVolume': 15.14,
            'vwap': 0.73645591,
            'open': 0.7154,
            'close': 0.7342,
            'last': 0.7342,
            'previousClose': 0.7189,
            'change': 0.0188,
            'percentage': 2.628,
            'average': None,
            'baseVolume': 439472.44,
            'quoteVolume': 323652.075405,
            'info': {}
        },
        'ADAHALF/USDT': {
            'symbol': 'ADAHALF/USDT',
            'timestamp': 1580469388244,
            'datetime': '2020-01-31T11:16:28.244Z',
            'high': None,
            'low': None,
            'bid': 0.7305,
            'bidVolume': None,
            'ask': 0.7342,
            'askVolume': None,
            'vwap': None,
            'open': None,
            'close': None,
            'last': None,
            'previousClose': None,
            'change': None,
            'percentage': 2.628,
            'average': None,
            'baseVolume': 0.0,
            'quoteVolume': 0.0,
            'info': {}
        },
        'ADADOUBLE/USDT': {
            'symbol': 'ADADOUBLE/USDT',
            'timestamp': 1580469388244,
            'datetime': '2020-01-31T11:16:28.244Z',
            'high': None,
            'low': None,
            'bid': 0.7305,
            'bidVolume': None,
            'ask': 0.7342,
            'askVolume': None,
            'vwap': None,
            'open': None,
            'close': None,
            'last': 0,
            'previousClose': None,
            'change': None,
            'percentage': 2.628,
            'average': None,
            'baseVolume': 0.0,
            'quoteVolume': 0.0,
            'info': {}
        }
    })


@pytest.fixture
def dataframe_1m(testdatadir: Path) -> pd.DataFrame:
    with (testdatadir / 'UNITTEST_BTC-1m.json').open('r') as data_file:
        data: Any = json.load(data_file)
    return ohlcv_to_dataframe(data, '1m', pair='UNITTEST/BTC', fill_missing=True)


@pytest.fixture(scope='function')
def trades_for_order() -> List[Dict[str, Any]]:
    return [
        {
            'info': {
                'id': 34567,
                'orderId': 123456,
                'price': '2.0',
                'qty': '8.00000000',
                'commission': '0.00800000',
                'commissionAsset': 'LTC',
                'time': 1521663363189,
                'isBuyer': True,
                'isMaker': False,
                'isBestMatch': True
            },
            'timestamp': 1521663363189,
            'datetime': '2018-03-21T20:16:03.189Z',
            'symbol': 'LTC/USDT',
            'id': '34567',
            'order': '123456',
            'type': None,
            'side': 'buy',
            'price': 2.0,
            'cost': 16.0,
            'amount': 8.0,
            'fee': {'cost': 0.008, 'currency': 'LTC'}
        }
    ]


@pytest.fixture(scope='function')
def trades_history() -> List[List[Union[int, str, Optional[str], float]]]:
    return [
        [1565798389463, '12618132aa9', None, 'buy', 0.019627, 0.04, 0.00078508],
        [1565798399629, '1261813bb30', None, 'buy', 0.019627, 0.244, 0.004788987999999999],
        [1565798399752, '1261813cc31', None, 'sell', 0.019626, 0.011, 0.00021588599999999999],
        [1565798399862, '126181cc332', None, 'sell', 0.019626, 0.011, 0.00021588599999999999],
        [1565798399862, '126181cc333', None, 'sell', 0.019626, 0.012, 0.00021588599999999999],
        [1565798399872, '1261aa81334', None, 'sell', 0.019626, 0.011, 0.00021588599999999999]
    ]


@pytest.fixture(scope='function')
def trades_history_df(trades_history: List[List[Union[int, str, Optional[str], float]]]) -> pd.DataFrame:
    trades: pd.DataFrame = trades_list_to_df(trades_history)
    trades['date'] = pd.to_datetime(trades['timestamp'], unit='ms', utc=True)
    return trades


@pytest.fixture(scope='function')
def fetch_trades_result() -> List[Dict[str, Any]]:
    return [
        {
            'info': ['0.01962700', '0.04000000', '1565798399.4631551', 'b', 'm', '', '126181329'],
            'timestamp': 1565798399463,
            'datetime': '2019-08-14T15:59:59.463Z',
            'symbol': 'ETH/BTC',
            'id': '126181329',
            'order': None,
            'type': None,
            'takerOrMaker': None,
            'side': 'buy',
            'price': 0.019627,
            'amount': 0.04,
            'cost': 0.00078508,
            'fee': None
        },
        {
            'info': ['0.01962700', '0.24400000', '1565798399.6291551', 'b', 'm', '', '126181330'],
            'timestamp': 1565798399629,
            'datetime': '2019-08-14T15:59:59.629Z',
            'symbol': 'ETH/BTC',
            'id': '126181330',
            'order': None,
            'type': None,
            'takerOrMaker': None,
            'side': 'buy',
            'price': 0.019627,
            'amount': 0.244,
            'cost': 0.004788987999999999,
            'fee': None
        },
        {
            'info': ['0.01962600', '0.01100000', '1565798399.7521551', 's', 'm', '', '126181331'],
            'timestamp': 1565798399752,
            'datetime': '2019-08-14T15:59:59.752Z',
            'symbol': 'ETH/BTC',
            'id': '126181331',
            'order': None,
            'type': None,
            'takerOrMaker': None,
            'side': 'sell',
            'price': 0.019626,
            'amount': 0.011,
            'cost': 0.00021588599999999999,
            'fee': None
        },
        {
            'info': ['0.01962600', '0.01100000', '1565798399.8621551', 's', 'm', '', '126181332'],
            'timestamp': 1565798399862,
            'datetime': '2019-08-14T15:59:59.862Z',
            'symbol': 'ETH/BTC',
            'id': '126181332',
            'order': None,
            'type': None,
            'takerOrMaker': None,
            'side': 'sell',
            'price': 0.019626,
            'amount': 0.011,
            'cost': 0.00021588599999999999,
            'fee': None
        },
        {
            'info': ['0.01952600', '0.01200000', '1565798399.8721551', 's', 'm', '', '126181333', 1565798399872512133],
            'timestamp': 1565798399872,
            'datetime': '2019-08-14T15:59:59.872Z',
            'symbol': 'ETH/BTC',
            'id': '126181333',
            'order': None,
            'type': None,
            'takerOrMaker': None,
            'side': 'sell',
            'price': 0.019626,
            'amount': 0.011,
            'cost': 0.00021588599999999999,
            'fee': None
        }
    ]


@pytest.fixture(scope='function')
def trades_for_order2() -> List[Dict[str, Any]]:
    return [
        {
            'info': {},
            'timestamp': 1521663363189,
            'datetime': '2018-03-21T20:16:03.189Z',
            'symbol': 'LTC/ETH',
            'id': '34567',
            'order': '123456',
            'type': None,
            'side': 'buy',
            'price': 0.245441,
            'cost': 1.963528,
            'amount': 4.0,
            'fee': {'cost': 0.004, 'currency': 'LTC'}
        },
        {
            'info': {},
            'timestamp': 1521663363189,
            'datetime': '2018-03-21T20:16:03.189Z',
            'symbol': 'LTC/ETH',
            'id': '34567',
            'order': '123456',
            'type': None,
            'side': 'buy',
            'price': 0.245441,
            'cost': 1.963528,
            'amount': 4.0,
            'fee': {'cost': 0.004, 'currency': 'LTC'}
        }
    ]


@pytest.fixture
def buy_order_fee() -> Dict[str, Any]:
    return {
        'id': 'mocked_limit_buy_old',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'timestamp': dt_ts(dt_now() - timedelta(minutes=601)),
        'datetime': (dt_now() - timedelta(minutes=601)).isoformat(),
        'price': 0.245441,
        'amount': 8.0,
        'cost': 1.963528,
        'remaining': 90.99181073,
        'status': 'closed',
        'fee': None
    }


@pytest.fixture(scope='function')
def edge_conf(default_conf: Dict[str, Any]) -> Dict[str, Any]:
    conf: Dict[str, Any] = deepcopy(default_conf)
    conf['runmode'] = RunMode.DRY_RUN
    conf['max_open_trades'] = -1
    conf['tradable_balance_ratio'] = 0.5
    conf['stake_amount'] = constants.UNLIMITED_STAKE_AMOUNT
    conf['edge'] = {
        'enabled': True,
        'process_throttle_secs': 1800,
        'calculate_since_number_of_days': 14,
        'allowed_risk': 0.01,
        'stoploss_range_min': -0.01,
        'stoploss_range_max': -0.1,
        'stoploss_range_step': -0.01,
        'maximum_winrate': 0.8,
        'minimum_expectancy': 0.2,
        'min_trade_number': 15,
        'max_trade_duration_minute': 1440,
        'remove_pumps': False
    }
    return conf


@pytest.fixture
def rpc_balance() -> Dict[str, Dict[str, float]]:
    return {
        'BTC': {'total': 12.0, 'free': 12.0, 'used': 0.0},
        'ETH': {'total': 0.0, 'free': 0.0, 'used': 0.0},
        'USDT': {'total': 10000.0, 'free': 10000.0, 'used': 0.0},
        'LTC': {'total': 10.0, 'free': 10.0, 'used': 0.0},
        'XRP': {'total': 0.1, 'free': 0.01, 'used': 0.0},
        'EUR': {'total': 10.0, 'free': 10.0, 'used': 0.0}
    }


@pytest.fixture
def testdatadir() -> Path:
    """Return the path where testdata files are stored"""
    return (Path(__file__).parent / 'testdata').resolve()


@pytest.fixture(scope='function')
def import_fails() -> None:
    import builtins
    realimport: Callable = builtins.__import__

    def mockedimport(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ['filelock', 'cysystemd.journal', 'uvloop']:
            raise ImportError(f"No module named '{name}'")
        return realimport(name, *args, **kwargs)
    builtins.__import__ = mockedimport
    yield
    builtins.__import__ = realimport


@pytest.fixture(scope='function')
def open_trade() -> Trade:
    trade: Trade = Trade(
        pair='ETH/BTC',
        open_rate=1.099e-05,
        exchange='binance',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=dt_now() - timedelta(minutes=601),
        is_open=True
    )
    trade.orders = [
        Order(
            ft_order_side='buy',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id='123456789',
            status='closed',
            symbol=trade.pair,
            order_type='market',
            side='buy',
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date
        )
    ]
    return trade


@pytest.fixture(scope='function')
def open_trade_usdt() -> Trade:
    trade: Trade = Trade(
        pair='ADA/USDT',
        open_rate=2.0,
        exchange='binance',
        amount=30.0,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=60.0,
        open_date=dt_now() - timedelta(minutes=601),
        is_open=True
    )
    trade.orders = [
        Order(
            ft_order_side='buy',
            ft_pair=trade.pair,
            ft_is_open=False,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id='123456789',
            status='closed',
            symbol=trade.pair,
            order_type='market',
            side='buy',
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date
        ),
        Order(
            ft_order_side='exit',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id='123456789_exit',
            status='open',
            symbol=trade.pair,
            order_type='limit',
            side='sell',
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date
        )
    ]
    return trade


@pytest.fixture(scope='function')
def limit_buy_order_usdt_open() -> Dict[str, Any]:
    return {
        'id': 'mocked_limit_buy_usdt',
        'type': 'limit',
        'side': 'buy',
        'symbol': 'mocked',
        'datetime': dt_now().isoformat(),
        'timestamp': dt_ts(),
        'price': 2.0,
        'average': 2.0,
        'amount': 30.0,
        'filled': 0.0,
        'cost': 60.0,
        'remaining': 30.0,
        'status': 'open'
    }


@pytest.fixture(scope='function')
def limit_buy_order_usdt(limit_buy_order_usdt_open: Dict[str, Any]) -> Dict[str, Any]:
    order: Dict[str, Any] = deepcopy(limit_buy_order_usdt_open)
    order['status'] = 'closed'
    order['filled'] = order['amount']
    order['remaining'] = 0.0
    return order


@pytest.fixture
def limit_sell_order_usdt_open() -> Dict[str, Any]:
    return {
        'id': 'mocked_limit_sell_usdt',
        'type': 'limit',
        'side': 'sell',
        'symbol': 'mocked',
        'datetime': dt_now().isoformat(),
        'timestamp': dt_ts(),
        'price': 2.2,
        'amount': 30.0,
        'cost': 66.0,
        'filled': 0.0,
        'remaining': 30.0,
        'status': 'open'
    }


@pytest.fixture
def limit_sell_order_usdt(limit_sell_order_usdt_open: Dict[str, Any]) -> Dict[str, Any]:
    order: Dict[str, Any] = deepcopy(limit_sell_order_usdt_open)
    order['remaining'] = 0.0
    order['filled'] = order['amount']
    order['status'] = 'closed'
    return order


@pytest.fixture(scope='function')
def market_buy_order_usdt() -> Dict[str, Any]:
    return {
        'id': 'mocked_market_buy',
        'type': 'market',
        'side': 'buy',
        'symbol': 'mocked',
        'timestamp': dt_ts(),
        'datetime': dt_now().isoformat(),
        'price': 2.0,
        'amount': 30.0,
        'filled': 30.0,
        'remaining': 0.0,
        'status': 'closed'
    }


@pytest.fixture
def market_buy_order_usdt_doublefee(market_buy_order_usdt: Dict[str, Any]) -> Dict[str, Any]:
    order: Dict[str, Any] = deepcopy(market_buy_order_usdt)
    order['fee'] = None
    order['fees'] = [
        {'cost': 0.00025125, 'currency': 'BNB'},
        {'cost': 0.05030681, 'currency': 'USDT'}
    ]
    order['trades'] = [
        {
            'timestamp': None,
            'datetime': None,
            'symbol': 'ETH/USDT',
            'id': None,
            'order': '123',
            'type': 'market',
            'side': 'sell',
            'takerOrMaker': None,
            'price': 2.01,
            'amount': 25.0,
            'cost': 50.25,
            'fee': {'cost': 0.00025125, 'currency': 'BNB'}
        },
        {
            'timestamp': None,
            'datetime': None,
            'symbol': 'ETH/USDT',
            'id': None,
            'order': '123',
            'type': 'market',
            'side': 'sell',
            'takerOrMaker': None,
            'price': 2.0,
            'amount': 5,
            'cost': 10,
            'fee': {'cost': 0.0100306, 'currency': 'USDT'}
        }
    ]
    return order


@pytest.fixture
def market_sell_order_usdt() -> Dict[str, Any]:
    return {
        'id': 'mocked_limit_sell',
        'type': 'market',
        'side': 'sell',
        'symbol': 'mocked',
        'timestamp': dt_ts(),
        'datetime': dt_now().isoformat(),
        'price': 2.2,
        'amount': 30.0,
        'filled': 30.0,
        'remaining': 0.0,
        'status': 'closed'
    }


@pytest.fixture(scope='function')
def limit_order(
    limit_buy_order_usdt: Dict[str, Any],
    limit_sell_order_usdt: Dict[str, Any]
) -> Dict[str, Any]:
    return {'buy': limit_buy_order_usdt, 'sell': limit_sell_order_usdt}


@pytest.fixture(scope='function')
def limit_order_open(
    limit_buy_order_usdt_open: Dict[str, Any],
    limit_sell_order_usdt_open: Dict[str, Any]
) -> Dict[str, Any]:
    return {'buy': limit_buy_order_usdt_open, 'sell': limit_sell_order_usdt_open}


@pytest.fixture(scope='function')
def mark_ohlcv() -> List[List[Union[int, float]]]:
    return [
        [1630454400000, 2.77, 2.77, 2.73, 2.73, 0],
        [1630458000000, 2.73, 2.76, 2.72, 2.74, 0],
        [1630461600000, 2.74, 2.76, 2.74, 2.76, 0],
        [1630465200000, 2.76, 2.76, 2.74, 2.76, 0],
        [1630468800000, 2.76, 2.77, 2.75, 2.77, 0],
        [1630472400000, 2.77, 2.79, 2.75, 2.78, 0],
        [1630476000000, 2.78, 2.8, 2.77, 2.77, 0],
        [1630479600000, 2.78, 2.79, 2.77, 2.77, 0],
        [1630483200000, 2.77, 2.79, 2.77, 2.78, 0],
        [1630486800000, 2.77, 2.84, 2.77, 2.84, 0],
        [1630490400000, 2.84, 2.85, 2.81, 2.81, 0],
        [1630494000000, 2.81, 2.83, 2.81, 2.81, 0],
        [1630497600000, 2.81, 2.84, 2.81, 2.82, 0],
        [1630501200000, 2.82, 2.83, 2.81, 2.81, 0]
    ]


@pytest.fixture(scope='function')
def funding_rate_history_hourly() -> List[Dict[str, Any]]:
    return [
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': -8e-06,
            'timestamp': 1630454400000,
            'datetime': '2021-09-01T00:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': -4e-06,
            'timestamp': 1630458000000,
            'datetime': '2021-09-01T01:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': 1.2e-05,
            'timestamp': 1630461600000,
            'datetime': '2021-09-01T02:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': -3e-06,
            'timestamp': 1630465200000,
            'datetime': '2021-09-01T03:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': -7e-06,
            'timestamp': 1630468800000,
            'datetime': '2021-09-01T04:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': 3e-06,
            'timestamp': 1630472400000,
            'datetime': '2021-09-01T05:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': 1.9e-05,
            'timestamp': 1630476000000,
            'datetime': '2021-09-01T06:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': 3e-06,
            'timestamp': 1630479600000,
            'datetime': '2021-09-01T07:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': -3e-06,
            'timestamp': 1630483200000,
            'datetime': '2021-09-01T08:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': 0,
            'timestamp': 1630486800000,
            'datetime': '2021-09-01T09:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': 1.3e-05,
            'timestamp': 1630490400000,
            'datetime': '2021-09-01T10:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': 7.7e-05,
            'timestamp': 1630494000000,
            'datetime': '2021-09-01T11:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': 7.2e-05,
            'timestamp': 1630497600000,
            'datetime': '2021-09-01T12:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': 9.7e-05,
            'timestamp': 1630501200000,
            'datetime': '2021-09-01T13:00:00.000Z'
        }
    ]


@pytest.fixture(scope='function')
def funding_rate_history_octohourly() -> List[Dict[str, Any]]:
    return [
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': -8e-06,
            'timestamp': 1630454400000,
            'datetime': '2021-09-01T00:00:00.000Z'
        },
        {
            'symbol': 'ADA/USDT:USDT',
            'fundingRate': -3e-06,
            'timestamp': 1630483200000,
            'datetime': '2021-09-01T08:00:00.000Z'
        }
    ]


@pytest.fixture(scope='function')
def leverage_tiers() -> Dict[str, List[Dict[str, Union[float, int]]]]:
    return {
        '1000SHIB/USDT:USDT': [
            {'minNotional': 0, 'maxNotional': 50000, 'maintenanceMarginRate': 0.01, 'maxLeverage': 50, 'maintAmt': 0.0},
            {'minNotional': 50000, 'maxNotional': 150000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 20, 'maintAmt': 750.0},
            {'minNotional': 150000, 'maxNotional': 250000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'maintAmt': 4500.0},
            {'minNotional': 250000, 'maxNotional': 500000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'maintAmt': 17000.0},
            {'minNotional': 500000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.125, 'maxLeverage': 4, 'maintAmt': 29500.0},
            {'minNotional': 1000000, 'maxNotional': 2000000, 'maintenanceMarginRate': 0.25, 'maxLeverage': 2, 'maintAmt': 154500.0},
            {'minNotional': 2000000, 'maxNotional': 30000000, 'maintenanceMarginRate': 0.5, 'maxLeverage': 1, 'maintAmt': 654500.0}
        ],
        '1INCH/USDT:USDT': [
            {'minNotional': 0, 'maxNotional': 5000, 'maintenanceMarginRate': 0.012, 'maxLeverage': 50, 'maintAmt': 0.0},
            {'minNotional': 5000, 'maxNotional': 25000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 20, 'maintAmt': 65.0},
            {'minNotional': 25000, 'maxNotional': 100000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'maintAmt': 690.0},
            {'minNotional': 100000, 'maxNotional': 250000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'maintAmt': 5690.0},
            {'minNotional': 250000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.125, 'maxLeverage': 2, 'maintAmt': 11940.0},
            {'minNotional': 1000000, 'maxNotional': 100000000, 'maintenanceMarginRate': 0.5, 'maxLeverage': 1, 'maintAmt': 386940.0}
        ],
        'AAVE/USDT:USDT': [
            {'minNotional': 0, 'maxNotional': 5000, 'maintenanceMarginRate': 0.01, 'maxLeverage': 50, 'maintAmt': 0.0},
            {'minNotional': 5000, 'maxNotional': 25000, 'maintenanceMarginRate': 0.02, 'maxLeverage': 25, 'maintAmt': 75.0},
            {'minNotional': 25000, 'maxNotional': 100000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'maintAmt': 700.0},
            {'minNotional': 100000, 'maxNotional': 250000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'maintAmt': 5700.0},
            {'minNotional': 250000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.125, 'maxLeverage': 4, 'maintAmt': 108035.0},
            {'minNotional': 10000000, 'maxNotional': 50000000, 'maintenanceMarginRate': 0.5, 'maxLeverage': 1, 'maintAmt': 386950.0}
        ],
        'ADA/USDT:USDT': [
            {'minNotional': 0, 'maxNotional': 100000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 20, 'maintAmt': 0.0},
            {'minNotional': 100000, 'maxNotional': 500000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'maintAmt': 2500.0},
            {'minNotional': 500000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'maintAmt': 27500.0},
            {'minNotional': 1000000, 'maxNotional': 2000000, 'maintenanceMarginRate': 0.15, 'maxLeverage': 3, 'maintAmt': 77500.0},
            {'minNotional': 2000000, 'maxNotional': 5000000, 'maintenanceMarginRate': 0.25, 'maxLeverage': 2, 'maintAmt': 277500.0},
            {'minNotional': 5000000, 'maxNotional': 30000000, 'maintenanceMarginRate': 0.5, 'maxLeverage': 1, 'maintAmt': 1527500.0}
        ],
        'XRP/USDT:USDT': [
            {'minNotional': 0, 'maxNotional': 100000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 20, 'maintAmt': 0.0},
            {'minNotional': 100000, 'maxNotional': 500000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'maintAmt': 2500.0},
            {'minNotional': 500000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'maintAmt': 27500.0},
            {'minNotional': 1000000, 'maxNotional': 2000000, 'maintenanceMarginRate': 0.15, 'maxLeverage': 3, 'maintAmt': 77500.0},
            {'minNotional': 2000000, 'maxNotional': 5000000, 'maintenanceMarginRate': 0.25, 'maxLeverage': 2, 'maintAmt': 277500.0},
            {'minNotional': 5000000, 'maxNotional': 30000000, 'maintenanceMarginRate': 0.5, 'maxLeverage': 1, 'maintAmt': 1527500.0}
        ],
        'BNB/USDT:USDT': [
            {'minNotional': 0, 'maxNotional': 10000, 'maintenanceMarginRate': 0.0065, 'maxLeverage': 75, 'maintAmt': 0.0},
            {'minNotional': 10000, 'maxNotional': 50000, 'maintenanceMarginRate': 0.01, 'maxLeverage': 50, 'maintAmt': 35.0},
            {'minNotional': 50000, 'maxNotional': 250000, 'maintenanceMarginRate': 0.02, 'maxLeverage': 25, 'maintAmt': 535.0},
            {'minNotional': 250000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'maintAmt': 8035.0},
            {'minNotional': 1000000, 'maxNotional': 20000000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'maintAmt': 58035.0},
            {'minNotional': 20000000, 'maxNotional': 50000000, 'maintenanceMarginRate': 0.5, 'maxLeverage': 1, 'maintAmt': 6233035.0}
        ],
        'BTC/USDT:USDT': [
            {'minNotional': 0, 'maxNotional': 50000, 'maintenanceMarginRate': 0.004, 'maxLeverage': 125, 'maintAmt': 0.0},
            {'minNotional': 50000, 'maxNotional': 250000, 'maintenanceMarginRate': 0.005, 'maxLeverage': 100, 'maintAmt': 50.0},
            {'minNotional': 250000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.01, 'maxLeverage': 50, 'maintAmt': 1300.0},
            {'minNotional': 1000000, 'maxNotional': 7500000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 20, 'maintAmt': 16300.0},
            {'minNotional': 7500000, 'maxNotional': 40000000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'maintAmt': 203800.0},
            {'minNotional': 40000000, 'maxNotional': 600000000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'maintAmt': 2203800.0},
            {'minNotional': 600000000, 'maxNotional': 1000000000, 'maintenanceMarginRate': 0.5, 'maxLeverage': 1, 'maintAmt': 199703800.0}
        ],
        'ZEC/USDT:USDT': [
            {'minNotional': 0, 'maxNotional': 50000, 'maintenanceMarginRate': 0.01, 'maxLeverage': 50, 'maintAmt': 0.0},
            {'minNotional': 50000, 'maxNotional': 150000, 'maintenanceMarginRate': 0.025, 'maxLeverage': 20, 'maintAmt': 750.0},
            {'minNotional': 150000, 'maxNotional': 250000, 'maintenanceMarginRate': 0.05, 'maxLeverage': 10, 'maintAmt': 4500.0},
            {'minNotional': 250000, 'maxNotional': 500000, 'maintenanceMarginRate': 0.1, 'maxLeverage': 5, 'maintAmt': 17000.0},
            {'minNotional': 500000, 'maxNotional': 1000000, 'maintenanceMarginRate': 0.125, 'maxLeverage': 4, 'maintAmt': 29500.0},
            {'minNotional': 1000000, 'maxNotional': 2000000, 'maintenanceMarginRate': 0.25, 'maxLeverage': 2, 'maintAmt': 154500.0},
            {'minNotional': 2000000, 'maxNotional': 30000000, 'maintenanceMarginRate': 0.5, 'maxLeverage': 1, 'maintAmt': 654500.0}
        ]
    }


@pytest.fixture
def limit_order_empty() -> Any:
    # Placeholder if needed
    pass


@pytest.fixture
def limit_order_data() -> Any:
    # Placeholder if needed
    pass


@pytest.fixture(scope='function')
def open_trade_with_orders() -> Trade:
    trade: Trade = Trade(
        pair='ETH/BTC',
        open_rate=1.099e-05,
        exchange='binance',
        amount=90.99181073,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=1,
        open_date=dt_now() - timedelta(minutes=601),
        is_open=True
    )
    trade.orders = [
        Order(
            ft_order_side='buy',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id='123456789',
            status='closed',
            symbol=trade.pair,
            order_type='market',
            side='buy',
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date
        )
    ]
    return trade


@pytest.fixture
def open_trade_usdt_with_exit() -> Trade:
    trade: Trade = Trade(
        pair='ADA/USDT',
        open_rate=2.0,
        exchange='binance',
        amount=30.0,
        fee_open=0.0,
        fee_close=0.0,
        stake_amount=60.0,
        open_date=dt_now() - timedelta(minutes=601),
        is_open=True
    )
    trade.orders = [
        Order(
            ft_order_side='buy',
            ft_pair=trade.pair,
            ft_is_open=False,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id='123456789',
            status='closed',
            symbol=trade.pair,
            order_type='market',
            side='buy',
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0.0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date
        ),
        Order(
            ft_order_side='exit',
            ft_pair=trade.pair,
            ft_is_open=True,
            ft_amount=trade.amount,
            ft_price=trade.open_rate,
            order_id='123456789_exit',
            status='open',
            symbol=trade.pair,
            order_type='limit',
            side='sell',
            price=trade.open_rate,
            average=trade.open_rate,
            filled=trade.amount,
            remaining=0.0,
            cost=trade.open_rate * trade.amount,
            order_date=trade.open_date,
            order_filled_date=trade.open_date
        )
    ]
    return trade
