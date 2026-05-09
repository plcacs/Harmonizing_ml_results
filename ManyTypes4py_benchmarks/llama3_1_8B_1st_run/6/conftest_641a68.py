import json
import logging
import platform
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock
import numpy as np
import pandas as pd
import pytest
from xdist.scheduler.loadscope import LoadScopeScheduling
from freqtrade import constants
from freqtrade import arguments
from freqtrade import configuration
from freqtrade import data
from freqtrade import enums
from freqtrade import exchange
from freqtrade import freqtradebot
from freqtrade import persistence
from freqtrade import resolvers
from freqtrade import util
from freqtrade import worker
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


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--longrun',
        action='store_true',
        dest='longrun',
        default=False,
        help='Enable long-run tests (ccxt compat)',
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line('markers', 'longrun: mark test that is running slowly and should not be run regularly')
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


def pytest_xdist_make_scheduler(config: pytest.Config, log: logging.Logger) -> LoadScopeScheduling:
    return FixtureScheduler(config, log)


def log_has(line: str, logs: pytest.LogCaptureFixture) -> bool:
    """Check if line is found on some caplog's message."""
    return any((line == message for message in logs.messages))


def log_has_when(line: str, logs: pytest.LogCaptureFixture, when: str) -> bool:
    """Check if line is found in caplog's messages during a specified stage"""
    return any((line == message.message for message in logs.get_records(when)))


def log_has_re(line: str, logs: pytest.LogCaptureFixture) -> bool:
    """Check if line matches some caplog's message."""
    return any((re.match(line, message) for message in logs.messages))


def num_log_has(line: str, logs: pytest.LogCaptureFixture) -> int:
    """Check how many times line is found in caplog's messages."""
    return sum((line == message for message in logs.messages))


def num_log_has_re(line: str, logs: pytest.LogCaptureFixture) -> int:
    """Check how many times line matches caplog's messages."""
    return sum((bool(re.match(line, message)) for message in logs.messages))


def get_args(args: str) -> arguments.Arguments:
    return arguments.Arguments(args).get_parsed_arg()


def generate_trades_history(
    n_rows: int,
    start_date: datetime | None = None,
    days: int = 5,
) -> pd.DataFrame:
    np.random.seed(42)
    if not start_date:
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = start_date + timedelta(days=days)
    _start_timestamp = start_date.timestamp()
    _end_timestamp = pd.to_datetime(end_date).timestamp()
    random_timestamps_in_seconds = np.random.uniform(_start_timestamp, _end_timestamp, n_rows)
    timestamp = pd.to_datetime(random_timestamps_in_seconds, unit='s')
    trade_id = [f'a{np.random.randint(1000000.0, 10000000.0 - 1)}cd{np.random.randint(100, 999)}' for _ in range(n_rows)]
    side = np.random.choice(['buy', 'sell'], n_rows)
    initial_price = 0.019626
    price_changes = np.random.normal(0, initial_price * 0.05, n_rows)
    price = np.cumsum(np.concatenate(([initial_price], price_changes)))[:n_rows]
    amount = np.random.uniform(0.011, 20, n_rows)
    cost = price * amount
    df = pd.DataFrame(
        {
            'timestamp': timestamp,
            'id': trade_id,
            'type': None,
            'side': side,
            'price': price,
            'amount': amount,
            'cost': cost,
        }
    )
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    assert list(df.columns) == constants.DEFAULT_TRADES_COLUMNS + ['date']
    return df


def generate_test_data(
    timeframe: str,
    size: int,
    start: str = '2020-07-05',
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)
    base = np.random.normal(20, 2, size=size)
    if timeframe == '1y':
        date = pd.date_range(start, periods=size, freq='1YS', tz='UTC')
    elif timeframe == '1M':
        date = pd.date_range(start, periods=size, freq='1MS', tz='UTC')
    elif timeframe == '3M':
        date = pd.date_range(start, periods=size, freq='3MS', tz='UTC')
    elif timeframe == '1w' or timeframe == '7d':
        date = pd.date_range(start, periods=size, freq='1W-MON', tz='UTC')
    else:
        tf_mins = timeframe_to_minutes(timeframe)
        if tf_mins >= 1:
            date = pd.date_range(start, periods=size, freq=f'{tf_mins}min', tz='UTC')
        else:
            tf_secs = timeframe_to_seconds(timeframe)
            date = pd.date_range(start, periods=size, freq=f'{tf_secs}s', tz='UTC')
    df = pd.DataFrame(
        {'date': date, 'open': base, 'high': base + np.random.normal(2, 1, size=size), 'low': base - np.random.normal(2, 1, size=size), 'close': base + np.random.normal(0, 1, size=size), 'volume': np.random.normal(200, size=size)}
    )
    df = df.dropna()
    return df


def generate_test_data_raw(
    timeframe: str,
    size: int,
    start: str = '2020-07-05',
    random_seed: int = 42,
) -> list:
    """Generates data in the ohlcv format used by ccxt"""
    df = generate_test_data(timeframe, size, start, random_seed)
    df['date'] = df.loc[:, 'date'].astype(np.int64) // 1000 // 1000
    return list((list(x) for x in zip(*(df[x].values.tolist() for x in df.columns), strict=False)))


def get_mock_coro(
    return_value: object | None = None,
    side_effect: object | list | None = None,
) -> Mock:
    async def mock_coro(*args, **kwargs):
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


def patched_configuration_load_config_file(
    mocker: pytest.Mock,
    config: dict,
) -> None:
    mocker.patch(
        'freqtrade.configuration.load_config.load_config_file',
        lambda *args, **kwargs: config,
    )


def patch_exchange(
    mocker: pytest.Mock,
    api_mock: exchange.Exchange | None = None,
    exchange: str = 'binance',
    mock_markets: bool | dict = True,
    mock_supported_modes: bool = True,
) -> None:
    mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.validate_config', MagicMock())
    mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.validate_timeframes', MagicMock())
    mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.id', PropertyMock(return_value=exchange))
    mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.name', PropertyMock(return_value=exchange.title()))
    mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.precisionMode', PropertyMock(return_value=2))
    mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.precision_mode_price', PropertyMock(return_value=2))
    mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.cache_leverage_tiers')
    if mock_markets:
        mocker.patch(
            f'{exchange.Exchange.__module__}.{exchange}.markets',
            PropertyMock(return_value=get_markets()),
        )
    if mock_supported_modes:
        mocker.patch(
            f'freqtrade.exchange.{exchange}.{exchange.capitalize()}._supported_trading_mode_margin_pairs',
            PropertyMock(return_value=[
                (enums.TradingMode.MARGIN, enums.MarginMode.CROSS),
                (enums.TradingMode.MARGIN, enums.MarginMode.ISOLATED),
                (enums.TradingMode.FUTURES, enums.MarginMode.CROSS),
                (enums.TradingMode.FUTURES, enums.MarginMode.ISOLATED),
            ]),
        )
    if api_mock:
        mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.get_fee', return_value=0.0025)
        mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.get_api', return_value=api_mock)
    else:
        mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.get_fee', return_value=0.0025)
        mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.get_api', MagicMock())
        mocker.patch(f'{exchange.Exchange.__module__}.{exchange}.timeframes', PropertyMock(return_value=['5m', '15m', '1h', '1d']))


def get_patched_exchange(
    mocker: pytest.Mock,
    config: dict,
    api_mock: exchange.Exchange | None = None,
    exchange: str = 'binance',
    mock_markets: bool | dict = True,
    mock_supported_modes: bool = True,
) -> exchange.Exchange:
    patch_exchange(mocker, api_mock, exchange, mock_markets, mock_supported_modes)
    config['exchange']['name'] = exchange
    try:
        exchange = resolvers.ExchangeResolver.load_exchange(config, load_leverage_tiers=True)
    except ImportError:
        exchange = exchange.Exchange(config)
    return exchange


def patch_wallet(mocker: pytest.Mock, free: float = 999.9) -> None:
    mocker.patch('freqtrade.wallets.Wallets.get_free', MagicMock(return_value=free))


def patch_whitelist(mocker: pytest.Mock, conf: dict) -> None:
    mocker.patch(
        'freqtrade.freqtradebot.FreqtradeBot._refresh_active_whitelist',
        MagicMock(return_value=conf['exchange']['pair_whitelist']),
    )


def patch_edge(mocker: pytest.Mock) -> None:
    mocker.patch(
        'freqtrade.edge.Edge._cached_pairs',
        PropertyMock(return_value={
            'NEO/BTC': enums.PairInfo(-0.2, 0.66, 3.71, 0.5, 1.71, 10, 25),
            'LTC/BTC': enums.PairInfo(-0.21, 0.66, 3.71, 0.5, 1.71, 11, 20),
        }),
    )
    mocker.patch('freqtrade.edge.Edge.calculate', MagicMock(return_value=True))


def patch_freqtradebot(
    mocker: pytest.Mock,
    config: dict,
) -> None:
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
    mocker.patch('freqtrade.configuration.config_validation._validate_consumers')


def get_patched_freqtradebot(
    mocker: pytest.Mock,
    config: dict,
) -> freqtradebot.FreqtradeBot:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: FreqtradeBot
    """
    patch_freqtradebot(mocker, config)
    return freqtradebot.FreqtradeBot(config)


def get_patched_worker(
    mocker: pytest.Mock,
    config: dict,
) -> worker.Worker:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: Worker
    """
    patch_freqtradebot(mocker, config)
    return worker.Worker(args=None, config=config)


def patch_get_signal(
    freqtrade: freqtradebot.FreqtradeBot,
    enter_long: bool = True,
    exit_long: bool = False,
    enter_short: bool = False,
    exit_short: bool = False,
    enter_tag: str | None = None,
    exit_tag: str | None = None,
) -> None:
    """
    :param mocker: mocker to patch IStrategy class
    :return: None
    """

    def patched_get_entry_signal(*args, **kwargs):
        direction = None
        if enter_long and (not any([exit_long, enter_short])):
            direction = enums.SignalDirection.LONG
        if enter_short and (not any([exit_short, enter_long])):
            direction = enums.SignalDirection.SHORT
        return (direction, enter_tag)

    freqtrade.strategy.get_entry_signal = patched_get_entry_signal

    def patched_get_exit_signal(
        pair: str,
        timeframe: str,
        dataframe: pd.DataFrame,
        is_short: bool,
    ) -> tuple:
        if is_short:
            return (enter_short, exit_short, exit_tag)
        else:
            return (enter_long, exit_long, exit_tag)

    freqtrade.strategy.get_exit_signal = patched_get_exit_signal
    freqtrade.exchange.refresh_latest_ohlcv = lambda p: None


def create_mock_trades(
    fee: float,
    is_short: bool | None = False,
    use_db: bool = True,
) -> None:
    """
    Create some fake trades ...
    :param is_short: Optional bool, None creates a mix of long and short trades.
    """

    def add_trade(trade: persistence.Trade):
        if use_db:
            persistence.Trade.session.add(trade)
        else:
            persistence.LocalTrade.add_bt_trade(trade)

    is_short1 = is_short if is_short is not None else True
    is_short2 = is_short if is_short is not None else False
    trade = mock_trade_1(fee, is_short1)
    add_trade(trade)
    trade = mock_trade_2(fee, is_short1)
    add_trade(trade)
    trade = mock_trade_3(fee, is_short2)
    add_trade(trade)
    trade = mock_trade_4(fee, is_short2)
    add_trade(trade)
    trade = mock_trade_5(fee, is_short2)
    add_trade(trade)
    trade = mock_trade_6(fee, is_short1)
    add_trade(trade)
    if use_db:
        persistence.Trade.commit()


def create_mock_trades_with_leverage(
    fee: float,
    use_db: bool = True,
) -> None:
    """
    Create some fake trades ...
    """
    if use_db:
        persistence.Trade.session.rollback()

    def add_trade(trade: persistence.Trade):
        if use_db:
            persistence.Trade.session.add(trade)
        else:
            persistence.LocalTrade.add_bt_trade(trade)

    trade = mock_trade_1(fee, False)
    add_trade(trade)
    trade = mock_trade_2(fee, False)
    add_trade(trade)
    trade = mock_trade_3(fee, False)
    add_trade(trade)
    trade = mock_trade_4(fee, False)
    add_trade(trade)
    trade = mock_trade_5(fee, False)
    add_trade(trade)
    trade = mock_trade_6(fee, False)
    add_trade(trade)
    trade = short_trade(fee)
    add_trade(trade)
    trade = leverage_trade(fee)
    add_trade(trade)
    if use_db:
        persistence.Trade.session.flush()


def create_mock_trades_usdt(
    fee: float,
    is_short: bool | None = False,
    use_db: bool = True,
) -> None:
    """
    Create some fake trades ...
    """

    def add_trade(trade: persistence.Trade):
        if use_db:
            persistence.Trade.session.add(trade)
        else:
            persistence.LocalTrade.add_bt_trade(trade)

    is_short1 = is_short if is_short is not None else True
    is_short2 = is_short if is_short is not None else False
    trade = mock_trade_usdt_1(fee, is_short1)
    add_trade(trade)
    trade = mock_trade_usdt_2(fee, is_short1)
    add_trade(trade)
    trade = mock_trade_usdt_3(fee, is_short1)
    add_trade(trade)
    trade = mock_trade_usdt_4(fee, is_short2)
    add_trade(trade)
    trade = mock_trade_usdt_5(fee, is_short2)
    add_trade(trade)
    trade = mock_trade_usdt_6(fee, is_short1)
    add_trade(trade)
    trade = mock_trade_usdt_7(fee, is_short1)
    add_trade(trade)
    if use_db:
        persistence.Trade.commit()


def is_arm() -> bool:
    machine = platform.machine()
    return 'arm' in machine or 'aarch64' in machine


def is_mac() -> bool:
    machine = platform.system()
    return 'Darwin' in machine


def get_default_conf(testdatadir: Path) -> dict:
    """Returns validated configuration suitable for most tests"""
    configuration = {
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
            'check_depth_of_market': {'enabled': False, 'bids_to_ask_delta': 1},
        },
        'exit_pricing': {'use_order_book': False, 'order_book_top': 1},
        'exchange': {
            'name': 'binance',
            'key': 'key',
            'enable_ws': False,
            'secret': 'secret',
            'pair_whitelist': ['ETH/BTC', 'LTC/BTC', 'XRP/BTC', 'NEO/BTC'],
            'pair_blacklist': ['DOGE/BTC', 'HOT/BTC'],
        },
        'pairlists': [{'method': 'StaticPairList'}],
        'telegram': {'enabled': False, 'token': 'token', 'chat_id': '1235', 'notification_settings': {}},
        'datadir': testdatadir,
        'initial_state': 'running',
        'db_url': 'sqlite://',
        'user_data_dir': Path('user_data'),
        'verbosity': 3,
        'strategy_path': str(Path(__file__).parent / 'strategy' / 'strats'),
        'strategy': 'StrategyTestV3',
        'disableparamexport': True,
        'internals': {},
        'export': 'none',
        'dataformat_ohlcv': 'feather',
        'dataformat_trades': 'feather',
        'runmode': 'dry_run',
        'trading_mode': 'spot',
        'margin_mode': '',
        'candle_type_def': enums.CandleType.SPOT,
    }
    return configuration


def get_default_conf_usdt(testdatadir: Path) -> dict:
    configuration = get_default_conf(testdatadir)
    configuration.update(
        {
            'stake_amount': 60.0,
            'stake_currency': 'USDT',
            'exchange': {
                'name': 'binance',
                'enabled': True,
                'key': 'key',
                'enable_ws': False,
                'secret': 'secret',
                'pair_whitelist': [
                    'ETH/USDT',
                    'LTC/USDT',
                    'XRP/USDT',
                    'NEO/USDT',
                    'TKN/USDT',
                ],
                'pair_blacklist': ['DOGE/USDT', 'HOT/USDT'],
            },
        }
    )
    return configuration


def get_patched_exchange(
    mocker: pytest.Mock,
    config: dict,
    api_mock: exchange.Exchange | None = None,
    exchange: str = 'binance',
    mock_markets: bool | dict = True,
    mock_supported_modes: bool = True,
) -> exchange.Exchange:
    patch_exchange(mocker, api_mock, exchange, mock_markets, mock_supported_modes)
    config['exchange']['name'] = exchange
    try:
        exchange = resolvers.ExchangeResolver.load_exchange(config, load_leverage_tiers=True)
    except ImportError:
        exchange = exchange.Exchange(config)
    return exchange


def get_patched_freqtradebot(
    mocker: pytest.Mock,
    config: dict,
) -> freqtradebot.FreqtradeBot:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: FreqtradeBot
    """
    patch_freqtradebot(mocker, config)
    return freqtradebot.FreqtradeBot(config)


def get_patched_worker(
    mocker: pytest.Mock,
    config: dict,
) -> worker.Worker:
    """
    This function patches _init_modules() to not call dependencies
    :param mocker: a Mocker object to apply patches
    :param config: Config to pass to the bot
    :return: Worker
    """
    patch_freqtradebot(mocker, config)
    return worker.Worker(args=None, config=config)


def get_markets() -> dict:
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
                'leverage': {'min': 1.0, 'max': 2.0},
            },
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
                'leverage': {'min': 1.0, 'max': 5.0},
            },
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
                'amount': {'min': 0.01, 'max': 100000000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000},
                'leverage': {'min': 1.0, 'max': 3.0},
            },
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
                'leverage': {'min': None, 'max': None},
            },
            'info': {},
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
                'leverage': {'min': None, 'max': None},
            },
            'info': {},
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
                'leverage': {'min': None, 'max': None},
            },
            'info': {},
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
            'precision': {'base': 8, 'quote': 8, 'amount': 0, 'price': 8},
            'limits': {
                'amount': {'min': 1.0, 'max': 90000000.0},
                'price': {'min': None, 'max': None},
                'cost': {'min': 0.0001, 'max': None},
                'leverage': {'min': None, 'max': None},
            },
            'info': {},
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
                'leverage': {'min': None, 'max': None},
            },
            'info': {},
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
            'inverse': None,
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
                'cost': {'min': None, 'max': None},
            },
            'info': {'maintenance_rate': '0.005'},
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
            'inverse': None,
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
                'cost': {'min': None, 'max': None},
            },
            'info': {'maintenance_rate': '0.005'},
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
                'amount': {'min': 0.06646786, 'max': None},
                'price': {'min': 1e-08, 'max': None},
                'leverage': {'min': None, 'max': None},
                'cost': {'min': None, 'max': None},
            },
            'info': {},
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
                'amount': {'min': 0.01, 'max': 1000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000},
            },
            'info': {},
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
            'type': 'spot',
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
                'cost': {'min': 0.0001, 'max': 500000},
            },
            'info': {},
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
                'amount': {'min': 0.01, 'max': 100000000000},
                'price': {'min': None, 'max': 500000},
                'cost': {'min': 0.0001, 'max': 500000},
                'leverage': {'min': None, 'max': None},
            },
            'info': {},
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
            'precision': {'amount': 8, 'price': 8},
            'limits': {
                'amount': {'min': 0.06646786, 'max': None},
                'price': {'min': 1e-08, 'max': None},
                'leverage': {'min': None, 'max': None},
                'cost': {'min': None, 'max': None},
            },
            'info': {},
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
                'cost': {'min': None, 'max': None},
            },
            'info': {},
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
                'cost': {'min': 0.01, 'max': None},
            },
            'info': {},
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
            'active': True,
            'expiry': None,
            'expiryDatetime': None,
            'strike': None,
            'optionType': None,
            'limits': {
                'leverage': {'min': 1, 'max': 100},
                'amount': {'min': 1, 'max': 300000},
                'price': {'min': None, 'max': None},
                'cost': {'min': None, 'max': None},
            },
            'precision': {'price': 0.05, 'amount': 1},
            'info': {},
        },
        'ADA/USDT:USDT': {
            'limits': {
                'leverage': {'min': 1, 'max': 20},
                'amount': {'min': 1, 'max': 1000000},
                'price': {'min': 0.52981, 'max': 1.58943},
                'cost': {'min': None, 'max': None},
            },
            'precision': {'amount': 1, 'price': 1e-05},
            'tierBased': True,
            'percentage': True,
            'taker': 7.5e-06,
            'maker': -2.5e-06,
            'feeSide': 'get',
            'tiers': {
                'maker': [[0, 0.002], [1.5, 0.00185], [3, 0.00175], [6, 0.00165], [12.5, 0.00155], [25, 0.00145], [75, 0.00135], [200, 0.00125], [500, 0.00115], [1250, 0.00105], [2500, 0.00095], [3000, 0.00085], [6000, 0.00075], [11000, 0.00065], [20000, 0.00055], [40000, 0.00055], [75000, 0.00055]],
                'taker': [[0, 0.002], [1.5, 0.00195], [3, 0.00185], [6, 0.00175], [12.5, 0.00165], [25, 0.00155], [75, 0.00145], [200, 0.00135], [500, 0.00125], [1250, 0.00115], [2500, 0.00105], [3000, 0.00095], [6000, 0.00085], [11000, 0.00075], [20000, 0.00065], [40000, 0.00065], [75000, 0.00065]],
            },
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
            'active': True,
            'contract': True,
            'linear': True,
            'inverse': False,
            'contractSize': 0.01,
            'expiry': None,
            'expiryDatetime': None,
            'strike': None,
            'optionType': None,
            'info': {},
        },
        'SOL/BUSD:BUSD': {
            'limits': {
                'leverage': {'min': None, 'max': None},
                'amount': {'min': 1, 'max': 1000000},
                'price': {'min': 0.04, 'max': 100000},
                'cost': {'min': 5, 'max': None},
                'market': {'min': 1, 'max': 1500},
            },
            'precision': {'amount': 0, 'price': 2, 'base': 8, 'quote': 8},
            'tierBased': False,
            'percentage': True,
            'taker': 0.0004,
            'maker': 0.0002,
            'feeSide': 'get',
            'id': 'SOLBUSD',
            'lowercaseId': 'solbusd',
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
                    {
                        'multiplierDown': '0.9500',
                        'multiplierUp': '1.0500',
                        'multiplierDecimal': '4',
                        'filterType': 'PERCENT_PRICE',
                    },
                ],
                'orderTypes': ['LIMIT', 'MARKET', 'STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET'],
                'timeInForce': ['GTC', 'IOC', 'FOK', 'GTX'],
            },
        },
    }
