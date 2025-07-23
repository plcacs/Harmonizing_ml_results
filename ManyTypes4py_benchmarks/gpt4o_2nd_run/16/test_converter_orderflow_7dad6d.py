import pandas as pd
import pytest
from freqtrade.constants import DEFAULT_TRADES_COLUMNS
from freqtrade.data.converter import populate_dataframe_with_trades
from freqtrade.data.converter.orderflow import ORDERFLOW_ADDED_COLUMNS, stacked_imbalance, timeframe_to_DateOffset, trades_to_volumeprofile_with_total_delta_bid_ask
from freqtrade.data.converter.trade_converter import trades_list_to_df
from freqtrade.data.dataprovider import DataProvider
from tests.strategy.strats.strategy_test_v3 import StrategyTestV3
from typing import Dict, List, Tuple
from pandas import DataFrame, Series

BIN_SIZE_SCALE: float = 0.5

def read_csv(filename: str) -> DataFrame:
    converter_columns: List[str] = ['side', 'type']
    return pd.read_csv(filename, skipinitialspace=True, index_col=0, parse_dates=True, date_format='ISO8601', converters={col: str.strip for col in converter_columns})

@pytest.fixture
def populate_dataframe_with_trades_dataframe(testdatadir: str) -> DataFrame:
    return pd.read_feather(testdatadir / 'orderflow/populate_dataframe_with_trades_DF.feather')

@pytest.fixture
def populate_dataframe_with_trades_trades(testdatadir: str) -> DataFrame:
    return pd.read_feather(testdatadir / 'orderflow/populate_dataframe_with_trades_TRADES.feather')

@pytest.fixture
def candles(testdatadir: str) -> DataFrame:
    return pd.read_json(testdatadir / 'orderflow/candles.json').copy()

@pytest.fixture
def public_trades_list(testdatadir: str) -> DataFrame:
    return read_csv(testdatadir / 'orderflow/public_trades_list.csv').copy()

@pytest.fixture
def public_trades_list_simple(testdatadir: str) -> DataFrame:
    return read_csv(testdatadir / 'orderflow/public_trades_list_simple_example.csv').copy()

def test_public_trades_columns_before_change(populate_dataframe_with_trades_dataframe: DataFrame, populate_dataframe_with_trades_trades: DataFrame) -> None:
    assert populate_dataframe_with_trades_dataframe.columns.tolist() == ['date', 'open', 'high', 'low', 'close', 'volume']
    assert populate_dataframe_with_trades_trades.columns.tolist() == ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost', 'date']

def test_public_trades_mock_populate_dataframe_with_trades__check_orderflow(populate_dataframe_with_trades_dataframe: DataFrame, populate_dataframe_with_trades_trades: DataFrame) -> None:
    dataframe: DataFrame = populate_dataframe_with_trades_dataframe.copy()
    trades: DataFrame = populate_dataframe_with_trades_trades.copy()
    dataframe['date'] = pd.to_datetime(dataframe['date'], unit='ms')
    dataframe = dataframe.copy().tail().reset_index(drop=True)
    config: Dict[str, Dict[str, float]] = {'timeframe': '5m', 'orderflow': {'cache_size': 1000, 'max_candles': 1500, 'scale': 0.005, 'imbalance_volume': 0, 'imbalance_ratio': 3, 'stacked_imbalance_range': 3}}
    df, _ = populate_dataframe_with_trades(None, config, dataframe, trades)
    results: Series = df.iloc[0]
    t: List[Dict[str, float]] = results['trades']
    of: Dict[str, float] = results['orderflow']
    assert 0 != len(results)
    assert 151 == len(t)
    assert 23 == len(of)
    assert isinstance(of, dict)
    of_values: List[Dict[str, float]] = list(of.values())
    assert of_values[0] == {'bid': 0.0, 'ask': 1.0, 'delta': 4.999, 'bid_amount': 0.0, 'ask_amount': 4.999, 'total_volume': 4.999, 'total_trades': 1}
    assert of_values[-1] == {'bid': 0.0, 'ask': 1.0, 'delta': 0.103, 'bid_amount': 0.0, 'ask_amount': 0.103, 'total_volume': 0.103, 'total_trades': 1}
    of = df.iloc[-1]['orderflow']
    assert 19 == len(of)
    of_values1: List[Dict[str, float]] = list(of.values())
    assert of_values1[0] == {'bid': 1.0, 'ask': 0.0, 'delta': -12.536, 'bid_amount': 12.536, 'ask_amount': 0.0, 'total_volume': 12.536, 'total_trades': 1}
    assert pytest.approx(of_values1[-1]) == {'bid': 4.0, 'ask': 3.0, 'delta': -40.948, 'bid_amount': 59.182, 'ask_amount': 18.23399, 'total_volume': 77.416, 'total_trades': 7}
    assert pytest.approx(results['delta']) == -50.519
    assert results['min_delta'] == -79.469
    assert results['max_delta'] == 17.298
    assert results['stacked_imbalances_bid'] == []
    assert results['stacked_imbalances_ask'] == []
    results = df.iloc[-2]
    assert pytest.approx(results['delta']) == -20.862
    assert pytest.approx(results['min_delta']) == -54.559999
    assert 82.842 == results['max_delta']
    assert results['stacked_imbalances_bid'] == [234.97]
    assert results['stacked_imbalances_ask'] == [234.94]
    results = df.iloc[-1]
    assert pytest.approx(results['delta']) == -49.302
    assert results['min_delta'] == -70.222
    assert pytest.approx(results['max_delta']) == 11.213
    assert results['stacked_imbalances_bid'] == []
    assert results['stacked_imbalances_ask'] == []

def test_public_trades_trades_mock_populate_dataframe_with_trades__check_trades(populate_dataframe_with_trades_dataframe: DataFrame, populate_dataframe_with_trades_trades: DataFrame) -> None:
    dataframe: DataFrame = populate_dataframe_with_trades_dataframe.copy()
    trades: DataFrame = populate_dataframe_with_trades_trades.copy()
    dataframe['date'] = pd.to_datetime(dataframe['date'], unit='ms')
    dataframe = dataframe.tail().reset_index(drop=True)
    trades = trades.loc[trades.date >= dataframe.date[0]]
    trades.reset_index(inplace=True, drop=True)
    assert trades['id'][0] == '313881442'
    config: Dict[str, Dict[str, float]] = {'timeframe': '5m', 'orderflow': {'cache_size': 1000, 'max_candles': 1500, 'scale': 0.5, 'imbalance_volume': 0, 'imbalance_ratio': 3, 'stacked_imbalance_range': 3}}
    df, _ = populate_dataframe_with_trades(None, config, dataframe, trades)
    row: Series = df.iloc[0]
    assert list(df.columns) == ['date', 'open', 'high', 'low', 'close', 'volume', 'trades', 'orderflow', 'imbalances', 'stacked_imbalances_bid', 'stacked_imbalances_ask', 'max_delta', 'min_delta', 'bid', 'ask', 'delta', 'total_trades']
    assert pytest.approx(row['delta']) == -50.519
    assert row['bid'] == 219.961
    assert row['ask'] == 169.442
    assert len(row['trades']) == 151
    t: Dict[str, float] = row['trades'][0]
    assert list(t.keys()) == ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost', 'date']
    assert trades['id'][0] == t['id']
    assert int(trades['timestamp'][0]) == int(t['timestamp'])
    assert t['side'] == 'sell'
    assert t['id'] == '313881442'
    assert t['price'] == 234.72

def test_public_trades_put_volume_profile_into_ohlcv_candles(public_trades_list_simple: DataFrame, candles: DataFrame) -> None:
    trades_df: DataFrame = trades_list_to_df(public_trades_list_simple[DEFAULT_TRADES_COLUMNS].values.tolist())
    df: DataFrame = trades_to_volumeprofile_with_total_delta_bid_ask(trades_df, scale=BIN_SIZE_SCALE)
    assert 0.14 == df.values.tolist()[1][2]
    assert 0.14 == df['delta'].iat[1]

def test_public_trades_binned_big_sample_list(public_trades_list: DataFrame) -> None:
    BIN_SIZE_SCALE: float = 0.05
    trades: DataFrame = trades_list_to_df(public_trades_list[DEFAULT_TRADES_COLUMNS].values.tolist())
    df: DataFrame = trades_to_volumeprofile_with_total_delta_bid_ask(trades, scale=BIN_SIZE_SCALE)
    assert df.columns.tolist() == ['bid', 'ask', 'delta', 'bid_amount', 'ask_amount', 'total_volume', 'total_trades']
    assert len(df) == 23
    assert all((df.index[i] < df.index[i + 1] for i in range(len(df) - 1)))
    assert df.index[0] + BIN_SIZE_SCALE == df.index[1]
    assert trades['price'].min() - BIN_SIZE_SCALE < df.index[0] < trades['price'].max()
    assert df.index[0] + BIN_SIZE_SCALE >= df.index[1]
    assert trades['price'].max() - BIN_SIZE_SCALE < df.index[-1] < trades['price'].max()
    assert 32 == df['bid'].iloc[0]
    assert 197.512 == df['bid_amount'].iloc[0]
    assert 88.98 == df['ask_amount'].iloc[0]
    assert 26 == df['ask'].iloc[0]
    assert -108.532 == pytest.approx(df['delta'].iloc[0])
    assert 3 == df['bid'].iloc[-1]
    assert 50.659 == df['bid_amount'].iloc[-1]
    assert 108.21 == df['ask_amount'].iloc[-1]
    assert 44 == df['ask'].iloc[-1]
    assert 57.551 == df['delta'].iloc[-1]
    BIN_SIZE_SCALE = 1
    df = trades_to_volumeprofile_with_total_delta_bid_ask(trades, scale=BIN_SIZE_SCALE)
    assert len(df) == 2
    assert all((df.index[i] < df.index[i + 1] for i in range(len(df) - 1)))
    assert trades['price'].min() - BIN_SIZE_SCALE < df.index[0] < trades['price'].max()
    assert df.index[0] + BIN_SIZE_SCALE >= df.index[1]
    assert trades['price'].max() - BIN_SIZE_SCALE < df.index[-1] < trades['price'].max()
    assert 1667.0 == df.index[-1]
    assert 710.98 == df['bid_amount'].iat[0]
    assert 111 == df['bid'].iat[0]
    assert 52.7199999 == pytest.approx(df['delta'].iat[0])

def test_public_trades_config_max_trades(default_conf: Dict[str, Dict[str, float]], populate_dataframe_with_trades_dataframe: DataFrame, populate_dataframe_with_trades_trades: DataFrame) -> None:
    dataframe: DataFrame = populate_dataframe_with_trades_dataframe.copy()
    trades: DataFrame = populate_dataframe_with_trades_trades.copy()
    default_conf['exchange']['use_public_trades'] = True
    orderflow_config: Dict[str, Dict[str, float]] = {'timeframe': '5m', 'orderflow': {'cache_size': 1000, 'max_candles': 1, 'scale': 0.005, 'imbalance_volume': 0, 'imbalance_ratio': 3, 'stacked_imbalance_range': 3}}
    df, _ = populate_dataframe_with_trades(None, default_conf | orderflow_config, dataframe, trades)
    assert df.delta.count() == 1

def test_public_trades_testdata_sanity(candles: DataFrame, public_trades_list: DataFrame, public_trades_list_simple: DataFrame, populate_dataframe_with_trades_dataframe: DataFrame, populate_dataframe_with_trades_trades: DataFrame) -> None:
    assert 10999 == len(candles)
    assert 1000 == len(public_trades_list)
    assert 999 == len(populate_dataframe_with_trades_dataframe)
    assert 293532 == len(populate_dataframe_with_trades_trades)
    assert 7 == len(public_trades_list_simple)
    assert 5 == public_trades_list_simple.loc[public_trades_list_simple['side'].str.contains('sell'), 'id'].count()
    assert 2 == public_trades_list_simple.loc[public_trades_list_simple['side'].str.contains('buy'), 'id'].count()
    assert public_trades_list.columns.tolist() == ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost', 'date']
    assert public_trades_list.columns.tolist() == ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost', 'date']
    assert public_trades_list_simple.columns.tolist() == ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost', 'date']
    assert populate_dataframe_with_trades_dataframe.columns.tolist() == ['date', 'open', 'high', 'low', 'close', 'volume']
    assert populate_dataframe_with_trades_trades.columns.tolist() == ['timestamp', 'id', 'type', 'side', 'price', 'amount', 'cost', 'date']

def test_analyze_with_orderflow(default_conf_usdt: Dict[str, Dict[str, float]], mocker, populate_dataframe_with_trades_dataframe: DataFrame, populate_dataframe_with_trades_trades: DataFrame) -> None:
    ohlcv_history: DataFrame = populate_dataframe_with_trades_dataframe
    strategy: StrategyTestV3 = StrategyTestV3(config=default_conf_usdt)
    strategy.dp = DataProvider(default_conf_usdt, None, None)
    mocker.patch.object(strategy.dp, 'trades', return_value=populate_dataframe_with_trades_trades)
    import freqtrade.data.converter.orderflow as orderflow_module
    spy = mocker.spy(orderflow_module, 'trades_to_volumeprofile_with_total_delta_bid_ask')
    pair: str = 'ETH/BTC'
    df: DataFrame = strategy.advise_indicators(ohlcv_history, {'pair:': pair})
    assert len(df) == len(ohlcv_history)
    assert 'open' in df.columns
    assert spy.call_count == 0
    for col in ORDERFLOW_ADDED_COLUMNS:
        assert col not in df.columns, f'Column {col} found in df.columns'
    default_conf_usdt['exchange']['use_public_trades'] = True
    default_conf_usdt['orderflow'] = {'cache_size': 5, 'max_candles': 5, 'scale': 0.005, 'imbalance_volume': 0, 'imbalance_ratio': 3, 'stacked_imbalance_range': 3}
    strategy.config = default_conf_usdt
    df1: DataFrame = strategy.advise_indicators(ohlcv_history, {'pair': pair})
    assert len(df1) == len(ohlcv_history)
    assert 'open' in df1.columns
    assert spy.call_count == 5
    for col in ORDERFLOW_ADDED_COLUMNS:
        assert col in df1.columns, f'Column {col} not found in df.columns'
        if col not in ('stacked_imbalances_bid', 'stacked_imbalances_ask'):
            assert df1[col].count() == 5, f'Column {col} has {df1[col].count()} non-NaN values'
    assert len(strategy._cached_grouped_trades_per_pair[pair]) == 5
    lastval_trades: List[Dict[str, float]] = df1.at[len(df1) - 1, 'trades']
    assert isinstance(lastval_trades, list)
    assert len(lastval_trades) == 122
    lastval_of: Dict[str, float] = df1.at[len(df1) - 1, 'orderflow']
    assert isinstance(lastval_of, dict)
    spy.reset_mock()
    df2: DataFrame = strategy.advise_indicators(ohlcv_history, {'pair': pair})
    assert len(df2) == len(ohlcv_history)
    assert 'open' in df2.columns
    assert spy.call_count == 0
    for col in ORDERFLOW_ADDED_COLUMNS:
        assert col in df2.columns, f'Round2: Column {col} not found in df.columns'
        if col not in ('stacked_imbalances_bid', 'stacked_imbalances_ask'):
            assert df2[col].count() == 5, f'Round2: Column {col} has {df2[col].count()} non-NaN values'
    lastval_trade2: List[Dict[str, float]] = df2.at[len(df2) - 1, 'trades']
    assert isinstance(lastval_trade2, list)
    assert len(lastval_trade2) == 122
    lastval_of2: Dict[str, float] = df2.at[len(df2) - 1, 'orderflow']
    assert isinstance(lastval_of2, dict)

def test_stacked_imbalances_multiple_prices() -> None:
    df_no_stacks: DataFrame = pd.DataFrame({'bid_imbalance': [False, False, True, False], 'ask_imbalance': [False, True, False, False]}, index=[234.95, 234.96, 234.97, 234.98])
    no_stacks: List[float] = stacked_imbalance(df_no_stacks, 'bid', stacked_imbalance_range=2)
    assert no_stacks == []
    df: DataFrame = pd.DataFrame({'bid_imbalance': [True, True, True, False, False, True, True, False, True], 'ask_imbalance': [False, False, True, True, True, False, False, True, True]}, index=[234.95, 234.96, 234.97, 234.98, 234.99, 235.0, 235.01, 235.02, 235.03])
    bid_prices: List[float] = stacked_imbalance(df, 'bid', stacked_imbalance_range=2)
    assert bid_prices == [234.95, 234.96, 235.0]
    ask_prices: List[float] = stacked_imbalance(df, 'ask', stacked_imbalance_range=2)
    assert ask_prices == [234.97, 234.98, 235.02]
    bid_prices_higher: List[float] = stacked_imbalance(df, 'bid', stacked_imbalance_range=3)
    assert bid_prices_higher == [234.95]

def test_timeframe_to_DateOffset() -> None:
    assert timeframe_to_DateOffset('1s') == pd.DateOffset(seconds=1)
    assert timeframe_to_DateOffset('1m') == pd.DateOffset(minutes=1)
    assert timeframe_to_DateOffset('5m') == pd.DateOffset(minutes=5)
    assert timeframe_to_DateOffset('1h') == pd.DateOffset(hours=1)
    assert timeframe_to_DateOffset('1d') == pd.DateOffset(days=1)
    assert timeframe_to_DateOffset('1w') == pd.DateOffset(weeks=1)
    assert timeframe_to_DateOffset('1M') == pd.DateOffset(months=1)
    assert timeframe_to_DateOffset('1y') == pd.DateOffset(years=1)
