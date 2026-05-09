import logging
from shutil import copyfile
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from freqtrade.configuration.timerange import TimeRange
from freqtrade.data.converter import convert_ohlcv_format, convert_trades_format, convert_trades_to_ohlcv, ohlcv_fill_up_missing_data, ohlcv_to_dataframe, reduce_dataframe_footprint, trades_df_remove_duplicates, trades_dict_to_list, trades_to_ohlcv, trim_dataframe
from freqtrade.data.history import get_timerange, load_data, load_pair_history, validate_backtest_data
from freqtrade.data.history.datahandlers import IDataHandler
from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_seconds
from tests.conftest import generate_test_data, generate_trades_history, log_has, log_has_re
from tests.data.test_history import _clean_test_file

def test_dataframe_correct_columns(dataframe_1m: pd.DataFrame) -> None:
    assert dataframe_1m.columns.tolist() == ['date', 'open', 'high', 'low', 'close', 'volume']

def test_ohlcv_to_dataframe(ohlcv_history_list: List[Any], caplog: Any) -> None:
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    caplog.set_level(logging.DEBUG)
    dataframe = ohlcv_to_dataframe(ohlcv_history_list, '5m', pair='UNITTEST/BTC', fill_missing=True)
    assert dataframe.columns.tolist() == columns
    assert log_has('Converting candle (OHLCV) data to dataframe for pair UNITTEST/BTC.', caplog)

def test_trades_to_ohlcv(trades_history_df: pd.DataFrame, caplog: Any) -> None:
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ValueError, match='Trade-list empty.'):
        trades_to_ohlcv(pd.DataFrame(columns=trades_history_df.columns), '1m')
    df = trades_to_ohlcv(trades_history_df, '1m')
    assert not df.empty
    assert len(df) == 1
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'close' in df.columns
    assert df.iloc[0, :]['high'] == 0.019627
    assert df.iloc[0, :]['low'] == 0.019626
    assert df.iloc[0, :]['date'] == pd.Timestamp('2019-08-14 15:59:00+0000')
    df_1h = trades_to_ohlcv(trades_history_df, '1h')
    assert len(df_1h) == 1
    assert df_1h.iloc[0, :]['high'] == 0.019627
    assert df_1h.iloc[0, :]['low'] == 0.019626
    assert df_1h.iloc[0, :]['date'] == pd.Timestamp('2019-08-14 15:00:00+0000')
    df_1s = trades_to_ohlcv(trades_history_df, '1s')
    assert len(df_1s) == 2
    assert df_1s.iloc[0, :]['high'] == 0.019627
    assert df_1s.iloc[0, :]['low'] == 0.019627
    assert df_1s.iloc[0, :]['date'] == pd.Timestamp('2019-08-14 15:59:49+0000')
    assert df_1s.iloc[-1, :]['date'] == pd.Timestamp('2019-08-14 15:59:59+0000')

@pytest.mark.parametrize('timeframe,rows,days,candles,start,end,weekday', [('1s', 20000, 5, 19522, '2020-01-01 00:00:05', '2020-01-05 23:59:27', None), ('1m', 20000, 5, 6745, '2020-01-01 00:00:00', '2020-01-05 23:59:00', None), ('5m', 20000, 5, 1440, '2020-01-01 00:00:00', '2020-01-05 23:55:00', None), ('15m', 20000, 5, 480, '2020-01-01 00:00:00', '2020-01-05 23:45:00', None), ('1h', 20000, 5, 120, '2020-01-01 00:00:00', '2020-01-05 23:00:00', None), ('2h', 20000, 5, 60, '2020-01-01 00:00:00', '2020-01-05 22:00:00', None), ('4h', 20000, 5, 30, '2020-01-01 00:00:00', '2020-01-05 20:00:00', None), ('8h', 20000, 5, 15, '2020-01-01 00:00:00', '2020-01-05 16:00:00', None), ('12h', 20000, 5, 10, '2020-01-01 00:00:00', '2020-01-05 12:00:00', None), ('1d', 20000, 5, 5, '2020-01-01 00:00:00', '2020-01-05 00:00:00', 'Sunday'), ('7d', 20000, 37, 6, '2020-01-06 00:00:00', '2020-02-10 00:00:00', 'Monday'), ('1w', 20000, 37, 6, '2020-01-06 00:00:00', '2020-02-10 00:00:00', 'Monday'), ('1M', 20000, 74, 3, '2020-01-01 00:00:00', '2020-03-01 00:00:00', None), ('3M', 20000, 100, 2, '2020-01-01 00:00:00', '2020-04-01 00:00:00', None), ('1y', 20000, 1000, 3, '2020-01-01 00:00:00', '2022-01-01 00:00:00', None)])
def test_trades_to_ohlcv_multi(timeframe: str, rows: int, days: int, candles: int, start: str, end: str, weekday: Optional[str]) -> None:
    trades_history = generate_trades_history(n_rows=rows, days=days)
    df = trades_to_ohlcv(trades_history, timeframe)
    assert not df.empty
    assert len(df) == candles
    assert df.iloc[0, :]['date'] == pd.Timestamp(f'{start}+0000')
    assert df.iloc[-1, :]['date'] == pd.Timestamp(f'{end}+0000')
    if weekday:
        assert df.iloc[-1, :]['date'].day_name() == weekday

def test_ohlcv_fill_up_missing_data(testdatadir: Any, caplog: Any) -> None:
    data = load_pair_history(datadir=testdatadir, timeframe='1m', pair='UNITTEST/BTC', fill_up_missing=False)
    caplog.set_level(logging.DEBUG)
    data2 = ohlcv_fill_up_missing_data(data, '1m', 'UNITTEST/BTC')
    assert len(data2) > len(data)
    assert (data.columns == data2.columns).all()
    assert log_has_re(f'Missing data fillup for UNITTEST/BTC, 1m: before: {len(data)} - after: {len(data2)}.*', caplog)
    min_date, max_date = get_timerange({'UNITTEST/BTC': data})
    assert validate_backtest_data(data, 'UNITTEST/BTC', min_date, max_date, 1)
    assert not validate_backtest_data(data2, 'UNITTEST/BTC', min_date, max_date, 1)

def test_ohlcv_fill_up_missing_data2(caplog: Any) -> None:
    timeframe = '5m'
    ticks = [[1511686200000, 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 2255], [1511686500000, 8.88e-05, 8.942e-05, 8.88e-05, 8.893e-05, 9911], [1511687100000, 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05, 2251], [1511687400000, 8.877e-05, 8.883e-05, 8.895e-05, 8.817e-05, 123551]]
    data = ohlcv_to_dataframe(ticks, timeframe, pair='UNITTEST/BTC', fill_missing=False)
    assert len(data) == 3
    caplog.set_level(logging.DEBUG)
    data2 = ohlcv_fill_up_missing_data(data, timeframe, 'UNITTEST/BTC')
    assert len(data2) == 4
    row = data2.loc[2, :]
    assert row['volume'] == 0
    assert row['close'] == data.loc[1, 'close']
    assert row['open'] == row['close']
    assert row['high'] == row['close']
    assert row['low'] == row['close']
    assert (data.columns == data2.columns).all()
    assert log_has_re(f'Missing data fillup for UNITTEST/BTC, {timeframe}: before: {len(data)} - after: {len(data2)}.*', caplog)

@pytest.mark.parametrize('timeframe', ['1s', '1m', '5m', '15m', '1h', '2h', '4h', '8h', '12h', '1d', '7d', '1w', '1M', '3M', '1y'])
def test_ohlcv_to_dataframe_multi(timeframe: str) -> None:
    data = generate_test_data(timeframe, 180)
    assert len(data) == 180
    df = ohlcv_to_dataframe(data, timeframe, 'UNITTEST/USDT')
    assert len(df) == len(data) - 1
    df1 = ohlcv_to_dataframe(data, timeframe, 'UNITTEST/USDT', drop_incomplete=False)
    assert len(df1) == len(data)
    assert data.equals(df1)
    data1 = data.copy()
    if timeframe in ('1M', '3M', '1y'):
        data1.loc[:, 'date'] = data1.loc[:, 'date'] + pd.to_timedelta('1w')
    else:
        data1.loc[:, 'date'] = data1.loc[:, 'date'] + pd.to_timedelta(timeframe) / 2
    df2 = ohlcv_to_dataframe(data1, timeframe, 'UNITTEST/USDT')
    assert len(df2) == len(data) - 1
    tfs = timeframe_to_seconds(timeframe)
    tfm = timeframe_to_minutes(timeframe)
    if 1 <= tfm < 10000:
        ohlcv_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        dfs = data1.resample(f'{tfs}s', on='date').agg(ohlcv_dict).reset_index(drop=False)
        dfm = data1.resample(f'{tfm}min', on='date').agg(ohlcv_dict).reset_index(drop=False)
        assert dfs.equals(dfm)
        assert dfs.equals(df1)

def test_ohlcv_to_dataframe_1M() -> None:
    ticks = [[1567296000000, 8042.08, 10475.54, 7700.67, 8041.96, 608742.1109999999], [1569888000000, 8285.31, 10408.48, 7172.76, 9150.0, 2439561.887], [1572566400000, 9149.88, 9550.0, 6510.19, 7542.93, 4042674.725], [1575158400000, 7541.08, 7800.0, 6427.0, 7189.0, 4063882.296], [1577836800000, 7189.43, 9599.0, 6863.44, 9364.51, 5165281.358], [1580515200000, 9364.5, 10540.0, 8450.0, 8531.98, 4581788.124], [1583020800000, 8532.5, 9204.0, 3621.81, 6407.1, 10859497.479], [1585699200000, 6407.1, 9479.77, 6140.0, 8624.76, 11276526.968], [1588291200000, 8623.61, 10080.0, 7940.0, 9446.43, 12469561.02], [1590969600000, 9446.49