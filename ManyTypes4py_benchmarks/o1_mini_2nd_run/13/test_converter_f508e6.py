import logging
from shutil import copyfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from freqtrade.configuration.timerange import TimeRange
from freqtrade.data.converter import (
    convert_ohlcv_format,
    convert_trades_format,
    convert_trades_to_ohlcv,
    ohlcv_fill_up_missing_data,
    ohlcv_to_dataframe,
    reduce_dataframe_footprint,
    trades_df_remove_duplicates,
    trades_dict_to_list,
    trades_to_ohlcv,
    trim_dataframe,
)
from freqtrade.data.history import (
    get_timerange,
    load_data,
    load_pair_history,
    validate_backtest_data,
)
from freqtrade.data.history.datahandlers import IDataHandler
from freqtrade.enums import CandleType
from freqtrade.exchange import (
    timeframe_to_minutes,
    timeframe_to_seconds,
)
from tests.conftest import (
    generate_test_data,
    generate_trades_history,
    log_has,
    log_has_re,
)
from tests.data.test_history import _clean_test_file


def test_dataframe_correct_columns(dataframe_1m: pd.DataFrame) -> None:
    assert dataframe_1m.columns.tolist() == ['date', 'open', 'high', 'low', 'close', 'volume']


def test_ohlcv_to_dataframe(
    ohlcv_history_list: List[List[Any]], caplog: pytest.LogCaptureFixture
) -> None:
    columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    caplog.set_level(logging.DEBUG)
    dataframe = ohlcv_to_dataframe(
        ohlcv_history_list, '5m', pair='UNITTEST/BTC', fill_missing=True
    )
    assert dataframe.columns.tolist() == columns
    assert log_has(
        'Converting candle (OHLCV) data to dataframe for pair UNITTEST/BTC.', caplog
    )


def test_trades_to_ohlcv(
    trades_history_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
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


@pytest.mark.parametrize(
    'timeframe,rows,days,candles,start,end,weekday',
    [
        ('1s', 20000, 5, 19522, '2020-01-01 00:00:05', '2020-01-05 23:59:27', None),
        ('1m', 20000, 5, 6745, '2020-01-01 00:00:00', '2020-01-05 23:59:00', None),
        ('5m', 20000, 5, 1440, '2020-01-01 00:00:00', '2020-01-05 23:55:00', None),
        ('15m', 20000, 5, 480, '2020-01-01 00:00:00', '2020-01-05 23:45:00', None),
        ('1h', 20000, 5, 120, '2020-01-01 00:00:00', '2020-01-05 23:00:00', None),
        ('2h', 20000, 5, 60, '2020-01-01 00:00:00', '2020-01-05 22:00:00', None),
        ('4h', 20000, 5, 30, '2020-01-01 00:00:00', '2020-01-05 20:00:00', None),
        ('8h', 20000, 5, 15, '2020-01-01 00:00:00', '2020-01-05 16:00:00', None),
        ('12h', 20000, 5, 10, '2020-01-01 00:00:00', '2020-01-05 12:00:00', None),
        ('1d', 20000, 5, 5, '2020-01-01 00:00:00', '2020-01-05 00:00:00', 'Sunday'),
        ('7d', 20000, 37, 6, '2020-01-06 00:00:00', '2020-02-10 00:00:00', 'Monday'),
        ('1w', 20000, 37, 6, '2020-01-06 00:00:00', '2020-02-10 00:00:00', 'Monday'),
        ('1M', 20000, 74, 3, '2020-01-01 00:00:00', '2020-03-01 00:00:00', None),
        ('3M', 20000, 100, 2, '2020-01-01 00:00:00', '2020-04-01 00:00:00', None),
        ('1y', 20000, 1000, 3, '2020-01-01 00:00:00', '2022-01-01 00:00:00', None),
    ],
)
def test_trades_to_ohlcv_multi(
    timeframe: str,
    rows: int,
    days: int,
    candles: int,
    start: str,
    end: str,
    weekday: Optional[str],
) -> None:
    trades_history = generate_trades_history(n_rows=rows, days=days)
    df = trades_to_ohlcv(trades_history, timeframe)
    assert not df.empty
    assert len(df) == candles
    assert df.iloc[0, :]['date'] == pd.Timestamp(f'{start}+0000')
    assert df.iloc[-1, :]['date'] == pd.Timestamp(f'{end}+0000')
    if weekday:
        assert df.iloc[-1, :]['date'].day_name() == weekday


def test_ohlcv_fill_up_missing_data(
    testdatadir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data = load_pair_history(
        datadir=testdatadir, timeframe='1m', pair='UNITTEST/BTC', fill_up_missing=False
    )
    caplog.set_level(logging.DEBUG)
    data2 = ohlcv_fill_up_missing_data(data, '1m', 'UNITTEST/BTC')
    assert len(data2) > len(data)
    assert (data.columns == data2.columns).all()
    assert log_has_re(
        f'Missing data fillup for UNITTEST/BTC, 1m: before: {len(data)} - after: {len(data2)}.*',
        caplog,
    )
    min_date, max_date = get_timerange({'UNITTEST/BTC': data})
    assert validate_backtest_data(data, 'UNITTEST/BTC', min_date, max_date, 1)
    assert not validate_backtest_data(data2, 'UNITTEST/BTC', min_date, max_date, 1)


def test_ohlcv_fill_up_missing_data2(caplog: pytest.LogCaptureFixture) -> None:
    timeframe = '5m'
    ticks = [
        [1511686200000, 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 2255],
        [1511686500000, 8.88e-05, 8.942e-05, 8.88e-05, 8.893e-05, 9911],
        [1511687100000, 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05, 2251],
        [1511687400000, 8.877e-05, 8.883e-05, 8.895e-05, 8.817e-05, 123551],
    ]
    data = ohlcv_to_dataframe(
        ticks, timeframe, pair='UNITTEST/BTC', fill_missing=False
    )
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
    assert log_has_re(
        f'Missing data fillup for UNITTEST/BTC, {timeframe}: before: {len(data)} - after: {len(data2)}.*',
        caplog,
    )


@pytest.mark.parametrize(
    'timeframe',
    [
        '1s',
        '1m',
        '5m',
        '15m',
        '1h',
        '2h',
        '4h',
        '8h',
        '12h',
        '1d',
        '7d',
        '1w',
        '1M',
        '3M',
        '1y',
    ],
)
def test_ohlcv_to_dataframe_multi(timeframe: str) -> None:
    data = generate_test_data(timeframe, 180)
    assert len(data) == 180
    df = ohlcv_to_dataframe(data, timeframe, 'UNITTEST/USDT')
    assert len(df) == len(data) - 1
    df1 = ohlcv_to_dataframe(
        data, timeframe, 'UNITTEST/USDT', drop_incomplete=False
    )
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
        ohlcv_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        dfs = (
            data1.resample(f'{tfs}s', on='date')
            .agg(ohlcv_dict)
            .reset_index(drop=False)
        )
        dfm = (
            data1.resample(f'{tfm}min', on='date')
            .agg(ohlcv_dict)
            .reset_index(drop=False)
        )
        assert dfs.equals(dfm)
        assert dfs.equals(df1)


def test_ohlcv_to_dataframe_1M() -> None:
    ticks = [
        [1567296000000, 8042.08, 10475.54, 7700.67, 8041.96, 608742.1109999999],
        [1569888000000, 8285.31, 10408.48, 7172.76, 9150.0, 2439561.887],
        [1572566400000, 9149.88, 9550.0, 6510.19, 7542.93, 4042674.725],
        [1575158400000, 7541.08, 7800.0, 6427.0, 7189.0, 4063882.296],
        [1577836800000, 7189.43, 9599.0, 6863.44, 9364.51, 5165281.358],
        [1580515200000, 9364.5, 10540.0, 8450.0, 8531.98, 4581788.124],
        [1583020800000, 8532.5, 9204.0, 3621.81, 6407.1, 10859497.479],
        [1585699200000, 6407.1, 9479.77, 6140.0, 8624.76, 11276526.968],
        [1588291200000, 8623.61, 10080.0, 7940.0, 9446.43, 12469561.02],
        [1590969600000, 9446.49, 10497.25, 8816.4, 9138.87, 6684044.201],
        [1593561600000, 9138.88, 11488.0, 8900.0, 11343.68, 5709327.926],
        [1596240000000, 11343.67, 12499.42, 10490.0, 11658.11, 6746487.129],
        [1598918400000, 11658.11, 12061.07, 9808.58, 10773.0, 6442697.051],
        [1601510400000, 10773.0, 14140.0, 10371.03, 13783.73, 7404103.004],
        [1604188800000, 13783.73, 19944.0, 13195.0, 19720.0, 12328272.549],
        [1606780800000, 19722.09, 29376.7, 17555.0, 28951.68, 10067314.24],
        [1609459200000, 28948.19, 42125.51, 27800.0, 33126.21, 12408873.079],
        [1612137600000, 33125.11, 58472.14, 32322.47, 45163.36, 8784474.482],
        [1614556800000, 45162.64, 61950.0, 44972.49, 58807.24, 9459821.267],
        [1617235200000, 58810.99, 64986.11, 46930.43, 57684.16, 7895051.389],
        [1619827200000, 57688.29, 59654.0, 28688.0, 37243.38, 16790964.443],
        [1622505600000, 37244.36, 41413.0, 28780.01, 35031.39, 23474519.886],
        [1625097600000, 35031.39, 48168.6, 29242.24, 41448.11, 16932491.175],
        [1627776000000, 41448.1, 50600.0, 37291.0, 47150.32, 13645800.254],
        [1630454400000, 47150.32, 52950.0, 39503.58, 43796.57, 10734742.869],
        [1633046400000, 43799.49, 67150.0, 43260.01, 61348.61, 9111112.847],
        [1635724800000, 61347.14, 69198.7, 53245.0, 56975.0, 7111424.463],
        [1638316800000, 56978.06, 59100.0, 40888.89, 46210.56, 8404449.024],
        [1640995200000, 46210.57, 48000.0, 32853.83, 38439.04, 11047479.277],
        [1643673600000, 38439.04, 45847.5, 34303.7, 43155.0, 10910339.91],
        [1646092800000, 43155.0, 48200.0, 37134.0, 45506.0, 10459721.586],
        [1648771200000, 45505.9, 47448.0, 37550.0, 37614.5, 8463568.862],
        [1651363200000, 37614.4, 40071.7, 26631.0, 31797.8, 14463715.774],
        [1654041600000, 31797.9, 31986.1, 17593.2, 19923.5, 20710810.306],
        [1656633600000, 19923.3, 24700.0, 18780.1, 23290.1, 20582518.513],
        [1659312000000, 23290.1, 25200.0, 19508.0, 20041.5, 17221921.557],
        [1661990400000, 20041.4, 22850.0, 18084.3, 19411.7, 21935261.414],
        [1664582400000, 19411.6, 21088.0, 17917.8, 20482.0, 16625843.584],
        [1667260800000, 20482.1, 21473.7, 15443.2, 17153.3, 18460614.013],
        [1669852800000, 17153.4, 18400.0, 16210.0, 16537.6, 9702408.711],
        [1672531200000, 16537.5, 23962.7, 16488.0, 23119.4, 14732180.645],
        [1675209600000, 23119.5, 25347.6, 21338.0, 23129.6, 15025197.415],
        [1677628800000, 23129.7, 29184.8, 19521.6, 28454.9, 23317458.541],
        [1680307200000, 28454.8, 31059.0, 26919.3, 29223.0, 14654208.219],
        [1682899200000, 29223.0, 29840.0, 25751.0, 27201.1, 13328157.284],
        [1685577600000, 27201.1, 31500.0, 24777.0, 30460.2, 14099299.273],
        [1688169600000, 30460.2, 31850.0, 28830.0, 29338.8, 8760361.377],
    ]
    data = ohlcv_to_dataframe(
        ticks,
        '1M',
        pair='UNITTEST/USDT',
        fill_missing=False,
        drop_incomplete=False,
    )
    assert len(data) == len(ticks)
    assert data.iloc[0]['date'].strftime('%Y-%m-%d') == '2019-09-01'
    assert data.iloc[-1]['date'].strftime('%Y-%m-%d') == '2023-07-01'
    data = ohlcv_to_dataframe(
        ticks, '1M', pair='UNITTEST/USDT', fill_missing=True, drop_incomplete=False
    )
    assert len(data) == len(ticks)
    assert data.iloc[0]['date'].strftime('%Y-%m-%d') == '2019-09-01'
    assert data.iloc[-1]['date'].strftime('%Y-%m-%d') == '2023-07-01'


def test_ohlcv_drop_incomplete(caplog: pytest.LogCaptureFixture) -> None:
    timeframe = '1d'
    ticks = [
        [1559750400000, 8.794e-05, 8.948e-05, 8.794e-05, 8.88e-05, 2255],
        [1559836800000, 8.88e-05, 8.942e-05, 8.88e-05, 8.893e-05, 9911],
        [1559923200000, 8.891e-05, 8.893e-05, 8.875e-05, 8.877e-05, 2251],
        [1560009600000, 8.877e-05, 8.883e-05, 8.895e-05, 8.817e-05, 123551],
    ]
    caplog.set_level(logging.DEBUG)
    data = ohlcv_to_dataframe(
        ticks, timeframe, pair='UNITTEST/BTC', fill_missing=False, drop_incomplete=False
    )
    assert len(data) == 4
    assert not log_has('Dropping last candle', caplog)
    data = ohlcv_to_dataframe(
        ticks, timeframe, pair='UNITTEST/BTC', fill_missing=False, drop_incomplete=True
    )
    assert len(data) == 3
    assert log_has('Dropping last candle', caplog)


def test_trim_dataframe(testdatadir: Path) -> None:
    data = load_data(
        datadir=testdatadir, timeframe='1m', pairs=['UNITTEST/BTC']
    )['UNITTEST/BTC']
    min_date = int(data.iloc[0]['date'].timestamp())
    max_date = int(data.iloc[-1]['date'].timestamp())
    data_modify = data.copy()
    tr = TimeRange('date', None, min_date + 1800, 0)
    data_modify = trim_dataframe(data_modify, tr)
    assert not data_modify.equals(data)
    assert len(data_modify) < len(data)
    assert len(data_modify) == len(data) - 30
    assert all(data_modify.iloc[-1] == data.iloc[-1])
    assert all(data_modify.iloc[0] == data.iloc[30])
    data_modify = data.copy()
    tr = TimeRange('date', None, min_date + 1800, 0)
    data_modify = trim_dataframe(data_modify, tr, startup_candles=20)
    assert not data_modify.equals(data)
    assert len(data_modify) < len(data)
    assert len(data_modify) == len(data) - 20
    assert all(data_modify.iloc[-1] == data.iloc[-1])
    assert all(data_modify.iloc[0] == data.iloc[20])
    data_modify = data.copy()
    tr = TimeRange(None, 'date', 0, max_date - 1800)
    data_modify = trim_dataframe(data_modify, tr)
    assert not data_modify.equals(data)
    assert len(data_modify) < len(data)
    assert len(data_modify) == len(data) - 30
    assert all(data_modify.iloc[0] == data.iloc[0])
    assert all(data_modify.iloc[-1] == data.iloc[-31])
    data_modify = data.copy()
    tr = TimeRange('date', 'date', min_date + 1500, max_date - 1800)
    data_modify = trim_dataframe(data_modify, tr)
    assert not data_modify.equals(data)
    assert len(data_modify) < len(data)
    assert len(data_modify) == len(data) - 55
    assert all(data_modify.iloc[0] == data.iloc[25])


def test_trades_df_remove_duplicates(trades_history_df: pd.DataFrame) -> None:
    trades_history1 = pd.concat(
        [trades_history_df, trades_history_df, trades_history_df]
    ).reset_index(drop=True)
    assert len(trades_history1) == len(trades_history_df) * 3
    res = trades_df_remove_duplicates(trades_history1)
    assert len(res) == len(trades_history_df)
    assert res.equals(trades_history_df)


def test_trades_dict_to_list(fetch_trades_result: List[Dict[str, Any]]) -> None:
    res = trades_dict_to_list(fetch_trades_result)
    assert isinstance(res, list)
    assert isinstance(res[0], list)
    for i, t in enumerate(res):
        assert t[0] == fetch_trades_result[i]['timestamp']
        assert t[1] == fetch_trades_result[i]['id']
        assert t[2] == fetch_trades_result[i]['type']
        assert t[3] == fetch_trades_result[i]['side']
        assert t[4] == fetch_trades_result[i]['price']
        assert t[5] == fetch_trades_result[i]['amount']
        assert t[6] == fetch_trades_result[i]['cost']


def test_convert_trades_format(
    default_conf: Dict[str, Any], testdatadir: Path, tmp_path: Path
) -> None:
    files = [
        {
            'old': tmp_path / 'XRP_ETH-trades.json.gz',
            'new': tmp_path / 'XRP_ETH-trades.json',
        },
        {
            'old': tmp_path / 'XRP_OLD-trades.json.gz',
            'new': tmp_path / 'XRP_OLD-trades.json',
        },
    ]
    for file in files:
        copyfile(testdatadir / file['old'].name, file['old'])
        assert not file['new'].exists()
    default_conf['datadir'] = tmp_path
    convert_trades_format(
        default_conf, convert_from='jsongz', convert_to='json', erase=False
    )
    for file in files:
        assert file['new'].exists()
        assert file['old'].exists()
        file['old'].unlink()
    convert_trades_format(
        default_conf, convert_from='json', convert_to='jsongz', erase=True
    )
    for file in files:
        assert file['old'].exists()
        assert not file['new'].exists()
        _clean_test_file(file['old'])
        if file['new'].exists():
            file['new'].unlink()


@pytest.mark.parametrize(
    'file_base,candletype',
    [
        (
            ['XRP_ETH-5m', 'XRP_ETH-1m'],
            CandleType.SPOT,
        ),
        (
            ['UNITTEST_USDT_USDT-1h-mark', 'XRP_USDT_USDT-1h-mark'],
            CandleType.MARK,
        ),
        (
            ['XRP_USDT_USDT-1h-futures'],
            CandleType.FUTURES,
        ),
    ],
)
def test_convert_ohlcv_format(
    default_conf: Dict[str, Any],
    testdatadir: Path,
    tmp_path: Path,
    file_base: List[str],
    candletype: CandleType,
) -> None:
    prependix = '' if candletype == CandleType.SPOT else 'futures/'
    files_orig = []
    files_temp = []
    files_new = []
    for file in file_base:
        file_orig = testdatadir / f'{prependix}{file}.feather'
        file_temp = tmp_path / f'{prependix}{file}.feather'
        file_new = tmp_path / f'{prependix}{file}.json.gz'
        IDataHandler.create_dir_if_needed(file_temp)
        copyfile(file_orig, file_temp)
        files_orig.append(file_orig)
        files_temp.append(file_temp)
        files_new.append(file_new)
    default_conf['datadir'] = tmp_path
    default_conf['candle_types'] = [candletype]
    if candletype == CandleType.SPOT:
        default_conf['pairs'] = ['XRP/ETH', 'XRP/USDT', 'UNITTEST/USDT']
    else:
        default_conf['pairs'] = [
            'XRP/ETH:ETH',
            'XRP/USDT:USDT',
            'UNITTEST/USDT:USDT',
        ]
    default_conf['timeframes'] = ['1m', '5m', '1h']
    for file_new in files_new:
        assert not file_new.exists()
    convert_ohlcv_format(
        default_conf, convert_from='feather', convert_to='jsongz', erase=False
    )
    for file in files_temp + files_new:
        assert file.exists()
    for file in files_temp:
        file.unlink()
    convert_ohlcv_format(
        default_conf, convert_from='jsongz', convert_to='feather', erase=True
    )
    for file in files_temp:
        assert file.exists()
    for file in files_new:
        assert not file.exists()


def test_reduce_dataframe_footprint() -> None:
    data = generate_test_data('15m', 40)
    data['open_copy'] = data['open']
    data['close_copy'] = data['close']
    data['close_copy'] = data['close']
    assert data['open'].dtype == np.float64
    assert data['open_copy'].dtype == np.float64
    assert data['close_copy'].dtype == np.float64
    df2 = reduce_dataframe_footprint(data)
    assert data['open'].dtype == np.float64
    assert data['open_copy'].dtype == np.float64
    assert data['close_copy'].dtype == np.float64
    assert df2['open'].dtype == np.float64
    assert df2['high'].dtype == np.float64
    assert df2['low'].dtype == np.float64
    assert df2['close'].dtype == np.float64
    assert df2['volume'].dtype == np.float64
    assert df2['open_copy'].dtype == np.float32
    assert df2['close_copy'].dtype == np.float32


def test_convert_trades_to_ohlcv(
    testdatadir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pair = 'XRP/ETH'
    file1 = tmp_path / 'XRP_ETH-1m.feather'
    file5 = tmp_path / 'XRP_ETH-5m.feather'
    filetrades = tmp_path / 'XRP_ETH-trades.json.gz'
    copyfile(testdatadir / file1.name, file1)
    copyfile(testdatadir / file5.name, file5)
    copyfile(testdatadir / filetrades.name, filetrades)
    dfbak_1m = load_pair_history(
        datadir=tmp_path, timeframe='1m', pair=pair
    )
    dfbak_5m = load_pair_history(
        datadir=tmp_path, timeframe='5m', pair=pair
    )
    tr = TimeRange.parse_timerange('20191011-20191012')
    convert_trades_to_ohlcv(
        [pair],
        timeframes=['1m', '5m'],
        data_format_trades='jsongz',
        datadir=tmp_path,
        timerange=tr,
        erase=True,
        data_format_ohlcv='feather',
        candle_type=CandleType.SPOT,
    )
    assert log_has('Deleting existing data for pair XRP/ETH, interval 1m.', caplog)
    df_1m = load_pair_history(datadir=tmp_path, timeframe='1m', pair=pair)
    df_5m = load_pair_history(datadir=tmp_path, timeframe='5m', pair=pair)
    assert_frame_equal(dfbak_1m, df_1m, check_exact=True)
    assert_frame_equal(dfbak_5m, df_5m, check_exact=True)
    msg = 'Could not convert NoDatapair to OHLCV.'
    assert not log_has(msg, caplog)
    convert_trades_to_ohlcv(
        ['NoDatapair'],
        timeframes=['1m', '5m'],
        data_format_trades='jsongz',
        datadir=tmp_path,
        timerange=tr,
        erase=True,
        data_format_ohlcv='feather',
        candle_type=CandleType.SPOT,
    )
    assert log_has(msg, caplog)
