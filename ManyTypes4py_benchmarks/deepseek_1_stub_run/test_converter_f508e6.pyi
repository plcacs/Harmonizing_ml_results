```python
import logging
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
import pytest
from freqtrade.configuration.timerange import TimeRange
from freqtrade.data.history.datahandlers import IDataHandler
from freqtrade.enums import CandleType

def test_dataframe_correct_columns(dataframe_1m: pd.DataFrame) -> None: ...

def test_ohlcv_to_dataframe(
    ohlcv_history_list: List[List[Any]],
    caplog: pytest.LogCaptureFixture
) -> None: ...

def test_trades_to_ohlcv(
    trades_history_df: pd.DataFrame,
    caplog: pytest.LogCaptureFixture
) -> None: ...

@pytest.mark.parametrize('timeframe,rows,days,candles,start,end,weekday', [('1s', 20000, 5, 19522, '2020-01-01 00:00:05', '2020-01-05 23:59:27', None), ('1m', 20000, 5, 6745, '2020-01-01 00:00:00', '2020-01-05 23:59:00', None), ('5m', 20000, 5, 1440, '2020-01-01 00:00:00', '2020-01-05 23:55:00', None), ('15m', 20000, 5, 480, '2020-01-01 00:00:00', '2020-01-05 23:45:00', None), ('1h', 20000, 5, 120, '2020-01-01 00:00:00', '2020-01-05 23:00:00', None), ('2h', 20000, 5, 60, '2020-01-01 00:00:00', '2020-01-05 22:00:00', None), ('4h', 20000, 5, 30, '2020-01-01 00:00:00', '2020-01-05 20:00:00', None), ('8h', 20000, 5, 15, '2020-01-01 00:00:00', '2020-01-05 16:00:00', None), ('12h', 20000, 5, 10, '2020-01-01 00:00:00', '2020-01-05 12:00:00', None), ('1d', 20000, 5, 5, '2020-01-01 00:00:00', '2020-01-05 00:00:00', 'Sunday'), ('7d', 20000, 37, 6, '2020-01-06 00:00:00', '2020-02-10 00:00:00', 'Monday'), ('1w', 20000, 37, 6, '2020-01-06 00:00:00', '2020-02-10 00:00:00', 'Monday'), ('1M', 20000, 74, 3, '2020-01-01 00:00:00', '2020-03-01 00:00:00', None), ('3M', 20000, 100, 2, '2020-01-01 00:00:00', '2020-04-01 00:00:00', None), ('1y', 20000, 1000, 3, '2020-01-01 00:00:00', '2022-01-01 00:00:00', None)])
def test_trades_to_ohlcv_multi(
    timeframe: str,
    rows: int,
    days: int,
    candles: int,
    start: str,
    end: str,
    weekday: Optional[str]
) -> None: ...

def test_ohlcv_fill_up_missing_data(
    testdatadir: Any,
    caplog: pytest.LogCaptureFixture
) -> None: ...

def test_ohlcv_fill_up_missing_data2(
    caplog: pytest.LogCaptureFixture
) -> None: ...

@pytest.mark.parametrize('timeframe', ['1s', '1m', '5m', '15m', '1h', '2h', '4h', '8h', '12h', '1d', '7d', '1w', '1M', '3M', '1y'])
def test_ohlcv_to_dataframe_multi(timeframe: str) -> None: ...

def test_ohlcv_to_dataframe_1M() -> None: ...

def test_ohlcv_drop_incomplete(caplog: pytest.LogCaptureFixture) -> None: ...

def test_trim_dataframe(testdatadir: Any) -> None: ...

def test_trades_df_remove_duplicates(trades_history_df: pd.DataFrame) -> None: ...

def test_trades_dict_to_list(fetch_trades_result: Any) -> None: ...

def test_convert_trades_format(
    default_conf: Any,
    testdatadir: Any,
    tmp_path: Any
) -> None: ...

@pytest.mark.parametrize('file_base,candletype', [(['XRP_ETH-5m', 'XRP_ETH-1m'], CandleType.SPOT), (['UNITTEST_USDT_USDT-1h-mark', 'XRP_USDT_USDT-1h-mark'], CandleType.MARK), (['XRP_USDT_USDT-1h-futures'], CandleType.FUTURES)])
def test_convert_ohlcv_format(
    default_conf: Any,
    testdatadir: Any,
    tmp_path: Any,
    file_base: List[str],
    candletype: CandleType
) -> None: ...

def test_reduce_dataframe_footprint() -> None: ...

def test_convert_trades_to_ohlcv(
    testdatadir: Any,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture
) -> None: ...
```