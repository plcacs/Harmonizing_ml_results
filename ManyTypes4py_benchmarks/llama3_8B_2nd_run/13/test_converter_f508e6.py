from typing import List

def test_dataframe_correct_columns(dataframe_1m: pd.DataFrame) -> None:
    assert dataframe_1m.columns.tolist() == ['date', 'open', 'high', 'low', 'close', 'volume']

def test_ohlcv_to_dataframe(ohlcv_history_list: List[List], timeframe: str, pair: str, caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_trades_to_ohlcv(trades_history_df: pd.DataFrame, timeframe: str, caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_ohlcv_fill_up_missing_data(testdatadir: str, caplog: pytest.LogCaptureFixture) -> None:
    ...

def test_ohlcv_to_dataframe_multi(timeframe: str) -> None:
    ...

def test_ohlcv_to_dataframe_1M() -> None:
    ...

def test_trim_dataframe(testdatadir: str) -> None:
    ...

def test_trades_df_remove_duplicates(trades_history_df: pd.DataFrame) -> None:
    ...

def test_trades_dict_to_list(fetch_trades_result: List[dict]) -> None:
    ...

def test_convert_trades_format(default_conf: dict, testdatadir: str, tmp_path: Path) -> None:
    ...

def test_convert_ohlcv_format(default_conf: dict, testdatadir: str, tmp_path: Path, file_base: List[str], candletype: CandleType) -> None:
    ...

def test_reduce_dataframe_footprint() -> None:
    ...

def test_convert_trades_to_ohlcv(testdatadir: str, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ...
