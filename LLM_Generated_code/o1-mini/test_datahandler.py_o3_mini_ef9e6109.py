# pragma pylint: disable=missing-docstring, protected-access, C0103

import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from typing import Any, List, Set, Tuple, Type

import pytest
from pandas import DataFrame, Timestamp
from pandas.testing import assert_frame_equal

from freqtrade.configuration import TimeRange
from freqtrade.constants import AVAILABLE_DATAHANDLERS
from freqtrade.data.history.datahandlers.featherdatahandler import FeatherDataHandler
from freqtrade.data.history.datahandlers.idatahandler import (
    IDataHandler,
    get_datahandler,
    get_datahandlerclass,
)
from freqtrade.data.history.datahandlers.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.data.history.datahandlers.parquetdatahandler import ParquetDataHandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from tests.conftest import log_has, log_has_re


def test_datahandler_ohlcv_get_pairs(testdatadir: Path) -> None:
    pairs: List[str] = FeatherDataHandler.ohlcv_get_pairs(testdatadir, "5m", candle_type=CandleType.SPOT)
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == {
        "UNITTEST/BTC",
        "XLM/BTC",
        "ETH/BTC",
        "TRX/BTC",
        "LTC/BTC",
        "XMR/BTC",
        "ZEC/BTC",
        "ADA/BTC",
        "ETC/BTC",
        "NXT/BTC",
        "DASH/BTC",
        "XRP/ETH",
    }

    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, "8m", candle_type=CandleType.SPOT)
    assert set(pairs) == {"UNITTEST/BTC"}

    pairs = FeatherDataHandler.ohlcv_get_pairs(testdatadir, "1h", candle_type=CandleType.MARK)
    assert set(pairs) == {"UNITTEST/USDT:USDT", "XRP/USDT:USDT"}

    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, "1h", candle_type=CandleType.FUTURES)
    assert set(pairs) == {"XRP/USDT:USDT"}


@pytest.mark.parametrize(
    "filename,pair,timeframe,candletype",
    [
        ("XMR_BTC-5m.json", "XMR_BTC", "5m", ""),
        ("XMR_USDT-1h.h5", "XMR_USDT", "1h", ""),
        ("BTC-PERP-1h.h5", "BTC-PERP", "1h", ""),
        ("BTC_USDT-2h.jsongz", "BTC_USDT", "2h", ""),
        ("BTC_USDT-2h-mark.jsongz", "BTC_USDT", "2h", "mark"),
        ("XMR_USDT-1h-mark.h5", "XMR_USDT", "1h", "mark"),
        ("XMR_USDT-1h-random.h5", "XMR_USDT", "1h", "random"),
        ("BTC-PERP-1h-index.h5", "BTC-PERP", "1h", "index"),
        ("XMR_USDT_USDT-1h-mark.h5", "XMR_USDT_USDT", "1h", "mark"),
    ],
)
def test_datahandler_ohlcv_regex(
    filename: str,
    pair: str,
    timeframe: str,
    candletype: str,
) -> None:
    regex: str = JsonDataHandler._OHLCV_REGEX

    match = re.search(regex, filename)
    assert match is not None
    assert len(match.groups()) > 1
    assert match.group(1) == pair
    assert match.group(2) == timeframe
    assert match.group(3) == candletype


@pytest.mark.parametrize(
    "pair,expected",
    [
        ("XMR_USDT", "XMR/USDT"),
        ("BTC_USDT", "BTC/USDT"),
        ("USDT_BUSD", "USDT/BUSD"),
        ("BTC_USDT_USDT", "BTC/USDT:USDT"),  # Futures
        ("XRP_USDT_USDT", "XRP/USDT:USDT"),  # futures
        ("BTC-PERP", "BTC-PERP"),
        ("BTC-PERP_USDT", "BTC-PERP:USDT"),
        ("UNITTEST_USDT", "UNITTEST/USDT"),
    ],
)
def test_rebuild_pair_from_filename(pair: str, expected: str) -> None:
    assert IDataHandler.rebuild_pair_from_filename(pair) == expected


def test_datahandler_ohlcv_get_available_data(testdatadir: Path) -> None:
    paircombs: List[Tuple[str, str, Any]] = FeatherDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    # Convert to set to avoid failures due to sorting
    assert set(paircombs) == {
        ("UNITTEST/BTC", "5m", CandleType.SPOT),
        ("ETH/BTC", "5m", CandleType.SPOT),
        ("XLM/BTC", "5m", CandleType.SPOT),
        ("TRX/BTC", "5m", CandleType.SPOT),
        ("LTC/BTC", "5m", CandleType.SPOT),
        ("XMR/BTC", "5m", CandleType.SPOT),
        ("ZEC/BTC", "5m", CandleType.SPOT),
        ("UNITTEST/BTC", "1m", CandleType.SPOT),
        ("ADA/BTC", "5m", CandleType.SPOT),
        ("ETC/BTC", "5m", CandleType.SPOT),
        ("NXT/BTC", "5m", CandleType.SPOT),
        ("DASH/BTC", "5m", CandleType.SPOT),
        ("XRP/ETH", "1m", CandleType.SPOT),
        ("XRP/ETH", "5m", CandleType.SPOT),
        ("UNITTEST/BTC", "30m", CandleType.SPOT),
        ("UNITTEST/BTC", "8m", CandleType.SPOT),
    }

    paircombs = FeatherDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.FUTURES)
    # Convert to set to avoid failures due to sorting
    assert set(paircombs) == {
        ("UNITTEST/USDT:USDT", "1h", "mark"),
        ("XRP/USDT:USDT", "5m", "futures"),
        ("XRP/USDT:USDT", "1h", "futures"),
        ("XRP/USDT:USDT", "1h", "mark"),
        ("XRP/USDT:USDT", "8h", "mark"),
        ("XRP/USDT:USDT", "8h", "funding_rate"),
    }

    paircombs = JsonGzDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {("UNITTEST/BTC", "8m", CandleType.SPOT)}


def test_jsondatahandler_ohlcv_purge(mocker: Any, testdatadir: Path) -> None:
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh: JsonGzDataHandler = JsonGzDataHandler(testdatadir)
    assert not dh.ohlcv_purge("UNITTEST/NONEXIST", "5m", "")
    assert not dh.ohlcv_purge("UNITTEST/NONEXIST", "5m", candle_type="mark")
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.ohlcv_purge("UNITTEST/NONEXIST", "5m", "")
    assert dh.ohlcv_purge("UNITTEST/NONEXIST", "5m", candle_type="mark")
    assert unlinkmock.call_count == 2


def test_jsondatahandler_ohlcv_load(testdatadir: Path, caplog: Any) -> None:
    dh: JsonDataHandler = JsonDataHandler(testdatadir)

    df: DataFrame = dh.ohlcv_load("UNITTEST/BTC", "1m", "spot")
    assert len(df) > 0

    # Failure case (empty array)
    df1: DataFrame = dh.ohlcv_load("NOPAIR/XXX", "4m", "spot")
    assert len(df1) == 0
    assert log_has("Could not load data for NOPAIR/XXX.", caplog)
    assert df.columns.equals(df1.columns)


def test_datahandler_ohlcv_data_min_max(testdatadir: Path) -> None:
    dh: JsonDataHandler = JsonDataHandler(testdatadir)
    min_max: Tuple[datetime, datetime, Any] = dh.ohlcv_data_min_max("UNITTEST/BTC", "5m", "spot")
    assert len(min_max) == 3

    # Empty pair
    min_max = dh.ohlcv_data_min_max("UNITTEST/BTC", "8m", "spot")
    assert len(min_max) == 3
    assert min_max[0] == datetime.fromtimestamp(0, tz=timezone.utc)
    assert min_max[0] == min_max[1]
    # Empty pair2
    min_max = dh.ohlcv_data_min_max("NOPAIR/XXX", "41m", "spot")
    assert len(min_max) == 3
    assert min_max[0] == datetime.fromtimestamp(0, tz=timezone.utc)
    assert min_max[0] == min_max[1]

    # Existing pair ...
    min_max = dh.ohlcv_data_min_max("UNITTEST/BTC", "1m", "spot")
    assert len(min_max) == 3
    assert min_max[0] == datetime(2017, 11, 4, 23, 2, tzinfo=timezone.utc)
    assert min_max[1] == datetime(2017, 11, 14, 22, 59, tzinfo=timezone.utc)


def test_datahandler__check_empty_df(testdatadir: Path, caplog: Any) -> None:
    dh: JsonDataHandler = JsonDataHandler(testdatadir)
    expected_text: str = r"Price jump in UNITTEST/USDT, 1h, spot between"
    df: DataFrame = DataFrame(
        [
            [
                1511686200000,  # 8:50:00
                8.794,  # open
                8.948,  # high
                8.794,  # low
                8.88,  # close
                2255,  # volume (in quote currency)
            ],
            [
                1511686500000,  # 8:55:00
                8.88,
                8.942,
                8.88,
                8.893,
                9911,
            ],
            [
                1511687100000,  # 9:05:00
                8.891,
                8.893,
                8.875,
                8.877,
                2251,
            ],
            [
                1511687400000,  # 9:10:00
                8.877,
                8.883,
                8.895,
                8.817,
                123551,
            ],
        ],
        columns=["date", "open", "high", "low", "close", "volume"],
    )

    dh._check_empty_df(df, "UNITTEST/USDT", "1h", CandleType.SPOT, True, True)
    assert not log_has_re(expected_text, caplog)
    df = DataFrame(
        [
            [
                1511686200000,  # 8:50:00
                8.794,  # open
                8.948,  # high
                8.794,  # low
                8.88,  # close
                2255,  # volume (in quote currency)
            ],
            [
                1511686500000,  # 8:55:00
                8.88,
                8.942,
                8.88,
                8.893,
                9911,
            ],
            [
                1511687100000,  # 9:05:00
                889.1,  # Price jump by several decimals
                889.3,
                887.5,
                887.7,
                2251,
            ],
            [
                1511687400000,  # 9:10:00
                8.877,
                8.883,
                8.895,
                8.817,
                123551,
            ],
        ],
        columns=["date", "open", "high", "low", "close", "volume"],
    )

    dh._check_empty_df(df, "UNITTEST/USDT", "1h", CandleType.SPOT, True, True)
    assert log_has_re(expected_text, caplog)


# @pytest.mark.parametrize('datahandler', [])
@pytest.mark.skip("All datahandlers currently support trades data.")
def test_datahandler_trades_not_supported(
    datahandler: Any,
    testdatadir: Path,
) -> None:
    # Currently disabled. Re-enable should a new provider not support trades data.
    dh: IDataHandler = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.trades_load("UNITTEST/ETH")
    with pytest.raises(NotImplementedError):
        dh.trades_store("UNITTEST/ETH", MagicMock())


def test_jsondatahandler_trades_load(testdatadir: Path, caplog: Any) -> None:
    dh: JsonGzDataHandler = JsonGzDataHandler(testdatadir)
    logmsg: str = "Old trades format detected - converting"
    dh.trades_load("XRP/ETH", TradingMode.SPOT)
    assert not log_has(logmsg, caplog)

    # Test conversation is happening
    dh.trades_load("XRP/OLD", TradingMode.SPOT)
    assert log_has(logmsg, caplog)


@pytest.mark.parametrize("datahandler", AVAILABLE_DATAHANDLERS)
def test_datahandler_ohlcv_append(
    datahandler: str,
    testdatadir: Path,
) -> None:
    dh: IDataHandler = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.ohlcv_append("UNITTEST/ETH", "5m", DataFrame(), CandleType.SPOT)
    with pytest.raises(NotImplementedError):
        dh.ohlcv_append("UNITTEST/ETH", "5m", DataFrame(), CandleType.MARK)


@pytest.mark.parametrize("datahandler", AVAILABLE_DATAHANDLERS)
def test_datahandler_trades_append(datahandler: str, testdatadir: Path) -> None:
    dh: IDataHandler = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.trades_append("UNITTEST/ETH", DataFrame())


@pytest.mark.parametrize(
    "datahandler,expected",
    [
        ("jsongz", {"XRP/ETH", "XRP/OLD"}),
        ("feather", {"XRP/ETH"}),
        ("parquet", {"XRP/ETH"}),
    ],
)
def test_datahandler_trades_get_pairs(testdatadir: Path, datahandler: str, expected: Set[str]) -> None:
    pairs: List[str] = get_datahandlerclass(datahandler).trades_get_pairs(testdatadir)
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == expected


def test_hdf5datahandler_deprecated(testdatadir: Path) -> None:
    with pytest.raises(
        OperationalException, match=r"DEPRECATED: The hdf5 dataformat is deprecated and has.*"
    ):
        get_datahandler(testdatadir, "hdf5")


@pytest.mark.parametrize(
    "pair,timeframe,candle_type,candle_append,startdt,enddt",
    [
        # Data goes from 2018-01-10 - 2018-01-30
        ("UNITTEST/BTC", "5m", "spot", "", "2018-01-15", "2018-01-19"),
        # Mark data goes from to 2021-11-15 2021-11-19
        ("UNITTEST/USDT:USDT", "1h", "mark", "-mark", "2021-11-16", "2021-11-18"),
    ],
)
@pytest.mark.parametrize("datahandler", ["feather", "parquet"])
def test_generic_datahandler_ohlcv_load_and_resave(
    datahandler: str,
    mocker: Any,
    testdatadir: Path,
    tmp_path: Path,
    pair: str,
    timeframe: str,
    candle_type: str,
    candle_append: str,
    startdt: str,
    enddt: str,
    caplog: Any,
) -> None:
    tmpdir2: Path = tmp_path
    if candle_type not in ("", "spot"):
        tmpdir2 = tmp_path / "futures"
        tmpdir2.mkdir()
    # Load data from one common file
    dhbase: IDataHandler = get_datahandler(testdatadir, "feather")
    ohlcv: DataFrame = dhbase._ohlcv_load(pair, timeframe, None, candle_type=candle_type)
    assert isinstance(ohlcv, DataFrame)
    assert len(ohlcv) > 0

    # Get data to test
    dh: IDataHandler = get_datahandler(testdatadir, datahandler)

    file: Path = tmpdir2 / f"UNITTEST_NEW-{timeframe}{candle_append}.{dh._get_file_extension()}"
    assert not file.is_file()

    dh1: IDataHandler = get_datahandler(tmp_path, datahandler)
    dh1.ohlcv_store("UNITTEST/NEW", timeframe, ohlcv, candle_type=candle_type)
    assert file.is_file()

    assert not ohlcv[ohlcv["date"] < startdt].empty

    timerange: TimeRange = TimeRange.parse_timerange(f"{startdt.replace('-', '')}-{enddt.replace('-', '')}")

    ohlcv = dhbase.ohlcv_load(pair, timeframe, timerange=timerange, candle_type=candle_type)
    ohlcv1: DataFrame = dh1.ohlcv_load("UNITTEST/NEW", timeframe, timerange=timerange, candle_type=candle_type)

    assert len(ohlcv) == len(ohlcv1)
    assert ohlcv.equals(ohlcv1)
    assert ohlcv[ohlcv["date"] < startdt].empty
    assert ohlcv[ohlcv["date"] > enddt].empty

    # Try loading inexisting file
    ohlcv = dh1.ohlcv_load("UNITTEST/NONEXIST", timeframe, candle_type=candle_type)
    assert ohlcv.empty

    # Try loading a file that exists but errors
    mocker.patch(
        "freqtrade.data.history.datahandlers.featherdatahandler.read_feather",
        side_effect=Exception("Test"),
    )
    mocker.patch(
        "freqtrade.data.history.datahandlers.parquetdatahandler.read_parquet",
        side_effect=Exception("Test"),
    )
    ohlcv_e: DataFrame = dh1.ohlcv_load("UNITTEST/NEW", timeframe, candle_type=candle_type)
    assert ohlcv_e.empty
    assert log_has_re("Error loading data from", caplog)


@pytest.mark.parametrize("datahandler", ["jsongz", "feather", "parquet"])
def test_datahandler_trades_load(testdatadir: Path, datahandler: str) -> None:
    dh: IDataHandler = get_datahandler(testdatadir, datahandler)
    trades: DataFrame = dh.trades_load("XRP/ETH", TradingMode.SPOT)
    assert isinstance(trades, DataFrame)
    assert trades.iloc[0]["timestamp"] == 1570752011620
    assert trades.iloc[0]["date"] == Timestamp("2019-10-11 00:00:11.620000+0000")
    assert trades.iloc[-1]["cost"] == 0.1986231

    trades1: DataFrame = dh.trades_load("UNITTEST/NONEXIST", TradingMode.SPOT)
    assert isinstance(trades1, DataFrame)
    assert trades1.empty


@pytest.mark.parametrize("datahandler", ["jsongz", "feather", "parquet"])
def test_datahandler_trades_store(testdatadir: Path, tmp_path: Path, datahandler: str) -> None:
    dh: IDataHandler = get_datahandler(testdatadir, datahandler)
    trades: DataFrame = dh.trades_load("XRP/ETH", TradingMode.SPOT)

    dh1: IDataHandler = get_datahandler(tmp_path, datahandler)
    dh1.trades_store("XRP/NEW", trades, TradingMode.SPOT)

    file: Path = tmp_path / f"XRP_NEW-trades.{dh1._get_file_extension()}"
    assert file.is_file()
    # Load trades back
    trades_new: DataFrame = dh1.trades_load("XRP/NEW", TradingMode.SPOT)
    assert_frame_equal(trades, trades_new, check_exact=True)
    assert len(trades_new) == len(trades)


@pytest.mark.parametrize("datahandler", ["jsongz", "feather", "parquet"])
def test_datahandler_trades_purge(mocker: Any, testdatadir: Path, datahandler: str) -> None:
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh: IDataHandler = get_datahandler(testdatadir, datahandler)
    assert not dh.trades_purge("UNITTEST/NONEXIST", TradingMode.SPOT)
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.trades_purge("UNITTEST/NONEXIST", TradingMode.SPOT)
    assert unlinkmock.call_count == 1


def test_datahandler_trades_get_available_data(testdatadir: Path) -> None:
    paircombs: List[str] = FeatherDataHandler.trades_get_available_data(testdatadir, TradingMode.SPOT)
    # Convert to set to avoid failures due to sorting
    assert set(paircombs) == {"XRP/ETH"}

    paircombs = FeatherDataHandler.trades_get_available_data(testdatadir, TradingMode.FUTURES)
    # Convert to set to avoid failures due to sorting
    assert set(paircombs) == set()

    paircombs = JsonGzDataHandler.trades_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {"XRP/ETH", "XRP/OLD"}


def test_datahandler_trades_data_min_max(testdatadir: Path) -> None:
    dh: IDataHandler = FeatherDataHandler(testdatadir)
    min_max: Tuple[datetime, datetime, Any] = dh.trades_data_min_max("XRP/ETH", TradingMode.SPOT)
    assert len(min_max) == 3

    # Empty pair
    min_max = dh.trades_data_min_max("NADA/ETH", TradingMode.SPOT)
    assert len(min_max) == 3
    assert min_max[0] == datetime.fromtimestamp(0, tz=timezone.utc)
    assert min_max[0] == min_max[1]

    # Existing pair ...
    min_max = dh.trades_data_min_max("XRP/ETH", TradingMode.SPOT)
    assert len(min_max) == 3
    assert min_max[0] == datetime(2019, 10, 11, 0, 0, 11, 620000, tzinfo=timezone.utc)
    assert min_max[1] == datetime(2019, 10, 13, 11, 19, 28, 844000, tzinfo=timezone.utc)


def test_gethandlerclass() -> None:
    cl: Type[IDataHandler] = get_datahandlerclass("json")
    assert cl == JsonDataHandler
    assert issubclass(cl, IDataHandler)

    cl = get_datahandlerclass("jsongz")
    assert cl == JsonGzDataHandler
    assert issubclass(cl, IDataHandler)
    assert issubclass(cl, JsonDataHandler)

    cl = get_datahandlerclass("feather")
    assert cl == FeatherDataHandler
    assert issubclass(cl, IDataHandler)

    cl = get_datahandlerclass("parquet")
    assert cl == ParquetDataHandler
    assert issubclass(cl, IDataHandler)

    with pytest.raises(ValueError, match=r"No datahandler for .*"):
        get_datahandlerclass("DeadBeef")


def test_get_datahandler(testdatadir: Path) -> None:
    dh: IDataHandler = get_datahandler(testdatadir, "json")
    assert isinstance(dh, JsonDataHandler)
    dh = get_datahandler(testdatadir, "jsongz")
    assert isinstance(dh, JsonGzDataHandler)
    dh1: IDataHandler = get_datahandler(testdatadir, "jsongz", dh)
    assert id(dh1) == id(dh)
