"""test parquet compat"""
import datetime
from decimal import Decimal
from io import BytesIO
import os
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Union, cast

import numpy as np
import pytest
from pandas._config import using_string_dtype
from pandas.compat import is_platform_windows
from pandas.compat.pyarrow import (
    pa_version_under11p0,
    pa_version_under13p0,
    pa_version_under15p0,
    pa_version_under17p0,
    pa_version_under19p0,
)
import pandas as pd
import pandas._testing as tm
from pandas.util.version import Version
from pandas.io.parquet import (
    FastParquetImpl,
    PyArrowImpl,
    get_engine,
    read_parquet,
    to_parquet,
)

try:
    import pyarrow
    _HAVE_PYARROW = True
except ImportError:
    _HAVE_PYARROW = False

try:
    import fastparquet
    _HAVE_FASTPARQUET = True
except ImportError:
    _HAVE_FASTPARQUET = False

pytestmark = [
    pytest.mark.filterwarnings("ignore:DataFrame._data is deprecated:FutureWarning"),
    pytest.mark.filterwarnings("ignore:Passing a BlockManager to DataFrame:DeprecationWarning"),
]

@pytest.fixture(
    params=[
        pytest.param(
            "fastparquet",
            marks=[
                pytest.mark.skipif(
                    not _HAVE_FASTPARQUET, reason="fastparquet is not installed"
                ),
                pytest.mark.xfail(
                    using_string_dtype(),
                    reason="TODO(infer_string) fastparquet",
                    strict=False,
                ),
            ],
        ),
        pytest.param(
            "pyarrow",
            marks=pytest.mark.skipif(
                not _HAVE_PYARROW, reason="pyarrow is not installed"
            ),
        ),
    ]
)
def engine(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def pa() -> str:
    if not _HAVE_PYARROW:
        pytest.skip("pyarrow is not installed")
    return "pyarrow"


@pytest.fixture
def fp(request: pytest.FixtureRequest) -> str:
    if not _HAVE_FASTPARQUET:
        pytest.skip("fastparquet is not installed")
    if using_string_dtype():
        request.applymarker(
            pytest.mark.xfail(reason="TODO(infer_string) fastparquet", strict=False)
        )
    return "fastparquet"


@pytest.fixture
def df_compat() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [1, 2, 3], "B": "foo"}, columns=pd.Index(["A", "B"])
    )


@pytest.fixture
def df_cross_compat() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "a": list("abc"),
            "b": list(range(1, 4)),
            "d": np.arange(4.0, 7.0, dtype="float64"),
            "e": [True, False, True],
            "f": pd.date_range("20130101", periods=3),
        }
    )
    return df


@pytest.fixture
def df_full() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "string": list("abc"),
            "string_with_nan": ["a", np.nan, "c"],
            "string_with_none": ["a", None, "c"],
            "bytes": [b"foo", b"bar", b"baz"],
            "unicode": ["foo", "bar", "baz"],
            "int": list(range(1, 4)),
            "uint": np.arange(3, 6).astype("u1"),
            "float": np.arange(4.0, 7.0, dtype="float64"),
            "float_with_nan": [2.0, np.nan, 3.0],
            "bool": [True, False, True],
            "datetime": pd.date_range("20130101", periods=3),
            "datetime_with_nat": [
                pd.Timestamp("20130101"),
                pd.NaT,
                pd.Timestamp("20130103"),
            ],
        }
    )


@pytest.fixture(
    params=[
        datetime.datetime.now(datetime.timezone.utc),
        datetime.datetime.now(datetime.timezone.min),
        datetime.datetime.now(datetime.timezone.max),
        datetime.datetime.strptime("2019-01-04T16:41:24+0200", "%Y-%m-%dT%H:%M:%S%z"),
        datetime.datetime.strptime("2019-01-04T16:41:24+0215", "%Y-%m-%dT%H:%M:%S%z"),
        datetime.datetime.strptime("2019-01-04T16:41:24-0200", "%Y-%m-%dT%H:%M:%S%z"),
        datetime.datetime.strptime("2019-01-04T16:41:24-0215", "%Y-%m-%dT%H:%M:%S%z"),
    ]
)
def timezone_aware_date_list(request: pytest.FixtureRequest) -> datetime.datetime:
    return request.param


def check_round_trip(
    df: pd.DataFrame,
    engine: Optional[str] = None,
    path: Optional[Union[str, pathlib.Path]] = None,
    write_kwargs: Optional[Dict[str, Any]] = None,
    read_kwargs: Optional[Dict[str, Any]] = None,
    expected: Optional[pd.DataFrame] = None,
    check_names: bool = True,
    check_like: bool = False,
    check_dtype: bool = True,
    repeat: int = 2,
) -> None:
    """Verify parquet serializer and deserializer produce the same results.

    Performs a pandas to disk and disk to pandas round trip,
    then compares the 2 resulting DataFrames to verify equality.

    Parameters
    ----------
    df: Dataframe
    engine: str, optional
        'pyarrow' or 'fastparquet'
    path: str, optional
    write_kwargs: dict of str:str, optional
    read_kwargs: dict of str:str, optional
    expected: DataFrame, optional
        Expected deserialization result, otherwise will be equal to `df`
    check_names: list of str, optional
        Closed set of column names to be compared
    check_like: bool, optional
        If True, ignore the order of index & columns.
    repeat: int, optional
        How many times to repeat the test
    """
    write_kwargs = write_kwargs or {"compression": None}
    read_kwargs = read_kwargs or {}
    if expected is None:
        expected = df
    if engine:
        write_kwargs["engine"] = engine
        read_kwargs["engine"] = engine

    def compare(repeat: int) -> None:
        for _ in range(repeat):
            df.to_parquet(path, **write_kwargs)
            actual = read_parquet(path, **read_kwargs)
            if "string_with_nan" in expected:
                expected.loc[1, "string_with_nan"] = None
            tm.assert_frame_equal(
                expected,
                actual,
                check_names=check_names,
                check_like=check_like,
                check_dtype=check_dtype,
            )

    if path is None:
        with tm.ensure_clean() as path:
            compare(repeat)
    else:
        compare(repeat)


def check_partition_names(path: Union[str, pathlib.Path], expected: Iterable[str]) -> None:
    """Check partitions of a parquet file are as expected.

    Parameters
    ----------
    path: str
        Path of the dataset.
    expected: iterable of str
        Expected partition names.
    """
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, partitioning="hive")
    assert dataset.partitioning.schema.names == expected


def test_invalid_engine(df_compat: pd.DataFrame) -> None:
    msg = "engine must be one of 'pyarrow', 'fastparquet'"
    with pytest.raises(ValueError, match=msg):
        check_round_trip(df_compat, "foo", "bar")


def test_options_py(df_compat: pd.DataFrame, pa: str, using_infer_string: bool) -> None:
    if using_infer_string and (not pa_version_under19p0):
        df_compat.columns = df_compat.columns.astype("str")
    with pd.option_context("io.parquet.engine", "pyarrow"):
        check_round_trip(df_compat)


def test_options_fp(df_compat: pd.DataFrame, fp: str) -> None:
    with pd.option_context("io.parquet.engine", "fastparquet"):
        check_round_trip(df_compat)


def test_options_auto(df_compat: pd.DataFrame, fp: str, pa: str) -> None:
    with pd.option_context("io.parquet.engine", "auto"):
        check_round_trip(df_compat)


def test_options_get_engine(fp: str, pa: str) -> None:
    assert isinstance(get_engine("pyarrow"), PyArrowImpl)
    assert isinstance(get_engine("fastparquet"), FastParquetImpl)
    with pd.option_context("io.parquet.engine", "pyarrow"):
        assert isinstance(get_engine("auto"), PyArrowImpl)
        assert isinstance(get_engine("pyarrow"), PyArrowImpl)
        assert isinstance(get_engine("fastparquet"), FastParquetImpl)
    with pd.option_context("io.parquet.engine", "fastparquet"):
        assert isinstance(get_engine("auto"), FastParquetImpl)
        assert isinstance(get_engine("pyarrow"), PyArrowImpl)
        assert isinstance(get_engine("fastparquet"), FastParquetImpl)
    with pd.option_context("io.parquet.engine", "auto"):
        assert isinstance(get_engine("auto"), PyArrowImpl)
        assert isinstance(get_engine("pyarrow"), PyArrowImpl)
        assert isinstance(get_engine("fastparquet"), FastParquetImpl)


def test_get_engine_auto_error_message() -> None:
    from pandas.compat._optional import VERSIONS

    pa_min_ver = VERSIONS.get("pyarrow")
    fp_min_ver = VERSIONS.get("fastparquet")
    have_pa_bad_version = (
        False
        if not _HAVE_PYARROW
        else Version(pyarrow.__version__) < Version(pa_min_ver)
    )
    have_fp_bad_version = (
        False
        if not _HAVE_FASTPARQUET
        else Version(fastparquet.__version__) < Version(fp_min_ver)
    )
    have_usable_pa = _HAVE_PYARROW and (not have_pa_bad_version)
    have_usable_fp = _HAVE_FASTPARQUET and (not have_fp_bad_version)
    if not have_usable_pa and (not have_usable_fp):
        if have_pa_bad_version:
            match = f"Pandas requires version .{pa_min_ver}. or newer of .pyarrow."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")
        else:
            match = "Missing optional dependency .pyarrow."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")
        if have_fp_bad_version:
            match = f"Pandas requires version .{fp_min_ver}. or newer of .fastparquet."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")
        else:
            match = "Missing optional dependency .fastparquet."
            with pytest.raises(ImportError, match=match):
                get_engine("auto")


def test_cross_engine_pa_fp(
    df_cross_compat: pd.DataFrame, pa: str, fp: str
) -> None:
    df = df_cross_compat
    with tm.ensure_clean() as path:
        df.to_parquet(path, engine=pa, compression=None)
        result = read_parquet(path, engine=fp)
        tm.assert_frame_equal(result, df)
        result = read_parquet(path, engine=fp, columns=["a", "d"])
        tm.assert_frame_equal(result, df[["a", "d"]])


def test_cross_engine_fp_pa(
    df_cross_compat: pd.DataFrame, pa: str, fp: str
) -> None:
    df = df_cross_compat
    with tm.ensure_clean() as path:
        df.to_parquet(path, engine=fp, compression=None)
        result = read_parquet(path, engine=pa)
        tm.assert_frame_equal(result, df)
        result = read_parquet(path, engine=pa, columns=["a", "d"])
        tm.assert_frame_equal(result, df[["a", "d"]])


class Base:
    def check_error_on_write(
        self,
        df: pd.DataFrame,
        engine: str,
        exc: type[Exception],
        err_msg: Optional[str],
    ) -> None:
        with tm.ensure_clean() as path:
            with pytest.raises(exc, match=err_msg):
                to_parquet(df, path, engine, compression=None)

    def check_external_error_on_write(
        self, df: pd.DataFrame, engine: str, exc: type[Exception]
    ) -> None:
        with tm.ensure_clean() as path:
            with tm.external_error_raised(exc):
                to_parquet(df, path, engine, compression=None)


class TestBasic(Base):
    def test_error(self, engine: str) -> None:
        for obj in [
            pd.Series([1, 2, 3]),
            1,
            "foo",
            pd.Timestamp("20130101"),
            np.array([1, 2, 3]),
        ]:
            msg = "to_parquet only supports IO with DataFrames"
            self.check_error_on_write(obj, engine, ValueError, msg)

    def test_columns_dtypes(self, engine: str) -> None:
        df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})
        df.columns = ["foo", "bar"]
        check_round_trip(df, engine)

    @pytest.mark.parametrize("compression", [None, "gzip", "snappy", "brotli"])
    def test_compression(self, engine: str, compression: Optional[str]) -> None:
        df = pd.DataFrame({"A": [1, 2, 3]})
        check_round_trip(df, engine, write_kwargs={"compression": compression})

    def test_read_columns(self, engine: str) -> None:
        df = pd.DataFrame({"string": list("abc"), "int": list(range(1, 4))})
        expected = pd.DataFrame({"string": list("abc")})
        check_round_trip(
            df, engine, expected=expected, read_kwargs={"columns": ["string"]}
        )

    def test_read_filters(
        self, engine: str, tmp_path: pathlib.Path
    ) -> None:
        df = pd.DataFrame({"int": list(range(4)), "part": list("aabb")})
        expected = pd.DataFrame({"int": [0, 1]})
        check_round_trip(
            df,
            engine,
            path=tmp_path,
            expected=expected,
            write_kwargs={"partition_cols": ["part"]},
            read_kwargs={"filters": [("part", "==", "a")], "columns": ["int"]},
            repeat=1,
        )

    def test_write_index(self) -> None:
        pytest.importorskip("pyarrow")
        df = pd.DataFrame({"A": [1, 2, 3]})
        check_round_trip(df, "pyarrow")
        indexes = [
            [2, 3, 4],
            pd.date_range("20130101", periods=3),
            list("abc"),
            [1, 3, 4],
        ]
        for index in indexes:
            df.index = index
            if isinstance(index, pd.DatetimeIndex):
                df.index = df.index._with_freq(None)
            check_round_trip(df, "pyarrow")
        df.index = [0, 1, 2]
        df.index.name = "foo"
        check_round_trip(df, "pyarrow")

    def test_write_multiindex(self, pa: str) -> None:
        engine = pa
        df = pd.DataFrame({"A": [1, 2, 3]})
        index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
        df.index = index
        check_round_trip(df, engine)

    def test_multiindex_with_columns(self, pa: str) -> None:
        engine = pa
        dates = pd.date_range("01-Jan-2018", "01-Dec-2018", freq="MS")
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((2 * len(dates), 3)),
            columns=list("ABC"),
        )
        index1 = pd.MultiIndex.from_product(
            [["Level1", "Level2"], dates], names=["level", "date"]
        )
        index2 = index1.copy(names=None)
        for index in [index1, index2]:
            df.index = index
            check_round_trip(df, engine)
            check_round_trip(
                df,
                engine,
                read_kwargs={"columns": ["A", "B"]},
                expected=df[["A", "B"]],
            )

    def test_write_ignoring_index(self, engine: str) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["q", "r", "s"]})
        write_kwargs = {"compression": None, "index": False}
        expected = df.reset_index(drop=True)
        check_round_trip(df, engine, write_kwargs=write_kwargs, expected=