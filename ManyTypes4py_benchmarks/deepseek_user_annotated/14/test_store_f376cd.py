import contextlib
import datetime as dt
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas.compat import PY312

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.conftest import has_pyarrow
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)

from pandas.io.pytables import (
    HDFStore,
    read_hdf,
)

pytestmark = [pytest.mark.single_cpu]

tables = pytest.importorskip("tables")


def test_context(setup_path: str) -> None:
    with tm.ensure_clean(setup_path) as path:
        try:
            with HDFStore(path) as tbl:
                raise ValueError("blah")
        except ValueError:
            pass
    with tm.ensure_clean(setup_path) as path:
        with HDFStore(path) as tbl:
            tbl["a"] = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )
            assert len(tbl) == 1
            assert type(tbl["a"]) == DataFrame


def test_no_track_times(tmp_path: Path, setup_path: str) -> None:
    def checksum(filename: str, hash_factory: Any = hashlib.md5, chunk_num_blocks: int = 128) -> Any:
        h = hash_factory()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b""):
                h.update(chunk)
        return h.digest()

    def create_h5_and_return_checksum(tmp_path: Path, track_times: bool) -> Any:
        path = tmp_path / setup_path
        df = DataFrame({"a": [1]})

        with HDFStore(path, mode="w") as hdf:
            hdf.put(
                "table",
                df,
                format="table",
                data_columns=True,
                index=None,
                track_times=track_times,
            )

        return checksum(path)

    checksum_0_tt_false = create_h5_and_return_checksum(tmp_path, track_times=False)
    checksum_0_tt_true = create_h5_and_return_checksum(tmp_path, track_times=True)

    time.sleep(1)

    checksum_1_tt_false = create_h5_and_return_checksum(tmp_path, track_times=False)
    checksum_1_tt_true = create_h5_and_return_checksum(tmp_path, track_times=True)

    assert checksum_0_tt_false == checksum_1_tt_false
    assert checksum_0_tt_true != checksum_1_tt_true


def test_iter_empty(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        assert list(store) == []


def test_repr(setup_path: str, performance_warning: Any, using_infer_string: bool) -> None:
    with ensure_clean_store(setup_path) as store:
        repr(store)
        store.info()
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        store["b"] = Series(
            range(10), dtype="float64", index=[f"i_{i}" for i in range(10)]
        )
        store["c"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )

        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        df["obj1"] = "foo"
        df["obj2"] = "bar"
        df["bool1"] = df["A"] > 0
        df["bool2"] = df["B"] > 0
        df["bool3"] = True
        df["int1"] = 1
        df["int2"] = 2
        df["timestamp1"] = Timestamp("20010102")
        df["timestamp2"] = Timestamp("20010103")
        df["datetime1"] = dt.datetime(2001, 1, 2, 0, 0)
        df["datetime2"] = dt.datetime(2001, 1, 3, 0, 0)
        df.loc[df.index[3:6], ["obj1"]] = np.nan
        df = df._consolidate()

        warning = None if using_infer_string else performance_warning
        msg = "cannot\nmap directly to c-types .* dtype='object'"
        with tm.assert_produces_warning(warning, match=msg):
            store["df"] = df

        store._handle.create_group(store._handle.root, "bah")

        assert store.filename in repr(store)
        assert store.filename in str(store)
        store.info()

    with ensure_clean_store(setup_path) as store:
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        store.append("df", df)

        s = store.get_storer("df")
        repr(s)
        str(s)


def test_contains(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        store["b"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        store["foo/bar"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        assert "a" in store
        assert "b" in store
        assert "c" not in store
        assert "foo/bar" in store
        assert "/foo/bar" in store
        assert "/foo/b" not in store
        assert "bar" not in store

        with tm.assert_produces_warning(
            tables.NaturalNameWarning, check_stacklevel=False
        ):
            store["node())"] = DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=Index([f"i-{i}" for i in range(30)], dtype=object),
            )
        assert "node())" in store


def test_versioning(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        store["a"] = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        store["b"] = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=20, freq="B"),
        )
        _maybe_remove(store, "df1")
        store.append("df1", df[:10])
        store.append("df1", df[10:])
        assert store.root.a._v_attrs.pandas_version == "0.15.2"
        assert store.root.b._v_attrs.pandas_version == "0.15.2"
        assert store.root.df1._v_attrs.pandas_version == "0.15.2"

        _maybe_remove(store, "df2")
        store.append("df2", df)

        store.get_node("df2")._v_attrs.pandas_version = None

        msg = "'NoneType' object has no attribute 'startswith'"

        with pytest.raises(Exception, match=msg):
            store.select("df2")


@pytest.mark.parametrize(
    "where, expected",
    [
        (
            "/",
            {
                "": ({"first_group", "second_group"}, set()),
                "/first_group": (set(), {"df1", "df2"}),
                "/second_group": ({"third_group"}, {"df3", "s1"}),
                "/second_group/third_group": (set(), {"df4"}),
            },
        ),
        (
            "/second_group",
            {
                "/second_group": ({"third_group"}, {"df3", "s1"}),
                "/second_group/third_group": (set(), {"df4"}),
            },
        ),
    ],
)
def test_walk(where: str, expected: Dict[str, Tuple[Set[str], Set[str]]]) -> None:
    objs = {
        "df1": DataFrame([1, 2, 3]),
        "df2": DataFrame([4, 5, 6]),
        "df3": DataFrame([6, 7, 8]),
        "df4": DataFrame([9, 10, 11]),
        "s1": Series([10, 9, 8]),
        "a1": np.array([[1, 2, 3], [4, 5, 6]]),
        "tb1": np.array([(1, 2, 3), (4, 5, 6)], dtype="i,i,i"),
        "tb2": np.array([(7, 8, 9), (10, 11, 12)], dtype="i,i,i"),
    }

    with ensure_clean_store("walk_groups.hdf", mode="w") as store:
        store.put("/first_group/df1", objs["df1"])
        store.put("/first_group/df2", objs["df2"])
        store.put("/second_group/df3", objs["df3"])
        store.put("/second_group/s1", objs["s1"])
        store.put("/second_group/third_group/df4", objs["df4"])
        store._handle.create_array("/first_group", "a1", objs["a1"])
        store._handle.create_table("/first_group", "tb1", obj=objs["tb1"])
        store._handle.create_table("/second_group", "tb2", obj=objs["tb2"])

        assert len(list(store.walk(where=where))) == len(expected)
        for path, groups, leaves in store.walk(where=where):
            assert path in expected
            expected_groups, expected_frames = expected[path]
            assert expected_groups == set(groups)
            assert expected_frames == set(leaves)
            for leaf in leaves:
                frame_path = "/".join([path, leaf])
                obj = store.get(frame_path)
                if "df" in leaf:
                    tm.assert_frame_equal(obj, objs[leaf])
                else:
                    tm.assert_series_equal(obj, objs[leaf])


def test_getattr(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:
        s = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        store["a"] = s

        result = store.a
        tm.assert_series_equal(result, s)
        result = store.a
        tm.assert_series_equal(result, s)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        store["df"] = df
        result = store.df
        tm.assert_frame_equal(result, df)

        for x in ["d", "mode", "path", "handle", "complib"]:
            msg = f"'HDFStore' object has no attribute '{x}'"
            with pytest.raises(AttributeError, match=msg):
                getattr(store, x)

        for x in ["mode", "path", "handle", "complib"]:
            getattr(store, f"_{x}")


def test_store_dropna(tmp_path: Path, setup_path: str) -> None:
    df_with_missing = DataFrame(
        {"col1": [0.0, np.nan, 2.0], "col2": [1.0, np.nan, np.nan]},
        index=list("abc"),
    )
    df_without_missing = DataFrame(
        {"col1": [0.0, 2.0], "col2": [1.0, np.nan]}, index=list("ac")
    )

    path = tmp_path / setup_path
    df_with_missing.to_hdf(path, key="df", format="table")
    reloaded = read_hdf(path, "df")
    tm.assert_frame_equal(df_with_missing, reloaded)

    path = tmp_path / setup_path
    df_with_missing.to_hdf(path, key="df", format="table", dropna=False)
    reloaded = read_hdf(path, "df")
    tm.assert_frame_equal(df_with_missing, reloaded)

    path = tmp_path / setup_path
    df_with_missing.to_hdf(path, key="df", format="table", dropna=True)
    reloaded = read_hdf(path, "df")
    tm.assert_frame_equal(df_without_missing, reloaded)


def test_to_hdf_with_min_itemsize(tmp_path: Path, setup_path: str) -> None:
    path = tmp_path / setup_path

    df = DataFrame(
        {
            "A": [0.0, 1.0, 2.0, 3.0, 4.0],
            "B": [0.0, 1.0, 0.0, 1.0, 0.0],
            "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"]),
            "D": date_range("20130101", periods=5),
        }
    ).set_index("C")
    df.to_hdf(path, key="ss3", format="table", min_itemsize={"index": 6})
    df2 = df.copy().reset_index().assign(C="longer").set_index("C")
    df2.to_hdf(path, key="ss3", append=True, format="table")
    tm.assert_frame_equal(read_hdf(path, "ss3"), concat([df, df2]))

    df["B"].to_hdf(path, key="ss4", format="table", min_itemsize={"index": 6})
    df2["B"].to_hdf(path, key="ss4", append=True, format="table")
    tm.assert_series_equal(read_hdf(path, "ss4"), concat([df["B"], df2["B"]]))


@pytest.mark.xfail(
    using_string_dtype() and has_pyarrow,
    reason="TODO(infer_string): can't encode '\ud800': surrogates not allowed",
)
@pytest.mark.parametrize("format", ["fixed", "table"])
def test_to_hdf_errors(tmp_path: Path, format: str, setup_path: str) -> None:
    data = ["\ud800foo"]
    ser = Series(data, index=Index(data))
    path = tmp_path / setup_path
    ser.to_hdf(path, key="table", format=format, errors="surrogatepass")

    result = read_hdf(path, "table", errors="surrogatepass")
    tm.assert_series_equal(result, ser)


def test_create_table_index(setup_path: str) -> None:
    with ensure_clean_store(setup_path) as store:

        def col(t: str, column: str) -> Any:
            return getattr(store.get_storer(t).table.cols, column)

        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD")),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        df["string"] = "foo"
        df["string2"] = "bar"
        store.append("f", df, data_columns=["string", "string2"])
        assert col("f", "index").is_indexed is True
        assert col("f", "string").is_indexed is True
        assert col("f", "string2").is_indexed is True

        store.append("f2", df, index=["string"], data_columns=["string", "string2"])
        assert col("f2", "index").is_indexed is False
        assert col("f2", "string").is_indexed is True
        assert col("f2", "string2").is_indexed is False

        _maybe_remove(store, "f2")
        store.put("f2", df)
        msg = "cannot create table index on a Fixed format store"
        with pytest.raises(TypeError, match=msg):
            store.create_table_index("f2")


def test_create_table_index_data_columns_argument(setup_path: str) -> None:
    with ensure_clean_store(set