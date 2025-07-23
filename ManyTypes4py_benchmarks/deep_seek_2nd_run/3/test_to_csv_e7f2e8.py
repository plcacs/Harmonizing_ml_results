import csv
from io import StringIO
import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, NaT, Series, Timestamp, date_range, period_range, read_csv, to_datetime
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle

class TestDataFrameToCSV:

    def read_csv(self, path: str, **kwargs: Any) -> DataFrame:
        params: Dict[str, Any] = {'index_col': 0}
        params.update(**kwargs)
        return read_csv(path, **params)

    def test_to_csv_from_csv1(self, temp_file: Any, float_frame: DataFrame) -> None:
        path = str(temp_file)
        float_frame.iloc[:5, float_frame.columns.get_loc('A')] = np.nan
        float_frame.to_csv(path)
        float_frame.to_csv(path, columns=['A', 'B'])
        float_frame.to_csv(path, header=False)
        float_frame.to_csv(path, index=False)

    def test_to_csv_from_csv1_datetime(self, temp_file: Any, datetime_frame: DataFrame) -> None:
        path = str(temp_file)
        datetime_frame.index = datetime_frame.index._with_freq(None)
        datetime_frame.to_csv(path)
        recons = self.read_csv(path, parse_dates=True)
        expected = datetime_frame.copy()
        expected.index = expected.index.as_unit('s')
        tm.assert_frame_equal(expected, recons)
        datetime_frame.to_csv(path, index_label='index')
        recons = self.read_csv(path, index_col=None, parse_dates=True)
        assert len(recons.columns) == len(datetime_frame.columns) + 1
        datetime_frame.to_csv(path, index=False)
        recons = self.read_csv(path, index_col=None, parse_dates=True)
        tm.assert_almost_equal(datetime_frame.values, recons.values)

    def test_to_csv_from_csv1_corner_case(self, temp_file: Any) -> None:
        path = str(temp_file)
        dm = DataFrame({'s1': Series(range(3), index=np.arange(3, dtype=np.int64)), 's2': Series(range(2), index=np.arange(2, dtype=np.int64))})
        dm.to_csv(path)
        recons = self.read_csv(path)
        tm.assert_frame_equal(dm, recons)

    def test_to_csv_from_csv2(self, temp_file: Any, float_frame: DataFrame) -> None:
        path = str(temp_file)
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=['a', 'a', 'b'], columns=['x', 'y', 'z'])
        df.to_csv(path)
        result = self.read_csv(path)
        tm.assert_frame_equal(result, df)
        midx = MultiIndex.from_tuples([('A', 1, 2), ('A', 1, 2), ('B', 1, 2)])
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=midx, columns=['x', 'y', 'z'])
        df.to_csv(path)
        result = self.read_csv(path, index_col=[0, 1, 2], parse_dates=False)
        tm.assert_frame_equal(result, df, check_names=False)
        col_aliases = Index(['AA', 'X', 'Y', 'Z'])
        float_frame.to_csv(path, header=col_aliases)
        rs = self.read_csv(path)
        xp = float_frame.copy()
        xp.columns = col_aliases
        tm.assert_frame_equal(xp, rs)
        msg = 'Writing 4 cols but got 2 aliases'
        with pytest.raises(ValueError, match=msg):
            float_frame.to_csv(path, header=['AA', 'X'])

    def test_to_csv_from_csv3(self, temp_file: Any) -> None:
        path = str(temp_file)
        df1 = DataFrame(np.random.default_rng(2).standard_normal((3, 1)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((3, 1)))
        df1.to_csv(path)
        df2.to_csv(path, mode='a', header=False)
        xp = pd.concat([df1, df2])
        rs = read_csv(path, index_col=0)
        rs.columns = [int(label) for label in rs.columns]
        xp.columns = [int(label) for label in xp.columns]
        tm.assert_frame_equal(xp, rs)

    def test_to_csv_from_csv4(self, temp_file: Any) -> None:
        path = str(temp_file)
        dt = pd.Timedelta(seconds=1)
        df = DataFrame({'dt_data': [i * dt for i in range(3)]}, index=Index([i * dt for i in range(3)], name='dt_index'))
        df.to_csv(path)
        result = read_csv(path, index_col='dt_index')
        result.index = pd.to_timedelta(result.index)
        result['dt_data'] = pd.to_timedelta(result['dt_data'])
        tm.assert_frame_equal(df, result, check_index_type=True)

    def test_to_csv_from_csv5(self, temp_file: Any, timezone_frame: DataFrame) -> None:
        path = str(temp_file)
        timezone_frame.to_csv(path)
        result = read_csv(path, index_col=0, parse_dates=['A'])
        converter = lambda c: to_datetime(result[c]).dt.tz_convert('UTC').dt.tz_convert(timezone_frame[c].dt.tz).dt.as_unit('ns')
        result['B'] = converter('B')
        result['C'] = converter('C')
        result['A'] = result['A'].dt.as_unit('ns')
        tm.assert_frame_equal(result, timezone_frame)

    def test_to_csv_cols_reordering(self, temp_file: Any) -> None:
        chunksize = 5
        N = int(chunksize * 2.5)
        df = DataFrame(np.ones((N, 3)), index=Index([f'i-{i}' for i in range(N)], name='a'), columns=Index([f'i-{i}' for i in range(3)], name='a'))
        cs = df.columns
        cols = [cs[2], cs[0]]
        path = str(temp_file)
        df.to_csv(path, columns=cols, chunksize=chunksize)
        rs_c = read_csv(path, index_col=0)
        tm.assert_frame_equal(df[cols], rs_c, check_names=False)

    @pytest.mark.parametrize('cols', [None, ['b', 'a']])
    def test_to_csv_new_dupe_cols(self, temp_file: Any, cols: Optional[List[str]]) -> None:
        chunksize = 5
        N = int(chunksize * 2.5)
        df = DataFrame(np.ones((N, 3)), index=Index([f'i-{i}' for i in range(N)], name='a'), columns=['a', 'a', 'b'])
        path = str(temp_file)
        df.to_csv(path, columns=cols, chunksize=chunksize)
        rs_c = read_csv(path, index_col=0)
        if cols is not None:
            if df.columns.is_unique:
                rs_c.columns = cols
            else:
                indexer, missing = df.columns.get_indexer_non_unique(cols)
                rs_c.columns = df.columns.take(indexer)
            for c in cols:
                obj_df = df[c]
                obj_rs = rs_c[c]
                if isinstance(obj_df, Series):
                    tm.assert_series_equal(obj_df, obj_rs)
                else:
                    tm.assert_frame_equal(obj_df, obj_rs, check_names=False)
        else:
            rs_c.columns = df.columns
            tm.assert_frame_equal(df, rs_c, check_names=False)

    @pytest.mark.slow
    def test_to_csv_dtnat(self, temp_file: Any) -> None:

        def make_dtnat_arr(n: int, nnat: Optional[int] = None) -> List[Union[Timestamp, NaT]]:
            if nnat is None:
                nnat = int(n * 0.1)
            s = list(date_range('2000', freq='5min', periods=n))
            if nnat:
                for i in np.random.default_rng(2).integers(0, len(s), nnat):
                    s[i] = NaT
                i = np.random.default_rng(2).integers(100)
                s[-i] = NaT
                s[i] = NaT
            return s
        chunksize = 1000
        s1 = make_dtnat_arr(chunksize + 5)
        s2 = make_dtnat_arr(chunksize + 5, 0)
        path = str(temp_file)
        df = DataFrame({'a': s1, 'b': s2})
        df.to_csv(path, chunksize=chunksize)
        result = self.read_csv(path).apply(to_datetime)
        expected = df[:]
        expected['a'] = expected['a'].astype('M8[s]')
        expected['b'] = expected['b'].astype('M8[s]')
        tm.assert_frame_equal(result, expected, check_names=False)

    def _return_result_expected(self, df: DataFrame, chunksize: int, r_dtype: Optional[str] = None, c_dtype: Optional[str] = None, rnlvl: Optional[int] = None, cnlvl: Optional[int] = None, dupe_col: bool = False) -> Tuple[DataFrame, DataFrame]:
        kwargs: Dict[str, Any] = {'parse_dates': False}
        if cnlvl:
            if rnlvl is not None:
                kwargs['index_col'] = list(range(rnlvl))
            kwargs['header'] = list(range(cnlvl))
            with tm.ensure_clean('__tmp_to_csv_moar__') as path:
                df.to_csv(path, encoding='utf8', chunksize=chunksize)
                recons = self.read_csv(path, **kwargs)
        else:
            kwargs['header'] = 0
            with tm.ensure_clean('__tmp_to_csv_moar__') as path:
                df.to_csv(path, encoding='utf8', chunksize=chunksize)
                recons = self.read_csv(path, **kwargs)

        def _to_uni(x: Any) -> Any:
            if not isinstance(x, str):
                return x.decode('utf8')
            return x
        if dupe_col:
            recons.columns = df.columns
        if rnlvl and (not cnlvl):
            delta_lvl = [recons.iloc[:, i].values for i in range(rnlvl - 1)]
            ix = MultiIndex.from_arrays([list(recons.index)] + delta_lvl)
            recons.index = ix
            recons = recons.iloc[:, rnlvl - 1:]
        type_map = {'i': 'i', 'f': 'f', 's': 'O', 'u': 'O', 'dt': 'O', 'p': 'O'}
        if r_dtype:
            if r_dtype == 'u':
                r_dtype = 'O'
                recons.index = np.array([_to_uni(label) for label in recons.index], dtype=r_dtype)
                df.index = np.array([_to_uni(label) for label in df.index], dtype=r_dtype)
            elif r_dtype == 'dt':
                r_dtype = 'O'
                recons.index = np.array([Timestamp(label) for label in recons.index], dtype=r_dtype)
                df.index = np.array([Timestamp(label) for label in df.index], dtype=r_dtype)
            elif r_dtype == 'p':
                r_dtype = 'O'
                idx_list = to_datetime(recons.index)
                recons.index = np.array([Timestamp(label) for label in idx_list], dtype=r_dtype)
                df.index = np.array(list(map(Timestamp, df.index.to_timestamp())), dtype=r_dtype)
            else:
                r_dtype = type_map.get(r_dtype)
                recons.index = np.array(recons.index, dtype=r_dtype)
                df.index = np.array(df.index, dtype=r_dtype)
        if c_dtype:
            if c_dtype == 'u':
                c_dtype = 'O'
                recons.columns = np.array([_to_uni(label) for label in recons.columns], dtype=c_dtype)
                df.columns = np.array([_to_uni(label) for label in df.columns], dtype=c_dtype)
            elif c_dtype == 'dt':
                c_dtype = 'O'
                recons.columns = np.array([Timestamp(label) for label in recons.columns], dtype=c_dtype)
                df.columns = np.array([Timestamp(label) for label in df.columns], dtype=c_dtype)
            elif c_dtype == 'p':
                c_dtype = 'O'
                col_list = to_datetime(recons.columns)
                recons.columns = np.array([Timestamp(label) for label in col_list], dtype=c_dtype)
                col_list = df.columns.to_timestamp()
                df.columns = np.array([Timestamp(label) for label in col_list], dtype=c_dtype)
            else:
                c_dtype = type_map.get(c_dtype)
                recons.columns = np.array(recons.columns, dtype=c_dtype)
                df.columns = np.array(df.columns, dtype=c_dtype)
        return (df, recons)

    @pytest.mark.slow
    @pytest.mark.parametrize('nrows', [2, 10, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251])
    def test_to_csv_nrows(self, nrows: int) -> None:
        df = DataFrame(np.ones((nrows, 4)), index=date_range('2020-01-01', periods=nrows), columns=Index(list('abcd'), dtype=object))
        result, expected = self._return_result_expected(df, 1000, 'dt', 's')
        expected.index = expected.index.astype('M8[ns]')
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize('nrows', [2, 10, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251])
    @pytest.mark.parametrize('r_idx_type, c_idx_type', [('i', 'i'), ('s', 's'), ('s', 'dt'), ('p', 'p')])
    @pytest.mark.parametrize('ncols', [1, 2, 3, 4])
    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_to_csv_idx_types(self, nrows: int, r_idx_type: str, c_idx_type: str, ncols: int) -> None:
        axes = {'i': lambda n: Index(np.arange(n), dtype=np.int64), 's': lambda n: Index([f'{i}_{chr(i)}' for i in range(97, 97 + n)]), 'dt': lambda n: date_range('2020-01-01', periods=n), 'p': lambda n: period_range('2020-01-01', periods=n, freq='D')}
        df = DataFrame(np.ones((nrows, ncols)), index=axes[r_idx_type](nrows), columns=axes[c_idx_type](ncols))
        result, expected = self._return_result_expected(df, 1000, r_idx_type, c_idx_type)
        if r_idx_type in ['dt', 'p']:
            expected.index = expected.index.astype('M8[ns]')
        if c_idx_type in ['dt', 'p']:
            expected.columns = expected.columns.astype('M8[ns]')
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize('nrows', [10, 98, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251])
    @pytest.mark.parametrize('ncols', [1, 2, 3, 4])
    def test_to_csv_idx_ncols(self, nrows: int, ncols: int) -> None:
        df = DataFrame(np.ones((nrows, ncols)), index=Index([f'i-{i}' for i in range(nrows)], name='a'), columns=Index([f'i-{i}' for i in range(ncols)], name='a'))
        result, expected = self._return_result_expected(df, 1000)
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize('nrows', [10, 98, 99, 100, 101, 102])
    def test_to_csv_dup_cols(self, nrows: int) -> None:
        df = DataFrame(np.ones((nrows, 3)), index=Index([f'i-{i}' for i in range(nrows)], name='a'), columns=Index([f'i-{i}' for i in range(3)], name='a'))
        cols = list(df.columns)
        cols[:2] = ['dupe', 'dupe']
        cols[-2:] = ['dupe', 'dupe']
        ix = list(df.index)
        ix[:2] = ['rdupe', 'rdupe']
        ix[-2:] = ['rdupe', 'rdupe']
        df.index = ix
        df.columns = cols
        result, expected = self._return_result_expected(df, 1000, dupe_col=True)
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    def test_to_csv_empty(self) -> None:
        df = DataFrame(index=np.arange(10, dtype=np.int64))
        result, expected = self._return_result_expected(df, 1000)
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.slow
    def test_to_csv_chunksize(self