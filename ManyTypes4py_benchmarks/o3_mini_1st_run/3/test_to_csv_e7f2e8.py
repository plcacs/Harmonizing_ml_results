from io import StringIO
import csv
import os
from typing import Any, Optional, Union, Tuple, List

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Index, MultiIndex, NaT, Series, Timestamp, date_range, period_range, read_csv, to_datetime
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle


class TestDataFrameToCSV:
    def read_csv(self, path: Union[str, StringIO], **kwargs: Any) -> pd.DataFrame:
        params: dict[str, Any] = {'index_col': 0}
        params.update(**kwargs)
        return read_csv(path, **params)

    def test_to_csv_from_csv1(self, temp_file: Any, float_frame: pd.DataFrame) -> None:
        path: str = str(temp_file)
        float_frame.iloc[:5, float_frame.columns.get_loc('A')] = np.nan
        float_frame.to_csv(path)
        float_frame.to_csv(path, columns=['A', 'B'])
        float_frame.to_csv(path, header=False)
        float_frame.to_csv(path, index=False)

    def test_to_csv_from_csv1_datetime(self, temp_file: Any, datetime_frame: pd.DataFrame) -> None:
        path: str = str(temp_file)
        datetime_frame.index = datetime_frame.index._with_freq(None)
        datetime_frame.to_csv(path)
        recons: pd.DataFrame = self.read_csv(path, parse_dates=True)
        expected: pd.DataFrame = datetime_frame.copy()
        expected.index = expected.index.as_unit('s')
        tm.assert_frame_equal(expected, recons)
        datetime_frame.to_csv(path, index_label='index')
        recons = self.read_csv(path, index_col=None, parse_dates=True)
        assert len(recons.columns) == len(datetime_frame.columns) + 1
        datetime_frame.to_csv(path, index=False)
        recons = self.read_csv(path, index_col=None, parse_dates=True)
        tm.assert_almost_equal(datetime_frame.values, recons.values)

    def test_to_csv_from_csv1_corner_case(self, temp_file: Any) -> None:
        path: str = str(temp_file)
        dm: pd.DataFrame = DataFrame({
            's1': Series(range(3), index=np.arange(3, dtype=np.int64)),
            's2': Series(range(2), index=np.arange(2, dtype=np.int64))
        })
        dm.to_csv(path)
        recons: pd.DataFrame = self.read_csv(path)
        tm.assert_frame_equal(dm, recons)

    def test_to_csv_from_csv2(self, temp_file: Any, float_frame: pd.DataFrame) -> None:
        path: str = str(temp_file)
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 3)),
                                     index=['a', 'a', 'b'], columns=['x', 'y', 'z'])
        df.to_csv(path)
        result: pd.DataFrame = self.read_csv(path)
        tm.assert_frame_equal(result, df)
        midx: MultiIndex = MultiIndex.from_tuples([('A', 1, 2), ('A', 1, 2), ('B', 1, 2)])
        df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)),
                       index=midx, columns=['x', 'y', 'z'])
        df.to_csv(path)
        result = self.read_csv(path, index_col=[0, 1, 2], parse_dates=False)
        tm.assert_frame_equal(result, df, check_names=False)
        col_aliases: Index = Index(['AA', 'X', 'Y', 'Z'])
        float_frame.to_csv(path, header=col_aliases)
        rs: pd.DataFrame = self.read_csv(path)
        xp: pd.DataFrame = float_frame.copy()
        xp.columns = col_aliases
        tm.assert_frame_equal(xp, rs)
        msg: str = 'Writing 4 cols but got 2 aliases'
        with pytest.raises(ValueError, match=msg):
            float_frame.to_csv(path, header=['AA', 'X'])

    def test_to_csv_from_csv3(self, temp_file: Any) -> None:
        path: str = str(temp_file)
        df1: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 1)))
        df2: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((3, 1)))
        df1.to_csv(path)
        df2.to_csv(path, mode='a', header=False)
        xp: pd.DataFrame = pd.concat([df1, df2])
        rs: pd.DataFrame = read_csv(path, index_col=0)
        rs.columns = [int(label) for label in rs.columns]
        xp.columns = [int(label) for label in xp.columns]
        tm.assert_frame_equal(xp, rs)

    def test_to_csv_from_csv4(self, temp_file: Any) -> None:
        path: str = str(temp_file)
        dt = pd.Timedelta(seconds=1)
        df: pd.DataFrame = DataFrame({'dt_data': [i * dt for i in range(3)]},
                                     index=Index([i * dt for i in range(3)], name='dt_index'))
        df.to_csv(path)
        result: pd.DataFrame = read_csv(path, index_col='dt_index')
        result.index = pd.to_timedelta(result.index)
        result['dt_data'] = pd.to_timedelta(result['dt_data'])
        tm.assert_frame_equal(df, result, check_index_type=True)

    def test_to_csv_from_csv5(self, temp_file: Any, timezone_frame: pd.DataFrame) -> None:
        path: str = str(temp_file)
        timezone_frame.to_csv(path)
        result: pd.DataFrame = read_csv(path, index_col=0, parse_dates=['A'])
        converter = lambda c: to_datetime(result[c]).dt.tz_convert('UTC').dt.tz_convert(timezone_frame[c].dt.tz).dt.as_unit('ns')
        result['B'] = converter('B')
        result['C'] = converter('C')
        result['A'] = result['A'].dt.as_unit('ns')
        tm.assert_frame_equal(result, timezone_frame)

    def test_to_csv_cols_reordering(self, temp_file: Any) -> None:
        chunksize: int = 5
        N: int = int(chunksize * 2.5)
        df: pd.DataFrame = DataFrame(np.ones((N, 3)),
                                     index=Index([f'i-{i}' for i in range(N)], name='a'),
                                     columns=Index([f'i-{i}' for i in range(3)], name='a'))
        cs: Index = df.columns
        cols: List[Any] = [cs[2], cs[0]]
        path: str = str(temp_file)
        df.to_csv(path, columns=cols, chunksize=chunksize)
        rs_c: pd.DataFrame = read_csv(path, index_col=0)
        tm.assert_frame_equal(df[cols], rs_c, check_names=False)

    @pytest.mark.parametrize('cols', [None, ['b', 'a']])
    def test_to_csv_new_dupe_cols(self, temp_file: Any, cols: Optional[List[str]]) -> None:
        chunksize: int = 5
        N: int = int(chunksize * 2.5)
        df: pd.DataFrame = DataFrame(np.ones((N, 3)),
                                     index=Index([f'i-{i}' for i in range(N)], name='a'),
                                     columns=['a', 'a', 'b'])
        path: str = str(temp_file)
        df.to_csv(path, columns=cols, chunksize=chunksize)
        rs_c: pd.DataFrame = read_csv(path, index_col=0)
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
        def make_dtnat_arr(n: int, nnat: Optional[int] = None) -> List[Any]:
            if nnat is None:
                nnat = int(n * 0.1)
            s: List[Any] = list(date_range('2000', freq='5min', periods=n))
            if nnat:
                for i in np.random.default_rng(2).integers(0, len(s), nnat):
                    s[i] = NaT
                i: int = int(np.random.default_rng(2).integers(100))
                s[-i] = NaT
                s[i] = NaT
            return s

        chunksize: int = 1000
        s1: List[Any] = make_dtnat_arr(chunksize + 5)
        s2: List[Any] = make_dtnat_arr(chunksize + 5, 0)
        path: str = str(temp_file)
        df: pd.DataFrame = DataFrame({'a': s1, 'b': s2})
        df.to_csv(path, chunksize=chunksize)
        result: pd.DataFrame = self.read_csv(path).apply(to_datetime)
        expected: pd.DataFrame = df.copy()
        expected['a'] = expected['a'].astype('M8[s]')
        expected['b'] = expected['b'].astype('M8[s]')
        tm.assert_frame_equal(result, expected, check_names=False)

    def _return_result_expected(
        self,
        df: pd.DataFrame,
        chunksize: int,
        r_dtype: Optional[str] = None,
        c_dtype: Optional[str] = None,
        rnlvl: Optional[int] = None,
        cnlvl: Optional[int] = None,
        dupe_col: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        kwargs: dict[str, Any] = {'parse_dates': False}
        if cnlvl:
            if rnlvl is not None:
                kwargs['index_col'] = list(range(rnlvl))
            kwargs['header'] = list(range(cnlvl))
            with tm.ensure_clean('__tmp_to_csv_moar__') as path:
                df.to_csv(path, encoding='utf8', chunksize=chunksize)
                recons: pd.DataFrame = self.read_csv(path, **kwargs)
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
            ix: MultiIndex = MultiIndex.from_arrays([list(recons.index)] + delta_lvl)
            recons.index = ix
            recons = recons.iloc[:, rnlvl - 1:]
        type_map: dict[str, str] = {'i': 'i', 'f': 'f', 's': 'O', 'u': 'O', 'dt': 'O', 'p': 'O'}
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
        df: pd.DataFrame = DataFrame(np.ones((nrows, 4)),
                                     index=date_range('2020-01-01', periods=nrows),
                                     columns=Index(list('abcd'), dtype=object))
        result, expected = self._return_result_expected(df, 1000, 'dt', 's')
        expected.index = expected.index.astype('M8[ns]')
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize('nrows', [2, 10, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251])
    @pytest.mark.parametrize('r_idx_type, c_idx_type', [('i', 'i'), ('s', 's'), ('s', 'dt'), ('p', 'p')])
    @pytest.mark.parametrize('ncols', [1, 2, 3, 4])
    @pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
    def test_to_csv_idx_types(self, nrows: int, r_idx_type: str, c_idx_type: str, ncols: int) -> None:
        axes: dict[str, Any] = {
            'i': lambda n: Index(np.arange(n), dtype=np.int64),
            's': lambda n: Index([f'{i}_{chr(i)}' for i in range(97, 97 + n)]),
            'dt': lambda n: date_range('2020-01-01', periods=n),
            'p': lambda n: period_range('2020-01-01', periods=n, freq='D')
        }
        df: pd.DataFrame = DataFrame(np.ones((nrows, ncols)),
                                     index=axes[r_idx_type](nrows),
                                     columns=axes[c_idx_type](ncols))
        result, expected = self._return_result_expected(df, 1000, r_idx_type, c_idx_type)
        if r_idx_type in ['dt', 'p']:
            expected.index = expected.index.astype('M8[ns]')
        if c_idx_type in ['dt', 'p']:
            expected.columns = expected.columns.astype('M8[ns]')
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize('nrows', [2, 10, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251])
    @pytest.mark.parametrize('ncols', [1, 2, 3, 4])
    def test_to_csv_idx_ncols(self, nrows: int, ncols: int) -> None:
        df: pd.DataFrame = DataFrame(np.ones((nrows, ncols)),
                                     index=Index([f'i-{i}' for i in range(nrows)], name='a'),
                                     columns=Index([f'i-{i}' for i in range(ncols)], name='a'))
        result, expected = self._return_result_expected(df, 1000)
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize('nrows', [10, 98, 99, 100, 101, 102])
    def test_to_csv_dup_cols(self, nrows: int) -> None:
        df: pd.DataFrame = DataFrame(np.ones((nrows, 3)),
                                     index=Index([f'i-{i}' for i in range(nrows)], name='a'),
                                     columns=Index([f'i-{i}' for i in range(3)], name='a'))
        cols: List[Any] = list(df.columns)
        cols[:2] = ['dupe', 'dupe']
        cols[-2:] = ['dupe', 'dupe']
        ix: List[Any] = list(df.index)
        ix[:2] = ['rdupe', 'rdupe']
        ix[-2:] = ['rdupe', 'rdupe']
        df.index = ix
        df.columns = cols
        result, expected = self._return_result_expected(df, 1000, dupe_col=True)
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    def test_to_csv_empty(self) -> None:
        df: pd.DataFrame = DataFrame(index=np.arange(10, dtype=np.int64))
        result, expected = self._return_result_expected(df, 1000)
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.slow
    def test_to_csv_chunksize(self, temp_file: Any) -> None:
        chunksize: int = 1000
        rows: int = chunksize // 2 + 1
        df: pd.DataFrame = DataFrame(np.ones((rows, 2)),
                                     columns=Index(list('ab')),
                                     index=MultiIndex.from_arrays([range(rows) for _ in range(2)]))
        result, expected = self._return_result_expected(df, chunksize, rnlvl=2)
        tm.assert_frame_equal(result, expected, check_names=False)

    @pytest.mark.slow
    @pytest.mark.parametrize('nrows', [10, 98, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251])
    @pytest.mark.parametrize('ncols', [2, 3, 4])
    @pytest.mark.parametrize('df_params, func_params', [
        ({'r_idx_nlevels': 2}, {'rnlvl': 2}),
        ({'c_idx_nlevels': 2}, {'cnlvl': 2}),
        ({'r_idx_nlevels': 2, 'c_idx_nlevels': 2}, {'rnlvl': 2, 'cnlvl': 2})
    ])
    def test_to_csv_params(self, nrows: int, df_params: dict[str, Any], func_params: dict[str, Any], ncols: int) -> None:
        if df_params.get('r_idx_nlevels'):
            index: Any = MultiIndex.from_arrays(([f'i-{i}' for i in range(nrows)]
                                                 for _ in range(df_params['r_idx_nlevels'])))
        else:
            index = None
        if df_params.get('c_idx_nlevels'):
            columns: Any = MultiIndex.from_arrays(([f'i-{i}' for i in range(ncols)]
                                                   for _ in range(df_params['c_idx_nlevels'])))
        else:
            columns = Index([f'i-{i}' for i in range(ncols)])
        df: pd.DataFrame = DataFrame(np.ones((nrows, ncols)), index=index, columns=columns)
        result, expected = self._return_result_expected(df, 1000, **func_params)
        tm.assert_frame_equal(result, expected, check_names=False)

    def test_to_csv_from_csv_w_some_infs(self, temp_file: Any, float_frame: pd.DataFrame) -> None:
        float_frame['G'] = np.nan
        f = lambda x: [np.inf, np.nan][np.random.default_rng(2).random() < 0.5]
        float_frame['h'] = float_frame.index.map(f)
        path: str = str(temp_file)
        float_frame.to_csv(path)
        recons: pd.DataFrame = self.read_csv(path)
        tm.assert_frame_equal(float_frame, recons)
        tm.assert_frame_equal(np.isinf(float_frame), np.isinf(recons))

    def test_to_csv_from_csv_w_all_infs(self, temp_file: Any, float_frame: pd.DataFrame) -> None:
        float_frame['E'] = np.inf
        float_frame['F'] = -np.inf
        path: str = str(temp_file)
        float_frame.to_csv(path)
        recons: pd.DataFrame = self.read_csv(path)
        tm.assert_frame_equal(float_frame, recons)
        tm.assert_frame_equal(np.isinf(float_frame), np.isinf(recons))

    def test_to_csv_no_index(self, temp_file: Any) -> None:
        path: str = str(temp_file)
        df: pd.DataFrame = DataFrame({'c1': [1, 2, 3], 'c2': [4, 5, 6]})
        df.to_csv(path, index=False)
        result: pd.DataFrame = read_csv(path)
        tm.assert_frame_equal(df, result)
        df['c3'] = Series([7, 8, 9], dtype='int64')
        df.to_csv(path, index=False)
        result = read_csv(path)
        tm.assert_frame_equal(df, result)

    def test_to_csv_with_mix_columns(self) -> None:
        df: pd.DataFrame = DataFrame({0: ['a', 'b', 'c'], 1: ['aa', 'bb', 'cc']})
        df['test'] = 'txt'
        assert df.to_csv() == df.to_csv(columns=[0, 1, 'test'])

    def test_to_csv_headers(self, temp_file: Any) -> None:
        from_df: pd.DataFrame = DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        to_df: pd.DataFrame = DataFrame([[1, 2], [3, 4]], columns=['X', 'Y'])
        path: str = str(temp_file)
        from_df.to_csv(path, header=['X', 'Y'])
        recons: pd.DataFrame = self.read_csv(path)
        tm.assert_frame_equal(to_df, recons)
        from_df.to_csv(path, index=False, header=['X', 'Y'])
        recons = self.read_csv(path)
        return_value = recons.reset_index(inplace=True)
        assert return_value is None
        tm.assert_frame_equal(to_df, recons)

    def test_to_csv_multiindex(self, temp_file: Any, float_frame: pd.DataFrame, datetime_frame: pd.DataFrame) -> None:
        frame: pd.DataFrame = float_frame
        old_index = frame.index
        arrays = np.arange(len(old_index) * 2, dtype=np.int64).reshape(2, -1)
        new_index: MultiIndex = MultiIndex.from_arrays(arrays, names=['first', 'second'])
        frame.index = new_index
        path: str = str(temp_file)
        frame.to_csv(path, header=False)
        frame.to_csv(path, columns=['A', 'B'])
        frame.to_csv(path)
        df: pd.DataFrame = self.read_csv(path, index_col=[0, 1], parse_dates=False)
        tm.assert_frame_equal(frame, df, check_names=False)
        assert frame.index.names == df.index.names
        float_frame.index = old_index
        tsframe: pd.DataFrame = datetime_frame
        old_index = tsframe.index
        new_index_list = [old_index, np.arange(len(old_index), dtype=np.int64)]
        tsframe.index = MultiIndex.from_arrays(new_index_list)
        tsframe.to_csv(path, index_label=['time', 'foo'])
        with tm.assert_produces_warning(UserWarning, match='Could not infer format'):
            recons: pd.DataFrame = self.read_csv(path, index_col=[0, 1], parse_dates=True)
        expected: pd.DataFrame = tsframe.copy()
        expected.index = MultiIndex.from_arrays([old_index.as_unit('s'), new_index_list[1]])
        tm.assert_frame_equal(recons, expected, check_names=False)
        tsframe.to_csv(path)
        recons = self.read_csv(path, index_col=None)
        assert len(recons.columns) == len(tsframe.columns) + 2
        tsframe.to_csv(path, index=False)
        recons = self.read_csv(path, index_col=None)
        tm.assert_almost_equal(recons.values, datetime_frame.values)
        datetime_frame.index = old_index
        with tm.ensure_clean('__tmp_to_csv_multiindex__') as path:
            def _make_frame(names: Union[bool, None]) -> pd.DataFrame:
                if names is True:
                    names_val: List[str] = ['first', 'second']
                else:
                    names_val = names  # type: ignore
                return DataFrame(
                    np.random.default_rng(2).integers(0, 10, size=(3, 3)),
                    columns=MultiIndex.from_tuples([('bah', 'foo'), ('bah', 'bar'), ('ban', 'baz')], names=names_val),
                    dtype='int64'
                )
            df: pd.DataFrame = DataFrame(
                np.ones((5, 3)),
                columns=MultiIndex.from_arrays([[f'i-{i}' for i in range(3)] for _ in range(4)], names=list('abcd')),
                index=MultiIndex.from_arrays([[f'i-{i}' for i in range(5)] for _ in range(2)], names=list('ab'))
            )
            df.to_csv(path)
            result: pd.DataFrame = read_csv(path, header=[0, 1, 2, 3], index_col=[0, 1])
            tm.assert_frame_equal(df, result)
            df = DataFrame(
                np.ones((5, 3)),
                columns=MultiIndex.from_arrays([[f'i-{i}' for i in range(3)] for _ in range(4)], names=list('abcd'))
            )
            df.to_csv(path)
            result = read_csv(path, header=[0, 1, 2, 3], index_col=0)
            tm.assert_frame_equal(df, result)
            df = DataFrame(
                np.ones((5, 3)),
                columns=MultiIndex.from_arrays([[f'i-{i}' for i in range(3)] for _ in range(4)], names=list('abcd')),
                index=MultiIndex.from_arrays([[f'i-{i}' for i in range(5)] for _ in range(3)], names=list('abc'))
            )
            df.to_csv(path)
            result = read_csv(path, header=[0, 1, 2, 3], index_col=[0, 1, 2])
            tm.assert_frame_equal(df, result)
            df = _make_frame(names=None)
            df.to_csv(path, index=False)
            result = read_csv(path, header=[0, 1])
            tm.assert_frame_equal(df, result)
            df = _make_frame(names=True)
            df.to_csv(path, index=False)
            result = read_csv(path, header=[0, 1])
            assert com.all_none(*result.columns.names)
            result.columns.names = df.columns.names
            tm.assert_frame_equal(df, result)
            df = _make_frame(names=None)
            df.to_csv(path)
            result = read_csv(path, header=[0, 1], index_col=[0])
            tm.assert_frame_equal(df, result)
            df = _make_frame(names=True)
            df.to_csv(path)
            result = read_csv(path, header=[0, 1], index_col=[0])
            tm.assert_frame_equal(df, result)
            df = _make_frame(names=True)
            df.to_csv(path)
            for i in [6, 7]:
                msg: str = f'len of {i}, but only 5 lines in file'
                with pytest.raises(ParserError, match=msg):
                    read_csv(path, header=list(range(i)), index_col=0)
            msg = 'cannot specify cols with a MultiIndex'
            with pytest.raises(TypeError, match=msg):
                df.to_csv(path, columns=['foo', 'bar'])
        with tm.ensure_clean('__tmp_to_csv_multiindex__') as path:
            tsframe[:0].to_csv(path)
            recons = self.read_csv(path)
            exp = tsframe[:0]
            exp.index = []
            tm.assert_index_equal(recons.columns, exp.columns)
            assert len(recons) == 0

    def test_to_csv_interval_index(self, temp_file: Any, using_infer_string: Any) -> None:
        df: pd.DataFrame = DataFrame({'A': list('abc'), 'B': range(3)},
                                      index=pd.interval_range(0, 3))
        path: str = str(temp_file)
        df.to_csv(path)
        result: pd.DataFrame = self.read_csv(path, index_col=0)
        expected: pd.DataFrame = df.copy()
        expected.index = expected.index.astype('str')
        tm.assert_frame_equal(result, expected)

    def test_to_csv_float32_nanrep(self, temp_file: Any) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((1, 4)).astype(np.float32))
        df[1] = np.nan
        path: str = str(temp_file)
        df.to_csv(path, na_rep=999)
        with open(path, encoding='utf-8') as f:
            lines: List[str] = f.readlines()
            assert lines[1].split(',')[2] == '999'

    def test_to_csv_withcommas(self, temp_file: Any) -> None:
        df: pd.DataFrame = DataFrame({'A': [1, 2, 3], 'B': ['5,6', '7,8', '9,0']})
        path: str = str(temp_file)
        df.to_csv(path)
        df2: pd.DataFrame = self.read_csv(path)
        tm.assert_frame_equal(df2, df)

    def test_to_csv_mixed(self, temp_file: Any) -> None:
        def create_cols(name: str) -> List[str]:
            return [f'{name}{i:03d}' for i in range(5)]
        df_float: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((100, 5)),
                                             dtype='float64', columns=create_cols('float'))
        df_int: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((100, 5)).astype('int64'),
                                           dtype='int64', columns=create_cols('int'))
        df_bool: pd.DataFrame = DataFrame(True, index=df_float.index, columns=create_cols('bool'))
        df_object: pd.DataFrame = DataFrame('foo', index=df_float.index, columns=create_cols('object'), dtype='object')
        df_dt: pd.DataFrame = DataFrame(Timestamp('20010101'), index=df_float.index, columns=create_cols('date'))
        df_float.iloc[30:50, 1:3] = np.nan
        df_dt.iloc[30:50, 1:3] = np.nan
        df: pd.DataFrame = pd.concat([df_float, df_int, df_bool, df_object, df_dt], axis=1)
        dtypes: dict[str, Any] = {}
        for n, dtype in [('float', np.float64), ('int', np.int64), ('bool', np.bool_), ('object', object)]:
            for c in create_cols(n):
                dtypes[c] = dtype
        path: str = str(temp_file)
        df.to_csv(path)
        rs: pd.DataFrame = read_csv(path, index_col=0, dtype=dtypes, parse_dates=create_cols('date'))
        tm.assert_frame_equal(rs, df)

    def test_to_csv_dups_cols(self, temp_file: Any) -> None:
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((1000, 30)),
                                     columns=list(range(15)) + list(range(15)), dtype='float64')
        path: str = str(temp_file)
        df.to_csv(path)
        result: pd.DataFrame = read_csv(path, index_col=0)
        result.columns = df.columns
        tm.assert_frame_equal(result, df)
        df_float: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((1000, 3)), dtype='float64')
        df_int: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((1000, 3))).astype('int64')
        df_bool: pd.DataFrame = DataFrame(True, index=df_float.index, columns=range(3))
        df_object: pd.DataFrame = DataFrame('foo', index=df_float.index, columns=range(3))
        df_dt: pd.DataFrame = DataFrame(Timestamp('20010101'), index=df_float.index, columns=range(3))
        df = pd.concat([df_float, df_int, df_bool, df_object, df_dt], axis=1, ignore_index=True)
        df.columns = [0, 1, 2] * 5
        with tm.ensure_clean() as filename:
            df.to_csv(filename)
            result = read_csv(filename, index_col=0)
            for i in ['0.4', '1.4', '2.4']:
                result[i] = to_datetime(result[i])
            result.columns = df.columns
            tm.assert_frame_equal(result, df)

    def test_to_csv_dups_cols2(self, temp_file: Any) -> None:
        df: pd.DataFrame = DataFrame(np.ones((5, 3)),
                                     index=Index([f'i-{i}' for i in range(5)], name='foo'),
                                     columns=Index(['a', 'a', 'b']))
        path: str = str(temp_file)
        df.to_csv(path)
        result: pd.DataFrame = read_csv(path, index_col=0)
        result = result.rename(columns={'a.1': 'a'})
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize('chunksize', [10000, 50000, 100000])
    def test_to_csv_chunking(self, chunksize: int, temp_file: Any) -> None:
        aa: pd.DataFrame = DataFrame({'A': range(100000)})
        aa['B'] = aa.A + 1.0
        aa['C'] = aa.A + 2.0
        aa['D'] = aa.A + 3.0
        path: str = str(temp_file)
        aa.to_csv(path, chunksize=chunksize)
        rs: pd.DataFrame = read_csv(path, index_col=0)
        tm.assert_frame_equal(rs, aa)

    @pytest.mark.slow
    def test_to_csv_wide_frame_formatting(self, temp_file: Any, monkeypatch: Any) -> None:
        chunksize: int = 100
        df: pd.DataFrame = DataFrame(np.random.default_rng(2).standard_normal((1, chunksize + 10)),
                                     columns=None, index=None)
        path: str = str(temp_file)
        with monkeypatch.context() as m:
            m.setattr('pandas.io.formats.csvs._DEFAULT_CHUNKSIZE_CELLS', chunksize)
            df.to_csv(path, header=False, index=False)
        rs: pd.DataFrame = read_csv(path, header=None)
        tm.assert_frame_equal(rs, df)

    def test_to_csv_bug(self, temp_file: Any) -> None:
        f1: StringIO = StringIO('a,1.0\nb,2.0')
        df: pd.DataFrame = self.read_csv(f1, header=None)
        newdf: pd.DataFrame = DataFrame({'t': df[df.columns[0]]})
        path: str = str(temp_file)
        newdf.to_csv(path)
        recons: pd.DataFrame = read_csv(path, index_col=0)
        tm.assert_frame_equal(recons, newdf, check_names=False)

    def test_to_csv_unicode(self, temp_file: Any) -> None:
        df: pd.DataFrame = DataFrame({'c/σ': [1, 2, 3]})
        path: str = str(temp_file)
        df.to_csv(path, encoding='UTF-8')
        df2: pd.DataFrame = read_csv(path, index_col=0, encoding='UTF-8')
        tm.assert_frame_equal(df, df2)
        df.to_csv(path, encoding='UTF-8', index=False)
        df2 = read_csv(path, index_col=None, encoding='UTF-8')
        tm.assert_frame_equal(df, df2)

    def test_to_csv_unicode_index_col(self) -> None:
        buf: StringIO = StringIO('')
        df: pd.DataFrame = DataFrame(
            [['א', 'd2', 'd3', 'd4'], ['a1', 'a2', 'a3', 'a4']],
            columns=['א', 'ב', 'ג', 'ד'],
            index=['א', 'ב']
        )
        df.to_csv(buf, encoding='UTF-8')
        buf.seek(0)
        df2: pd.DataFrame = read_csv(buf, index_col=0, encoding='UTF-8')
        tm.assert_frame_equal(df, df2)

    def test_to_csv_stringio(self, float_frame: pd.DataFrame) -> None:
        buf: StringIO = StringIO()
        float_frame.to_csv(buf)
        buf.seek(0)
        recons: pd.DataFrame = read_csv(buf, index_col=0)
        tm.assert_frame_equal(recons, float_frame)

    def test_to_csv_float_format(self, temp_file: Any) -> None:
        df: pd.DataFrame = DataFrame([[0.123456, 0.234567, 0.567567],
                                      [12.32112, 123123.2, 321321.2]],
                                     index=['A', 'B'], columns=['X', 'Y', 'Z'])
        path: str = str(temp_file)
        df.to_csv(path, float_format='%.2f')
        rs: pd.DataFrame = read_csv(path, index_col=0)
        xp: pd.DataFrame = DataFrame([[0.12, 0.23, 0.57],
                                      [12.32, 123123.2, 321321.2]],
                                     index=['A', 'B'], columns=['X', 'Y', 'Z'])
        tm.assert_frame_equal(rs, xp)

    def test_to_csv_float_format_over_decimal(self) -> None:
        df: pd.DataFrame = DataFrame({'a': [0.5, 1.0]})
        result: str = df.to_csv(decimal=',', float_format=lambda x: np.format_float_positional(x, trim='-'), index=False)
        expected_rows: List[str] = ['a', '0.5', '1']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_to_csv_unicodewriter_quoting(self) -> None:
        df: pd.DataFrame = DataFrame({'A': [1, 2, 3], 'B': ['foo', 'bar', 'baz']})
        buf: StringIO = StringIO()
        df.to_csv(buf, index=False, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
        result: str = buf.getvalue()
        expected_rows: List[str] = ['"A","B"', '1,"foo"', '2,"bar"', '3,"baz"']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    @pytest.mark.parametrize('encoding', [None, 'utf-8'])
    def test_to_csv_quote_none(self, encoding: Optional[str]) -> None:
        df: pd.DataFrame = DataFrame({'A': ['hello', '{"hello"}']})
        buf: StringIO = StringIO()
        df.to_csv(buf, quoting=csv.QUOTE_NONE, encoding=encoding, index=False)
        result: str = buf.getvalue()
        expected_rows: List[str] = ['A', 'hello', '{"hello"}']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_to_csv_index_no_leading_comma(self) -> None:
        df: pd.DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]},
                                     index=['one', 'two', 'three'])
        buf: StringIO = StringIO()
        df.to_csv(buf, index_label=False)
        expected_rows: List[str] = ['A,B', 'one,1,4', 'two,2,5', 'three,3,6']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        assert buf.getvalue() == expected

    def test_to_csv_lineterminators(self, temp_file: Any) -> None:
        df: pd.DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]},
                                     index=['one', 'two', 'three'])
        path: str = str(temp_file)
        df.to_csv(path, lineterminator='\r\n')
        expected: bytes = b',A,B\r\none,1,4\r\ntwo,2,5\r\nthree,3,6\r\n'
        with open(path, mode='rb') as f:
            assert f.read() == expected

    def test_to_csv_lineterminators2(self, temp_file: Any) -> None:
        df: pd.DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]},
                                     index=['one', 'two', 'three'])
        path: str = str(temp_file)
        df.to_csv(path, lineterminator='\n')
        expected: bytes = b',A,B\none,1,4\ntwo,2,5\nthree,3,6\n'
        with open(path, mode='rb') as f:
            assert f.read() == expected

    def test_to_csv_lineterminators3(self, temp_file: Any) -> None:
        df: pd.DataFrame = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]},
                                     index=['one', 'two', 'three'])
        path: str = str(temp_file)
        df.to_csv(path)
        os_linesep: bytes = os.linesep.encode('utf-8')
        expected: bytes = b',A,B' + os_linesep + b'one,1,4' + os_linesep + b'two,2,5' + os_linesep + b'three,3,6' + os_linesep
        with open(path, mode='rb') as f:
            assert f.read() == expected

    def test_to_csv_from_csv_categorical(self) -> None:
        s: pd.Series = Series(pd.Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']))
        s2: pd.Series = Series(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'])
        res: StringIO = StringIO()
        s.to_csv(res, header=False)
        exp: StringIO = StringIO()
        s2.to_csv(exp, header=False)
        assert res.getvalue() == exp.getvalue()
        df: pd.DataFrame = DataFrame({'s': s})
        df2: pd.DataFrame = DataFrame({'s': s2})
        res = StringIO()
        df.to_csv(res)
        exp = StringIO()
        df2.to_csv(exp)
        assert res.getvalue() == exp.getvalue()

    def test_to_csv_path_is_none(self, float_frame: pd.DataFrame) -> None:
        csv_str: str = float_frame.to_csv(path_or_buf=None)
        assert isinstance(csv_str, str)
        recons: pd.DataFrame = read_csv(StringIO(csv_str), index_col=0)
        tm.assert_frame_equal(float_frame, recons)

    @pytest.mark.parametrize('df,encoding', [
        (DataFrame([[0.123456, 0.234567, 0.567567],
                    [12.32112, 123123.2, 321321.2]],
                   index=['A', 'B'], columns=['X', 'Y', 'Z']), None),
        (DataFrame([['abc', 'def', 'ghi']], columns=['X', 'Y', 'Z']), 'ascii'),
        (DataFrame(5 * [[123, '你好', '世界']], columns=['X', 'Y', 'Z']), 'gb2312'),
        (DataFrame(5 * [[123, 'Γειά σου', 'Κόσμε']], columns=['X', 'Y', 'Z']), 'cp737')
    ])
    def test_to_csv_compression(
        self,
        temp_file: Any,
        df: pd.DataFrame,
        encoding: Optional[str],
        compression: Optional[str]
    ) -> None:
        path: str = str(temp_file)
        df.to_csv(path, compression=compression, encoding=encoding)
        result: pd.DataFrame = read_csv(path, compression=compression, index_col=0, encoding=encoding)
        tm.assert_frame_equal(df, result)
        with get_handle(path, 'w', compression=compression, encoding=encoding) as handles:
            df.to_csv(handles.handle, encoding=encoding)
            assert not handles.handle.closed
        result = read_csv(path, compression=compression, encoding=encoding, index_col=0).squeeze('columns')
        tm.assert_frame_equal(df, result)
        with tm.decompress_file(path, compression) as fh:
            text: str = fh.read().decode(encoding or 'utf8')
            for col in df.columns:
                assert col in text
        with tm.decompress_file(path, compression) as fh:
            tm.assert_frame_equal(df, read_csv(fh, index_col=0, encoding=encoding))

    def test_to_csv_date_format(self, temp_file: Any, datetime_frame: pd.DataFrame) -> None:
        path: str = str(temp_file)
        dt_index = datetime_frame.index
        datetime_frame = DataFrame({'A': dt_index, 'B': dt_index.shift(1)}, index=dt_index)
        datetime_frame.to_csv(path, date_format='%Y%m%d')
        test: pd.DataFrame = read_csv(path, index_col=0)
        datetime_frame_int: pd.DataFrame = datetime_frame.map(lambda x: int(x.strftime('%Y%m%d')))
        datetime_frame_int.index = datetime_frame_int.index.map(lambda x: int(x.strftime('%Y%m%d')))
        tm.assert_frame_equal(test, datetime_frame_int)
        datetime_frame.to_csv(path, date_format='%Y-%m-%d')
        test = read_csv(path, index_col=0)
        datetime_frame_str: pd.DataFrame = datetime_frame.map(lambda x: x.strftime('%Y-%m-%d'))
        datetime_frame_str.index = datetime_frame_str.index.map(lambda x: x.strftime('%Y-%m-%d'))
        tm.assert_frame_equal(test, datetime_frame_str)
        datetime_frame_columns: pd.DataFrame = datetime_frame.T
        datetime_frame_columns.to_csv(path, date_format='%Y%m%d')
        test = read_csv(path, index_col=0)
        datetime_frame_columns = datetime_frame_columns.map(lambda x: int(x.strftime('%Y%m%d')))
        datetime_frame_columns.columns = datetime_frame_columns.columns.map(lambda x: x.strftime('%Y%m%d'))
        tm.assert_frame_equal(test, datetime_frame_columns)
        nat_index = to_datetime(['NaT'] * 10 + ['2000-01-01', '2000-01-01', '2000-01-01'])
        nat_frame: pd.DataFrame = DataFrame({'A': nat_index}, index=nat_index)
        nat_frame.to_csv(path, date_format='%Y-%m-%d')
        test = read_csv(path, parse_dates=[0, 1], index_col=0)
        tm.assert_frame_equal(test, nat_frame)

    @pytest.mark.parametrize('td', [pd.Timedelta(0), pd.Timedelta('10s')])
    def test_to_csv_with_dst_transitions(self, td: pd.Timedelta, temp_file: Any) -> None:
        path: str = str(temp_file)
        times = date_range('2013-10-26 23:00', '2013-10-27 01:00', tz='Europe/London', freq='h', ambiguous='infer')
        i = times + td
        i = i._with_freq(None)
        time_range: np.ndarray = np.array(range(len(i)), dtype='int64')
        df: pd.DataFrame = DataFrame({'A': time_range}, index=i)
        df.to_csv(path, index=True)
        result: pd.DataFrame = read_csv(path, index_col=0)
        result.index = to_datetime(result.index, utc=True).tz_convert('Europe/London').as_unit('ns')
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize('start,end', [['2015-03-29', '2015-03-30'], ['2015-10-25', '2015-10-26']])
    def test_to_csv_with_dst_transitions_with_pickle(self, start: str, end: str, temp_file: Any) -> None:
        idx = date_range(start, end, freq='h', tz='Europe/Paris')
        idx = idx._with_freq(None)
        idx._data._freq = None
        df: pd.DataFrame = DataFrame({'values': 1, 'idx': idx}, index=idx)
        with tm.ensure_clean('csv_date_format_with_dst') as path:
            df.to_csv(path, index=True)
            result: pd.DataFrame = read_csv(path, index_col=0)
            result.index = to_datetime(result.index, utc=True).tz_convert('Europe/Paris').as_unit('ns')
            result['idx'] = to_datetime(result['idx'], utc=True).astype('datetime64[ns, Europe/Paris]')
            tm.assert_frame_equal(result, df)
        df.astype(str)
        path = str(temp_file)
        df.to_pickle(path)
        result = pd.read_pickle(path)
        tm.assert_frame_equal(result, df)

    def test_to_csv_quoting(self) -> None:
        df: pd.DataFrame = DataFrame({'c_bool': [True, False],
                                      'c_float': [1.0, 3.2],
                                      'c_int': [42, np.nan],
                                      'c_string': ['a', 'b,c']})
        expected_rows: List[str] = [',c_bool,c_float,c_int,c_string', '0,True,1.0,42.0,a', '1,False,3.2,,"b,c"']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        result: str = df.to_csv()
        assert result == expected
        result = df.to_csv(quoting=None)
        assert result == expected
        expected_rows = [',c_bool,c_float,c_int,c_string', '0,True,1.0,42.0,a', '1,False,3.2,,"b,c"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_MINIMAL)
        assert result == expected
        expected_rows = ['"","c_bool","c_float","c_int","c_string"',
                         '"0","True","1.0","42.0","a"',
                         '"1","False","3.2","","b,c"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_ALL)
        assert result == expected
        expected_rows = ['"","c_bool","c_float","c_int","c_string"',
                         '0,True,1.0,42.0,"a"',
                         '1,False,3.2,"","b,c"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_NONNUMERIC)
        assert result == expected
        msg: str = 'need to escape, but no escapechar set'
        with pytest.raises(csv.Error, match=msg):
            df.to_csv(quoting=csv.QUOTE_NONE)
        with pytest.raises(csv.Error, match=msg):
            df.to_csv(quoting=csv.QUOTE_NONE, escapechar=None)
        expected_rows = [',c_bool,c_float,c_int,c_string', '0,True,1.0,42.0,a', '1,False,3.2,,b!,c']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_NONE, escapechar='!')
        assert result == expected
        expected_rows = [',c_bool,c_ffloat,c_int,c_string', '0,True,1.0,42.0,a', '1,False,3.2,,bf,c']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        result = df.to_csv(quoting=csv.QUOTE_NONE, escapechar='f')
        assert result == expected
        text_rows = ['a,b,c', '1,"test \r\n",3']
        text: str = tm.convert_rows_list_to_csv_str(text_rows)
        df = read_csv(StringIO(text))
        buf = StringIO()
        df.to_csv(buf, encoding='utf-8', index=False)
        assert buf.getvalue() == text
        df = DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        df = df.set_index(['a', 'b'])
        expected_rows = ['"a","b","c"', '"1","3","5"', '"2","4","6"']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(quoting=csv.QUOTE_ALL) == expected

    def test_period_index_date_overflow(self) -> None:
        dates: List[Union[str, NaT]] = ['1990-01-01', '2000-01-01', '3005-01-01']
        index = pd.PeriodIndex(dates, freq='D')
        df: pd.DataFrame = DataFrame([4, 5, 6], index=index)
        result: str = df.to_csv()
        expected_rows: List[str] = [',0', '1990-01-01,4', '2000-01-01,5', '3005-01-01,6']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected
        date_format: str = '%m-%d-%Y'
        result = df.to_csv(date_format=date_format)
        expected_rows = [',0', '01-01-1990,4', '01-01-2000,5', '01-01-3005,6']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected
        dates = ['1990-01-01', NaT, '3005-01-01']
        index = pd.PeriodIndex(dates, freq='D')
        df = DataFrame([4, 5, 6], index=index)
        result = df.to_csv()
        expected_rows = [',0', '1990-01-01,4', ',5', '3005-01-01,6']
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_multi_index_header(self) -> None:
        columns: MultiIndex = MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
        df: pd.DataFrame = DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
        df.columns = columns
        header: List[str] = ['a', 'b', 'c', 'd']
        result: str = df.to_csv(header=header)
        expected_rows: List[str] = [',a,b,c,d', '0,1,2,3,4', '1,5,6,7,8']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_to_csv_single_level_multi_index(self) -> None:
        index: Index = Index([(1,), (2,), (3,)])
        df: pd.DataFrame = DataFrame([[1, 2, 3]], columns=index)
        df = df.reindex(columns=[(1,), (3,)])
        expected: str = ',1,3\n0,1,3\n'
        result: str = df.to_csv(lineterminator='\n')
        tm.assert_almost_equal(result, expected)

    def test_gz_lineend(self, tmp_path: Any) -> None:
        df: pd.DataFrame = DataFrame({'a': [1, 2]})
        expected_rows: List[str] = ['a', '1', '2']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        file_path = tmp_path / '__test_gz_lineend.csv.gz'
        file_path.touch()
        path: str = str(file_path)
        df.to_csv(path, index=False)
        with tm.decompress_file(path, compression='gzip') as f:
            result: str = f.read().decode('utf-8')
        assert result == expected

    def test_to_csv_numpy_16_bug(self) -> None:
        frame: pd.DataFrame = DataFrame({'a': date_range('1/1/2000', periods=10)})
        buf: StringIO = StringIO()
        frame.to_csv(buf)
        result: str = buf.getvalue()
        assert '2000-01-01' in result

    def test_to_csv_na_quoting(self) -> None:
        result: str = DataFrame([None, None]).to_csv(None, header=False, index=False, na_rep='').replace('\r\n', '\n')
        expected: str = '""\n""\n'
        assert result == expected

    def test_to_csv_categorical_and_ea(self) -> None:
        df: pd.DataFrame = DataFrame({'a': 'x', 'b': [1, pd.NA]})
        df['b'] = df['b'].astype('Int16')
        df['b'] = df['b'].astype('category')
        result: str = df.to_csv()
        expected_rows: List[str] = [',a,b', '0,x,1', '1,x,']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_to_csv_categorical_and_interval(self) -> None:
        df: pd.DataFrame = DataFrame({'a': [pd.Interval(Timestamp('2020-01-01'), Timestamp('2020-01-02'), closed='both')]})
        df['a'] = df['a'].astype('category')
        result: str = df.to_csv()
        expected_rows: List[str] = [',a', '0,"[2020-01-01 00:00:00, 2020-01-02 00:00:00]"']
        expected: str = tm.convert_rows_list_to_csv_str(expected_rows)
        assert result == expected

    def test_to_csv_warn_when_zip_tar_and_append_mode(self, tmp_path: Any) -> None:
        df: pd.DataFrame = DataFrame({'a': [1, 2, 3]})
        msg: str = "zip and tar do not support mode 'a' properly. This combination will result in multiple files with same name being added to the archive"
        zip_path = tmp_path / 'test.zip'
        tar_path = tmp_path / 'test.tar'
        with tm.assert_produces_warning(RuntimeWarning, match=msg, raise_on_extra_warnings=False):
            df.to_csv(zip_path, mode='a')
        with tm.assert_produces_warning(RuntimeWarning, match=msg, raise_on_extra_warnings=False):
            df.to_csv(tar_path, mode='a')