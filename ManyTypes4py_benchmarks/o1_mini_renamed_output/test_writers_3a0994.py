from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import partial
from io import BytesIO
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    date_range,
    option_context,
    period_range,
)
import pandas._testing as tm
from pandas.io.excel import (
    ExcelFile,
    ExcelWriter,
    _OpenpyxlWriter,
    _XlsxWriter,
    register_writer,
)
from pandas.io.excel._util import _writers


def get_exp_unit(path: str) -> str:
    if path.endswith('.ods'):
        return 's'
    return 'us'


@pytest.fixture
def frame(float_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the first ten items in fixture "float_frame".
    """
    return float_frame[:10]


@pytest.fixture
def merge_cells(request: Any) -> Union[bool, str]:
    return request.param


@pytest.fixture
def tmp_excel(ext: str, tmp_path: Any) -> str:
    """
    Fixture to open file for use in each test case.
    """
    tmp = tmp_path / f'{uuid.uuid4()}{ext}'
    tmp.touch()
    return str(tmp)


@pytest.fixture
def set_engine(engine: str, ext: str) -> Any:
    """
    Fixture to set engine for use in each test case.

    Rather than requiring `engine=...` to be provided explicitly as an
    argument in each test, this fixture sets a global option to dictate
    which engine should be used to write Excel files. After executing
    the test it rolls back said change to the global option.
    """
    option_name = f'io.excel.{ext.strip(".")}.writer'
    with option_context(option_name, engine):
        yield


@pytest.mark.parametrize(
    'ext',
    [
        pytest.param(
            '.xlsx',
            marks=[
                td.skip_if_no('openpyxl'),
                td.skip_if_no('xlrd'),
            ],
        ),
        pytest.param(
            '.xlsm',
            marks=[
                td.skip_if_no('openpyxl'),
                td.skip_if_no('xlrd'),
            ],
        ),
        pytest.param(
            '.xlsx',
            marks=[
                td.skip_if_no('xlsxwriter'),
                td.skip_if_no('xlrd'),
            ],
        ),
        pytest.param(
            '.ods',
            marks=td.skip_if_no('odf'),
        ),
    ],
)
class TestRoundTrip:

    @pytest.mark.parametrize(
        'header,expected',
        [
            (None, [np.nan] * 4),
            (
                0,
                {'Unnamed: 0': [np.nan] * 3},
            ),
        ],
    )
    def test_read_one_empty_col_no_header(
        self,
        tmp_excel: str,
        header: Optional[int],
        expected: Union[List[float], Dict[str, List[float]]],
    ) -> None:
        filename = 'no_header'
        df: DataFrame = DataFrame([['', 1, 100], ['', 2, 200], ['', 3, 300], ['', 4, 400]])
        df.to_excel(tmp_excel, sheet_name=filename, index=False, header=False)
        result: DataFrame = pd.read_excel(tmp_excel, sheet_name=filename, usecols=[0], header=header)
        expected_df: DataFrame = DataFrame(expected)
        tm.assert_frame_equal(result, expected_df)

    @pytest.mark.parametrize(
        'header,expected_extra',
        [
            (None, [0]),
            (0, []),
        ],
    )
    def test_read_one_empty_col_with_header(
        self,
        tmp_excel: str,
        header: Optional[int],
        expected_extra: List[int],
    ) -> None:
        filename = 'with_header'
        df: DataFrame = DataFrame([['', 1, 100], ['', 2, 200], ['', 3, 300], ['', 4, 400]])
        df.to_excel(tmp_excel, sheet_name='with_header', index=False, header=True)
        result: DataFrame = pd.read_excel(tmp_excel, sheet_name=filename, usecols=[0], header=header)
        expected: DataFrame = DataFrame(expected_extra + [np.nan] * 4)
        tm.assert_frame_equal(result, expected)

    def test_set_column_names_in_parameter(self, tmp_excel: str) -> None:
        refdf: DataFrame = DataFrame([[1, 'foo'], [2, 'bar'], [3, 'baz']], columns=['a', 'b'])
        with ExcelWriter(tmp_excel) as writer:
            refdf.to_excel(writer, sheet_name='Data_no_head', header=False, index=False)
            refdf.to_excel(writer, sheet_name='Data_with_head', index=False)
        refdf.columns = ['A', 'B']
        with ExcelFile(tmp_excel) as reader:
            xlsdf_no_head: DataFrame = pd.read_excel(reader, sheet_name='Data_no_head', header=None, names=['A', 'B'])
            xlsdf_with_head: DataFrame = pd.read_excel(reader, sheet_name='Data_with_head', index_col=None, names=['A', 'B'])
        tm.assert_frame_equal(xlsdf_no_head, refdf)
        tm.assert_frame_equal(xlsdf_with_head, refdf)

    def test_creating_and_reading_multiple_sheets(self, tmp_excel: str) -> None:

        def tdf(col_sheet_name: str) -> DataFrame:
            d, i = ([11, 22, 33], [1, 2, 3])
            return DataFrame(d, i, columns=[col_sheet_name])

        sheets: List[str] = ['AAA', 'BBB', 'CCC']
        dfs: List[DataFrame] = [tdf(s) for s in sheets]
        dfs_dict: Dict[str, DataFrame] = dict(zip(sheets, dfs))
        with ExcelWriter(tmp_excel) as ew:
            for sheetname, df in dfs_dict.items():
                df.to_excel(ew, sheet_name=sheetname)
        sheets_returned: Dict[str, DataFrame] = pd.read_excel(tmp_excel, sheet_name=sheets, index_col=0)
        for s in sheets:
            tm.assert_frame_equal(dfs_dict[s], sheets_returned[s])

    def test_read_excel_multiindex_empty_level(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            {
                ('One', 'x'): {0: 1},
                ('Two', 'X'): {0: 3},
                ('Two', 'Y'): {0: 7},
                ('Zero', ''): {0: 0},
            }
        )
        expected: DataFrame = DataFrame(
            {
                ('One', 'x'): {0: 1},
                ('Two', 'X'): {0: 3},
                ('Two', 'Y'): {0: 7},
                ('Zero', 'Unnamed: 4_level_1'): {0: 0},
            }
        )
        df.to_excel(tmp_excel)
        actual: DataFrame = pd.read_excel(tmp_excel, header=[0, 1], index_col=0)
        tm.assert_frame_equal(actual, expected)
        df = DataFrame(
            {
                ('Beg', ''): {0: 0},
                ('Middle', 'x'): {0: 1},
                ('Tail', 'X'): {0: 3},
                ('Tail', 'Y'): {0: 7},
            }
        )
        expected = DataFrame(
            {
                ('Beg', 'Unnamed: 1_level_1'): {0: 0},
                ('Middle', 'x'): {0: 1},
                ('Tail', 'X'): {0: 3},
                ('Tail', 'Y'): {0: 7},
            }
        )
        df.to_excel(tmp_excel)
        actual = pd.read_excel(tmp_excel, header=[0, 1], index_col=0)
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        'c_idx_names', ['a', None],
    )
    @pytest.mark.parametrize(
        'r_idx_names', ['b', None],
    )
    @pytest.mark.parametrize(
        'c_idx_levels', [1, 3],
    )
    @pytest.mark.parametrize(
        'r_idx_levels', [1, 3],
    )
    def test_excel_multindex_roundtrip(
        self,
        tmp_excel: str,
        c_idx_names: Optional[str],
        r_idx_names: Optional[str],
        c_idx_levels: int,
        r_idx_levels: int,
    ) -> None:
        check_names: bool = bool(r_idx_names) or r_idx_levels <= 1
        if c_idx_levels == 1:
            columns: Index = Index(list('abcde'))
        else:
            columns = MultiIndex.from_arrays(
                [range(5) for _ in range(c_idx_levels)],
                names=[f'{c_idx_names}-{i}' for i in range(c_idx_levels)],
            )
        if r_idx_levels == 1:
            index: Index = Index(list('ghijk'))
        else:
            index = MultiIndex.from_arrays(
                [range(5) for _ in range(r_idx_levels)],
                names=[f'{r_idx_names}-{i}' for i in range(r_idx_levels)],
            )
        df: DataFrame = DataFrame(1.1 * np.ones((5, 5)), columns=columns, index=index)
        df.to_excel(tmp_excel)
        act: DataFrame = pd.read_excel(
            tmp_excel,
            index_col=list(range(r_idx_levels)),
            header=list(range(c_idx_levels)),
        )
        tm.assert_frame_equal(df, act, check_names=check_names)
        df.iloc[0, :] = np.nan
        df.to_excel(tmp_excel)
        act = pd.read_excel(
            tmp_excel,
            index_col=list(range(r_idx_levels)),
            header=list(range(c_idx_levels)),
        )
        tm.assert_frame_equal(df, act, check_names=check_names)
        df.iloc[-1, :] = np.nan
        df.to_excel(tmp_excel)
        act = pd.read_excel(
            tmp_excel,
            index_col=list(range(r_idx_levels)),
            header=list(range(c_idx_levels)),
        )
        tm.assert_frame_equal(df, act, check_names=check_names)

    def test_read_excel_parse_dates(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            {
                'col': [1, 2, 3],
                'date_strings': date_range('2012-01-01', periods=3),
            }
        )
        df2: DataFrame = df.copy()
        df2['date_strings'] = df2['date_strings'].dt.strftime('%m/%d/%Y')
        df2.to_excel(tmp_excel)
        res: DataFrame = pd.read_excel(tmp_excel, index_col=0)
        tm.assert_frame_equal(df2, res)
        res = pd.read_excel(tmp_excel, parse_dates=['date_strings'], index_col=0)
        expected: DataFrame = df[:]
        expected['date_strings'] = expected['date_strings'].astype('M8[s]')
        tm.assert_frame_equal(res, expected)
        res = pd.read_excel(
            tmp_excel,
            parse_dates=['date_strings'],
            date_format='%m/%d/%Y',
            index_col=0,
        )
        expected['date_strings'] = expected['date_strings'].astype('M8[s]')
        tm.assert_frame_equal(expected, res)

    def test_multiindex_interval_datetimes(self, tmp_excel: str) -> None:
        midx: MultiIndex = MultiIndex.from_arrays(
            [
                range(4),
                pd.interval_range(
                    start=pd.Timestamp('2020-01-01'), periods=4, freq='6ME'
                ),
            ]
        )
        df: DataFrame = DataFrame(range(4), index=midx)
        df.to_excel(tmp_excel)
        result: DataFrame = pd.read_excel(tmp_excel, index_col=[0, 1])
        expected: DataFrame = DataFrame(
            range(4),
            MultiIndex.from_arrays(
                [
                    range(4),
                    [
                        '(2020-01-31 00:00:00, 2020-07-31 00:00:00)',
                        '(2020-07-31 00:00:00, 2021-01-31 00:00:00)',
                        '(2021-01-31 00:00:00, 2021-07-31 00:00:00)',
                        '(2021-07-31 00:00:00, 2022-01-31 00:00:00]',
                    ],
                ]
            ),
            columns=Index([0]),
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'merge_cells',
        [True, False, 'columns'],
    )
    def test_excel_round_trip_with_periodindex(
        self,
        tmp_excel: str,
        merge_cells: Union[bool, str],
    ) -> None:
        df: DataFrame = DataFrame(
            {
                'A': [1, 2],
                'B': [3, 4],
            },
            index=MultiIndex.from_arrays(
                [
                    period_range(start='2006-10-06', end='2006-10-07', freq='D'),
                    ['X', 'Y'],
                ],
                names=['date', 'category'],
            ),
        )
        df.to_excel(tmp_excel, merge_cells=merge_cells)
        result: DataFrame = pd.read_excel(tmp_excel, index_col=[0, 1])
        expected: DataFrame = DataFrame(
            {'A': [1, 2]},
            MultiIndex.from_arrays(
                [
                    pd.to_datetime(['2006-10-06 00:00:00', '2006-10-07 00:00:00']),
                    ['X', 'Y'],
                ],
                names=['date', 'category'],
            ),
        )
        time_format: str = 'datetime64[s]' if tmp_excel.endswith('.ods') else 'datetime64[us]'
        expected.index = expected.index.set_levels(
            expected.index.levels[0].astype(time_format), level=0
        )
        tm.assert_frame_equal(result, expected)

@pytest.mark.parametrize(
    'engine,ext',
    [
        pytest.param(
            'openpyxl',
            '.xlsx',
            marks=[td.skip_if_no('openpyxl'), td.skip_if_no('xlrd')],
        ),
        pytest.param(
            'openpyxl',
            '.xlsm',
            marks=[td.skip_if_no('openpyxl'), td.skip_if_no('xlrd')],
        ),
        pytest.param(
            'xlsxwriter',
            '.xlsx',
            marks=[td.skip_if_no('xlsxwriter'), td.skip_if_no('xlrd')],
        ),
        pytest.param(
            'odf',
            '.ods',
            marks=td.skip_if_no('odf'),
        ),
    ],
)
@pytest.mark.usefixtures('set_engine')
class TestExcelWriter:

    def test_excel_sheet_size(self, tmp_excel: str) -> None:
        breaking_row_count: int = 2 ** 20 + 1
        breaking_col_count: int = 2 ** 14 + 1
        row_arr: np.ndarray = np.zeros(shape=(breaking_row_count, 1))
        col_arr: np.ndarray = np.zeros(shape=(1, breaking_col_count))
        row_df: DataFrame = DataFrame(row_arr)
        col_df: DataFrame = DataFrame(col_arr)
        msg: str = 'sheet is too large'
        with pytest.raises(ValueError, match=msg):
            row_df.to_excel(tmp_excel)
        with pytest.raises(ValueError, match=msg):
            col_df.to_excel(tmp_excel)

    def test_excel_sheet_by_name_raise(self, tmp_excel: str) -> None:
        gt: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=Index(list(range(10))),
        )
        gt.to_excel(tmp_excel)
        with ExcelFile(tmp_excel) as xl:
            df: DataFrame = pd.read_excel(xl, sheet_name=0, index_col=0)
        tm.assert_frame_equal(gt, df)
        msg: str = "Worksheet named '0' not found"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(xl, '0')

    def test_excel_writer_context_manager(
        self, frame: DataFrame, tmp_excel: str
    ) -> None:
        with ExcelWriter(tmp_excel) as writer:
            frame.to_excel(writer, sheet_name='Data1')
            frame2: DataFrame = frame.copy()
            frame2.columns = frame.columns[::-1]
            frame2.to_excel(writer, sheet_name='Data2')
        with ExcelFile(tmp_excel) as reader:
            found_df: DataFrame = pd.read_excel(reader, sheet_name='Data1', index_col=0)
            found_df2: DataFrame = pd.read_excel(reader, sheet_name='Data2', index_col=0)
            tm.assert_frame_equal(found_df, frame)
            tm.assert_frame_equal(found_df2, frame2)

    def test_roundtrip(self, frame: DataFrame, tmp_excel: str) -> None:
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.to_excel(tmp_excel, sheet_name='test1')
        frame.to_excel(tmp_excel, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(tmp_excel, sheet_name='test1', header=False)
        frame.to_excel(tmp_excel, sheet_name='test1', index=False)
        frame.to_excel(tmp_excel, sheet_name='test1')
        recons: DataFrame = pd.read_excel(tmp_excel, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(tmp_excel, sheet_name='test1', index=False)
        recons = pd.read_excel(tmp_excel, sheet_name='test1', index_col=None)
        recons.index = frame.index
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(tmp_excel, sheet_name='test1', na_rep='NA')
        recons = pd.read_excel(tmp_excel, sheet_name='test1', index_col=0, na_values=['NA'])
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(tmp_excel, sheet_name='test1', na_rep='88')
        recons = pd.read_excel(tmp_excel, sheet_name='test1', index_col=0, na_values=['88'])
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(tmp_excel, sheet_name='test1', na_rep='88')
        recons = pd.read_excel(tmp_excel, sheet_name='test1', index_col=0, na_values=[88, 88.0])
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(tmp_excel, sheet_name='Sheet1')
        recons = pd.read_excel(tmp_excel, index_col=0)
        tm.assert_frame_equal(frame, recons)
        frame.to_excel(tmp_excel, sheet_name='0')
        recons = pd.read_excel(tmp_excel, index_col=0)
        tm.assert_frame_equal(frame, recons)
        s: pd.Series = frame['A']
        s.to_excel(tmp_excel)
        recons = pd.read_excel(tmp_excel, index_col=0)
        tm.assert_frame_equal(s.to_frame(), recons)

    def test_mixed(self, frame: DataFrame, tmp_excel: str) -> None:
        mixed_frame: DataFrame = frame.copy()
        mixed_frame['foo'] = 'bar'
        mixed_frame.to_excel(tmp_excel, sheet_name='test1')
        with ExcelFile(tmp_excel) as reader:
            recons: DataFrame = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(mixed_frame, recons)

    def test_ts_frame(self, tmp_excel: str) -> None:
        unit: str = get_exp_unit(tmp_excel)
        df: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list('ABCD')),
            index=date_range('2000-01-01', periods=5, freq='B'),
        )
        index: pd.DatetimeIndex = pd.DatetimeIndex(np.asarray(df.index), freq=None)
        df.index = index
        expected: DataFrame = df[:]
        expected.index = expected.index.as_unit(unit)
        df.to_excel(tmp_excel, sheet_name='test1')
        with ExcelFile(tmp_excel) as reader:
            recons: DataFrame = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(expected, recons)

    def test_basics_with_nan(self, frame: DataFrame, tmp_excel: str) -> None:
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.to_excel(tmp_excel, sheet_name='test1')
        frame.to_excel(tmp_excel, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(tmp_excel, sheet_name='test1', header=False)
        frame.to_excel(tmp_excel, sheet_name='test1', index=False)

    @pytest.mark.parametrize(
        'np_type',
        [np.int8, np.int16, np.int32, np.int64],
    )
    def test_int_types(
        self,
        np_type: type,
        tmp_excel: str,
    ) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).integers(-10, 10, size=(10, 2)),
            dtype=np_type,
            index=Index(list(range(10))),
        )
        df.to_excel(tmp_excel, sheet_name='test1')
        with ExcelFile(tmp_excel) as reader:
            recons: DataFrame = pd.read_excel(reader, sheet_name='test1', index_col=0)
        int_frame: DataFrame = df.astype(np.int64)
        tm.assert_frame_equal(int_frame, recons)
        with ExcelFile(tmp_excel) as reader:
            recons2: DataFrame = pd.read_excel(reader, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(int_frame, recons2)

    @pytest.mark.parametrize(
        'np_type',
        [np.float16, np.float32, np.float64],
    )
    def test_float_types(
        self,
        np_type: type,
        tmp_excel: str,
    ) -> None:
        df: DataFrame = DataFrame(
            np.random.default_rng(2).random(10),
            dtype=np_type,
            index=Index(list(range(10))),
        )
        df.to_excel(tmp_excel, sheet_name='test1')
        with ExcelFile(tmp_excel) as reader:
            recons: DataFrame = pd.read_excel(
                reader,
                sheet_name='test1',
                index_col=0,
            ).astype(np_type)
        tm.assert_frame_equal(df, recons)

    def test_bool_types(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            [1, 0, True, False],
            dtype=np.bool_,
            index=Index(list(range(4))),
        )
        df.to_excel(tmp_excel, sheet_name='test1')
        with ExcelFile(tmp_excel) as reader:
            recons: DataFrame = pd.read_excel(
                reader,
                sheet_name='test1',
                index_col=0,
            ).astype(np.bool_)
        tm.assert_frame_equal(df, recons)

    def test_inf_roundtrip(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            [(1, np.inf), (2, 3), (5, -np.inf)],
            index=Index(list(range(3))),
        )
        df.to_excel(tmp_excel, sheet_name='test1')
        with ExcelFile(tmp_excel) as reader:
            recons: DataFrame = pd.read_excel(
                reader,
                sheet_name='test1',
                index_col=0,
            )
        tm.assert_frame_equal(df, recons)

    def test_sheets(self, frame: DataFrame, tmp_excel: str) -> None:
        unit: str = get_exp_unit(tmp_excel)
        tsframe: DataFrame = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list('ABCD')),
            index=date_range('2000-01-01', periods=5, freq='B'),
        )
        index: pd.DatetimeIndex = pd.DatetimeIndex(np.asarray(tsframe.index), freq=None)
        tsframe.index = index
        expected: DataFrame = tsframe[:]
        expected.index = expected.index.as_unit(unit)
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.to_excel(tmp_excel, sheet_name='test1')
        frame.to_excel(tmp_excel, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(tmp_excel, sheet_name='test1', header=False)
        frame.to_excel(tmp_excel, sheet_name='test1', index=False)
        with ExcelWriter(tmp_excel) as writer:
            frame.to_excel(writer, sheet_name='test1')
            tsframe.to_excel(writer, sheet_name='test2')
        with ExcelFile(tmp_excel) as reader:
            recons: DataFrame = pd.read_excel(reader, sheet_name='test1', index_col=0)
            tm.assert_frame_equal(frame, recons)
            recons = pd.read_excel(reader, sheet_name='test2', index_col=0)
        tm.assert_frame_equal(expected, recons)
        assert 2 == len(reader.sheet_names)
        assert 'test1' == reader.sheet_names[0]
        assert 'test2' == reader.sheet_names[1]

    def test_colaliases(self, frame: DataFrame, tmp_excel: str) -> None:
        frame = frame.copy()
        frame.iloc[:5, frame.columns.get_loc('A')] = np.nan
        frame.to_excel(tmp_excel, sheet_name='test1')
        frame.to_excel(tmp_excel, sheet_name='test1', columns=['A', 'B'])
        frame.to_excel(tmp_excel, sheet_name='test1', header=False)
        frame.to_excel(tmp_excel, sheet_name='test1', index=False)
        col_aliases: Index = Index(['AA', 'X', 'Y', 'Z'])
        frame.to_excel(tmp_excel, sheet_name='test1', header=col_aliases)
        with ExcelFile(tmp_excel) as reader:
            rs: DataFrame = pd.read_excel(reader, sheet_name='test1', index_col=0)
        xp: DataFrame = frame.copy()
        xp.columns = col_aliases
        tm.assert_frame_equal(xp, rs)

    @pytest.mark.parametrize(
        'c_idx_nlevels',
        [1, 2, 3],
    )
    @pytest.mark.parametrize(
        'r_idx_nlevels',
        [1, 2, 3],
    )
    @pytest.mark.parametrize(
        'use_headers',
        [True, False],
    )
    def test_excel_010_hemstring(
        self,
        merge_cells: Union[bool, str],
        c_idx_nlevels: int,
        r_idx_nlevels: int,
        use_headers: bool,
        tmp_excel: str,
    ) -> None:

        def roundtrip(
            data: DataFrame,
            header: bool = True,
            parser_hdr: Optional[int] = 0,
            index: bool = True,
        ) -> DataFrame:
            data.to_excel(
                tmp_excel,
                header=header,
                merge_cells=merge_cells,
                index=index,
            )
            with ExcelFile(tmp_excel) as xf:
                return pd.read_excel(
                    xf,
                    sheet_name=xf.sheet_names[0],
                    header=parser_hdr,
                )

        parser_header: Optional[int] = 0 if use_headers else None
        res: DataFrame = roundtrip(DataFrame([0]), use_headers, parser_header)
        assert res.shape == (1, 2)
        assert res.iloc[0, 0] is not np.nan
        nrows: int = 5
        ncols: int = 3
        if c_idx_nlevels == 1:
            columns: Index = Index([f'a-{i}' for i in range(ncols)], dtype=object)
        else:
            columns = MultiIndex.from_arrays(
                [range(ncols) for _ in range(c_idx_nlevels)],
                names=[f'i-{i}' for i in range(c_idx_nlevels)],
            )
        if r_idx_nlevels == 1:
            index: Index = Index([f'b-{i}' for i in range(nrows)], dtype=object)
        else:
            index = MultiIndex.from_arrays(
                [range(nrows) for _ in range(r_idx_nlevels)],
                names=[f'j-{i}' for i in range(r_idx_nlevels)],
            )
        df: DataFrame = DataFrame(
            np.ones((nrows, ncols)),
            columns=columns,
            index=index,
        )
        if c_idx_nlevels > 1:
            msg: str = (
                "Writing to Excel with MultiIndex columns and no index "
                r"\('index'=False\) is not yet implemented."
            )
            with pytest.raises(NotImplementedError, match=msg):
                roundtrip(df, use_headers, index=False)
        else:
            res = roundtrip(df, use_headers)
            if use_headers:
                assert res.shape == (nrows, ncols + r_idx_nlevels)
            else:
                assert res.shape == (nrows - 1, ncols + r_idx_nlevels)
            for r in range(len(res.index)):
                for c in range(len(res.columns)):
                    assert res.iloc[r, c] is not np.nan

    def test_duplicated_columns(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            columns=['A', 'B', 'B'],
        )
        df.to_excel(tmp_excel, sheet_name='test1')
        expected: DataFrame = DataFrame(
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
            columns=['A', 'B', 'B.1'],
        )
        result: DataFrame = pd.read_excel(tmp_excel, sheet_name='test1', index_col=0)
        tm.assert_frame_equal(result, expected)
        df = DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            columns=['A', 'B', 'A', 'B'],
        )
        df.to_excel(tmp_excel, sheet_name='test1')
        result = pd.read_excel(tmp_excel, sheet_name='test1', index_col=0)
        expected = DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            columns=['A', 'B', 'A.1', 'B.1'],
        )
        tm.assert_frame_equal(result, expected)
        df.to_excel(tmp_excel, sheet_name='test1', index=False, header=False)
        result = pd.read_excel(tmp_excel, sheet_name='test1', header=None)
        expected = DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
        )
        tm.assert_frame_equal(result, expected)

    def test_swapped_columns(self, tmp_excel: str) -> None:
        write_frame: DataFrame = DataFrame({'A': [1, 1, 1], 'B': [2, 2, 2]})
        write_frame.to_excel(tmp_excel, sheet_name='test1', columns=['B', 'A'])
        read_frame: DataFrame = pd.read_excel(tmp_excel, sheet_name='test1', header=0)
        tm.assert_series_equal(write_frame['A'], read_frame['A'])
        tm.assert_series_equal(write_frame['B'], read_frame['B'])

    def test_invalid_columns(self, tmp_excel: str) -> None:
        write_frame: DataFrame = DataFrame({'A': [1, 1, 1], 'B': [2, 2, 2]})
        with pytest.raises(KeyError, match='Not all names specified'):
            write_frame.to_excel(tmp_excel, sheet_name='test1', columns=['B', 'C'])
        with pytest.raises(
            KeyError,
            match="'passes columns are not ALL present dataframe'",
        ):
            write_frame.to_excel(tmp_excel, sheet_name='test1', columns=['C', 'D'])

    @pytest.mark.parametrize(
        'to_excel_index,read_excel_index_col',
        [(True, 0), (False, None)],
    )
    def test_write_subset_columns(
        self,
        tmp_excel: str,
        to_excel_index: bool,
        read_excel_index_col: Optional[int],
    ) -> None:
        write_frame: DataFrame = DataFrame(
            {'A': [1, 1, 1], 'B': [2, 2, 2], 'C': [3, 3, 3]},
        )
        write_frame.to_excel(
            tmp_excel,
            sheet_name='col_subset_bug',
            columns=['A', 'B'],
            index=to_excel_index,
        )
        expected: DataFrame = write_frame[['A', 'B']]
        read_frame: DataFrame = pd.read_excel(
            tmp_excel,
            sheet_name='col_subset_bug',
            index_col=read_excel_index_col,
        )
        tm.assert_frame_equal(expected, read_frame)

    def test_comment_arg(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            {'A': ['one', '#one', 'one'], 'B': ['two', 'two', '#two']}
        )
        df.to_excel(tmp_excel, sheet_name='test_c')
        result1: DataFrame = pd.read_excel(tmp_excel, sheet_name='test_c', index_col=0)
        result1.iloc[1, 0] = None
        result1.iloc[1, 1] = None
        result1.iloc[2, 1] = None
        result2: DataFrame = pd.read_excel(
            tmp_excel, sheet_name='test_c', comment='#', index_col=0
        )
        tm.assert_frame_equal(result1, result2)

    def test_comment_default(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            {'A': ['one', '#one', 'one'], 'B': ['two', 'two', '#two']}
        )
        df.to_excel(tmp_excel, sheet_name='test_c')
        result1: DataFrame = pd.read_excel(tmp_excel, sheet_name='test_c')
        result2: DataFrame = pd.read_excel(
            tmp_excel, sheet_name='test_c', comment=None
        )
        tm.assert_frame_equal(result1, result2)

    def test_comment_used(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            {'A': ['one', '#one', 'one'], 'B': ['two', 'two', '#two']}
        )
        df.to_excel(tmp_excel, sheet_name='test_c')
        expected: DataFrame = DataFrame(
            {'A': ['one', None, 'one'], 'B': ['two', None, None]}
        )
        result: DataFrame = pd.read_excel(
            tmp_excel, sheet_name='test_c', comment='#', index_col=0
        )
        tm.assert_frame_equal(result, expected)

    def test_comment_empty_line(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            {'a': ['1', '#2'], 'b': ['2', '3']}
        )
        df.to_excel(tmp_excel, index=False)
        expected: DataFrame = DataFrame({'a': [1], 'b': [2]})
        result: DataFrame = pd.read_excel(tmp_excel, comment='#')
        tm.assert_frame_equal(result, expected)

    def test_datetimes(self, tmp_excel: str) -> None:
        unit: str = get_exp_unit(tmp_excel)
        datetimes: List[datetime] = [
            datetime(2013, 1, 13, 1, 2, 3),
            datetime(2013, 1, 13, 2, 45, 56),
            datetime(2013, 1, 13, 4, 29, 49),
            datetime(2013, 1, 13, 6, 13, 42),
            datetime(2013, 1, 13, 7, 57, 35),
            datetime(2013, 1, 13, 9, 41, 28),
            datetime(2013, 1, 13, 11, 25, 21),
            datetime(2013, 1, 13, 13, 9, 14),
            datetime(2013, 1, 13, 14, 53, 7),
            datetime(2013, 1, 13, 16, 37, 0),
            datetime(2013, 1, 13, 18, 20, 52),
        ]
        write_frame: DataFrame = DataFrame(
            {'A': datetimes},
        )
        write_frame.to_excel(tmp_excel, sheet_name='Sheet1')
        read_frame: DataFrame = pd.read_excel(
            tmp_excel,
            sheet_name='Sheet1',
            header=0,
        )
        expected: DataFrame = write_frame.astype(f'M8[{unit}]')
        tm.assert_series_equal(expected['A'], read_frame['A'])

    def test_bytes_io(self, engine: str) -> None:
        with BytesIO() as bio:
            df: DataFrame = DataFrame(
                np.random.default_rng(2).standard_normal((10, 2))
            )
            with ExcelWriter(bio, engine=engine) as writer:
                df.to_excel(writer)
            bio.seek(0)
            reread_df: DataFrame = pd.read_excel(bio, index_col=0)
            tm.assert_frame_equal(df, reread_df)

    def test_engine_kwargs(self, engine: str, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            [{'A': 1, 'B': 2}, {'A': 3, 'B': 4}]
        )
        msgs: Dict[str, str] = {
            'odf': "OpenDocumentSpreadsheet() got an unexpected keyword argument 'foo'",
            'openpyxl': "__init__() got an unexpected keyword argument 'foo'",
            'xlsxwriter': "__init__() got an unexpected keyword argument 'foo'",
        }
        msgs['openpyxl'] = "Workbook.__init__() got an unexpected keyword argument 'foo'"
        msgs['xlsxwriter'] = "Workbook.__init__() got an unexpected keyword argument 'foo'"
        if engine == 'openpyxl' and (not os.path.exists(tmp_excel)):
            msgs['openpyxl'] = "load_workbook() got an unexpected keyword argument 'foo'"
        with pytest.raises(TypeError, match=re.escape(msgs[engine])):
            df.to_excel(tmp_excel, engine=engine, engine_kwargs={'foo': 'bar'})

    def test_write_lists_dict(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            {
                'mixed': ['a', ['b', 'c'], {'d': 'e', 'f': 2}],
                'numeric': [1, 2, 3.0],
                'str': ['apple', 'banana', 'cherry'],
            }
        )
        df.to_excel(tmp_excel, sheet_name='Sheet1')
        read: DataFrame = pd.read_excel(tmp_excel, sheet_name='Sheet1', header=0, index_col=0)
        expected: DataFrame = df.copy()
        expected.mixed = expected.mixed.apply(str)
        expected.numeric = expected.numeric.astype('int64')
        tm.assert_frame_equal(read, expected)

    def test_render_as_column_name(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            {'render': [1, 2], 'data': [3, 4]}
        )
        df.to_excel(tmp_excel, sheet_name='Sheet1')
        read: DataFrame = pd.read_excel(tmp_excel, 'Sheet1', index_col=0)
        expected: DataFrame = df
        tm.assert_frame_equal(read, expected)

    def test_true_and_false_value_options(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            [['foo', 'bar']],
            columns=['col1', 'col2'],
            dtype=object,
        )
        with option_context(
            'future.no_silent_downcasting', True
        ):
            expected: DataFrame = df.replace({'foo': True, 'bar': False}).astype('bool')
        df.to_excel(tmp_excel)
        read_frame: DataFrame = pd.read_excel(
            tmp_excel,
            true_values=['foo'],
            false_values=['bar'],
            index_col=0,
        )
        tm.assert_frame_equal(read_frame, expected)

    def test_freeze_panes(self, tmp_excel: str) -> None:
        expected: DataFrame = DataFrame(
            [[1, 2], [3, 4]],
            columns=['col1', 'col2'],
        )
        expected.to_excel(
            tmp_excel,
            sheet_name='Sheet1',
            freeze_panes=(1, 1),
        )
        result: DataFrame = pd.read_excel(tmp_excel, index_col=0)
        tm.assert_frame_equal(result, expected)

    def test_path_path_lib(self, engine: str, ext: str, tmp_path: Any) -> None:
        df: DataFrame = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list('ABCD')),
            index=Index([f'i-{i}' for i in range(30)]),
        )
        writer = partial(df.to_excel, engine=engine)
        reader = partial(pd.read_excel, index_col=0)
        result: DataFrame = tm.round_trip_pathlib(
            writer, reader, path=f'foo{ext}'
        )
        tm.assert_frame_equal(result, df)

    def test_merged_cell_custom_objects(self, tmp_excel: str) -> None:
        mi: MultiIndex = MultiIndex.from_tuples(
            [
                (pd.Period('2018'), pd.Period('2018Q1')),
                (pd.Period('2018'), pd.Period('2018Q2')),
            ]
        )
        expected: DataFrame = DataFrame(
            np.ones((2, 2), dtype='int64'),
            columns=mi,
        )
        expected.to_excel(tmp_excel)
        result: DataFrame = pd.read_excel(
            tmp_excel,
            header=[0, 1],
            index_col=0,
        )
        expected.columns = expected.columns.set_levels(
            [[str(i) for i in mi.levels[0]], [str(i) for i in mi.levels[1]]],
            level=[0, 1],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        'dtype',
        [None, object],
    )
    def test_raise_when_saving_timezones(
        self,
        dtype: Optional[type],
        tz_aware_fixture: Any,
        tmp_excel: str,
    ) -> None:
        tz: Any = tz_aware_fixture
        data: pd.Timestamp = pd.Timestamp('2019', tz=tz)
        df: DataFrame = DataFrame([data], dtype=dtype)
        with pytest.raises(
            ValueError,
            match='Excel does not support',
        ):
            df.to_excel(tmp_excel)
        data = data.to_pydatetime()
        df = DataFrame([data], dtype=dtype)
        with pytest.raises(
            ValueError,
            match='Excel does not support',
        ):
            df.to_excel(tmp_excel)

    def test_excel_duplicate_columns_with_names(self, tmp_excel: str) -> None:
        df: DataFrame = DataFrame(
            {'A': [0, 1], 'B': [10, 11]}
        )
        df.to_excel(tmp_excel, columns=['A', 'B', 'A'], index=False)
        result: DataFrame = pd.read_excel(tmp_excel)
        expected: DataFrame = DataFrame(
            [[0, 10, 0], [1, 11, 1]],
            columns=['A', 'B', 'A.1'],
        )
        tm.assert_frame_equal(result, expected)

    def test_if_sheet_exists_raises(self, tmp_excel: str) -> None:
        msg: str = "if_sheet_exists is only valid in append mode (mode='a')"
        with pytest.raises(ValueError, match=re.escape(msg)):
            ExcelWriter(tmp_excel, if_sheet_exists='replace')

    def test_excel_writer_empty_frame(
        self,
        engine: str,
        tmp_excel: str,
    ) -> None:
        with ExcelWriter(tmp_excel, engine=engine) as writer:
            DataFrame().to_excel(writer)
        result: DataFrame = pd.read_excel(tmp_excel)
        expected: DataFrame = DataFrame()
        tm.assert_frame_equal(result, expected)

    def test_to_excel_empty_frame(
        self,
        engine: str,
        tmp_excel: str,
    ) -> None:
        DataFrame().to_excel(tmp_excel, engine=engine)
        result: DataFrame = pd.read_excel(tmp_excel)
        expected: DataFrame = DataFrame()
        tm.assert_frame_equal(result, expected)

    def test_to_excel_raising_warning_when_cell_character_exceed_limit(self) -> None:
        df: DataFrame = DataFrame({'A': ['a' * 32768]})
        msg: str = 'Cell contents too long \\(32768\\), truncated to 32767 characters'
        with tm.assert_produces_warning(
            UserWarning, match=msg, raise_on_extra_warnings=False
        ):
            buf: BytesIO = BytesIO()
            df.to_excel(buf)


class TestExcelWriterEngineTests:

    @pytest.mark.parametrize(
        'klass,ext',
        [
            pytest.param(
                _XlsxWriter,
                '.xlsx',
                marks=td.skip_if_no('xlsxwriter'),
            ),
            pytest.param(
                _OpenpyxlWriter,
                '.xlsx',
                marks=td.skip_if_no('openpyxl'),
            ),
        ],
    )
    def test_ExcelWriter_dispatch(
        self,
        klass: Any,
        ext: str,
        tmp_excel: str,
    ) -> None:
        with ExcelWriter(tmp_excel) as writer:
            if ext == '.xlsx' and bool(
                import_optional_dependency('xlsxwriter', errors='ignore')
            ):
                assert isinstance(writer, _XlsxWriter)
            else:
                assert isinstance(writer, klass)

    def test_ExcelWriter_dispatch_raises(self) -> None:
        with pytest.raises(ValueError, match='No engine'):
            ExcelWriter('nothing')

    def test_register_writer(self, tmp_path: Any) -> None:

        class DummyClass(ExcelWriter):
            called_save: bool = False
            called_write_cells: bool = False
            called_sheets: bool = False
            _supported_extensions: Tuple[str, ...] = ('xlsx', 'xls')
            _engine: str = 'dummy'

            def book(self) -> None:
                pass

            def _save(self) -> None:
                type(self).called_save = True

            def _write_cells(self, *args: Any, **kwargs: Any) -> None:
                type(self).called_write_cells = True

            @property
            def sheets(self) -> Any:
                type(self).called_sheets = True

            @classmethod
            def assert_called_and_reset(cls) -> None:
                assert cls.called_save
                assert cls.called_write_cells
                assert not cls.called_sheets
                cls.called_save = False
                cls.called_write_cells = False

        register_writer(DummyClass)
        with option_context('io.excel.xlsx.writer', 'dummy'):
            filepath = tmp_path / 'something.xlsx'
            filepath.touch()
            with ExcelWriter(filepath) as writer:
                assert isinstance(writer, DummyClass)
            df: DataFrame = DataFrame(
                ['a'],
                columns=Index(['b'], name='foo'),
                index=Index(['c'], name='bar'),
            )
            df.to_excel(filepath)
            DummyClass.assert_called_and_reset()
        filepath2 = tmp_path / 'something2.xlsx'
        filepath2.touch()
        df.to_excel(filepath2, engine='dummy')
        DummyClass.assert_called_and_reset()


@td.skip_if_no('xlrd')
@td.skip_if_no('openpyxl')
class TestFSPath:

    def test_excelfile_fspath(self, tmp_path: Any) -> None:
        path = tmp_path / 'foo.xlsx'
        path.touch()
        df: DataFrame = DataFrame({'A': [1, 2]})
        df.to_excel(path)
        with ExcelFile(path) as xl:
            result: str = os.fspath(xl)
        assert result == str(path)

    def test_excelwriter_fspath(self, tmp_path: Any) -> None:
        path = tmp_path / 'foo.xlsx'
        path.touch()
        with ExcelWriter(path) as writer:
            assert os.fspath(writer) == str(path)


@pytest.mark.parametrize(
    'klass',
    list(_writers.values()),
)
def test_subclass_attr(klass: Any) -> None:
    attrs_base: set = {name for name in dir(ExcelWriter) if not name.startswith('_')}
    attrs_klass: set = {name for name in dir(klass) if not name.startswith('_')}
    assert not attrs_base.symmetric_difference(attrs_klass)
