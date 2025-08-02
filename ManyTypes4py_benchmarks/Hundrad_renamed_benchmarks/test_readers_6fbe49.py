from __future__ import annotations
from datetime import datetime, time
from functools import partial
from io import BytesIO
import os
from pathlib import Path
import platform
import re
from urllib.error import URLError
import uuid
from zipfile import BadZipFile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series, read_csv
import pandas._testing as tm
read_ext_params = ['.xls', '.xlsx', '.xlsm', '.xlsb', '.ods']
engine_params = [pytest.param('xlrd', marks=[td.skip_if_no('xlrd')]),
    pytest.param('openpyxl', marks=[td.skip_if_no('openpyxl')]), pytest.
    param(None, marks=[td.skip_if_no('xlrd')]), pytest.param('pyxlsb',
    marks=td.skip_if_no('pyxlsb')), pytest.param('odf', marks=td.skip_if_no
    ('odf')), pytest.param('calamine', marks=td.skip_if_no('python_calamine'))]


def func_8c5a9y9x(engine, read_ext):
    """
    Filter out invalid (engine, ext) pairs instead of skipping, as that
    produces 500+ pytest.skips.
    """
    engine = engine.values[0]
    if engine == 'openpyxl' and read_ext == '.xls':
        return False
    if engine == 'odf' and read_ext != '.ods':
        return False
    if read_ext == '.ods' and engine not in {'odf', 'calamine'}:
        return False
    if engine == 'pyxlsb' and read_ext != '.xlsb':
        return False
    if read_ext == '.xlsb' and engine not in {'pyxlsb', 'calamine'}:
        return False
    if engine == 'xlrd' and read_ext != '.xls':
        return False
    return True


def func_jds5urnz(engine, read_ext):
    """
    engine gives us a pytest.param object with some marks, read_ext is just
    a string.  We need to generate a new pytest.param inheriting the marks.
    """
    values = engine.values + (read_ext,)
    new_param = pytest.param(values, marks=engine.marks)
    return new_param


@pytest.fixture(params=[func_jds5urnz(eng, ext) for eng in engine_params for
    ext in read_ext_params if func_8c5a9y9x(eng, ext)], ids=str)
def func_kghfv52i(request):
    """
    Fixture for Excel reader engine and read_ext, only including valid pairs.
    """
    return request.param


@pytest.fixture
def func_d7yn5ojf(engine_and_read_ext):
    engine, read_ext = engine_and_read_ext
    return engine


@pytest.fixture
def func_xzsbpzmy(engine_and_read_ext):
    engine, read_ext = engine_and_read_ext
    return read_ext


@pytest.fixture
def func_hn75yqc5(read_ext, tmp_path):
    tmp = tmp_path / f'{uuid.uuid4()}{read_ext}'
    tmp.touch()
    return str(tmp)


@pytest.fixture
def func_u8i3bhxt(datapath):
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    filepath = datapath('io', 'data', 'csv', 'test1.csv')
    df_ref = read_csv(filepath, index_col=0, parse_dates=True, engine='python')
    return df_ref


def func_h5xd31uv(read_ext, engine):
    unit = 'us'
    if (read_ext == '.ods') ^ (engine == 'calamine'):
        unit = 's'
    return unit


def func_r8n3x9o9(expected, read_ext, engine):
    expected.index.name = None
    unit = func_h5xd31uv(read_ext, engine)
    expected.index = expected.index.as_unit(unit)


def func_2tw9ycnx(engine, request):
    if engine == 'pyxlsb':
        request.applymarker(pytest.mark.xfail(reason=
            'Sheets containing datetimes not supported by pyxlsb'))


class TestReaders:

    @pytest.mark.parametrize('col', [[True, None, False], [True], [True, 
        False]])
    def func_9jcm22ji(self, col, tmp_excel, read_ext):
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({'bool_column': col}, dtype='boolean')
        df.to_excel(tmp_excel, index=False)
        df2 = pd.read_excel(tmp_excel, dtype={'bool_column': 'boolean'})
        tm.assert_frame_equal(df, df2)

    def func_rbbtqw94(self, datapath):
        f_path = datapath('io', 'data', 'excel', 'test_none_type.xlsx')
        with pd.ExcelFile(f_path) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=True, na_values=['nan', 'None', 'abcd'],
                dtype='boolean', engine='openpyxl')
        expected = DataFrame({'Test': [True, None, False, None, False, None,
            True]}, dtype='boolean')
        tm.assert_frame_equal(parsed, expected)

    @pytest.fixture(autouse=True)
    def func_pqpvebzn(self, engine, datapath, monkeypatch):
        """
        Change directory and set engine for read_excel calls.
        """
        func = partial(pd.read_excel, engine=engine)
        monkeypatch.chdir(datapath('io', 'data', 'excel'))
        monkeypatch.setattr(pd, 'read_excel', func)

    def func_xvopw874(self, read_ext, engine, monkeypatch):

        def func_u2bfvezs(self, *args, **kwargs):
            return self.engine
        monkeypatch.setattr(pd.ExcelFile, 'parse', parser)
        expected_defaults = {'xlsx': 'openpyxl', 'xlsm': 'openpyxl', 'xlsb':
            'pyxlsb', 'xls': 'xlrd', 'ods': 'odf'}
        with open('test1' + read_ext, 'rb') as f:
            result = pd.read_excel(f)
        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def func_h5dyecrz(self, read_ext, engine):
        expected_defaults = {'xlsx': {'foo': 'abcd'}, 'xlsm': {'foo': 123},
            'xlsb': {'foo': 'True'}, 'xls': {'foo': True}, 'ods': {'foo':
            'abcd'}}
        if engine in {'xlrd', 'pyxlsb'}:
            msg = re.escape(
                "open_workbook() got an unexpected keyword argument 'foo'")
        elif engine == 'odf':
            msg = re.escape("load() got an unexpected keyword argument 'foo'")
        else:
            msg = re.escape(
                "load_workbook() got an unexpected keyword argument 'foo'")
        if engine is not None:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
                    index_col=0, engine_kwargs=expected_defaults[read_ext[1:]])

    def func_oj5as2b3(self, read_ext):
        msg = 'Passing an integer for `usecols`'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
                index_col=0, usecols=3)
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet2', skiprows
                =[1], index_col=0, usecols=3)

    def func_z3koxpzd(self, request, engine, read_ext, df_ref):
        func_2tw9ycnx(engine, request)
        expected = df_ref[['B', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        df1 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols=[0, 2, 3])
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0, usecols=[0, 2, 3])
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def func_yxcwh5jq(self, request, engine, read_ext, df_ref):
        func_2tw9ycnx(engine, request)
        expected = df_ref[['A', 'B', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols='A:D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0, usecols='A:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        expected = df_ref[['B', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols='A,C,D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0, usecols='A,C,D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols='A,C:D')
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0, usecols='A,C:D')
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize('usecols', [[0, 1, 3], [0, 3, 1], [1, 0, 3], [
        1, 3, 0], [3, 0, 1], [3, 1, 0]])
    def func_gamky2lo(self, request, engine, read_ext, usecols, df_ref):
        func_2tw9ycnx(engine, request)
        expected = df_ref[['A', 'C']]
        func_r8n3x9o9(expected, read_ext, engine)
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols=usecols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('usecols', [['B', 'D'], ['D', 'B']])
    def func_0t8qtwg1(self, read_ext, usecols, df_ref):
        expected = df_ref[['B', 'D']]
        expected.index = range(len(expected))
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            usecols=usecols)
        tm.assert_frame_equal(result, expected)

    def func_prw3pvl9(self, request, engine, read_ext, df_ref):
        func_2tw9ycnx(engine, request)
        expected = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0)
        tm.assert_frame_equal(result, expected)

    def func_ot2ryb70(self, request, engine, read_ext, df_ref):
        func_2tw9ycnx(engine, request)
        expected = df_ref[['C', 'D']]
        func_r8n3x9o9(expected, read_ext, engine)
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, usecols='A,D:E')
        tm.assert_frame_equal(result, expected)

    def func_odz2bv7c(self, read_ext):
        msg = 'Invalid column name: E1'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1', usecols=
                'D:E1')

    def func_e7xv9x7m(self, read_ext):
        msg = 'list indices must be integers.*, not str'
        with pytest.raises(TypeError, match=msg):
            pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
                index_col=['A'], usecols=['A', 'C'])

    def func_8u4hi162(self, read_ext):
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet3',
            index_col='A')
        expected = DataFrame(columns=['B', 'C', 'D', 'E', 'F'], index=Index
            ([], name='A'))
        tm.assert_frame_equal(result, expected)

    def func_btud0p8t(self, read_ext):
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet3',
            index_col=['A', 'B', 'C'])
        expected = DataFrame(columns=['D', 'E', 'F'], index=MultiIndex(
            levels=[[]] * 3, codes=[[]] * 3, names=['A', 'B', 'C']))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('index_col', [None, 2])
    def func_fc989g5u(self, read_ext, index_col):
        result = pd.read_excel('test1' + read_ext, sheet_name='Sheet4',
            index_col=index_col)
        expected = DataFrame([['i1', 'a', 'x'], ['i2', 'b', 'y']], columns=
            ['Unnamed: 0', 'col1', 'col2'])
        if index_col:
            expected = expected.set_index(expected.columns[index_col])
        tm.assert_frame_equal(result, expected)

    def func_1skx9sju(self, read_ext):
        msg = (
            "Usecols do not match columns, columns expected but not found: \\['E'\\]"
            )
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, usecols=['E'])

    def func_tlzzjw1a(self, read_ext):
        msg = (
            "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
            )
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, usecols=['E1', 0])

    def func_lhnpaax8(self, read_ext):
        parsed = pd.read_excel('test2' + read_ext, sheet_name='Sheet1')
        expected = DataFrame([['aaaa', 'bbbbb']], columns=['Test', 'Test1'])
        tm.assert_frame_equal(parsed, expected)

    def func_xd4a5cnq(self, request, engine, read_ext):
        func_2tw9ycnx(engine, request)
        if engine == 'calamine' and read_ext == '.ods':
            request.applymarker(pytest.mark.xfail(reason=
                "Calamine can't extract error from ods files"))
        parsed = pd.read_excel('test3' + read_ext, sheet_name='Sheet1')
        expected = DataFrame([[np.nan]], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)

    def func_dvjwgrg7(self, request, engine, read_ext, df_ref):
        func_2tw9ycnx(engine, request)
        expected = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        df1 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0)
        df2 = pd.read_excel('test1' + read_ext, sheet_name='Sheet2',
            skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        df3 = pd.read_excel('test1' + read_ext, sheet_name='Sheet1',
            index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def func_2pkvsvn9(self, request, engine, read_ext):
        func_2tw9ycnx(engine, request)
        unit = func_h5xd31uv(read_ext, engine)
        expected = DataFrame.from_dict({'IntCol': [1, 2, -3, 4, 0],
            'FloatCol': [1.25, 2.25, 1.83, 1.92, 5e-10], 'BoolCol': [True, 
            False, True, True, False], 'StrCol': [1, 2, 3, 4, 5], 'Str2Col':
            ['a', 3, 'c', 'd', 'e'], 'DateCol': Index([datetime(2013, 10, 
            30), datetime(2013, 10, 31), datetime(1905, 1, 1), datetime(
            2013, 12, 14), datetime(2015, 3, 14)], dtype=f'M8[{unit}]')})
        basename = 'test_types'
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, expected)
        float_expected = expected.copy()
        float_expected.loc[float_expected.index[1], 'Str2Col'] = 3.0
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, float_expected)
        for icol, name in enumerate(expected.columns):
            actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1',
                index_col=icol)
            exp = expected.set_index(name)
            tm.assert_frame_equal(actual, exp)
        expected['StrCol'] = expected['StrCol'].apply(str)
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1',
            converters={'StrCol': str})
        tm.assert_frame_equal(actual, expected)

    def func_87zqxtpf(self, read_ext):
        basename = 'test_converters'
        expected = DataFrame.from_dict({'IntCol': [1, 2, -3, -1000, 0],
            'FloatCol': [12.5, np.nan, 18.3, 19.2, 5e-09], 'BoolCol': [
            'Found', 'Found', 'Found', 'Not found', 'Found'], 'StrCol': [
            '1', np.nan, '3', '4', '5']})
        converters = {'IntCol': lambda x: int(x) if x != '' else -1000,
            'FloatCol': lambda x: 10 * x if x else np.nan, (2): lambda x: 
            'Found' if x != '' else 'Not found', (3): lambda x: str(x) if x
             else ''}
        actual = pd.read_excel(basename + read_ext, sheet_name='Sheet1',
            converters=converters)
        tm.assert_frame_equal(actual, expected)

    def func_5a7nwomg(self, read_ext):
        basename = 'testdtype'
        actual = pd.read_excel(basename + read_ext)
        expected = DataFrame({'a': [1, 2, 3, 4], 'b': [2.5, 3.5, 4.5, 5.5],
            'c': [1, 2, 3, 4], 'd': [1.0, 2.0, np.nan, 4.0]})
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(basename + read_ext, dtype={'a': 'float64',
            'b': 'float32', 'c': str})
        expected['a'] = expected['a'].astype('float64')
        expected['b'] = expected['b'].astype('float32')
        expected['c'] = Series(['001', '002', '003', '004'], dtype='str')
        tm.assert_frame_equal(actual, expected)
        msg = 'Unable to convert column d to type int64'
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(basename + read_ext, dtype={'d': 'int64'})

    @pytest.mark.parametrize('dtype,expected', [(None, {'a': [1, 2, 3, 4],
        'b': [2.5, 3.5, 4.5, 5.5], 'c': [1, 2, 3, 4], 'd': [1.0, 2.0, np.
        nan, 4.0]}), ({'a': 'float64', 'b': 'float32', 'c': str, 'd': str},
        {'a': Series([1, 2, 3, 4], dtype='float64'), 'b': Series([2.5, 3.5,
        4.5, 5.5], dtype='float32'), 'c': Series(['001', '002', '003',
        '004'], dtype='str'), 'd': Series(['1', '2', np.nan, '4'], dtype=
        'str')})])
    def func_gfrjq93o(self, read_ext, dtype, expected):
        basename = 'testdtype'
        actual = pd.read_excel(basename + read_ext, dtype=dtype)
        expected = DataFrame(expected)
        tm.assert_frame_equal(actual, expected)

    def func_6qav95dx(self, read_ext, dtype_backend, engine, tmp_excel):
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({'a': Series([1, 3], dtype='Int64'), 'b': Series([
            2.5, 4.5], dtype='Float64'), 'c': Series([True, False], dtype=
            'boolean'), 'd': Series(['a', 'b'], dtype='string'), 'e':
            Series([pd.NA, 6], dtype='Int64'), 'f': Series([pd.NA, 7.5],
            dtype='Float64'), 'g': Series([pd.NA, True], dtype='boolean'),
            'h': Series([pd.NA, 'a'], dtype='string'), 'i': Series([pd.
            Timestamp('2019-12-31')] * 2), 'j': Series([pd.NA, pd.NA],
            dtype='Int64')})
        df.to_excel(tmp_excel, sheet_name='test', index=False)
        result = pd.read_excel(tmp_excel, sheet_name='test', dtype_backend=
            dtype_backend)
        if dtype_backend == 'pyarrow':
            import pyarrow as pa
            from pandas.arrays import ArrowExtensionArray
            expected = DataFrame({col: ArrowExtensionArray(pa.array(df[col],
                from_pandas=True)) for col in df.columns})
            expected['i'] = ArrowExtensionArray(expected['i'].array.
                _pa_array.cast(pa.timestamp(unit='us')))
            expected['j'] = ArrowExtensionArray(pa.array([None, None]))
        else:
            expected = df
            unit = func_h5xd31uv(read_ext, engine)
            expected['i'] = expected['i'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(result, expected)

    def func_hr5mddk1(self, read_ext, tmp_excel):
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({'a': [np.nan, 1.0], 'b': [2.5, np.nan]})
        df.to_excel(tmp_excel, sheet_name='test', index=False)
        result = pd.read_excel(tmp_excel, sheet_name='test', dtype_backend=
            'numpy_nullable', dtype='float64')
        tm.assert_frame_equal(result, df)

    def func_2pxrdduy(self, read_ext, string_storage, tmp_excel):
        if read_ext in ('.xlsb', '.xls'):
            pytest.skip(f"No engine for filetype: '{read_ext}'")
        df = DataFrame({'a': np.array(['a', 'b'], dtype=np.object_), 'b':
            np.array(['x', pd.NA], dtype=np.object_)})
        df.to_excel(tmp_excel, sheet_name='test', index=False)
        with pd.option_context('mode.string_storage', string_storage):
            result = pd.read_excel(tmp_excel, sheet_name='test',
                dtype_backend='numpy_nullable')
        expected = DataFrame({'a': Series(['a', 'b'], dtype=pd.StringDtype(
            string_storage)), 'b': Series(['x', None], dtype=pd.StringDtype
            (string_storage))})
        tm.assert_frame_equal(result, expected, check_column_type=False)

    @pytest.mark.parametrize('dtypes, exp_value', [({}, 1), ({'a.1':
        'int64'}, 1)])
    def func_vvj9ao3r(self, read_ext, dtypes, exp_value):
        basename = 'df_mangle_dup_col_dtypes'
        dtype_dict = {'a': object, **dtypes}
        dtype_dict_copy = dtype_dict.copy()
        result = pd.read_excel(basename + read_ext, dtype=dtype_dict)
        expected = DataFrame({'a': Series([1], dtype=object), 'a.1': Series
            ([exp_value], dtype=object if not dtypes else None)})
        assert dtype_dict == dtype_dict_copy, 'dtype dict changed'
        tm.assert_frame_equal(result, expected)

    def func_2h3l68t3(self, read_ext):
        basename = 'test_spaces'
        actual = pd.read_excel(basename + read_ext)
        expected = DataFrame({'testcol': ['this is great', '4    spaces',
            '1 trailing ', ' 1 leading', '2  spaces  multiple  times']})
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize('basename,expected', [('gh-35802', DataFrame({
        'COLUMN': ['Test (1)']})), ('gh-36122', DataFrame(columns=[
        'got 2nd sa']))])
    def func_n28pbj5x(self, engine, read_ext, basename, expected):
        if engine != 'odf':
            pytest.skip(f'Skipped for engine: {engine}')
        actual = pd.read_excel(basename + read_ext)
        tm.assert_frame_equal(actual, expected)

    def func_aozvz3g7(self, read_ext):
        basename = 'test_multisheet'
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        expected_keys = ['Charlie', 'Alpha', 'Beta']
        tm.assert_contains_all(expected_keys, dfs.keys())
        assert expected_keys == list(dfs.keys())

    def func_hjnou5kb(self, read_ext):
        basename = 'test_multisheet'
        expected_keys = [2, 'Charlie', 'Charlie']
        dfs = pd.read_excel(basename + read_ext, sheet_name=expected_keys)
        expected_keys = list(set(expected_keys))
        tm.assert_contains_all(expected_keys, dfs.keys())
        assert len(expected_keys) == len(dfs.keys())

    def func_rq0mhu8p(self, read_ext):
        basename = 'blank_with_header'
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        expected_keys = ['Sheet1', 'Sheet2', 'Sheet3']
        tm.assert_contains_all(expected_keys, dfs.keys())

    def func_ppwy56rz(self, read_ext):
        actual = pd.read_excel('blank' + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, DataFrame())

    def func_pnm7bky4(self, read_ext):
        expected = DataFrame(columns=['col_1', 'col_2'])
        actual = pd.read_excel('blank_with_header' + read_ext, sheet_name=
            'Sheet1')
        tm.assert_frame_equal(actual, expected)

    def func_n1j51rgs(self, read_ext):
        with pytest.raises(ValueError, match=' \\(sheet: Sheet1\\)$'):
            pd.read_excel('blank_with_header' + read_ext, header=[1],
                sheet_name=None)
        with pytest.raises(ZeroDivisionError, match=' \\(sheet: Sheet1\\)$'):
            pd.read_excel('test1' + read_ext, usecols=lambda x: 1 / 0,
                sheet_name=None)

    @pytest.mark.filterwarnings('ignore:Cell A4 is marked:UserWarning:openpyxl'
        )
    def func_kl4f470s(self, request, engine, read_ext):
        func_2tw9ycnx(engine, request)
        expected = DataFrame([[pd.Timestamp('2016-03-12'), 'Marc Johnson'],
            [pd.Timestamp('2016-03-16'), 'Jack Black'], [1e+20,
            'Timothy Brown']], columns=['DateColWithBigInt', 'StringCol'])
        if engine == 'openpyxl':
            request.applymarker(pytest.mark.xfail(reason=
                'Maybe not supported by openpyxl'))
        if engine is None and read_ext in ('.xlsx', '.xlsm'):
            request.applymarker(pytest.mark.xfail(reason=
                'Defaults to openpyxl, maybe not supported'))
        result = pd.read_excel('testdateoverflow' + read_ext)
        tm.assert_frame_equal(result, expected)

    def func_x29f0xa4(self, request, read_ext, engine, df_ref):
        func_2tw9ycnx(engine, request)
        filename = 'test1'
        sheet_name = 'Sheet1'
        expected = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        df1 = pd.read_excel(filename + read_ext, sheet_name=sheet_name,
            index_col=0)
        df2 = pd.read_excel(filename + read_ext, index_col=0, sheet_name=
            sheet_name)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def func_370zx14s(self, read_ext):
        pth = 'test1' + read_ext
        expected = pd.read_excel(pth, sheet_name='Sheet1', index_col=0)
        with open(pth, 'rb') as f:
            actual = pd.read_excel(f, sheet_name='Sheet1', index_col=0)
            tm.assert_frame_equal(expected, actual)

    def func_vibxezpe(self):
        bad_engine = 'foo'
        with pytest.raises(ValueError, match='Unknown engine: foo'):
            pd.read_excel('', engine=bad_engine)

    @pytest.mark.parametrize('sheet_name', [3, [0, 3], [3, 0], 'Sheet4', [
        'Sheet1', 'Sheet4'], ['Sheet4', 'Sheet1']])
    def func_b2xk3vtr(self, read_ext, sheet_name):
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('blank' + read_ext, sheet_name=sheet_name)

    def func_hn9gi63w(self, read_ext):
        bad_file = f'foo{read_ext}'
        match = '|'.join(['(No such file or directory', '没有那个文件或目录',
            'File o directory non esistente)'])
        with pytest.raises(FileNotFoundError, match=match):
            pd.read_excel(bad_file)

    def func_s9em03wo(self, engine):
        bad_stream = b'foo'
        if engine is None:
            error = ValueError
            msg = (
                'Excel file format cannot be determined, you must specify an engine manually.'
                )
        elif engine == 'xlrd':
            from xlrd import XLRDError
            error = XLRDError
            msg = (
                "Unsupported format, or corrupt file: Expected BOF record; found b'foo'"
                )
        elif engine == 'calamine':
            from python_calamine import CalamineError
            error = CalamineError
            msg = 'Cannot detect file format'
        else:
            error = BadZipFile
            msg = 'File is not a zip file'
        with pytest.raises(error, match=msg):
            pd.read_excel(BytesIO(bad_stream))

    @pytest.mark.network
    @pytest.mark.single_cpu
    def func_jtey8pb6(self, httpserver, read_ext):
        with open('test1' + read_ext, 'rb') as f:
            httpserver.serve_content(content=f.read())
        url_table = pd.read_excel(httpserver.url)
        local_table = pd.read_excel('test1' + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @td.skip_if_not_us_locale
    @pytest.mark.single_cpu
    def func_m5y9whc0(self, read_ext, s3_public_bucket, s3so):
        with open('test1' + read_ext, 'rb') as f:
            s3_public_bucket.put_object(Key='test1' + read_ext, Body=f)
        url = f's3://{s3_public_bucket.name}/test1' + read_ext
        url_table = pd.read_excel(url, storage_options=s3so)
        local_table = pd.read_excel('test1' + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.single_cpu
    def func_jsrouadv(self, read_ext, s3_public_bucket, s3so):
        with open('test1' + read_ext, 'rb') as f:
            s3_public_bucket.put_object(Key='test1' + read_ext, Body=f)
        import s3fs
        s3 = s3fs.S3FileSystem(**s3so)
        with s3.open(f's3://{s3_public_bucket.name}/test1' + read_ext) as f:
            url_table = pd.read_excel(f)
        local_table = pd.read_excel('test1' + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.slow
    def func_tlwbkj5s(self, read_ext, datapath):
        localtable = os.path.join(datapath('io', 'data', 'excel'), 'test1' +
            read_ext)
        local_table = pd.read_excel(localtable)
        try:
            url_table = pd.read_excel('file://localhost/' + localtable)
        except URLError:
            platform_info = ' '.join(platform.uname()).strip()
            pytest.skip(f'failing on {platform_info}')
        tm.assert_frame_equal(url_table, local_table)

    def func_501hh6xm(self, read_ext):
        str_path = 'test1' + read_ext
        expected = pd.read_excel(str_path, sheet_name='Sheet1', index_col=0)
        path_obj = Path('test1' + read_ext)
        actual = pd.read_excel(path_obj, sheet_name='Sheet1', index_col=0)
        tm.assert_frame_equal(expected, actual)

    def func_oxm84imj(self, read_ext):
        str_path = os.path.join('test1' + read_ext)
        with open(str_path, 'rb') as f:
            x = pd.read_excel(f, sheet_name='Sheet1', index_col=0)
            del x
            f.read()

    def func_r9frcbla(self, request, engine, read_ext):
        func_2tw9ycnx(engine, request)
        if engine == 'calamine' and read_ext == '.ods':
            request.applymarker(pytest.mark.xfail(reason=
                'ODS file contains bad datetime (seconds as text)'))
        expected = DataFrame.from_dict({'Time': [time(1, 2, 3), time(2, 45,
            56, 100000), time(4, 29, 49, 200000), time(6, 13, 42, 300000),
            time(7, 57, 35, 400000), time(9, 41, 28, 500000), time(11, 25, 
            21, 600000), time(13, 9, 14, 700000), time(14, 53, 7, 800000),
            time(16, 37, 0, 900000), time(18, 20, 54)]})
        actual = pd.read_excel('times_1900' + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel('times_1904' + read_ext, sheet_name='Sheet1')
        tm.assert_frame_equal(actual, expected)

    def func_wq9miazp(self, request, engine, read_ext):
        func_2tw9ycnx(engine, request)
        unit = func_h5xd31uv(read_ext, engine)
        mi = MultiIndex.from_product([['foo', 'bar'], ['a', 'b']])
        mi_file = 'testmultiindex' + read_ext
        expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [
            2, 3.5, pd.Timestamp('2015-01-02'), False], [3, 4.5, pd.
            Timestamp('2015-01-03'), False], [4, 5.5, pd.Timestamp(
            '2015-01-04'), True]], columns=mi)
        expected[mi[2]] = expected[mi[2]].astype(f'M8[{unit}]')
        actual = pd.read_excel(mi_file, sheet_name='mi_column', header=[0, 
            1], index_col=0)
        tm.assert_frame_equal(actual, expected)
        expected.index = mi
        expected.columns = ['a', 'b', 'c', 'd']
        actual = pd.read_excel(mi_file, sheet_name='mi_index', index_col=[0, 1]
            )
        tm.assert_frame_equal(actual, expected)
        expected.columns = mi
        actual = pd.read_excel(mi_file, sheet_name='both', index_col=[0, 1],
            header=[0, 1])
        tm.assert_frame_equal(actual, expected)
        expected.columns = ['a', 'b', 'c', 'd']
        expected.index = mi.set_names(['ilvl1', 'ilvl2'])
        actual = pd.read_excel(mi_file, sheet_name='mi_index_name',
            index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)
        expected.index = range(4)
        expected.columns = mi.set_names(['c1', 'c2'])
        actual = pd.read_excel(mi_file, sheet_name='mi_column_name', header
            =[0, 1], index_col=0)
        tm.assert_frame_equal(actual, expected)
        expected.columns = mi.set_levels([1, 2], level=1).set_names(['c1',
            'c2'])
        actual = pd.read_excel(mi_file, sheet_name='name_with_int',
            index_col=0, header=[0, 1])
        tm.assert_frame_equal(actual, expected)
        expected.columns = mi.set_names(['c1', 'c2'])
        expected.index = mi.set_names(['ilvl1', 'ilvl2'])
        actual = pd.read_excel(mi_file, sheet_name='both_name', index_col=[
            0, 1], header=[0, 1])
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel(mi_file, sheet_name='both_name_skiprows',
            index_col=[0, 1], header=[0, 1], skiprows=2)
        tm.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize('sheet_name,idx_lvl2', [(
        'both_name_blank_after_mi_name', [np.nan, 'b', 'a', 'b']), (
        'both_name_multiple_blanks', [np.nan] * 4)])
    def func_zgekno2g(self, request, engine, read_ext, sheet_name, idx_lvl2):
        func_2tw9ycnx(engine, request)
        mi_file = 'testmultiindex' + read_ext
        mi = MultiIndex.from_product([['foo', 'bar'], ['a', 'b']], names=[
            'c1', 'c2'])
        unit = func_h5xd31uv(read_ext, engine)
        expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [
            2, 3.5, pd.Timestamp('2015-01-02'), False], [3, 4.5, pd.
            Timestamp('2015-01-03'), False], [4, 5.5, pd.Timestamp(
            '2015-01-04'), True]], columns=mi, index=MultiIndex.from_arrays
            ((['foo', 'foo', 'bar', 'bar'], idx_lvl2), names=['ilvl1',
            'ilvl2']))
        expected[mi[2]] = expected[mi[2]].astype(f'M8[{unit}]')
        result = pd.read_excel(mi_file, sheet_name=sheet_name, index_col=[0,
            1], header=[0, 1])
        tm.assert_frame_equal(result, expected)

    def func_dw08fyac(self, read_ext):
        mi_file = 'testmultiindex' + read_ext
        result = pd.read_excel(mi_file, sheet_name='index_col_none', header
            =[0, 1])
        exp_columns = MultiIndex.from_product([('A', 'B'), ('key', 'val')])
        expected = DataFrame([[1, 2, 3, 4]] * 2, columns=exp_columns)
        tm.assert_frame_equal(result, expected)

    def func_yph7gn3n(self, read_ext):
        filename = 'test_index_name_pre17' + read_ext
        data = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan], ['R0C0',
            'R0C1', 'R0C2', 'R0C3', 'R0C4'], ['R1C0', 'R1C1', 'R1C2',
            'R1C3', 'R1C4'], ['R2C0', 'R2C1', 'R2C2', 'R2C3', 'R2C4'], [
            'R3C0', 'R3C1', 'R3C2', 'R3C3', 'R3C4'], ['R4C0', 'R4C1',
            'R4C2', 'R4C3', 'R4C4']], dtype=object)
        columns = ['C_l0_g0', 'C_l0_g1', 'C_l0_g2', 'C_l0_g3', 'C_l0_g4']
        mi = MultiIndex(levels=[['R0', 'R_l0_g0', 'R_l0_g1', 'R_l0_g2',
            'R_l0_g3', 'R_l0_g4'], ['R1', 'R_l1_g0', 'R_l1_g1', 'R_l1_g2',
            'R_l1_g3', 'R_l1_g4']], codes=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3,
            4, 5]], names=[None, None])
        si = Index(['R0', 'R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3',
            'R_l0_g4'], name=None)
        expected = DataFrame(data, index=si, columns=columns)
        actual = pd.read_excel(filename, sheet_name='single_names', index_col=0
            )
        tm.assert_frame_equal(actual, expected)
        expected.index = mi
        actual = pd.read_excel(filename, sheet_name='multi_names',
            index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)
        data = np.array([['R0C0', 'R0C1', 'R0C2', 'R0C3', 'R0C4'], ['R1C0',
            'R1C1', 'R1C2', 'R1C3', 'R1C4'], ['R2C0', 'R2C1', 'R2C2',
            'R2C3', 'R2C4'], ['R3C0', 'R3C1', 'R3C2', 'R3C3', 'R3C4'], [
            'R4C0', 'R4C1', 'R4C2', 'R4C3', 'R4C4']])
        columns = ['C_l0_g0', 'C_l0_g1', 'C_l0_g2', 'C_l0_g3', 'C_l0_g4']
        mi = MultiIndex(levels=[['R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3',
            'R_l0_g4'], ['R_l1_g0', 'R_l1_g1', 'R_l1_g2', 'R_l1_g3',
            'R_l1_g4']], codes=[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], names=[
            None, None])
        si = Index(['R_l0_g0', 'R_l0_g1', 'R_l0_g2', 'R_l0_g3', 'R_l0_g4'],
            name=None)
        expected = DataFrame(data, index=si, columns=columns)
        actual = pd.read_excel(filename, sheet_name='single_no_names',
            index_col=0)
        tm.assert_frame_equal(actual, expected)
        expected.index = mi
        actual = pd.read_excel(filename, sheet_name='multi_no_names',
            index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

    def func_9tnv8b2s(self, read_ext):
        msg = 'Passing a bool to header is invalid'
        for arg in [True, False]:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel('test1' + read_ext, header=arg)

    def func_v1wa7qzy(self, request, engine, read_ext):
        func_2tw9ycnx(engine, request)
        unit = func_h5xd31uv(read_ext, engine)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name=
            'skiprows_list', skiprows=[0, 2])
        expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [
            2, 3.5, pd.Timestamp('2015-01-02'), False], [3, 4.5, pd.
            Timestamp('2015-01-03'), False], [4, 5.5, pd.Timestamp(
            '2015-01-04'), True]], columns=['a', 'b', 'c', 'd'])
        expected['c'] = expected['c'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name=
            'skiprows_list', skiprows=np.array([0, 2]))
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name=
            'skiprows_list', skiprows=lambda x: x in [0, 2])
        tm.assert_frame_equal(actual, expected)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name=
            'skiprows_list', skiprows=3, names=['a', 'b', 'c', 'd'])
        expected = DataFrame([[2, 3.5, pd.Timestamp('2015-01-02'), False],
            [3, 4.5, pd.Timestamp('2015-01-03'), False], [4, 5.5, pd.
            Timestamp('2015-01-04'), True]], columns=['a', 'b', 'c', 'd'])
        expected['c'] = expected['c'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(actual, expected)

    def func_h06071fy(self, request, engine, read_ext):
        func_2tw9ycnx(engine, request)
        unit = func_h5xd31uv(read_ext, engine)
        actual = pd.read_excel('testskiprows' + read_ext, sheet_name=
            'skiprows_list', skiprows=lambda x: x not in [1, 3, 5])
        expected = DataFrame([[1, 2.5, pd.Timestamp('2015-01-01'), True], [
            3, 4.5, pd.Timestamp('2015-01-03'), False]], columns=['a', 'b',
            'c', 'd'])
        expected['c'] = expected['c'].astype(f'M8[{unit}]')
        tm.assert_frame_equal(actual, expected)

    def func_6jynh7cx(self, read_ext):
        num_rows_to_pull = 5
        actual = pd.read_excel('test1' + read_ext, nrows=num_rows_to_pull)
        expected = pd.read_excel('test1' + read_ext)
        expected = expected[:num_rows_to_pull]
        tm.assert_frame_equal(actual, expected)

    def func_0on7x3fy(self, read_ext):
        expected = pd.read_excel('test1' + read_ext)
        num_records_in_file = len(expected)
        num_rows_to_pull = num_records_in_file + 10
        actual = pd.read_excel('test1' + read_ext, nrows=num_rows_to_pull)
        tm.assert_frame_equal(actual, expected)

    def func_m072iuyh(self, read_ext):
        msg = "'nrows' must be an integer >=0"
        with pytest.raises(ValueError, match=msg):
            pd.read_excel('test1' + read_ext, nrows='5')

    @pytest.mark.parametrize('filename,sheet_name,header,index_col,skiprows',
        [('testmultiindex', 'mi_column', [0, 1], 0, None), (
        'testmultiindex', 'mi_index', None, [0, 1], None), (
        'testmultiindex', 'both', [0, 1], [0, 1], None), ('testmultiindex',
        'mi_column_name', [0, 1], 0, None), ('testskiprows',
        'skiprows_list', None, None, [0, 2]), ('testskiprows',
        'skiprows_list', None, None, lambda x: x in (0, 2))])
    def func_jfsn0t2f(self, read_ext, filename, sheet_name, header,
        index_col, skiprows):
        """
        For various parameters, we should get the same result whether we
        limit the rows during load (nrows=3) or after (df.iloc[:3]).
        """
        expected = pd.read_excel(filename + read_ext, sheet_name=sheet_name,
            header=header, index_col=index_col, skiprows=skiprows).iloc[:3]
        actual = pd.read_excel(filename + read_ext, sheet_name=sheet_name,
            header=header, index_col=index_col, skiprows=skiprows, nrows=3)
        tm.assert_frame_equal(actual, expected)

    def func_4jk18wqj(self, read_ext):
        with pytest.raises(TypeError, match='but 3 positional arguments'):
            pd.read_excel('test1' + read_ext, 'Sheet1', 0)

    def func_3i2lmdos(self, read_ext):
        file_name = 'testmultiindex' + read_ext
        data = [('B', 'B'), ('key', 'val'), (3, 4), (3, 4)]
        idx = MultiIndex.from_tuples([('A', 'A'), ('key', 'val'), (1, 2), (
            1, 2)], names=(0, 1))
        expected = DataFrame(data, index=idx, columns=(2, 3))
        result = pd.read_excel(file_name, sheet_name='index_col_none',
            index_col=[0, 1], header=None)
        tm.assert_frame_equal(expected, result)

    def func_nmgsfcgq(self, read_ext):
        file_name = 'one_col_blank_line' + read_ext
        data = [0.5, np.nan, 1, 2]
        expected = DataFrame(data, columns=['numbers'])
        result = pd.read_excel(file_name)
        tm.assert_frame_equal(result, expected)

    def func_y4lyz2hg(self, read_ext):
        file_name = 'testmultiindex' + read_ext
        columns = MultiIndex.from_tuples([('a', 'A'), ('b', 'B')])
        data = [[np.nan, np.nan], [np.nan, np.nan], [1, 3], [2, 4]]
        expected = DataFrame(data, columns=columns)
        result = pd.read_excel(file_name, sheet_name='mi_column_empty_rows',
            header=[0, 1])
        tm.assert_frame_equal(result, expected)

    def func_r0zdgb0q(self, read_ext):
        """
        Sheets can contain blank cells with no data. Some of our readers
        were including those cells, creating many empty rows and columns
        """
        file_name = 'trailing_blanks' + read_ext
        result = pd.read_excel(file_name)
        assert result.shape == (3, 3)

    def func_m40ma66s(self, request, engine, read_ext):
        if read_ext == '.ods':
            pytest.skip('chartsheets do not exist in the ODF format')
        if engine == 'pyxlsb':
            request.applymarker(pytest.mark.xfail(reason=
                "pyxlsb can't distinguish chartsheets from worksheets"))
        with pytest.raises(ValueError, match=
            "Worksheet named 'Chart1' not found"):
            pd.read_excel('chartsheet' + read_ext, sheet_name='Chart1')

    def func_ed9oh69s(self, request, engine, read_ext):
        if read_ext == '.ods':
            pytest.skip('chartsheets do not exist in the ODF format')
        if engine == 'pyxlsb':
            request.applymarker(pytest.mark.xfail(reason=
                "pyxlsb can't distinguish chartsheets from worksheets"))
        with pytest.raises(ValueError, match=
            'Worksheet index 1 is invalid, 1 worksheets found'):
            pd.read_excel('chartsheet' + read_ext, sheet_name=1)

    def func_8i3tgebh(self, read_ext):
        result = pd.read_excel('test_decimal' + read_ext, decimal=',',
            skiprows=1)
        expected = DataFrame([[1, 1521.1541, 187101.9543, 'ABC', 'poi', 
            4.738797819], [2, 121.12, 14897.76, 'DEF', 'uyt', 0.377320872],
            [3, 878.158, 108013.434, 'GHI', 'rez', 2.735694704]], columns=[
            'Id', 'Number1', 'Number2', 'Text1', 'Text2', 'Number3'])
        tm.assert_frame_equal(result, expected)


class TestExcelFileRead:

    def func_v03b6exe(self, engine, read_ext):
        msg = 'Expected file path name or file-like object'
        with pytest.raises(TypeError, match=msg):
            with open('test1' + read_ext, 'rb') as f:
                pd.read_excel(f.read(), engine=engine)

    @pytest.fixture(autouse=True)
    def func_pqpvebzn(self, engine, datapath, monkeypatch):
        """
        Change directory and set engine for ExcelFile objects.
        """
        func = partial(pd.ExcelFile, engine=engine)
        monkeypatch.chdir(datapath('io', 'data', 'excel'))
        monkeypatch.setattr(pd, 'ExcelFile', func)

    def func_xvopw874(self, read_ext, engine):
        expected_defaults = {'xlsx': 'openpyxl', 'xlsm': 'openpyxl', 'xlsb':
            'pyxlsb', 'xls': 'xlrd', 'ods': 'odf'}
        with pd.ExcelFile('test1' + read_ext) as excel:
            result = excel.engine
        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def func_mluoiufw(self, read_ext):
        with pd.ExcelFile('test4' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=False, na_values=['apple'])
        expected = DataFrame([['NA'], [1], ['NA'], [np.nan], ['rabbit']],
            columns=['Test'])
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile('test4' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=True, na_values=['apple'])
        expected = DataFrame([[np.nan], [1], [np.nan], [np.nan], ['rabbit']
            ], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile('test5' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=False, na_values=['apple'])
        expected = DataFrame([['1.#QNAN'], [1], ['nan'], [np.nan], [
            'rabbit']], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)
        with pd.ExcelFile('test5' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=True, na_values=['apple'])
        expected = DataFrame([[np.nan], [1], [np.nan], [np.nan], ['rabbit']
            ], columns=['Test'])
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('na_filter', [None, True, False])
    def func_qm14tkmz(self, read_ext, na_filter):
        kwargs = {}
        if na_filter is not None:
            kwargs['na_filter'] = na_filter
        with pd.ExcelFile('test5' + read_ext) as excel:
            parsed = pd.read_excel(excel, sheet_name='Sheet1',
                keep_default_na=True, na_values=['apple'], **kwargs)
        if na_filter is False:
            expected = [['1.#QNAN'], [1], ['nan'], ['apple'], ['rabbit']]
        else:
            expected = [[np.nan], [1], [np.nan], [np.nan], ['rabbit']]
        expected = DataFrame(expected, columns=['Test'])
        tm.assert_frame_equal(parsed, expected)

    def func_k07ibrmj(self, request, engine, read_ext, df_ref):
        func_2tw9ycnx(engine, request)
        expected = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        with pd.ExcelFile('test1' + read_ext) as excel:
            df1 = pd.read_excel(excel, sheet_name=0, index_col=0)
            df2 = pd.read_excel(excel, sheet_name=1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        with pd.ExcelFile('test1' + read_ext) as excel:
            df1 = excel.parse(0, index_col=0)
            df2 = excel.parse(1, skiprows=[1], index_col=0)
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
        with pd.ExcelFile('test1' + read_ext) as excel:
            df3 = pd.read_excel(excel, sheet_name=0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])
        with pd.ExcelFile('test1' + read_ext) as excel:
            df3 = excel.parse(0, index_col=0, skipfooter=1)
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def func_x29f0xa4(self, request, engine, read_ext, df_ref):
        func_2tw9ycnx(engine, request)
        expected = df_ref
        func_r8n3x9o9(expected, read_ext, engine)
        filename = 'test1'
        sheet_name = 'Sheet1'
        with pd.ExcelFile(filename + read_ext) as excel:
            df1_parse = excel.parse(sheet_name=sheet_name, index_col=0)
        with pd.ExcelFile(filename + read_ext) as excel:
            df2_parse = excel.parse(index_col=0, sheet_name=sheet_name)
        tm.assert_frame_equal(df1_parse, expected)
        tm.assert_frame_equal(df2_parse, expected)

    @pytest.mark.parametrize('sheet_name', [3, [0, 3], [3, 0], 'Sheet4', [
        'Sheet1', 'Sheet4'], ['Sheet4', 'Sheet1']])
    def func_b2xk3vtr(self, read_ext, sheet_name):
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        with pytest.raises(ValueError, match=msg):
            with pd.ExcelFile('blank' + read_ext) as excel:
                excel.parse(sheet_name=sheet_name)

    def func_370zx14s(self, engine, read_ext):
        pth = 'test1' + read_ext
        expected = pd.read_excel(pth, sheet_name='Sheet1', index_col=0,
            engine=engine)
        with open(pth, 'rb') as f:
            with pd.ExcelFile(f) as xls:
                actual = pd.read_excel(xls, sheet_name='Sheet1', index_col=0)
        tm.assert_frame_equal(expected, actual)

    def func_tfxs9u18(self, engine, read_ext):
        with open('test1' + read_ext, 'rb') as f:
            with pd.ExcelFile(f) as xlsx:
                pd.read_excel(xlsx, sheet_name='Sheet1', index_col=0,
                    engine=engine)
        assert f.closed

    def func_7kdf83v8(self, read_ext):
        msg = 'Engine should not be specified when passing an ExcelFile'
        with pd.ExcelFile('test1' + read_ext) as xl:
            with pytest.raises(ValueError, match=msg):
                pd.read_excel(xl, engine='foo')

    def func_hszzaef4(self, engine, read_ext):
        expected = pd.read_excel('test1' + read_ext, engine=engine)
        with open('test1' + read_ext, 'rb') as f:
            data = f.read()
        actual = pd.read_excel(BytesIO(data), engine=engine)
        tm.assert_frame_equal(expected, actual)

    def func_7ikmkc9m(self, read_ext, engine):
        with open('test1' + read_ext, 'rb') as f:
            result = pd.read_excel(f, engine=engine)
        expected = pd.read_excel('test1' + read_ext, engine=engine)
        tm.assert_frame_equal(result, expected)

    def func_t6ju4ctv(self, engine):
        with open('df_header_oob.xlsx', 'rb') as f:
            with pytest.raises(ValueError, match='exceeds maximum'):
                pd.read_excel(f, header=[0, 1])

    @pytest.mark.parametrize('filename', ['df_empty.xlsx', 'df_equals.xlsx'])
    def func_4sqi4ln2(self, filename):
        idx = Index(['Z'], name='I2')
        cols = MultiIndex.from_tuples([('A', 'B'), ('A', 'B.1')], names=[
            'I11', 'I12'])
        expected = DataFrame([[1, 3]], index=idx, columns=cols, dtype='int64')
        result = pd.read_excel(filename, sheet_name='Sheet1', index_col=0,
            header=[0, 1])
        tm.assert_frame_equal(expected, result)

    def func_hg41h2x8(self, request, engine, read_ext):
        func_2tw9ycnx(engine, request)
        f = 'test_datetime_mi' + read_ext
        with pd.ExcelFile(f) as excel:
            actual = pd.read_excel(excel, header=[0, 1], index_col=0,
                engine=engine)
        unit = func_h5xd31uv(read_ext, engine)
        dti = pd.DatetimeIndex(['2020-02-29', '2020-03-01'], dtype=
            f'M8[{unit}]')
        expected_column_index = MultiIndex.from_arrays([dti[:1], dti[1:]],
            names=[dti[0].to_pydatetime(), dti[1].to_pydatetime()])
        expected = DataFrame([], index=[], columns=expected_column_index)
        tm.assert_frame_equal(expected, actual)

    def func_rpca9dli(self, read_ext):
        with pytest.raises(ValueError, match='Value must be one of *'):
            with pd.option_context(f'io.excel{read_ext}.reader', 'abc'):
                pass

    def func_oviszlf1(self, request, engine, read_ext):
        if read_ext == '.ods':
            pytest.skip('chartsheets do not exist in the ODF format')
        if engine == 'pyxlsb':
            request.applymarker(pytest.mark.xfail(reason=
                "pyxlsb can't distinguish chartsheets from worksheets"))
        with pd.ExcelFile('chartsheet' + read_ext) as excel:
            assert excel.sheet_names == ['Sheet1']

    def func_n4h0gny4(self, engine, tmp_excel):
        errors = BadZipFile,
        if engine is None:
            pytest.skip(f'Invalid test for engine={engine}')
        elif engine == 'xlrd':
            import xlrd
            errors = BadZipFile, xlrd.biffh.XLRDError
        elif engine == 'calamine':
            from python_calamine import CalamineError
            errors = CalamineError,
        Path(tmp_excel).write_text('corrupt', encoding='utf-8')
        with tm.assert_produces_warning(False):
            try:
                pd.ExcelFile(tmp_excel, engine=engine)
            except errors:
                pass
