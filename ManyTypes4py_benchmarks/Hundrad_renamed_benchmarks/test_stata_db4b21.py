import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import itertools
import os
import string
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import DataFrame, Series
from pandas.io.parsers import read_csv
from pandas.io.stata import CategoricalConversionWarning, InvalidColumnName, PossiblePrecisionLoss, StataMissingValue, StataReader, StataWriter, StataWriterUTF8, ValueLabelTypeMismatch, read_stata


@pytest.fixture
def func_gr71xg09():
    return DataFrame({'a': [1, 2, 3, 4], 'b': [1.0, 3.0, 27.0, 81.0], 'c':
        ['Atlanta', 'Birmingham', 'Cincinnati', 'Detroit']})


@pytest.fixture
def func_ifrfjz5i(datapath):
    dta14_114 = datapath('io', 'data', 'stata', 'stata5_114.dta')
    parsed_114 = read_stata(dta14_114, convert_dates=True)
    parsed_114.index.name = 'index'
    return parsed_114


class TestStata:

    def func_faakeqon(self, file):
        return read_stata(file, convert_dates=True)

    def func_nk169mcs(self, file):
        return func_nk169mcs(file, parse_dates=True)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_49sliwby(self, version, temp_file):
        empty_ds = DataFrame(columns=['unit'])
        path = temp_file
        empty_ds.to_stata(path, write_index=False, version=version)
        empty_ds2 = read_stata(path)
        tm.assert_frame_equal(empty_ds, empty_ds2)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_chbphp4q(self, version, temp_file):
        empty_df_typed = DataFrame({'i8': np.array([0], dtype=np.int8),
            'i16': np.array([0], dtype=np.int16), 'i32': np.array([0],
            dtype=np.int32), 'i64': np.array([0], dtype=np.int64), 'u8': np
            .array([0], dtype=np.uint8), 'u16': np.array([0], dtype=np.
            uint16), 'u32': np.array([0], dtype=np.uint32), 'u64': np.array
            ([0], dtype=np.uint64), 'f32': np.array([0], dtype=np.float32),
            'f64': np.array([0], dtype=np.float64)})
        path = temp_file
        empty_df_typed.to_stata(path, write_index=False, version=version)
        empty_reread = read_stata(path)
        expected = empty_df_typed
        expected['u8'] = expected['u8'].astype(np.int8)
        expected['u16'] = expected['u16'].astype(np.int16)
        expected['u32'] = expected['u32'].astype(np.int32)
        expected['u64'] = expected['u64'].astype(np.int32)
        expected['i64'] = expected['i64'].astype(np.int32)
        tm.assert_frame_equal(expected, empty_reread)
        tm.assert_series_equal(expected.dtypes, empty_reread.dtypes)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_ajcvbvgy(self, version, temp_file):
        df = DataFrame({'a': range(5), 'b': ['b1', 'b2', 'b3', 'b4', 'b5']})
        path = temp_file
        df.to_stata(path, write_index=False, version=version)
        read_df = read_stata(path)
        assert isinstance(read_df.index, pd.RangeIndex)
        expected = df
        expected['a'] = expected['a'].astype(np.int32)
        tm.assert_frame_equal(read_df, expected, check_index_type=True)

    @pytest.mark.parametrize('version', [102, 103, 104, 105, 108, 110, 111,
        113, 114, 115, 117, 118, 119])
    def func_2lghgygb(self, version, datapath):
        file = datapath('io', 'data', 'stata', f'stata1_{version}.dta')
        parsed = self.read_dta(file)
        expected = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)],
            columns=['float_miss', 'double_miss', 'byte_miss', 'int_miss',
            'long_miss'])
        expected['float_miss'] = expected['float_miss'].astype(np.float32)
        if version <= 108:
            expected = expected.rename(columns={'float_miss': 'f_miss',
                'double_miss': 'd_miss', 'byte_miss': 'b_miss', 'int_miss':
                'i_miss', 'long_miss': 'l_miss'})
        tm.assert_frame_equal(parsed, expected)

    def func_06ufazzl(self, datapath):
        expected = DataFrame.from_records([(datetime(2006, 11, 19, 23, 13, 
            20), 1479596223000, datetime(2010, 1, 20), datetime(2010, 1, 8),
            datetime(2010, 1, 1), datetime(1974, 7, 1), datetime(2010, 1, 1
            ), datetime(2010, 1, 1)), (datetime(1959, 12, 31, 20, 3, 20), -
            1479590, datetime(1953, 10, 2), datetime(1948, 6, 10), datetime
            (1955, 1, 1), datetime(1955, 7, 1), datetime(1955, 1, 1),
            datetime(2, 1, 1)), (pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd
            .NaT, pd.NaT, pd.NaT)], columns=['datetime_c', 'datetime_big_c',
            'date', 'weekly_date', 'monthly_date', 'quarterly_date',
            'half_yearly_date', 'yearly_date'])
        expected['datetime_c'] = expected['datetime_c'].astype('M8[ms]')
        expected['date'] = expected['date'].astype('M8[s]')
        expected['weekly_date'] = expected['weekly_date'].astype('M8[s]')
        expected['monthly_date'] = expected['monthly_date'].astype('M8[s]')
        expected['quarterly_date'] = expected['quarterly_date'].astype('M8[s]')
        expected['half_yearly_date'] = expected['half_yearly_date'].astype(
            'M8[s]')
        expected['yearly_date'] = expected['yearly_date'].astype('M8[s]')
        path1 = datapath('io', 'data', 'stata', 'stata2_114.dta')
        path2 = datapath('io', 'data', 'stata', 'stata2_115.dta')
        path3 = datapath('io', 'data', 'stata', 'stata2_117.dta')
        msg = 'Leaving in Stata Internal Format'
        with tm.assert_produces_warning(UserWarning, match=msg):
            parsed_114 = self.read_dta(path1)
        with tm.assert_produces_warning(UserWarning, match=msg):
            parsed_115 = self.read_dta(path2)
        with tm.assert_produces_warning(UserWarning, match=msg):
            parsed_117 = self.read_dta(path3)
        tm.assert_frame_equal(parsed_114, expected)
        tm.assert_frame_equal(parsed_115, expected)
        tm.assert_frame_equal(parsed_117, expected)

    @pytest.mark.parametrize('file', ['stata3_113', 'stata3_114',
        'stata3_115', 'stata3_117'])
    def func_bjts3m28(self, file, datapath):
        file = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = self.read_dta(file)
        expected = self.read_csv(datapath('io', 'data', 'stata', 'stata3.csv'))
        expected = expected.astype(np.float32)
        expected['year'] = expected['year'].astype(np.int16)
        expected['quarter'] = expected['quarter'].astype(np.int8)
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [110, 111, 113, 114, 115, 117])
    def func_s6fdhyad(self, version, datapath):
        file = datapath('io', 'data', 'stata', f'stata4_{version}.dta')
        parsed = self.read_dta(file)
        expected = DataFrame.from_records([['one', 'ten', 'one', 'one',
            'one'], ['two', 'nine', 'two', 'two', 'two'], ['three', 'eight',
            'three', 'three', 'three'], ['four', 'seven', 4, 'four', 'four'
            ], ['five', 'six', 5, np.nan, 'five'], ['six', 'five', 6, np.
            nan, 'six'], ['seven', 'four', 7, np.nan, 'seven'], ['eight',
            'three', 8, np.nan, 'eight'], ['nine', 'two', 9, np.nan, 'nine'
            ], ['ten', 'one', 'ten', np.nan, 'ten']], columns=[
            'fully_labeled', 'fully_labeled2', 'incompletely_labeled',
            'labeled_with_missings', 'float_labelled'])
        for col in expected:
            orig = expected[col].copy()
            categories = np.asarray(expected['fully_labeled'][orig.notna()])
            if col == 'incompletely_labeled':
                categories = orig
            cat = orig.astype('category')._values
            cat = cat.set_categories(categories, ordered=True)
            cat.categories.rename(None, inplace=True)
            expected[col] = cat
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [102, 103, 104, 105, 108])
    def func_pbel0v4l(self, version, datapath):
        file = datapath('io', 'data', 'stata', f'stata4_{version}.dta')
        parsed = self.read_dta(file)
        expected = DataFrame.from_records([['one', 'ten', 'one', 'one',
            'one'], ['two', 'nine', 'two', 'two', 'two'], ['three', 'eight',
            'three', 'three', 'three'], ['four', 'seven', 4, 'four', 'four'
            ], ['five', 'six', 5, np.nan, 'five'], ['six', 'five', 6, np.
            nan, 'six'], ['seven', 'four', 7, np.nan, 'seven'], ['eight',
            'three', 8, np.nan, 'eight'], ['nine', 'two', 9, np.nan, 'nine'
            ], ['ten', 'one', 'ten', np.nan, 'ten']], columns=['fulllab',
            'fulllab2', 'incmplab', 'misslab', 'floatlab'])
        for col in expected:
            orig = expected[col].copy()
            categories = np.asarray(expected['fulllab'][orig.notna()])
            if col == 'incmplab':
                categories = orig
            cat = orig.astype('category')._values
            cat = cat.set_categories(categories, ordered=True)
            cat.categories.rename(None, inplace=True)
            expected[col] = cat
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('file', ['stata12_117', 'stata12_be_117',
        'stata12_118', 'stata12_be_118', 'stata12_119', 'stata12_be_119'])
    def func_bsgyyv0u(self, file, datapath):
        parsed = self.read_dta(datapath('io', 'data', 'stata', f'{file}.dta'))
        expected = DataFrame.from_records([[1, 'abc', 'abcdefghi'], [3,
            'cba', 'qwertywertyqwerty'], [93, '', 'strl']], columns=['x',
            'y', 'z'])
        tm.assert_frame_equal(parsed, expected, check_dtype=False)

    @pytest.mark.parametrize('file', ['stata14_118', 'stata14_be_118',
        'stata14_119', 'stata14_be_119'])
    def func_0fcmg1ho(self, file, datapath):
        parsed_118 = self.read_dta(datapath('io', 'data', 'stata',
            f'{file}.dta'))
        parsed_118['Bytes'] = parsed_118['Bytes'].astype('O')
        expected = DataFrame.from_records([['Cat', 'Bogota', 'Bogotá', 1, 
            1.0, 'option b Ünicode', 1.0], ['Dog', 'Boston', 'Uzunköprü',
            np.nan, np.nan, np.nan, np.nan], ['Plane', 'Rome', 'Tromsø', 0,
            0.0, 'option a', 0.0], ['Potato', 'Tokyo', 'Elâzığ', -4, 4.0, 4,
            4], ['', '', '', 0, 0.3332999, 'option a', 1 / 3.0]], columns=[
            'Things', 'Cities', 'Unicode_Cities_Strl', 'Ints', 'Floats',
            'Bytes', 'Longs'])
        expected['Floats'] = expected['Floats'].astype(np.float32)
        for col in parsed_118.columns:
            tm.assert_almost_equal(parsed_118[col], expected[col])
        with StataReader(datapath('io', 'data', 'stata', f'{file}.dta')
            ) as rdr:
            vl = rdr.variable_labels()
            vl_expected = {'Unicode_Cities_Strl':
                'Here are some strls with Ünicode chars', 'Longs':
                'long data', 'Things': 'Here are some things', 'Bytes':
                'byte data', 'Ints': 'int data', 'Cities':
                'Here are some cities', 'Floats': 'float data'}
            tm.assert_dict_equal(vl, vl_expected)
            assert rdr.data_label == 'This is a  Ünicode data label'

    def func_g4eoaimm(self, temp_file):
        original = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)],
            columns=['float_miss', 'double_miss', 'byte_miss', 'int_miss',
            'long_miss'])
        original.index.name = 'index'
        path = temp_file
        original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
        expected = original
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    def func_x12h35pi(self, datapath, temp_file):
        original = self.read_csv(datapath('io', 'data', 'stata', 'stata3.csv'))
        original.index.name = 'index'
        original.index = original.index.astype(np.int32)
        original['year'] = original['year'].astype(np.int32)
        original['quarter'] = original['quarter'].astype(np.int32)
        path = temp_file
        original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            original, check_index_type=False)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_jdme1aqo(self, version, temp_file, using_infer_string):
        original = DataFrame(data=[['string', 'object', 1, 1.1, np.
            datetime64('2003-12-25')]], columns=['string', 'object',
            'integer', 'floating', 'datetime'])
        original['object'] = Series(original['object'], dtype=object)
        original.index.name = 'index'
        original.index = original.index.astype(np.int32)
        original['integer'] = original['integer'].astype(np.int32)
        path = temp_file
        original.to_stata(path, convert_dates={'datetime': 'tc'}, version=
            version)
        written_and_read_again = self.read_dta(path)
        expected = original.copy()
        expected['datetime'] = expected['datetime'].astype('M8[ms]')
        if using_infer_string:
            expected['object'] = expected['object'].astype('str')
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    def func_emlcb2uy(self, temp_file):
        path = temp_file
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)),
            columns=list('AB'))
        df.to_stata(path)

    def func_f7hpbgxu(self, temp_file):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)),
            columns=list('abcd'))
        df.loc[2, 'a':'c'] = np.nan
        df_copy = df.copy()
        path = temp_file
        df.to_stata(path, write_index=False)
        tm.assert_frame_equal(df, df_copy)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_x5wmt25j(self, version, datapath, temp_file):
        raw = read_stata(datapath('io', 'data', 'stata', 'stata1_encoding.dta')
            )
        encoded = read_stata(datapath('io', 'data', 'stata',
            'stata1_encoding.dta'))
        result = encoded.kreis1849[0]
        expected = raw.kreis1849[0]
        assert result == expected
        assert isinstance(result, str)
        path = temp_file
        encoded.to_stata(path, write_index=False, version=version)
        reread_encoded = read_stata(path)
        tm.assert_frame_equal(encoded, reread_encoded)

    def func_ouiv6lqg(self, temp_file):
        original = DataFrame([(1, 2, 3, 4)], columns=['good', 'bäd',
            '8number', 'astringwithmorethan32characters______'])
        formatted = DataFrame([(1, 2, 3, 4)], columns=['good', 'b_d',
            '_8number', 'astringwithmorethan32characters_'])
        formatted.index.name = 'index'
        formatted = formatted.astype(np.int32)
        path = temp_file
        msg = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
        expected = formatted
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_ia3gsweg(self, version, temp_file):
        original = DataFrame([(1, 2, 3, 4, 5, 6)], columns=[
            'astringwithmorethan32characters_1',
            'astringwithmorethan32characters_2', '+', '-', 'short', 'delete'])
        formatted = DataFrame([(1, 2, 3, 4, 5, 6)], columns=[
            'astringwithmorethan32characters_',
            '_0astringwithmorethan32character', '_', '_1_', '_short',
            '_delete'])
        formatted.index.name = 'index'
        formatted = formatted.astype(np.int32)
        path = temp_file
        msg = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates=None, version=version)
        written_and_read_again = self.read_dta(path)
        expected = formatted
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    def func_8np405d0(self, temp_file):
        s1 = Series(2 ** 9, dtype=np.int16)
        s2 = Series(2 ** 17, dtype=np.int32)
        s3 = Series(2 ** 33, dtype=np.int64)
        original = DataFrame({'int16': s1, 'int32': s2, 'int64': s3})
        original.index.name = 'index'
        formatted = original
        formatted['int64'] = formatted['int64'].astype(np.float64)
        path = temp_file
        original.to_stata(path)
        written_and_read_again = self.read_dta(path)
        expected = formatted
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    @pytest.mark.parametrize('file', ['stata5_113', 'stata5_114',
        'stata5_115', 'stata5_117'])
    def func_a1lt1keo(self, file, parsed_114, version, datapath, temp_file):
        file = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = self.read_dta(file)
        parsed.index.name = 'index'
        tm.assert_frame_equal(parsed_114, parsed)
        path = temp_file
        func_ifrfjz5i.to_stata(path, convert_dates={'date_td': 'td'},
            version=version)
        written_and_read_again = self.read_dta(path)
        expected = func_ifrfjz5i.copy()
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    @pytest.mark.parametrize('file', ['stata6_113', 'stata6_114',
        'stata6_115', 'stata6_117'])
    def func_6jyhr1gh(self, file, datapath):
        expected = self.read_csv(datapath('io', 'data', 'stata', 'stata6.csv'))
        expected['byte_'] = expected['byte_'].astype(np.int8)
        expected['int_'] = expected['int_'].astype(np.int16)
        expected['long_'] = expected['long_'].astype(np.int32)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['double_'] = expected['double_'].astype(np.float64)
        arr = expected['date_td'].astype('Period[D]')._values.asfreq('s',
            how='S')
        expected['date_td'] = arr.view('M8[s]')
        file = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = self.read_dta(file)
        tm.assert_frame_equal(expected, parsed)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_cjdas6n9(self, version, temp_file):
        original = DataFrame([(1,)], columns=['variable'])
        time_stamp = datetime(2000, 2, 29, 14, 21)
        data_label = 'This is a data file.'
        path = temp_file
        original.to_stata(path, time_stamp=time_stamp, data_label=
            data_label, version=version)
        with StataReader(path) as reader:
            assert reader.time_stamp == '29 Feb 2000 14:21'
            assert reader.data_label == data_label

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_rt9xljsr(self, version, temp_file):
        original = DataFrame([(1,)], columns=['variable'])
        time_stamp = '01 Jan 2000, 00:00:00'
        path = temp_file
        msg = 'time_stamp should be datetime type'
        with pytest.raises(ValueError, match=msg):
            original.to_stata(path, time_stamp=time_stamp, version=version)
        assert not os.path.isfile(path)

    def func_u02qrrjt(self, temp_file):
        original = DataFrame(np.reshape(np.arange(25.0), (5, 5)))
        original.index.name = 'index'
        path = temp_file
        msg = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path)
        written_and_read_again = self.read_dta(path)
        written_and_read_again = written_and_read_again.set_index('index')
        columns = list(written_and_read_again.columns)
        convert_col_name = lambda x: int(x[1])
        written_and_read_again.columns = map(convert_col_name, columns)
        expected = original
        tm.assert_frame_equal(expected, written_and_read_again)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_bbpcggau(self, version, temp_file):
        s1 = Series(np.arange(4.0), dtype=np.float32)
        s2 = Series(np.arange(4.0), dtype=np.float64)
        s1[::2] = np.nan
        s2[1::2] = np.nan
        original = DataFrame({'s1': s1, 's2': s2})
        original.index.name = 'index'
        path = temp_file
        original.to_stata(path, version=version)
        written_and_read_again = self.read_dta(path)
        written_and_read_again = written_and_read_again.set_index('index')
        expected = original
        tm.assert_frame_equal(written_and_read_again, expected)

    def func_emb7vzlz(self, temp_file):
        columns = ['x', 'y']
        original = DataFrame(np.reshape(np.arange(10.0), (5, 2)), columns=
            columns)
        original.index.name = 'index_not_written'
        path = temp_file
        original.to_stata(path, write_index=False)
        written_and_read_again = self.read_dta(path)
        with pytest.raises(KeyError, match=original.index.name):
            written_and_read_again['index_not_written']

    def func_brme0kgw(self, temp_file):
        s1 = Series(['a', 'A longer string'])
        s2 = Series([1.0, 2.0], dtype=np.float64)
        original = DataFrame({'s1': s1, 's2': s2})
        original.index.name = 'index'
        path = temp_file
        original.to_stata(path)
        written_and_read_again = self.read_dta(path)
        expected = original
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    def func_lbrlbjrs(self, temp_file):
        s0 = Series([1, 99], dtype=np.int8)
        s1 = Series([1, 127], dtype=np.int8)
        s2 = Series([1, 2 ** 15 - 1], dtype=np.int16)
        s3 = Series([1, 2 ** 63 - 1], dtype=np.int64)
        original = DataFrame({'s0': s0, 's1': s1, 's2': s2, 's3': s3})
        original.index.name = 'index'
        path = temp_file
        with tm.assert_produces_warning(PossiblePrecisionLoss, match=
            'from int64 to'):
            original.to_stata(path)
        written_and_read_again = self.read_dta(path)
        modified = original
        modified['s1'] = Series(modified['s1'], dtype=np.int16)
        modified['s2'] = Series(modified['s2'], dtype=np.int32)
        modified['s3'] = Series(modified['s3'], dtype=np.float64)
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            modified)

    def func_wjou9xnh(self, temp_file):
        original = DataFrame([datetime(2006, 11, 19, 23, 13, 20)])
        original.index.name = 'index'
        path = temp_file
        msg = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates={(0): 'tc'})
        written_and_read_again = self.read_dta(path)
        expected = original.copy()
        expected.columns = ['_0']
        expected.index = original.index.astype(np.int32)
        expected['_0'] = expected['_0'].astype('M8[ms]')
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    def func_5q9gndmf(self, datapath):
        dpath = datapath('io', 'data', 'stata', 'S4_EDUC1.dta')
        df = read_stata(dpath)
        df0 = [[1, 1, 3, -2], [2, 1, 2, -2], [4, 1, 1, -2]]
        df0 = DataFrame(df0)
        df0.columns = ['clustnum', 'pri_schl', 'psch_num', 'psch_dis']
        df0['clustnum'] = df0['clustnum'].astype(np.int16)
        df0['pri_schl'] = df0['pri_schl'].astype(np.int8)
        df0['psch_num'] = df0['psch_num'].astype(np.int8)
        df0['psch_dis'] = df0['psch_dis'].astype(np.float32)
        tm.assert_frame_equal(df.head(3), df0)

    def func_054shn9w(self, datapath):
        dpath = datapath('io', 'data', 'stata', 'S4_EDUC1.dta')
        with StataReader(dpath) as reader:
            assert reader.value_labels() == {}

    def func_c8plrxc2(self, temp_file):
        columns = ['tc', 'td', 'tw', 'tm', 'tq', 'th', 'ty']
        conversions = {c: c for c in columns}
        data = [datetime(2006, 11, 20, 23, 13, 20)] * len(columns)
        original = DataFrame([data], columns=columns)
        original.index.name = 'index'
        expected_values = [datetime(2006, 11, 20, 23, 13, 20), datetime(
            2006, 11, 20), datetime(2006, 11, 19), datetime(2006, 11, 1),
            datetime(2006, 10, 1), datetime(2006, 7, 1), datetime(2006, 1, 1)]
        expected = DataFrame([expected_values], index=pd.Index([0], dtype=
            np.int32, name='index'), columns=columns, dtype='M8[s]')
        expected['tc'] = expected['tc'].astype('M8[ms]')
        path = temp_file
        original.to_stata(path, convert_dates=conversions)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    def func_na0b86xl(self, temp_file):
        original = DataFrame([['1'], [None]], columns=['foo'])
        expected = DataFrame([['1'], ['']], index=pd.RangeIndex(2, name=
            'index'), columns=['foo'])
        path = temp_file
        original.to_stata(path)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    @pytest.mark.parametrize('byteorder', ['>', '<'])
    def func_fod4kijf(self, byteorder, version, temp_file):
        s0 = Series([0, 1, True], dtype=np.bool_)
        s1 = Series([0, 1, 100], dtype=np.uint8)
        s2 = Series([0, 1, 255], dtype=np.uint8)
        s3 = Series([0, 1, 2 ** 15 - 100], dtype=np.uint16)
        s4 = Series([0, 1, 2 ** 16 - 1], dtype=np.uint16)
        s5 = Series([0, 1, 2 ** 31 - 100], dtype=np.uint32)
        s6 = Series([0, 1, 2 ** 32 - 1], dtype=np.uint32)
        original = DataFrame({'s0': s0, 's1': s1, 's2': s2, 's3': s3, 's4':
            s4, 's5': s5, 's6': s6})
        original.index.name = 'index'
        path = temp_file
        original.to_stata(path, byteorder=byteorder, version=version)
        written_and_read_again = self.read_dta(path)
        written_and_read_again = written_and_read_again.set_index('index')
        expected = original
        expected_types = (np.int8, np.int8, np.int16, np.int16, np.int32,
            np.int32, np.float64)
        for c, t in zip(expected.columns, expected_types):
            expected[c] = expected[c].astype(t)
        tm.assert_frame_equal(written_and_read_again, expected)

    def func_5mhi3rzp(self, datapath):
        with StataReader(datapath('io', 'data', 'stata', 'stata7_115.dta')
            ) as rdr:
            sr_115 = rdr.variable_labels()
        with StataReader(datapath('io', 'data', 'stata', 'stata7_117.dta')
            ) as rdr:
            sr_117 = rdr.variable_labels()
        keys = 'var1', 'var2', 'var3'
        labels = 'label1', 'label2', 'label3'
        for k, v in sr_115.items():
            assert k in sr_117
            assert v == sr_117[k]
            assert k in keys
            assert v in labels

    def func_7i0isxww(self, temp_file):
        str_lens = 1, 100, 244
        s = {}
        for str_len in str_lens:
            s['s' + str(str_len)] = Series(['a' * str_len, 'b' * str_len, 
                'c' * str_len])
        original = DataFrame(s)
        path = temp_file
        original.to_stata(path, write_index=False)
        with StataReader(path) as sr:
            sr._ensure_open()
            for variable, fmt, typ in zip(sr._varlist, sr._fmtlist, sr._typlist
                ):
                assert int(variable[1:]) == int(fmt[1:-1])
                assert int(variable[1:]) == typ

    def func_htnsqjad(self, temp_file):
        str_lens = 1, 244, 500
        s = {}
        for str_len in str_lens:
            s['s' + str(str_len)] = Series(['a' * str_len, 'b' * str_len, 
                'c' * str_len])
        original = DataFrame(s)
        msg = (
            "Fixed width strings in Stata \\.dta files are limited to 244 \\(or fewer\\)\\ncharacters\\.  Column 's500' does not satisfy this restriction\\. Use the\\n'version=117' parameter to write the newer \\(Stata 13 and later\\) format\\."
            )
        with pytest.raises(ValueError, match=msg):
            path = temp_file
            original.to_stata(path)

    def func_0hgkxukf(self, temp_file):
        types = 'b', 'h', 'l'
        df = DataFrame([[0.0]], columns=['float_'])
        path = temp_file
        df.to_stata(path)
        with StataReader(path) as rdr:
            valid_range = rdr.VALID_RANGE
        expected_values = [('.' + chr(97 + i)) for i in range(26)]
        expected_values.insert(0, '.')
        for t in types:
            offset = valid_range[t][1]
            for i in range(27):
                val = StataMissingValue(offset + 1 + i)
                assert val.string == expected_values[i]
        val = StataMissingValue(struct.unpack('<f', b'\x00\x00\x00\x7f')[0])
        assert val.string == '.'
        val = StataMissingValue(struct.unpack('<f', b'\x00\xd0\x00\x7f')[0])
        assert val.string == '.z'
        val = StataMissingValue(struct.unpack('<d',
            b'\x00\x00\x00\x00\x00\x00\xe0\x7f')[0])
        assert val.string == '.'
        val = StataMissingValue(struct.unpack('<d',
            b'\x00\x00\x00\x00\x00\x1a\xe0\x7f')[0])
        assert val.string == '.z'

    @pytest.mark.parametrize('version', [113, 115, 117])
    def func_o5wkqtj7(self, version, datapath):
        columns = ['int8_', 'int16_', 'int32_', 'float32_', 'float64_']
        smv = StataMissingValue(101)
        keys = sorted(smv.MISSING_VALUES.keys())
        data = []
        for i in range(27):
            row = [StataMissingValue(keys[i + j * 27]) for j in range(5)]
            data.append(row)
        expected = DataFrame(data, columns=columns)
        parsed = read_stata(datapath('io', 'data', 'stata',
            f'stata8_{version}.dta'), convert_missing=True)
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [104, 105, 108, 110, 111])
    def func_wbqxtehc(self, version, datapath):
        columns = ['int8_', 'int16_', 'int32_', 'float32_', 'float64_']
        smv = StataMissingValue(101)
        keys = sorted(smv.MISSING_VALUES.keys())
        data = []
        row = [StataMissingValue(keys[j * 27]) for j in range(5)]
        data.append(row)
        expected = DataFrame(data, columns=columns)
        parsed = read_stata(datapath('io', 'data', 'stata',
            f'stata8_{version}.dta'), convert_missing=True)
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [102, 103])
    def func_cgcymr9p(self, version, datapath):
        columns = ['int8_', 'int16_', 'int32_', 'float32_', 'float64_']
        smv = StataMissingValue(101)
        keys = sorted(smv.MISSING_VALUES.keys())
        data = []
        row = [StataMissingValue(keys[j * 27]) for j in [1, 1, 2, 3, 4]]
        data.append(row)
        expected = DataFrame(data, columns=columns)
        parsed = read_stata(datapath('io', 'data', 'stata',
            f'stata8_{version}.dta'), convert_missing=True)
        tm.assert_frame_equal(parsed, expected)

    def func_ol1zgnwq(self, datapath, temp_file):
        yr = [1960, 2000, 9999, 100, 2262, 1677]
        mo = [1, 1, 12, 1, 4, 9]
        dd = [1, 1, 31, 1, 22, 23]
        hr = [0, 0, 23, 0, 0, 0]
        mm = [0, 0, 59, 0, 0, 0]
        ss = [0, 0, 59, 0, 0, 0]
        expected = []
        for year, month, day, hour, minute, second in zip(yr, mo, dd, hr,
            mm, ss):
            row = []
            for j in range(7):
                if j == 0:
                    row.append(datetime(year, month, day, hour, minute, second)
                        )
                elif j == 6:
                    row.append(datetime(year, 1, 1))
                else:
                    row.append(datetime(year, month, day))
            expected.append(row)
        expected.append([pd.NaT] * 7)
        columns = ['date_tc', 'date_td', 'date_tw', 'date_tm', 'date_tq',
            'date_th', 'date_ty']
        expected[2][2] = datetime(9999, 12, 24)
        expected[2][3] = datetime(9999, 12, 1)
        expected[2][4] = datetime(9999, 10, 1)
        expected[2][5] = datetime(9999, 7, 1)
        expected[4][2] = datetime(2262, 4, 16)
        expected[4][3] = expected[4][4] = datetime(2262, 4, 1)
        expected[4][5] = expected[4][6] = datetime(2262, 1, 1)
        expected[5][2] = expected[5][3] = expected[5][4] = datetime(1677, 10, 1
            )
        expected[5][5] = expected[5][6] = datetime(1678, 1, 1)
        expected = DataFrame(expected, columns=columns, dtype=object)
        expected['date_tc'] = expected['date_tc'].astype('M8[ms]')
        expected['date_td'] = expected['date_td'].astype('M8[s]')
        expected['date_tm'] = expected['date_tm'].astype('M8[s]')
        expected['date_tw'] = expected['date_tw'].astype('M8[s]')
        expected['date_tq'] = expected['date_tq'].astype('M8[s]')
        expected['date_th'] = expected['date_th'].astype('M8[s]')
        expected['date_ty'] = expected['date_ty'].astype('M8[s]')
        parsed_115 = read_stata(datapath('io', 'data', 'stata',
            'stata9_115.dta'))
        parsed_117 = read_stata(datapath('io', 'data', 'stata',
            'stata9_117.dta'))
        tm.assert_frame_equal(expected, parsed_115)
        tm.assert_frame_equal(expected, parsed_117)
        date_conversion = {c: c[-2:] for c in columns}
        path = temp_file
        expected.index.name = 'index'
        expected.to_stata(path, convert_dates=date_conversion)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected.set_index(expected.index.astype(np.int32)))

    def func_tdg31z96(self, datapath):
        expected = self.read_csv(datapath('io', 'data', 'stata', 'stata6.csv'))
        expected['byte_'] = expected['byte_'].astype(np.int8)
        expected['int_'] = expected['int_'].astype(np.int16)
        expected['long_'] = expected['long_'].astype(np.int32)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['double_'] = expected['double_'].astype(np.float64)
        expected['date_td'] = expected['date_td'].astype('M8[s]')
        no_conversion = read_stata(datapath('io', 'data', 'stata',
            'stata6_117.dta'), convert_dates=True)
        tm.assert_frame_equal(expected, no_conversion)
        conversion = read_stata(datapath('io', 'data', 'stata',
            'stata6_117.dta'), convert_dates=True, preserve_dtypes=False)
        expected2 = self.read_csv(datapath('io', 'data', 'stata', 'stata6.csv')
            )
        expected2['date_td'] = expected['date_td']
        tm.assert_frame_equal(expected2, conversion)

    def func_b4gj91wm(self, datapath):
        expected = self.read_csv(datapath('io', 'data', 'stata', 'stata6.csv'))
        expected['byte_'] = expected['byte_'].astype(np.int8)
        expected['int_'] = expected['int_'].astype(np.int16)
        expected['long_'] = expected['long_'].astype(np.int32)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['double_'] = expected['double_'].astype(np.float64)
        expected['date_td'] = expected['date_td'].apply(datetime.strptime,
            args=('%Y-%m-%d',))
        columns = ['byte_', 'int_', 'long_']
        expected = expected[columns]
        dropped = read_stata(datapath('io', 'data', 'stata',
            'stata6_117.dta'), convert_dates=True, columns=columns)
        tm.assert_frame_equal(expected, dropped)
        columns = ['int_', 'long_', 'byte_']
        expected = expected[columns]
        reordered = read_stata(datapath('io', 'data', 'stata',
            'stata6_117.dta'), convert_dates=True, columns=columns)
        tm.assert_frame_equal(expected, reordered)
        msg = 'columns contains duplicate entries'
        with pytest.raises(ValueError, match=msg):
            read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'),
                convert_dates=True, columns=['byte_', 'byte_'])
        msg = (
            'The following columns were not found in the Stata data set: not_found'
            )
        with pytest.raises(ValueError, match=msg):
            read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'),
                convert_dates=True, columns=['byte_', 'int_', 'long_',
                'not_found'])

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    @pytest.mark.filterwarnings(
        'ignore:\\nStata value:pandas.io.stata.ValueLabelTypeMismatch')
    def func_jgz6o7hy(self, version, temp_file):
        original = DataFrame.from_records([['one', 'ten', 'one', 'one',
            'one', 1], ['two', 'nine', 'two', 'two', 'two', 2], ['three',
            'eight', 'three', 'three', 'three', 3], ['four', 'seven', 4,
            'four', 'four', 4], ['five', 'six', 5, np.nan, 'five', 5], [
            'six', 'five', 6, np.nan, 'six', 6], ['seven', 'four', 7, np.
            nan, 'seven', 7], ['eight', 'three', 8, np.nan, 'eight', 8], [
            'nine', 'two', 9, np.nan, 'nine', 9], ['ten', 'one', 'ten', np.
            nan, 'ten', 10]], columns=['fully_labeled', 'fully_labeled2',
            'incompletely_labeled', 'labeled_with_missings',
            'float_labelled', 'unlabeled'])
        path = temp_file
        original.astype('category').to_stata(path, version=version)
        written_and_read_again = self.read_dta(path)
        res = written_and_read_again.set_index('index')
        expected = original
        expected.index = expected.index.set_names('index')
        expected['incompletely_labeled'] = expected['incompletely_labeled'
            ].apply(str)
        expected['unlabeled'] = expected['unlabeled'].apply(str)
        for col in expected:
            orig = expected[col]
            cat = orig.astype('category')._values
            cat = cat.as_ordered()
            if col == 'unlabeled':
                cat = cat.set_categories(orig, ordered=True)
            cat.categories.rename(None, inplace=True)
            expected[col] = cat
        tm.assert_frame_equal(res, expected)

    def func_2o0n5jcf(self, temp_file):
        original = DataFrame.from_records([['a'], ['b'], ['c'], ['d'], [1]],
            columns=['Too_long']).astype('category')
        msg = (
            'data file created has not lost information due to duplicate labels'
            )
        with tm.assert_produces_warning(ValueLabelTypeMismatch, match=msg):
            original.to_stata(temp_file)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_ipjwjp4o(self, version, temp_file):
        values = [['a' + str(i)] for i in range(120)]
        values.append([np.nan])
        original = DataFrame.from_records(values, columns=['many_labels'])
        original = pd.concat([original[col].astype('category') for col in
            original], axis=1)
        original.index.name = 'index'
        path = temp_file
        original.to_stata(path, version=version)
        written_and_read_again = self.read_dta(path)
        res = written_and_read_again.set_index('index')
        expected = original
        for col in expected:
            cat = expected[col]._values
            new_cats = cat.remove_unused_categories().categories
            cat = cat.set_categories(new_cats, ordered=True)
            expected[col] = cat
        tm.assert_frame_equal(res, expected)

    @pytest.mark.parametrize('file', ['stata10_115', 'stata10_117'])
    def func_mnn9zfup(self, file, datapath):
        expected = [(True, 'ordered', ['a', 'b', 'c', 'd', 'e'], np.arange(
            5)), (True, 'reverse', ['a', 'b', 'c', 'd', 'e'], np.arange(5)[
            ::-1]), (True, 'noorder', ['a', 'b', 'c', 'd', 'e'], np.array([
            2, 1, 4, 0, 3])), (True, 'floating', ['a', 'b', 'c', 'd', 'e'],
            np.arange(0, 5)), (True, 'float_missing', ['a', 'd', 'e'], np.
            array([0, 1, 2, -1, -1])), (False, 'nolabel', [1.0, 2.0, 3.0, 
            4.0, 5.0], np.arange(5)), (True, 'int32_mixed', ['d', 2, 'e',
            'b', 'a'], np.arange(5))]
        cols = []
        for is_cat, col, labels, codes in expected:
            if is_cat:
                cols.append((col, pd.Categorical.from_codes(codes, labels,
                    ordered=True)))
            else:
                cols.append((col, Series(labels, dtype=np.float32)))
        expected = DataFrame.from_dict(dict(cols))
        file = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = read_stata(file)
        tm.assert_frame_equal(expected, parsed)
        for col in expected:
            if isinstance(expected[col].dtype, CategoricalDtype):
                tm.assert_series_equal(expected[col].cat.codes, parsed[col]
                    .cat.codes)
                tm.assert_index_equal(expected[col].cat.categories, parsed[
                    col].cat.categories)

    @pytest.mark.parametrize('file', ['stata11_115', 'stata11_117'])
    def func_crdr5ywi(self, file, datapath):
        parsed = read_stata(datapath('io', 'data', 'stata', f'{file}.dta'))
        parsed = parsed.sort_values('srh', na_position='first')
        parsed.index = pd.RangeIndex(len(parsed))
        codes = [-1, -1, 0, 1, 1, 1, 2, 2, 3, 4]
        categories = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
        cat = pd.Categorical.from_codes(codes=codes, categories=categories,
            ordered=True)
        expected = Series(cat, name='srh')
        tm.assert_series_equal(expected, parsed['srh'])

    @pytest.mark.parametrize('file', ['stata10_115', 'stata10_117'])
    def func_85xr0sl6(self, file, datapath):
        file = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = read_stata(file)
        parsed_unordered = read_stata(file, order_categoricals=False)
        for col in parsed:
            if not isinstance(parsed[col].dtype, CategoricalDtype):
                continue
            assert parsed[col].cat.ordered
            assert not parsed_unordered[col].cat.ordered

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('file', ['stata1_117', 'stata2_117',
        'stata3_117', 'stata4_117', 'stata5_117', 'stata6_117',
        'stata7_117', 'stata8_117', 'stata9_117', 'stata10_117', 'stata11_117']
        )
    @pytest.mark.parametrize('chunksize', [1, 2])
    @pytest.mark.parametrize('convert_categoricals', [False, True])
    @pytest.mark.parametrize('convert_dates', [False, True])
    def func_3s0muga7(self, file, chunksize, convert_categoricals,
        convert_dates, datapath):
        fname = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = read_stata(fname, convert_categoricals=
            convert_categoricals, convert_dates=convert_dates)
        with read_stata(fname, iterator=True, convert_categoricals=
            convert_categoricals, convert_dates=convert_dates) as itr:
            pos = 0
            for j in range(5):
                try:
                    chunk = itr.read(chunksize)
                except StopIteration:
                    break
                from_frame = parsed.iloc[pos:pos + chunksize, :].copy()
                from_frame = self._convert_categorical(from_frame)
                tm.assert_frame_equal(from_frame, chunk, check_dtype=False)
                pos += chunksize

    @staticmethod
    def func_b106tjue(from_frame):
        """
        Emulate the categorical casting behavior we expect from roundtripping.
        """
        for col in from_frame:
            ser = from_frame[col]
            if isinstance(ser.dtype, CategoricalDtype):
                cat = ser._values.remove_unused_categories()
                if cat.categories.dtype == object:
                    categories = pd.Index._with_infer(cat.categories._values)
                    cat = cat.set_categories(categories)
                elif cat.categories.dtype == 'string' and len(cat.categories
                    ) == 0:
                    categories = cat.categories.astype(object)
                    cat = cat.set_categories(categories)
                from_frame[col] = cat
        return from_frame

    def func_4iz5hn62(self, datapath):
        fname = datapath('io', 'data', 'stata', 'stata3_117.dta')
        parsed = read_stata(fname)
        with read_stata(fname, iterator=True) as itr:
            chunk = itr.read(5)
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)
        with read_stata(fname, chunksize=5) as itr:
            chunk = list(itr)
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk[0])
        with read_stata(fname, iterator=True) as itr:
            chunk = itr.get_chunk(5)
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)
        with read_stata(fname, chunksize=5) as itr:
            chunk = itr.get_chunk()
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)
        with read_stata(fname, chunksize=4) as itr:
            from_chunks = pd.concat(itr)
        tm.assert_frame_equal(parsed, from_chunks)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('file', ['stata2_115', 'stata3_115',
        'stata4_115', 'stata5_115', 'stata6_115', 'stata7_115',
        'stata8_115', 'stata9_115', 'stata10_115', 'stata11_115'])
    @pytest.mark.parametrize('chunksize', [1, 2])
    @pytest.mark.parametrize('convert_categoricals', [False, True])
    @pytest.mark.parametrize('convert_dates', [False, True])
    def func_78lwftgv(self, file, chunksize, convert_categoricals,
        convert_dates, datapath):
        fname = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = read_stata(fname, convert_categoricals=
            convert_categoricals, convert_dates=convert_dates)
        with read_stata(fname, iterator=True, convert_dates=convert_dates,
            convert_categoricals=convert_categoricals) as itr:
            pos = 0
            for j in range(5):
                try:
                    chunk = itr.read(chunksize)
                except StopIteration:
                    break
                from_frame = parsed.iloc[pos:pos + chunksize, :].copy()
                from_frame = self._convert_categorical(from_frame)
                tm.assert_frame_equal(from_frame, chunk, check_dtype=False)
                pos += chunksize

    def func_g8gzcfou(self, datapath):
        fname = datapath('io', 'data', 'stata', 'stata3_117.dta')
        columns = ['quarter', 'cpi', 'm1']
        chunksize = 2
        parsed = read_stata(fname, columns=columns)
        with read_stata(fname, iterator=True) as itr:
            pos = 0
            for j in range(5):
                chunk = itr.read(chunksize, columns=columns)
                if chunk is None:
                    break
                from_frame = parsed.iloc[pos:pos + chunksize, :]
                tm.assert_frame_equal(from_frame, chunk, check_dtype=False)
                pos += chunksize

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_5d7dc35f(self, version, mixed_frame, temp_file):
        mixed_frame.index.name = 'index'
        variable_labels = {'a': 'City Rank', 'b': 'City Exponent', 'c': 'City'}
        path = temp_file
        func_gr71xg09.to_stata(path, variable_labels=variable_labels,
            version=version)
        with StataReader(path) as sr:
            read_labels = sr.variable_labels()
        expected_labels = {'index': '', 'a': 'City Rank', 'b':
            'City Exponent', 'c': 'City'}
        assert read_labels == expected_labels
        variable_labels['index'] = 'The Index'
        path = temp_file
        func_gr71xg09.to_stata(path, variable_labels=variable_labels,
            version=version)
        with StataReader(path) as sr:
            read_labels = sr.variable_labels()
        assert read_labels == variable_labels

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_feanqcxa(self, version, mixed_frame, temp_file):
        mixed_frame.index.name = 'index'
        variable_labels = {'a': 'very long' * 10, 'b': 'City Exponent', 'c':
            'City'}
        path = temp_file
        msg = 'Variable labels must be 80 characters or fewer'
        with pytest.raises(ValueError, match=msg):
            func_gr71xg09.to_stata(path, variable_labels=variable_labels,
                version=version)

    @pytest.mark.parametrize('version', [114, 117])
    def func_aom6y664(self, version, mixed_frame, temp_file):
        mixed_frame.index.name = 'index'
        variable_labels = {'a': 'very long' * 10, 'b': 'City Exponent', 'c':
            'City'}
        variable_labels['a'] = 'invalid character Œ'
        path = temp_file
        with pytest.raises(ValueError, match=
            'Variable labels must contain only characters'):
            func_gr71xg09.to_stata(path, variable_labels=variable_labels,
                version=version)

    def func_1gcao5qj(self, mixed_frame, temp_file):
        values = ['Ρ', 'Α', 'Ν', 'Δ', 'Α', 'Σ']
        variable_labels_utf8 = {'a': 'City Rank', 'b': 'City Exponent', 'c':
            ''.join(values)}
        msg = (
            'Variable labels must contain only characters that can be encoded in Latin-1'
            )
        with pytest.raises(ValueError, match=msg):
            path = temp_file
            func_gr71xg09.to_stata(path, variable_labels=variable_labels_utf8)
        variable_labels_long = {'a': 'City Rank', 'b': 'City Exponent', 'c':
            'A very, very, very long variable label that is too long for Stata which means that it has more than 80 characters'
            }
        msg = 'Variable labels must be 80 characters or fewer'
        with pytest.raises(ValueError, match=msg):
            path = temp_file
            func_gr71xg09.to_stata(path, variable_labels=variable_labels_long)

    def func_v8pixc9g(self, temp_file):
        dates = [dt.datetime(1999, 12, 31, 12, 12, 12, 12000), dt.datetime(
            2012, 12, 21, 12, 21, 12, 21000), dt.datetime(1776, 7, 4, 7, 4,
            7, 4000)]
        original = DataFrame({'nums': [1.0, 2.0, 3.0], 'strs': ['apple',
            'banana', 'cherry'], 'dates': dates})
        expected = original[:]
        expected['dates'] = expected['dates'].astype('M8[ms]')
        path = temp_file
        original.to_stata(path, write_index=False)
        reread = read_stata(path, convert_dates=True)
        tm.assert_frame_equal(expected, reread)
        original.to_stata(path, write_index=False, convert_dates={'dates':
            'tc'})
        direct = read_stata(path, convert_dates=True)
        tm.assert_frame_equal(reread, direct)
        dates_idx = original.columns.tolist().index('dates')
        original.to_stata(path, write_index=False, convert_dates={dates_idx:
            'tc'})
        direct = read_stata(path, convert_dates=True)
        tm.assert_frame_equal(reread, direct)

    def func_2m3juvey(self, temp_file):
        original = DataFrame({'a': [1 + 2.0j, 2 + 4.0j]})
        msg = 'Data type complex128 not supported'
        with pytest.raises(NotImplementedError, match=msg):
            path = temp_file
            original.to_stata(path)

    def func_smlk26vx(self, temp_file):
        dates = [dt.datetime(1999, 12, 31, 12, 12, 12, 12000), dt.datetime(
            2012, 12, 21, 12, 21, 12, 21000), dt.datetime(1776, 7, 4, 7, 4,
            7, 4000)]
        original = DataFrame({'nums': [1.0, 2.0, 3.0], 'strs': ['apple',
            'banana', 'cherry'], 'dates': dates})
        msg = 'Format %tC not implemented'
        with pytest.raises(NotImplementedError, match=msg):
            path = temp_file
            original.to_stata(path, convert_dates={'dates': 'tC'})
        dates = pd.date_range('1-1-1990', periods=3, tz='Asia/Hong_Kong')
        original = DataFrame({'nums': [1.0, 2.0, 3.0], 'strs': ['apple',
            'banana', 'cherry'], 'dates': dates})
        with pytest.raises(NotImplementedError, match='Data type datetime64'):
            path = temp_file
            original.to_stata(path)

    def func_d6kplmyo(self, datapath):
        msg = """
Value labels for column ethnicsn are not unique. These cannot be converted to
pandas categoricals.

Either read the file with `convert_categoricals` set to False or use the
low level interface in `StataReader` to separately read the values and the
value_labels.

The repeated labels are:
-+
wolof
"""
        with pytest.raises(ValueError, match=msg):
            read_stata(datapath('io', 'data', 'stata', 'stata15.dta'),
                convert_categoricals=True)

    def func_hdmgpvdk(self, datapath):
        df = read_stata(datapath('io', 'data', 'stata', 'stata7_111.dta'))
        original = DataFrame({'y': [1, 1, 1, 1, 1, 0, 0, np.nan, 0, 0], 'x':
            [1, 2, 1, 3, np.nan, 4, 3, 5, 1, 6], 'w': [2, np.nan, 5, 2, 4, 
            4, 3, 1, 2, 3], 'z': ['a', 'b', 'c', 'd', 'e', '', 'g', 'h',
            'i', 'j']})
        original = original[['y', 'x', 'w', 'z']]
        tm.assert_frame_equal(original, df)

    def func_xdkpn4a6(self, temp_file):
        df = DataFrame({'ColumnOk': [0.0, np.finfo(np.double).eps, 
            4.49423283715579e+307], 'ColumnTooBig': [0.0, np.finfo(np.
            double).eps, np.finfo(np.double).max]})
        msg = (
            'Column ColumnTooBig has a maximum value \\(.+\\) outside the range supported by Stata \\(.+\\)'
            )
        with pytest.raises(ValueError, match=msg):
            path = temp_file
            df.to_stata(path)

    def func_g27yuulf(self, temp_file):
        original = DataFrame({'ColumnOk': [0.0, np.finfo(np.float32).eps, 
            np.finfo(np.float32).max / 10.0], 'ColumnTooBig': [0.0, np.
            finfo(np.float32).eps, np.finfo(np.float32).max]})
        original.index.name = 'index'
        for col in original:
            original[col] = original[col].astype(np.float32)
        path = temp_file
        original.to_stata(path)
        reread = read_stata(path)
        original['ColumnTooBig'] = original['ColumnTooBig'].astype(np.float64)
        expected = original
        tm.assert_frame_equal(reread.set_index('index'), expected)

    @pytest.mark.parametrize('infval', [np.inf, -np.inf])
    def func_df8j227q(self, infval, temp_file):
        df = DataFrame({'WithoutInf': [0.0, 1.0], 'WithInf': [2.0, infval]})
        msg = (
            'Column WithInf contains infinity or -infinitywhich is outside the range supported by Stata.'
            )
        with pytest.raises(ValueError, match=msg):
            path = temp_file
            df.to_stata(path)

    def func_1dvo3n2j(self):
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.
            Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
        df.index.name = 'index'
        reader = lambda x: read_stata(x).set_index('index')
        result = tm.round_trip_pathlib(df.to_stata, reader)
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize('write_index', [True, False])
    def func_cp2uhjrs(self, write_index, temp_file):
        d = {'A': ['B', 'E', 'C', 'A', 'E']}
        df = DataFrame(data=d)
        df['A'] = df['A'].astype('category')
        path = temp_file
        df.to_stata(path, write_index=write_index)
        with read_stata(path, iterator=True) as dta_iter:
            value_labels = dta_iter.value_labels()
        assert value_labels == {'A': {(0): 'A', (1): 'B', (2): 'C', (3): 'E'}}

    def func_bzs23qap(self, temp_file):
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.
            Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
        df.index.name = 'index'
        path = temp_file
        df.to_stata(path)
        reread = read_stata(path, index_col='index')
        tm.assert_frame_equal(df, reread)

    @pytest.mark.parametrize('column', ['ms', 'day', 'week', 'month', 'qtr',
        'half', 'yr'])
    def func_epa2g90u(self, column, datapath):
        df = read_stata(datapath('io', 'data', 'stata', 'stata13_dates.dta'))
        unformatted = df.loc[0, column]
        formatted = df.loc[0, column + '_fmt']
        assert unformatted == formatted

    @pytest.mark.parametrize('byteorder', ['little', 'big'])
    def func_9bstwxpi(self, byteorder, temp_file, using_infer_string):
        original = DataFrame(data=[['string', 'object', 1, 1, 1, 1.1, 1.1,
            np.datetime64('2003-12-25'), 'a', 'a' * 2045, 'a' * 5000, 'a'],
            ['string-1', 'object-1', 1, 1, 1, 1.1, 1.1, np.datetime64(
            '2003-12-26'), 'b', 'b' * 2045, '', '']], columns=['string',
            'object', 'int8', 'int16', 'int32', 'float32', 'float64',
            'datetime', 's1', 's2045', 'srtl', 'forced_strl'])
        original['object'] = Series(original['object'], dtype=object)
        original['int8'] = Series(original['int8'], dtype=np.int8)
        original['int16'] = Series(original['int16'], dtype=np.int16)
        original['int32'] = original['int32'].astype(np.int32)
        original['float32'] = Series(original['float32'], dtype=np.float32)
        original.index.name = 'index'
        copy = original.copy()
        path = temp_file
        original.to_stata(path, convert_dates={'datetime': 'tc'}, byteorder
            =byteorder, convert_strl=['forced_strl'], version=117)
        written_and_read_again = self.read_dta(path)
        expected = original[:]
        expected['datetime'] = expected['datetime'].astype('M8[ms]')
        if using_infer_string:
            expected['object'] = expected['object'].astype('str')
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)
        tm.assert_frame_equal(original, copy)

    def func_8xpa4mjs(self, temp_file):
        original = DataFrame([['a' * 3000, 'A', 'apple'], ['b' * 1000, 'B',
            'banana']], columns=['long1' * 10, 'long', 1])
        original.index.name = 'index'
        msg = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            path = temp_file
            original.to_stata(path, convert_strl=['long', 1], version=117)
            reread = self.read_dta(path)
            reread = reread.set_index('index')
            reread.columns = original.columns
            tm.assert_frame_equal(reread, original, check_index_type=False)

    def func_l0m4bwzz(self, temp_file):
        dates = [dt.datetime(1999, 12, 31, 12, 12, 12, 12000), dt.datetime(
            2012, 12, 21, 12, 21, 12, 21000), dt.datetime(1776, 7, 4, 7, 4,
            7, 4000)]
        original = DataFrame({'nums': [1.0, 2.0, 3.0], 'strs': ['apple',
            'banana', 'cherry'], 'dates': dates})
        path = temp_file
        msg = 'convert_dates key must be a column or an integer'
        with pytest.raises(ValueError, match=msg):
            original.to_stata(path, convert_dates={'wrong_name': 'tc'})

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_5h4fl59f(self, version, temp_file):
        bio = io.BytesIO()
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.
            Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
        df.index.name = 'index'
        path = temp_file
        df.to_stata(bio, version=version)
        bio.seek(0)
        with open(path, 'wb') as dta:
            dta.write(bio.read())
        reread = read_stata(path, index_col='index')
        tm.assert_frame_equal(df, reread)

    def func_nkar8cit(self, temp_file):
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.
            Index(list('ABCD')), index=pd.Index([f'i-{i}' for i in range(30)]))
        df.index.name = 'index'
        path = temp_file
        with gzip.GzipFile(path, 'wb') as gz:
            df.to_stata(gz, version=114)
        with gzip.GzipFile(path, 'rb') as gz:
            reread = read_stata(gz, index_col='index')
        tm.assert_frame_equal(df, reread)

    @pytest.mark.parametrize('file', ['stata16_118', 'stata16_be_118',
        'stata16_119', 'stata16_be_119'])
    def func_hfyeovfv(self, file, datapath):
        unicode_df = self.read_dta(datapath('io', 'data', 'stata',
            f'{file}.dta'))
        columns = ['utf8', 'latin1', 'ascii', 'utf8_strl', 'ascii_strl']
        values = [['ραηδας', 'PÄNDÄS', 'p', 'ραηδας', 'p'], ['ƤĀńĐąŜ', 'Ö',
            'a', 'ƤĀńĐąŜ', 'a'], ['ᴘᴀᴎᴅᴀS', 'Ü', 'n', 'ᴘᴀᴎᴅᴀS', 'n'], [
            '      ', '      ', 'd', '      ', 'd'], [' ', '', 'a', ' ',
            'a'], ['', '', 's', '', 's'], ['', '', ' ', '', ' ']]
        expected = DataFrame(values, columns=columns)
        tm.assert_frame_equal(unicode_df, expected)

    def func_prxz30nr(self, temp_file, using_infer_string):
        output = [{'mixed': 'string' * 500, 'number': 0}, {'mixed': None,
            'number': 1}]
        output = DataFrame(output)
        output.number = output.number.astype('int32')
        path = temp_file
        output.to_stata(path, write_index=False, version=117)
        reread = read_stata(path)
        expected = output.fillna('')
        tm.assert_frame_equal(reread, expected)
        output['mixed'] = None
        output.to_stata(path, write_index=False, convert_strl=['mixed'],
            version=117)
        reread = read_stata(path)
        expected = output.fillna('')
        if using_infer_string:
            expected['mixed'] = expected['mixed'].astype('str')
        tm.assert_frame_equal(reread, expected)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_1iwdj7o4(self, version, temp_file):
        output = [{'none': 'none', 'number': 0}, {'none': None, 'number': 1}]
        output = DataFrame(output)
        output['none'] = None
        with pytest.raises(ValueError, match='Column `none` cannot be exported'
            ):
            output.to_stata(temp_file, version=version)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_h6itf2qh(self, version, temp_file):
        content = 'Here is one __�__ Another one __·__ Another one __½__'
        df = DataFrame([content], columns=['invalid'])
        msg1 = (
            "'latin-1' codec can't encode character '\\\\ufffd' in position 14: ordinal not in range\\(256\\)"
            )
        msg2 = (
            "'ascii' codec can't decode byte 0xef in position 14: ordinal not in range\\(128\\)"
            )
        with pytest.raises(UnicodeEncodeError, match=f'{msg1}|{msg2}'):
            df.to_stata(temp_file)

    def func_9pimtmz6(self, temp_file):
        output = DataFrame([['pandas'] * 2, ['þâÑÐÅ§'] * 2], columns=[
            'var_str', 'var_strl'])
        output.to_stata(temp_file, version=117, convert_strl=['var_strl'])
        with open(temp_file, 'rb') as reread:
            content = reread.read()
            expected = 'þâÑÐÅ§'
            assert expected.encode('latin-1') in content
            assert expected.encode('utf-8') in content
            gsos = content.split(b'strls')[1][1:-2]
            for gso in gsos.split(b'GSO')[1:]:
                val = gso.split(b'\x00')[-2]
                size = gso[gso.find(b'\x82') + 1]
                assert len(val) == size - 1

    def func_98hz7yw5(self, datapath):
        msg = """
One or more strings in the dta file could not be decoded using utf-8, and
so the fallback encoding of latin-1 is being used.  This can happen when a file
has been incorrectly encoded by Stata or some other software. You should verify
the string values returned are correct."""
        path = datapath('io', 'data', 'stata', 'stata1_encoding_118.dta')
        with tm.assert_produces_warning(UnicodeWarning, filter_level='once'
            ) as w:
            encoded = read_stata(path)
            assert len(w) == 1
            assert w[0].message.args[0] == msg
        expected = DataFrame([['Düsseldorf']] * 151, columns=['kreis1849'])
        tm.assert_frame_equal(encoded, expected)

    @pytest.mark.slow
    def func_iqrn0r5k(self, datapath):
        with gzip.open(datapath('io', 'data', 'stata', 'stata1_119.dta.gz'),
            'rb') as gz:
            with StataReader(gz) as reader:
                reader._ensure_open()
                assert reader._nvar == 32999

    @pytest.mark.parametrize('version', [118, 119, None])
    @pytest.mark.parametrize('byteorder', ['little', 'big'])
    def func_m15x5moc(self, version, byteorder, temp_file):
        cat = pd.Categorical(['a', 'β', 'ĉ'], ordered=True)
        data = DataFrame([[1.0, 1, 'ᴬ', 'ᴀ relatively long ŝtring'], [2.0, 
            2, 'ᴮ', ''], [3.0, 3, 'ᴰ', None]], columns=['Å', 'β', 'ĉ', 'strls']
            )
        data['ᴐᴬᵀ'] = cat
        variable_labels = {'Å': 'apple', 'β': 'ᵈᵉᵊ', 'ĉ': 'ᴎტჄႲႳႴႶႺ',
            'strls': 'Long Strings', 'ᴐᴬᵀ': ''}
        data_label = 'ᴅaᵀa-label'
        value_labels = {'β': {(1): 'label', (2): 'æøå', (3):
            'ŋot valid latin-1'}}
        data['β'] = data['β'].astype(np.int32)
        writer = StataWriterUTF8(temp_file, data, data_label=data_label,
            convert_strl=['strls'], variable_labels=variable_labels,
            write_index=False, byteorder=byteorder, version=version,
            value_labels=value_labels)
        writer.write_file()
        reread_encoded = read_stata(temp_file)
        data['strls'] = data['strls'].fillna('')
        data['β'] = data['β'].replace(value_labels['β']).astype('category'
            ).cat.as_ordered()
        tm.assert_frame_equal(data, reread_encoded)
        with StataReader(temp_file) as reader:
            assert reader.data_label == data_label
            assert reader.variable_labels() == variable_labels
        data.to_stata(temp_file, version=version, write_index=False)
        reread_to_stata = read_stata(temp_file)
        tm.assert_frame_equal(data, reread_to_stata)

    def func_keel45v2(self, temp_file):
        df = DataFrame(np.zeros((1, 33000), dtype=np.int8))
        with pytest.raises(ValueError, match=
            'version must be either 118 or 119.'):
            StataWriterUTF8(temp_file, df, version=117)
        with pytest.raises(ValueError, match='You must use version 119'):
            StataWriterUTF8(temp_file, df, version=118)

    @pytest.mark.parametrize('dtype_backend', ['numpy_nullable', pytest.
        param('pyarrow', marks=td.skip_if_no('pyarrow'))])
    def func_pypsh4n0(self, dtype_backend, temp_file, tmp_path):
        df = DataFrame({'a': [1, 2, None], 'b': ['a', 'b', 'c'], 'c': [True,
            False, None], 'd': [1.5, 2.5, 3.5], 'e': pd.date_range(
            '2020-12-31', periods=3, freq='D')}, index=pd.Index([0, 1, 2],
            name='index'))
        df = df.convert_dtypes(dtype_backend=dtype_backend)
        stata_path = tmp_path / 'test_stata.dta'
        df.to_stata(stata_path, version=118)
        df.to_stata(temp_file)
        written_and_read_again = self.read_dta(temp_file)
        expected = DataFrame({'a': [1, 2, np.nan], 'b': ['a', 'b', 'c'],
            'c': [1.0, 0, np.nan], 'd': [1.5, 2.5, 3.5], 'e': pd.date_range
            ('2020-12-31', periods=3, freq='D', unit='ms')}, index=pd.
            RangeIndex(range(3), name='index'))
        tm.assert_frame_equal(written_and_read_again.set_index('index'),
            expected)

    @pytest.mark.parametrize('version', [113, 114, 115, 117, 118, 119])
    def func_4ddc6x81(self, version, datapath):
        expected = DataFrame({'byte': np.array([-127, 100], dtype=np.int8),
            'int': np.array([-32767, 32740], dtype=np.int16), 'long': np.
            array([-2147483647, 2147483620], dtype=np.int32)})
        parsed = read_stata(datapath('io', 'data', 'stata',
            f'stata_int_validranges_{version}.dta'))
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [104, 105, 108, 110, 111])
    def func_fehpabjm(self, version, datapath):
        expected = DataFrame({'byte': np.array([-128, 126], dtype=np.int8),
            'int': np.array([-32768, 32766], dtype=np.int16), 'long': np.
            array([-2147483648, 2147483646], dtype=np.int32)})
        parsed = read_stata(datapath('io', 'data', 'stata',
            f'stata_int_validranges_{version}.dta'))
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [102, 103])
    def func_u97i6tlv(self, version, datapath):
        expected = DataFrame({'byte': np.array([-128, 126], dtype=np.int16),
            'int': np.array([-32768, 32766], dtype=np.int16), 'long': np.
            array([-2147483648, 2147483646], dtype=np.int32)})
        parsed = read_stata(datapath('io', 'data', 'stata',
            f'stata_int_validranges_{version}.dta'))
        tm.assert_frame_equal(parsed, expected)


@pytest.mark.parametrize('version', [105, 108, 110, 111, 113, 114])
def func_fjx32lzb(version, datapath):
    data_base = datapath('io', 'data', 'stata')
    ref = os.path.join(data_base, 'stata-compat-118.dta')
    old = os.path.join(data_base, f'stata-compat-{version}.dta')
    expected = read_stata(ref)
    old_dta = read_stata(old)
    tm.assert_frame_equal(old_dta, expected, check_dtype=False)


@pytest.mark.parametrize('version', [103, 104])
def func_h0h1u1be(version, datapath):
    data_base = datapath('io', 'data', 'stata')
    ref = os.path.join(data_base, 'stata-compat-118.dta')
    old = os.path.join(data_base, f'stata-compat-{version}.dta')
    expected = read_stata(ref, convert_dates=False)
    old_dta = read_stata(old, convert_dates=False)
    tm.assert_frame_equal(old_dta, expected, check_dtype=False)


@pytest.mark.parametrize('version', [102])
def func_aib381z9(version, datapath):
    ref = datapath('io', 'data', 'stata', 'stata-compat-118.dta')
    old = datapath('io', 'data', 'stata', f'stata-compat-{version}.dta')
    expected = read_stata(ref, convert_dates=False)
    expected = expected.drop(columns=['s10'])
    old_dta = read_stata(old, convert_dates=False)
    tm.assert_frame_equal(old_dta, expected, check_dtype=False)


@pytest.mark.parametrize('version', [105, 108, 110, 111, 113, 114, 118])
def func_u3ihnlgj(version, datapath):
    ref = datapath('io', 'data', 'stata', f'stata-compat-{version}.dta')
    big = datapath('io', 'data', 'stata', f'stata-compat-be-{version}.dta')
    expected = read_stata(ref)
    big_dta = read_stata(big)
    tm.assert_frame_equal(big_dta, expected)


@pytest.mark.parametrize('version', [103, 104])
def func_iofdc6us(version, datapath):
    ref = datapath('io', 'data', 'stata', f'stata-compat-{version}.dta')
    big = datapath('io', 'data', 'stata', f'stata-compat-be-{version}.dta')
    expected = read_stata(ref, convert_dates=False)
    big_dta = read_stata(big, convert_dates=False)
    tm.assert_frame_equal(big_dta, expected)


def func_wcrq7zoy(datapath, monkeypatch):
    file_path = datapath('io', 'data', 'stata', 'stata-compat-118.dta')
    with StataReader(file_path) as reader:
        assert not reader.read().empty
        assert not isinstance(reader._path_or_buf, io.BytesIO)
    with open(file_path, 'rb') as fp:
        with StataReader(fp) as reader:
            assert not reader.read().empty
            assert reader._path_or_buf is fp
    with open(file_path, 'rb') as fp:
        with io.BytesIO(fp.read()) as bio:
            with StataReader(bio) as reader:
                assert not reader.read().empty
                assert reader._path_or_buf is bio


@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
@pytest.mark.parametrize('use_dict', [True, False])
@pytest.mark.parametrize('infer', [True, False])
def func_f7228i19(compression, version, use_dict, infer,
    compression_to_extension, tmp_path):
    file_name = 'dta_inferred_compression.dta'
    if compression:
        if use_dict:
            file_ext = compression
        else:
            file_ext = compression_to_extension[compression]
        file_name += f'.{file_ext}'
    compression_arg = compression
    if infer:
        compression_arg = 'infer'
    if use_dict:
        compression_arg = {'method': compression}
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)),
        columns=list('AB'))
    df.index.name = 'index'
    path = tmp_path / file_name
    path.touch()
    df.to_stata(path, version=version, compression=compression_arg)
    if compression == 'gzip':
        with gzip.open(path, 'rb') as comp:
            fp = io.BytesIO(comp.read())
    elif compression == 'zip':
        with zipfile.ZipFile(path, 'r') as comp:
            fp = io.BytesIO(comp.read(comp.filelist[0]))
    elif compression == 'tar':
        with tarfile.open(path) as tar:
            fp = io.BytesIO(tar.extractfile(tar.getnames()[0]).read())
    elif compression == 'bz2':
        with bz2.open(path, 'rb') as comp:
            fp = io.BytesIO(comp.read())
    elif compression == 'zstd':
        zstd = pytest.importorskip('zstandard')
        with zstd.open(path, 'rb') as comp:
            fp = io.BytesIO(comp.read())
    elif compression == 'xz':
        lzma = pytest.importorskip('lzma')
        with lzma.open(path, 'rb') as comp:
            fp = io.BytesIO(comp.read())
    elif compression is None:
        fp = path
    reread = read_stata(fp, index_col='index')
    expected = df
    tm.assert_frame_equal(reread, expected)


@pytest.mark.parametrize('method', ['zip', 'infer'])
@pytest.mark.parametrize('file_ext', [None, 'dta', 'zip'])
def func_gf6yov6j(method, file_ext, tmp_path):
    file_name = f'test.{file_ext}'
    archive_name = 'test.dta'
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)),
        columns=list('AB'))
    df.index.name = 'index'
    compression = {'method': method, 'archive_name': archive_name}
    path = tmp_path / file_name
    path.touch()
    df.to_stata(path, compression=compression)
    if method == 'zip' or file_ext == 'zip':
        with zipfile.ZipFile(path, 'r') as zp:
            assert len(zp.filelist) == 1
            assert zp.filelist[0].filename == archive_name
            fp = io.BytesIO(zp.read(zp.filelist[0]))
    else:
        fp = path
    reread = read_stata(fp, index_col='index')
    expected = df
    tm.assert_frame_equal(reread, expected)


@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
def func_243kgcbb(version, temp_file):
    df = DataFrame({'cats': Series(['a', 'b', 'a', 'b', 'c'], dtype=
        'category')})
    df.index.name = 'index'
    expected = df.copy()
    df.to_stata(temp_file, version=version)
    with StataReader(temp_file, chunksize=2, order_categoricals=False
        ) as reader:
        for i, block in enumerate(reader):
            block = block.set_index('index')
            assert 'cats' in block
            tm.assert_series_equal(block.cats, expected.cats.iloc[2 * i:2 *
                (i + 1)], check_index_type=len(block) > 1)


def func_bsg8879a(datapath):
    dta_file = datapath('io', 'data', 'stata',
        'stata-dta-partially-labeled.dta')
    values = ['a', 'b', 'a', 'b', 3.0]
    msg = 'series with value labels are not fully labeled'
    with StataReader(dta_file, chunksize=2) as reader:
        with tm.assert_produces_warning(CategoricalConversionWarning, match=msg
            ):
            for i, block in enumerate(reader):
                assert list(block.cats) == values[2 * i:2 * (i + 1)]
                if i < 2:
                    idx = pd.Index(['a', 'b'])
                else:
                    idx = pd.Index([3.0], dtype='float64')
                tm.assert_index_equal(block.cats.cat.categories, idx)
    with tm.assert_produces_warning(CategoricalConversionWarning, match=msg):
        with StataReader(dta_file, chunksize=5) as reader:
            large_chunk = reader.__next__()
    direct = read_stata(dta_file)
    tm.assert_frame_equal(direct, large_chunk)


@pytest.mark.parametrize('chunksize', (-1, 0, 'apple'))
def func_l90a5hgc(datapath, chunksize):
    dta_file = datapath('io', 'data', 'stata',
        'stata-dta-partially-labeled.dta')
    with pytest.raises(ValueError, match='chunksize must be a positive'):
        with StataReader(dta_file, chunksize=chunksize):
            pass


def func_zqfpskod(temp_file):
    values = ['c_label', 'b_label'] + ['a_label'] * 500
    df = DataFrame({f'col{k}': pd.Categorical(values, ordered=True) for k in
        range(2)})
    df.to_stata(temp_file, write_index=False)
    expected = pd.Index(['a_label', 'b_label', 'c_label'])
    with read_stata(temp_file, chunksize=100) as reader:
        for j, chunk in enumerate(reader):
            for i in range(2):
                tm.assert_index_equal(chunk.dtypes.iloc[i].categories, expected
                    )
            tm.assert_frame_equal(chunk, df.iloc[j * 100:(j + 1) * 100])


def func_djztgt0v(temp_file):
    df = DataFrame([[sum(2 ** i for i in range(60)), sum(2 ** i for i in
        range(52))]], columns=['big', 'little'])
    with tm.assert_produces_warning(PossiblePrecisionLoss, match=
        'Column converted from int64 to float64'):
        df.to_stata(temp_file, write_index=False)
    reread = read_stata(temp_file)
    expected_dt = Series([np.float64, np.float64], index=['big', 'little'])
    tm.assert_series_equal(reread.dtypes, expected_dt)
    assert reread.loc[0, 'little'] == df.loc[0, 'little']
    assert reread.loc[0, 'big'] == float(df.loc[0, 'big'])


def func_zo929boq(compression, temp_file):
    df = DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 
        321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
    df.index.name = 'index'
    df.to_stata(temp_file, compression=compression)
    reread = read_stata(temp_file, compression=compression, index_col='index')
    tm.assert_frame_equal(df, reread)
    with tm.decompress_file(temp_file, compression) as fh:
        contents = io.BytesIO(fh.read())
    reread = read_stata(contents, index_col='index')
    tm.assert_frame_equal(df, reread)


@pytest.mark.parametrize('to_infer', [True, False])
@pytest.mark.parametrize('read_infer', [True, False])
def func_zl7wxwd1(compression_only, read_infer, to_infer,
    compression_to_extension, tmp_path):
    compression = compression_only
    ext = compression_to_extension[compression]
    filename = f'test.{ext}'
    df = DataFrame([[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 
        321321.2]], index=['A', 'B'], columns=['X', 'Y', 'Z'])
    df.index.name = 'index'
    to_compression = 'infer' if to_infer else compression
    read_compression = 'infer' if read_infer else compression
    path = tmp_path / filename
    path.touch()
    df.to_stata(path, compression=to_compression)
    result = read_stata(path, compression=read_compression, index_col='index')
    tm.assert_frame_equal(result, df)


def func_c5p2qo2l(temp_file):
    data = DataFrame({'fully_labelled': [1, 2, 3, 3, 1],
        'partially_labelled': [1.0, 2.0, np.nan, 9.0, np.nan], 'Y': [7, 7, 
        9, 8, 10], 'Z': pd.Categorical(['j', 'k', 'l', 'k', 'j'])})
    path = temp_file
    value_labels = {'fully_labelled': {(1): 'one', (2): 'two', (3): 'three'
        }, 'partially_labelled': {(1.0): 'one', (2.0): 'two'}}
    expected = {**value_labels, 'Z': {(0): 'j', (1): 'k', (2): 'l'}}
    writer = StataWriter(path, data, value_labels=value_labels)
    writer.write_file()
    with StataReader(path) as reader:
        reader_value_labels = reader.value_labels()
        assert reader_value_labels == expected
    msg = "Can't create value labels for notY, it wasn't found in the dataset."
    value_labels = {'notY': {(7): 'label1', (8): 'label2'}}
    with pytest.raises(KeyError, match=msg):
        StataWriter(path, data, value_labels=value_labels)
    msg = (
        "Can't create value labels for Z, value labels can only be applied to numeric columns."
        )
    value_labels = {'Z': {(1): 'a', (2): 'k', (3): 'j', (4): 'i'}}
    with pytest.raises(ValueError, match=msg):
        StataWriter(path, data, value_labels=value_labels)


def func_r0s8g4w1(temp_file):
    data = DataFrame({'invalid~!': [1, 1, 2, 3, 5, 8], '6_invalid': [1, 1, 
        2, 3, 5, 8], 'invalid_name_longer_than_32_characters': [8, 8, 9, 9,
        8, 8], 'aggregate': [2, 5, 5, 6, 6, 9], (1, 2): [1, 2, 3, 4, 5, 6]})
    value_labels = {'invalid~!': {(1): 'label1', (2): 'label2'},
        '6_invalid': {(1): 'label1', (2): 'label2'},
        'invalid_name_longer_than_32_characters': {(8): 'eight', (9):
        'nine'}, 'aggregate': {(5): 'five'}, (1, 2): {(3): 'three'}}
    expected = {'invalid__': {(1): 'label1', (2): 'label2'}, '_6_invalid':
        {(1): 'label1', (2): 'label2'}, 'invalid_name_longer_than_32_char':
        {(8): 'eight', (9): 'nine'}, '_aggregate': {(5): 'five'}, '_1__2_':
        {(3): 'three'}}
    msg = 'Not all pandas column names were valid Stata variable names'
    with tm.assert_produces_warning(InvalidColumnName, match=msg):
        data.to_stata(temp_file, value_labels=value_labels)
    with StataReader(temp_file) as reader:
        reader_value_labels = reader.value_labels()
        assert reader_value_labels == expected


def func_cr2881w9(temp_file):
    value_labels = {'repeated_labels': {(10): 'Ten', (20): 'More than ten',
        (40): 'More than ten'}}
    data = DataFrame({'repeated_labels': [10, 10, 20, 20, 40, 40]})
    data.to_stata(temp_file, value_labels=value_labels)
    with StataReader(temp_file, convert_categoricals=False) as reader:
        reader_value_labels = reader.value_labels()
    assert reader_value_labels == value_labels
    col = 'repeated_labels'
    repeats = '-' * 80 + '\n' + '\n'.join(['More than ten'])
    msg = f"""
Value labels for column {col} are not unique. These cannot be converted to
pandas categoricals.

Either read the file with `convert_categoricals` set to False or use the
low level interface in `StataReader` to separately read the values and the
value_labels.

The repeated labels are:
{repeats}
"""
    with pytest.raises(ValueError, match=msg):
        read_stata(temp_file, convert_categoricals=True)


@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
@pytest.mark.parametrize('dtype', [pd.BooleanDtype, pd.Int8Dtype, pd.
    Int16Dtype, pd.Int32Dtype, pd.Int64Dtype, pd.UInt8Dtype, pd.UInt16Dtype,
    pd.UInt32Dtype, pd.UInt64Dtype])
def func_1a6sp1w0(dtype, version, temp_file):
    df = DataFrame({'a': Series([1.0, 2.0, 3.0]), 'b': Series([1, pd.NA, pd
        .NA], dtype=dtype.name), 'c': Series(['a', 'b', None])})
    dtype_name = df.b.dtype.numpy_dtype.name
    dtype_name = dtype_name.replace('u', '')
    if dtype_name == 'int64':
        dtype_name = 'int32'
    elif dtype_name == 'bool':
        dtype_name = 'int8'
    value = StataMissingValue.BASE_MISSING_VALUES[dtype_name]
    smv = StataMissingValue(value)
    expected_b = Series([1, smv, smv], dtype=object, name='b')
    expected_c = Series(['a', 'b', ''], name='c')
    df.to_stata(temp_file, write_index=False, version=version)
    reread = read_stata(temp_file, convert_missing=True)
    tm.assert_series_equal(df.a, reread.a)
    tm.assert_series_equal(reread.b, expected_b)
    tm.assert_series_equal(reread.c, expected_c)


def func_kfxj4n12(temp_file):
    df = DataFrame(data={'a': range(3), 'b': [1.0, 2.0, 3.0]}).head(0)
    path = temp_file
    df.to_stata(path, write_index=False, version=117)
    df2 = read_stata(path)
    assert 'b' in df2
    dtypes = Series({'a': np.dtype('int32'), 'b': np.dtype('float64')})
    tm.assert_series_equal(df2.dtypes, dtypes)
    df3 = read_stata(path, columns=['a'])
    assert 'b' not in df3
    tm.assert_series_equal(df3.dtypes, dtypes.loc[['a']])


@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
def func_jehm85j3(temp_file, version):
    n = 65534
    df = DataFrame(np.arange(n), columns=['col'])
    lbls = [''.join(v) for v in itertools.product(*([string.ascii_letters] *
        3))]
    value_labels = {'col': {i: lbls[i] for i in range(n)}}
    df.to_stata(temp_file, value_labels=value_labels, version=version)
