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
def mixed_frame() -> DataFrame:
    return DataFrame({'a': [1, 2, 3, 4], 'b': [1.0, 3.0, 27.0, 81.0], 'c': ['Atlanta', 'Birmingham', 'Cincinnati', 'Detroit']})

@pytest.fixture
def parsed_114(datapath: str) -> DataFrame:
    dta14_114 = datapath('io', 'data', 'stata', 'stata5_114.dta')
    parsed_114 = read_stata(dta14_114, convert_dates=True)
    parsed_114.index.name = 'index'
    return parsed_114

class TestStata:

    def read_dta(self, file: str) -> pd.DataFrame:
        return read_stata(file, convert_dates=True)

    def read_csv(self, file: str) -> pd.DataFrame:
        return read_csv(file, parse_dates=True)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def test_read_empty_dta(self, version: int | None, temp_file: str) -> None:
        empty_ds = DataFrame(columns=['unit'])
        path = temp_file
        empty_ds.to_stata(path, write_index=False, version=version)
        empty_ds2 = read_stata(path)
        tm.assert_frame_equal(empty_ds, empty_ds2)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def test_read_empty_dta_with_dtypes(self, version: int | None, temp_file: str) -> None:
        empty_df_typed = DataFrame({'i8': np.array([0], dtype=np.int8), 'i16': np.array([0], dtype=np.int16), 'i32': np.array([0], dtype=np.int32), 'i64': np.array([0], dtype=np.int64), 'u8': np.array([0], dtype=np.uint8), 'u16': np.array([0], dtype=np.uint16), 'u32': np.array([0], dtype=np.uint32), 'u64': np.array([0], dtype=np.uint64), 'f32': np.array([0], dtype=np.float32), 'f64': np.array([0], dtype=np.float64)})
        path = temp_file
        empty_df_typed.to_stata(path, write_index=False, version=version)
        empty_reread = read_stata(path)
        expected = empty_df_typed
        expected['u8'] = expected['u8'].astype(np.int8)
        expected['u16'] = expected['u16'].astype(np.int16)
        expected['u32'] = expected['u32'].astype(np.int32)
        expected['u64'] = expected['u64'].astype(np.int32)
        expected['i64'] = expected['i64'].astype(np.int32)
        tm.assert_frame_equal(empty_reread, expected)
        tm.assert_series_equal(expected.dtypes, empty_reread.dtypes)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def test_read_index_col_none(self, version: int | None, temp_file: str) -> None:
        df = DataFrame({'a': range(5), 'b': ['b1', 'b2', 'b3', 'b4', 'b5']})
        path = temp_file
        df.to_stata(path, write_index=False, version=version)
        read_df = read_stata(path)
        assert isinstance(read_df.index, pd.RangeIndex)
        expected = df
        expected['a'] = expected['a'].astype(np.int32)
        tm.assert_frame_equal(read_df, expected, check_index_type=True)

    @pytest.mark.parametrize('version', [102, 103, 104, 105, 108, 110, 111, 113, 114, 115, 117, 118, 119])
    def test_read_dta1(self, version: int, datapath: str) -> None:
        file = datapath('io', 'data', 'stata', f'stata1_{version}.dta')
        parsed = self.read_dta(file)
        expected = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)], columns=['float_miss', 'double_miss', 'byte_miss', 'int_miss', 'long_miss'])
        expected['float_miss'] = expected['float_miss'].astype(np.float32)
        if version <= 108:
            expected = expected.rename(columns={'float_miss': 'f_miss', 'double_miss': 'd_miss', 'byte_miss': 'b_miss', 'int_miss': 'i_miss', 'long_miss': 'l_miss'})
        tm.assert_frame_equal(parsed, expected)

    def test_read_dta2(self, datapath: str) -> None:
        expected = DataFrame.from_records([(datetime(2006, 11, 19, 23, 13, 20), 1479596223000, datetime(2010, 1, 20), datetime(2010, 1, 8), datetime(2010, 1, 1), datetime(1974, 7, 1), datetime(2010, 1, 1), datetime(2010, 1, 1)), (datetime(1959, 12, 31, 20, 3, 20), -1479590, datetime(1953, 10, 2), datetime(1948, 6, 10), datetime(1955, 1, 1), datetime(1955, 7, 1), datetime(1955, 1, 1), datetime(2, 1, 1)), (pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT)], columns=['datetime_c', 'datetime_big_c', 'date', 'weekly_date', 'monthly_date', 'quarterly_date', 'half_yearly_date', 'yearly_date'])
        expected['datetime_c'] = expected['datetime_c'].astype('M8[ms]')
        expected['date'] = expected['date'].astype('M8[s]')
        expected['weekly_date'] = expected['weekly_date'].astype('M8[s]')
        expected['monthly_date'] = expected['monthly_date'].astype('M8[s]')
        expected['quarterly_date'] = expected['quarterly_date'].astype('M8[s]')
        expected['half_yearly_date'] = expected['half_yearly_date'].astype('M8[s]')
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

    @pytest.mark.parametrize('file', ['stata3_113', 'stata3_114', 'stata3_115', 'stata3_117'])
    def test_read_dta3(self, file: str, datapath: str) -> None:
        file = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = self.read_dta(file)
        expected = self.read_csv(datapath('io', 'data', 'stata', 'stata3.csv'))
        expected = expected.astype(np.float32)
        expected['year'] = expected['year'].astype(np.int16)
        expected['quarter'] = expected['quarter'].astype(np.int8)
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [110, 111, 113, 114, 115, 117])
    def test_read_dta4(self, version: int, datapath: str) -> None:
        file = datapath('io', 'data', 'stata', f'stata4_{version}.dta')
        parsed = self.read_dta(file)
        expected = DataFrame.from_records([['one', 'ten', 'one', 'one', 'one'], ['two', 'nine', 'two', 'two', 'two'], ['three', 'eight', 'three', 'three', 'three'], ['four', 'seven', 4, 'four', 'four'], ['five', 'six', 5, np.nan, 'five'], ['six', 'five', 6, np.nan, 'six'], ['seven', 'four', 7, np.nan, 'seven'], ['eight', 'three', 8, np.nan, 'eight'], ['nine', 'two', 9, np.nan, 'nine'], ['ten', 'one', 'ten', np.nan, 'ten']], columns=['fully_labeled', 'fully_labeled2', 'incompletely_labeled', 'labeled_with_missings', 'float_labelled'])
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
    def test_readold_dta4(self, version: int, datapath: str) -> None:
        file = datapath('io', 'data', 'stata', f'stata4_{version}.dta')
        parsed = self.read_dta(file)
        expected = DataFrame.from_records([['one', 'ten', 'one', 'one', 'one'], ['two', 'nine', 'two', 'two', 'two'], ['three', 'eight', 'three', 'three', 'three'], ['four', 'seven', 4, 'four', 'four'], ['five', 'six', 5, np.nan, 'five'], ['six', 'five', 6, np.nan, 'six'], ['seven', 'four', 7, np.nan, 'seven'], ['eight', 'three', 8, np.nan, 'eight'], ['nine', 'two', 9, np.nan, 'nine'], ['ten', 'one', 'ten', np.nan, 'ten']], columns=['fulllab', 'fulllab2', 'incmplab', 'misslab', 'floatlab'])
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

    @pytest.mark.parametrize('file', ['stata12_117', 'stata12_be_117', 'stata12_118', 'stata12_be_118', 'stata12_119', 'stata12_be_119'])
    def test_read_dta_strl(self, file: str, datapath: str) -> None:
        parsed = self.read_dta(datapath('io', 'data', 'stata', f'{file}.dta'))
        expected = DataFrame.from_records([[1, 'abc', 'abcdefghi'], [3, 'cba', 'qwertywertyqwerty'], [93, '', 'strl']], columns=['x', 'y', 'z'])
        tm.assert_frame_equal(parsed, expected, check_dtype=False)

    @pytest.mark.parametrize('file', ['stata14_118', 'stata14_be_118', 'stata14_119', 'stata14_be_119'])
    def test_read_dta118_119(self, file: str, datapath: str) -> None:
        parsed_118 = self.read_dta(datapath('io', 'data', 'stata', f'{file}.dta'))
        parsed_118['Bytes'] = parsed_118['Bytes'].astype('O')
        expected = DataFrame.from_records([['Cat', 'Bogota', 'Bogotá', 1, 1.0, 'option b Ünicode', 1.0], ['Dog', 'Boston', 'Uzunköprü', np.nan, np.nan, np.nan, np.nan], ['Plane', 'Rome', 'Tromsø', 0, 0.0, 'option a', 0.0], ['Potato', 'Tokyo', 'Elâzığ', -4, 4.0, 4, 4], ['', '', '', 0, 0.3332999, 'option a', 1 / 3.0]], columns=['Things', 'Cities', 'Unicode_Cities_Strl', 'Ints', 'Floats', 'Bytes', 'Longs'])
        expected['Floats'] = expected['Floats'].astype(np.float32)
        for col in parsed_118.columns:
            tm.assert_almost_equal(parsed_118[col], expected[col])
        with StataReader(datapath('io', 'data', 'stata', f'{file}.dta')) as rdr:
            vl = rdr.variable_labels()
            vl_expected = {'Unicode_Cities_Strl': 'Here are some strls with Ünicode chars', 'Longs': 'long data', 'Things': 'Here are some things', 'Bytes': 'byte data', 'Ints': 'int data', 'Cities': 'Here are some cities', 'Floats': 'float data'}
            tm.assert_dict_equal(vl, vl_expected)
            assert rdr.data_label == 'This is a  Ünicode data label'

    def test_read_write_dta5(self, temp_file: str) -> None:
        original = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)], columns=['float_miss', 'double_miss', 'byte_miss', 'int_miss', 'long_miss'])
        original.index.name = 'index'
        path = temp_file
        original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
        expected = original
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def test_write_dta6(self, datapath: str, temp_file: str) -> None:
        original = self.read_csv(datapath('io', 'data', 'stata', 'stata3.csv'))
        original.index.name = 'index'
        original.index = original.index.astype(np.int32)
        original['year'] = original['year'].astype(np.int32)
        original['quarter'] = original['quarter'].astype(np.int32)
        path = temp_file
        original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), original, check_index_type=False)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def test_read_write_dta10(self, version: int | None, temp_file: str, using_infer_string: bool) -> None:
        original = DataFrame(data=[['string', 'object', 1, 1.1, np.datetime64('2003-12-25')]], columns=['string', 'object', 'integer', 'floating', 'datetime'])
        original['object'] = Series(original['object'], dtype=object)
        original.index.name = 'index'
        original.index = original.index.astype(np.int32)
        original['integer'] = original['integer'].astype(np.int32)
        path = temp_file
        original.to_stata(path, convert_dates={'datetime': 'tc'}, version=version)
        written_and_read_again = self.read_dta(path)
        expected = original.copy()
        expected['datetime'] = expected['datetime'].astype('M8[ms]')
        if using_infer_string:
            expected['object'] = expected['object'].astype('str')
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def test_stata_doc_examples(self, temp_file: str) -> None:
        path = temp_file
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=list('AB'))
        df.to_stata(path)

    def test_write_preserves_original(self, temp_file: str) -> None:
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), columns=list('abcd'))
        df.loc[2, 'a':'c'] = np.nan
        df_copy = df.copy()
        path = temp_file
        df.to_stata(path, write_index=False)
        tm.assert_frame_equal(df, df_copy)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def test_encoding(self, version: int | None, datapath: str, temp_file: str) -> None:
        raw = read_stata(datapath('io', 'data', 'stata', 'stata1_encoding.dta'))
        encoded = read_stata(datapath('io', 'data', 'stata', 'stata1_encoding.dta'))
        result = encoded.kreis1849[0]
        expected = raw.kreis1849[0]
        assert result == expected
        assert isinstance(result, str)
        path = temp_file
        encoded.to_stata(path, write_index=False, version=version)
        reread_encoded = read_stata(path)
        tm.assert_frame_equal(encoded, reread_encoded)

    def test_read_write_dta11(self, temp_file: str) -> None:
        original = DataFrame([(1, 2, 3, 4)], columns=['astringwithmorethan32characters______'])
        formatted = DataFrame([(1, 2, 3, 4)], columns=['astringwithmorethan32characters_'])
        formatted.index.name = 'index'
        formatted = formatted.astype(np.int32)
        path = temp_file
        msg = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
        expected = formatted
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def test_read_write_dta12(self, version: int | None, temp_file: str) -> None:
        original = DataFrame([(1, 2, 3, 4, 5, 6)], columns=['astringwithmorethan32characters_1', 'astringwithmorethan32characters_2', '+', '-', 'short', 'delete'])
        formatted = DataFrame([(1, 2, 3, 4, 5, 6)], columns=['astringwithmorethan32characters_', '_0astringwithmorethan32character', '_', '_1_', '_short', '_delete'])
        formatted.index.name = 'index'
        formatted = formatted.astype(np.int32)
        path = temp_file
        msg = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates=None, version=version)
        written_and_read_again = self.read_dta(path)
        expected = formatted
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def test_read_write_dta13(self, temp_file: str) -> None:
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
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    @pytest.mark.parametrize('file', ['stata5_113', 'stata5_114', 'stata5_115', 'stata5_117'])
    def test_read_write_reread_dta14(self, file: str, parsed_114: DataFrame, version: int | None, datapath: str, temp_file: str) -> None:
        file = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = self.read_dta(file)
        parsed.index.name = 'index'
        tm.assert_frame_equal(parsed_114, parsed)
        path = temp_file
        parsed_114.to_stata(path, convert_dates={'date_td': 'td'}, version=version)
        written_and_read_again = self.read_dta(path)
        expected = parsed_114.copy()
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    @pytest.mark.parametrize('file', ['stata6_113', 'stata6_114', 'stata6_115', 'stata6_117'])
    def test_read_write_reread_dta15(self, file: str, datapath: str) -> None:
        expected = self.read_csv(datapath('io', 'data', 'stata', 'stata6.csv'))
        expected['byte_'] = expected['byte_'].astype(np.int8)
        expected['int_'] = expected['int_'].astype(np.int16)
        expected['long_'] = expected['long_'].astype(np.int32)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['double_'] = expected['double_'].astype(np.float64)
        arr = expected['date_td'].astype('Period[D]')._values.asfreq('s', how='S')
        expected['date_td'] = arr.view('M8[s]')
        file = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed = self.read_dta(file)
        tm.assert_frame_equal(expected, parsed)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def test_timestamp_and_label(self, version: int | None, temp_file: str) -> None:
        original = DataFrame([(1,)], columns=['variable'])
        time_stamp = datetime(2000, 2, 29, 14, 21)
        data_label = 'This is a data file.'
        path = temp_file
        original.to_stata(path, time_stamp=time_stamp, data_label=data_label, version=version)
        with StataReader(path) as reader:
            assert reader.time_stamp == '29 Feb 2000 14:21'
            assert reader.data_label == data_label

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def test_invalid_timestamp(self, version: int | None, temp_file: str) -> None:
        original = DataFrame([(1,)], columns=['variable'])
        time_stamp = '01 Jan 2000, 00:00:00'
        path = temp_file
        msg = 'time_stamp should be datetime type'
        with pytest.raises(ValueError, match=msg):
            original.to_stata(path, time_stamp=time_stamp, version=version)
        assert not os.path.isfile(path)

    def test_numeric_column_names(self, temp_file: str) -> None:
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
    def test_nan_to_missing_value(self, version: int | None, temp_file: str) -> None:
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

    def test_no_index(self, temp_file: str) -> None:
        columns = ['x', 'y']
        original = DataFrame(np.reshape(np.arange(10.0), (5, 2)), columns=columns)
        original.index.name = 'index_not_written'
        path = temp_file
        original.to_stata(path, write_index=False)
        written_and_read_again = self.read_dta(path)
        with pytest.raises(KeyError, match=original.index.name):
            written_and_read_again['index_not_written']

    def test_string_no_dates(self, temp_file: str) -> None:
        s1 = Series(['a', 'A longer string'])
        s2 = Series([1.0, 2.0], dtype=np.float64)
        original = DataFrame({'s1': s1, 's2': s2})
        original.index.name = 'index'
        path = temp_file
        original.to_stata(path)
        written_and_read_again = self.read_dta(path)
        expected = original
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def test_large_value_conversion(self, temp_file: str) -> None:
        s0 = Series([1, 99], dtype=np.int8)
        s1 = Series([1, 127], dtype=np.int8)
        s2 = Series([1, 2 ** 15 - 1], dtype=np.int16)
        s3 = Series([1, 2 ** 63 - 1], dtype=np.int64)
        original = DataFrame({'s0': s0, 's1': s1, 's2': s2, 's3': s3})
        original.index.name = 'index'
        path = temp_file
        with tm.assert_produces_warning(PossiblePrecisionLoss, match='from int64 to'):
            original.to_stata(path)
        written_and_read_again = self.read_dta(path)
        modified = original
        modified['s1'] = Series(modified['s1'], dtype=np.int16)
        modified['s2'] = Series(modified['s2'], dtype=np.int32)
        modified['s3'] = Series(modified['s3'], dtype=np.float64)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), modified)

    def test_dates_invalid_column(self, temp_file: str) -> None:
        original = DataFrame([datetime(2006, 11, 19, 23, 13, 20)])
        original.index.name = 'index'
        path = temp_file
        msg = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates={0: 'tc'})
        written_and_read_again = self.read_dta(path)
        expected = original.copy()
        expected.columns = ['_0']
        expected.index = original.index.astype(np.int32)
        expected['_0'] = expected['_0'].astype('M8[ms]')
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def test_105(self, datapath: str) -> None:
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

    def test_value_labels_old_format(self, datapath: str) -> None:
        dpath = datapath('io', 'data', 'stata', 'S4_EDUC1.dta')
        with StataReader(dpath) as reader:
            assert reader.value_labels() == {}

    def test_date_export_formats(self, temp_file: str) -> None:
        columns = ['tc', 'td', 'tw', 'tm', 'tq', 'th', 'ty']
        conversions = {c: c for c in columns}
        data = [datetime(2006, 11, 20, 23, 13, 20)] * len(columns)
        original = DataFrame([data], columns=columns)
        original.index.name = 'index'
        expected_values = [datetime(2006, 11, 20, 23, 13, 20), datetime(2006, 11, 20), datetime(2006, 11, 19), datetime(2006, 11, 1), datetime(2006, 10, 1), datetime(2006, 7, 1), datetime(2006, 1, 1)]
        expected = DataFrame([expected_values], index=pd.Index([0], dtype=np.int32, name='index'), columns=columns, dtype='M8[s]')
        expected['tc'] = expected['tc'].astype('M8[ms]')
        path = temp_file
        original.to_stata(path, convert_dates=conversions)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def test_write_missing_strings(self, temp_file: str) -> None:
        original = DataFrame([['1'], [None]], columns=['foo'])
        expected = DataFrame([['1'], ['']], index=pd.RangeIndex(2, name='index'), columns=['foo'])
        path = temp_file
        original.to_stata(path)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    @pytest.mark.parametrize('byteorder', ['>', '<'])
    def test_bool_uint(self, byteorder: str, version: int | None, temp_file: str) -> None:
        s0 = Series([0, 1, True], dtype=np.bool_)
        s1 = Series([0, 1, 100], dtype=np.uint8)
        s2 = Series([0, 1, 255], dtype=np.uint8)
        s3 = Series([0, 1, 2 ** 15 - 100], dtype=np.uint16)
        s4 = Series([0, 1, 2 ** 16 - 1], dtype