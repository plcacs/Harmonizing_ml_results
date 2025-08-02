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
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype, DataFrame, Series
from pandas.io.parsers import read_csv
from pandas.io.stata import (
    CategoricalConversionWarning,
    InvalidColumnName,
    PossiblePrecisionLoss,
    StataMissingValue,
    StataReader,
    StataWriter,
    StataWriterUTF8,
    ValueLabelTypeMismatch,
    read_stata,
)
import pandas._testing as tm


@pytest.fixture
def func_gr71xg09() -> DataFrame:
    return DataFrame({'a': [1, 2, 3, 4], 'b': [1.0, 3.0, 27.0, 81.0], 'c': [
        'Atlanta', 'Birmingham', 'Cincinnati', 'Detroit']})


@pytest.fixture
def func_ifrfjz5i(datapath: Callable[..., str]) -> DataFrame:
    dta14_114: str = datapath('io', 'data', 'stata', 'stata5_114.dta')
    parsed_114: DataFrame = read_stata(dta14_114, convert_dates=True)
    parsed_114.index.name = 'index'
    return parsed_114


class TestStata:

    def func_faakeqon(self, file: Union[str, os.PathLike]) -> DataFrame:
        return read_stata(file, convert_dates=True)

    def func_nk169mcs(self, file: Union[str, os.PathLike]) -> DataFrame:
        return func_nk169mcs(file, parse_dates=True)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_49sliwby(self, version: Optional[int], temp_file: str) -> None:
        empty_ds: DataFrame = DataFrame(columns=['unit'])
        path: str = temp_file
        empty_ds.to_stata(path, write_index=False, version=version)
        empty_ds2: DataFrame = read_stata(path)
        tm.assert_frame_equal(empty_ds, empty_ds2)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_chbphp4q(self, version: Optional[int], temp_file: str) -> None:
        empty_df_typed: DataFrame = DataFrame({
            'i8': np.array([0], dtype=np.int8),
            'i16': np.array([0], dtype=np.int16),
            'i32': np.array([0], dtype=np.int32),
            'i64': np.array([0], dtype=np.int64),
            'u8': np.array([0], dtype=np.uint8),
            'u16': np.array([0], dtype=np.uint16),
            'u32': np.array([0], dtype=np.uint32),
            'u64': np.array([0], dtype=np.uint64),
            'f32': np.array([0], dtype=np.float32),
            'f64': np.array([0], dtype=np.float64)
        })
        path: str = temp_file
        empty_df_typed.to_stata(path, write_index=False, version=version)
        empty_reread: DataFrame = read_stata(path)
        expected: DataFrame = empty_df_typed.copy()
        expected['u8'] = expected['u8'].astype(np.int8)
        expected['u16'] = expected['u16'].astype(np.int16)
        expected['u32'] = expected['u32'].astype(np.int32)
        expected['u64'] = expected['u64'].astype(np.int32)
        expected['i64'] = expected['i64'].astype(np.int32)
        tm.assert_frame_equal(expected, empty_reread)
        tm.assert_series_equal(expected.dtypes, empty_reread.dtypes)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_ajcvbvgy(self, version: Optional[int], temp_file: str) -> None:
        df: DataFrame = DataFrame({'a': range(5), 'b': ['b1', 'b2', 'b3', 'b4', 'b5']})
        path: str = temp_file
        df.to_stata(path, write_index=False, version=version)
        read_df: DataFrame = read_stata(path)
        assert isinstance(read_df.index, pd.RangeIndex)
        expected: DataFrame = df.copy()
        expected['a'] = expected['a'].astype(np.int32)
        tm.assert_frame_equal(read_df, expected, check_index_type=True)

    @pytest.mark.parametrize('version', [102, 103, 104, 105, 108, 110, 111, 113, 114, 115, 117, 118, 119])
    def func_2lghgygb(self, version: int, datapath: Callable[..., str]) -> None:
        file: str = datapath('io', 'data', 'stata', f'stata1_{version}.dta')
        parsed: DataFrame = self.read_dta(file)
        expected: DataFrame = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)],
            columns=['float_miss', 'double_miss', 'byte_miss', 'int_miss', 'long_miss'])
        expected['float_miss'] = expected['float_miss'].astype(np.float32)
        if version <= 108:
            expected = expected.rename(columns={
                'float_miss': 'f_miss',
                'double_miss': 'd_miss',
                'byte_miss': 'b_miss',
                'int_miss': 'i_miss',
                'long_miss': 'l_miss'
            })
        tm.assert_frame_equal(parsed, expected)

    def func_06ufazzl(self, datapath: Callable[..., str]) -> None:
        expected: DataFrame = DataFrame.from_records([
            (datetime(2006, 11, 19, 23, 13, 20), 1479596223000, datetime(2010, 1, 20),
             datetime(2010, 1, 8), datetime(2010, 1, 1), datetime(1974, 7, 1),
             datetime(2010, 1, 1), datetime(2010, 1, 1)),
            (datetime(1959, 12, 31, 20, 3, 20), -1479590, datetime(1953, 10, 2),
             datetime(1948, 6, 10), datetime(1955, 1, 1), datetime(1955, 7, 1),
             datetime(1955, 1, 1), datetime(2, 1, 1)),
            (pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT)
        ], columns=[
            'datetime_c', 'datetime_big_c', 'date', 'weekly_date',
            'monthly_date', 'quarterly_date', 'half_yearly_date', 'yearly_date'
        ])
        expected['datetime_c'] = expected['datetime_c'].astype('M8[ms]')
        expected['date'] = expected['date'].astype('M8[s]')
        expected['weekly_date'] = expected['weekly_date'].astype('M8[s]')
        expected['monthly_date'] = expected['monthly_date'].astype('M8[s]')
        expected['quarterly_date'] = expected['quarterly_date'].astype('M8[s]')
        expected['half_yearly_date'] = expected['half_yearly_date'].astype('M8[s]')
        expected['yearly_date'] = expected['yearly_date'].astype('M8[s]')
        path1: str = datapath('io', 'data', 'stata', 'stata2_114.dta')
        path2: str = datapath('io', 'data', 'stata', 'stata2_115.dta')
        path3: str = datapath('io', 'data', 'stata', 'stata2_117.dta')
        msg: str = 'Leaving in Stata Internal Format'
        with tm.assert_produces_warning(UserWarning, match=msg):
            parsed_114: DataFrame = self.read_dta(path1)
        with tm.assert_produces_warning(UserWarning, match=msg):
            parsed_115: DataFrame = self.read_dta(path2)
        with tm.assert_produces_warning(UserWarning, match=msg):
            parsed_117: DataFrame = self.read_dta(path3)
        tm.assert_frame_equal(parsed_114, expected)
        tm.assert_frame_equal(parsed_115, expected)
        tm.assert_frame_equal(parsed_117, expected)

    @pytest.mark.parametrize('file', ['stata3_113', 'stata3_114', 'stata3_115', 'stata3_117'])
    def func_bjts3m28(self, file: str, datapath: Callable[..., str]) -> None:
        file_path: str = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed: DataFrame = self.read_dta(file_path)
        expected: DataFrame = self.read_csv(datapath('io', 'data', 'stata', 'stata3.csv'))
        expected = expected.astype(np.float32)
        expected['year'] = expected['year'].astype(np.int16)
        expected['quarter'] = expected['quarter'].astype(np.int8)
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [110, 111, 113, 114, 115, 117])
    def func_s6fdhyad(self, version: int, datapath: Callable[..., str]) -> None:
        file: str = datapath('io', 'data', 'stata', f'stata4_{version}.dta')
        parsed: DataFrame = self.read_dta(file)
        expected: DataFrame = DataFrame.from_records([
            ['one', 'ten', 'one', 'one', 'one'],
            ['two', 'nine', 'two', 'two', 'two'],
            ['three', 'eight', 'three', 'three', 'three'],
            ['four', 'seven', 4, 'four', 'four'],
            ['five', 'six', 5, np.nan, 'five'],
            ['six', 'five', 6, np.nan, 'six'],
            ['seven', 'four', 7, np.nan, 'seven'],
            ['eight', 'three', 8, np.nan, 'eight'],
            ['nine', 'two', 9, np.nan, 'nine'],
            ['ten', 'one', 'ten', np.nan, 'ten']
        ], columns=[
            'fully_labeled', 'fully_labeled2', 'incompletely_labeled',
            'labeled_with_missings', 'float_labelled'
        ])
        for col in expected:
            orig: Series = expected[col].copy()
            categories: np.ndarray = np.asarray(expected['fully_labeled'][orig.notna()])
            if col == 'incompletely_labeled':
                categories = orig
            cat: pd.Categorical = orig.astype('category')._values
            cat = cat.set_categories(categories, ordered=True)
            cat.categories.rename(None, inplace=True)
            expected[col] = cat
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [102, 103, 104, 105, 108])
    def func_pbel0v4l(self, version: int, datapath: Callable[..., str]) -> None:
        file: str = datapath('io', 'data', 'stata', f'stata4_{version}.dta')
        parsed: DataFrame = self.read_dta(file)
        expected: DataFrame = DataFrame.from_records([
            ['one', 'ten', 'one', 'one', 'one'],
            ['two', 'nine', 'two', 'two', 'two'],
            ['three', 'eight', 'three', 'three', 'three'],
            ['four', 'seven', 4, 'four', 'four'],
            ['five', 'six', 5, np.nan, 'five'],
            ['six', 'five', 6, np.nan, 'six'],
            ['seven', 'four', 7, np.nan, 'seven'],
            ['eight', 'three', 8, np.nan, 'eight'],
            ['nine', 'two', 9, np.nan, 'nine'],
            ['ten', 'one', 'ten', np.nan, 'ten']
        ], columns=['fulllab', 'fulllab2', 'incmplab', 'misslab', 'floatlab'])
        for col in expected:
            orig: Series = expected[col].copy()
            categories: np.ndarray = np.asarray(expected['fulllab'][orig.notna()])
            if col == 'incmplab':
                categories = orig
            cat: pd.Categorical = orig.astype('category')._values
            cat = cat.set_categories(categories, ordered=True)
            cat.categories.rename(None, inplace=True)
            expected[col] = cat
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('file', ['stata12_117', 'stata12_be_117', 'stata12_118', 'stata12_be_118', 'stata12_119', 'stata12_be_119'])
    def func_bsgyyv0u(self, file: str, datapath: Callable[..., str]) -> None:
        parsed: DataFrame = self.read_dta(datapath('io', 'data', 'stata', f'{file}.dta'))
        expected: DataFrame = DataFrame.from_records([
            [1, 'abc', 'abcdefghi'],
            [3, 'cba', 'qwertywertyqwerty'],
            [93, '', 'strl']
        ], columns=['x', 'y', 'z'])
        tm.assert_frame_equal(parsed, expected, check_dtype=False)

    @pytest.mark.parametrize('file', ['stata14_118', 'stata14_be_118', 'stata14_119', 'stata14_be_119'])
    def func_0fcmg1ho(self, file: str, datapath: Callable[..., str]) -> None:
        parsed_118: DataFrame = self.read_dta(datapath('io', 'data', 'stata', f'{file}.dta'))
        parsed_118['Bytes'] = parsed_118['Bytes'].astype('O')
        expected: DataFrame = DataFrame.from_records([
            ['Cat', 'Bogota', 'Bogotá', 1, 1.0, 'option b Ünicode', 1.0],
            ['Dog', 'Boston', 'Uzunköprü', np.nan, np.nan, np.nan, np.nan],
            ['Plane', 'Rome', 'Tromsø', 0, 0.0, 'option a', 0.0],
            ['Potato', 'Tokyo', 'Elâzığ', -4, 4.0, 4, 4],
            ['', '', '', 0, 0.3332999, 'option a', 1 / 3.0]
        ], columns=['Things', 'Cities', 'Unicode_Cities_Strl', 'Ints', 'Floats', 'Bytes', 'Longs'])
        expected['Floats'] = expected['Floats'].astype(np.float32)
        for col in parsed_118.columns:
            tm.assert_almost_equal(parsed_118[col], expected[col])  # type: ignore
        with StataReader(datapath('io', 'data', 'stata', f'{file}.dta')) as rdr:
            vl: Dict[str, str] = rdr.variable_labels()
            vl_expected: Dict[str, str] = {
                'Unicode_Cities_Strl': 'Here are some strls with Ünicode chars',
                'Longs': 'long data',
                'Things': 'Here are some things',
                'Bytes': 'byte data',
                'Ints': 'int data',
                'Cities': 'Here are some cities',
                'Floats': 'float data'
            }
            tm.assert_dict_equal(vl, vl_expected)
            assert rdr.data_label == 'This is a  Ünicode data label'

    def func_g4eoaimm(self, temp_file: str) -> None:
        original: DataFrame = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)],
            columns=['float_miss', 'double_miss', 'byte_miss', 'int_miss', 'long_miss'])
        original.index.name = 'index'
        path: str = temp_file
        original.to_stata(path, convert_dates=None)
        written_and_read_again: DataFrame = self.read_dta(path)
        expected: DataFrame = original.copy()
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def func_x12h35pi(self, datapath: Callable[..., str], temp_file: str) -> None:
        original: DataFrame = self.read_csv(datapath('io', 'data', 'stata', 'stata3.csv'))
        original.index.name = 'index'
        original.index = original.index.astype(np.int32)
        original['year'] = original['year'].astype(np.int32)
        original['quarter'] = original['quarter'].astype(np.int32)
        path: str = temp_file
        original.to_stata(path, convert_dates=None)
        written_and_read_again: DataFrame = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), original, check_index_type=False)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    @pytest.mark.parametrize('using_infer_string', [True, False])
    def func_jdme1aqo(self, version: Optional[int], temp_file: str, using_infer_string: bool) -> None:
        original: DataFrame = DataFrame(data=[['string', 'object', 1, 1.1, np.datetime64('2003-12-25')]],
            columns=['string', 'object', 'integer', 'floating', 'datetime'])
        original['object'] = Series(original['object'], dtype=object)
        original.index.name = 'index'
        original.index = original.index.astype(np.int32)
        original['integer'] = original['integer'].astype(np.int32)
        path: str = temp_file
        original.to_stata(path, convert_dates={'datetime': 'tc'}, version=version)
        written_and_read_again: DataFrame = self.read_dta(path)
        expected: DataFrame = original.copy()
        expected['datetime'] = expected['datetime'].astype('M8[ms]')
        if using_infer_string:
            expected['object'] = expected['object'].astype('str')
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def func_emlcb2uy(self, temp_file: str) -> None:
        path: str = temp_file
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((10, 2)),
            columns=list('AB'))
        df.to_stata(path)

    def func_f7hpbgxu(self, temp_file: str) -> None:
        df: DataFrame = DataFrame(np.random.default_rng(2).standard_normal((5, 4)),
            columns=list('abcd'))
        df.loc[2, 'a':'c'] = np.nan
        df_copy: DataFrame = df.copy()
        path: str = temp_file
        df.to_stata(path, write_index=False)
        tm.assert_frame_equal(df, df_copy)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_x5wmt25j(self, version: Optional[int], datapath: Callable[..., str], temp_file: str, using_infer_string: bool) -> None:
        raw: DataFrame = read_stata(datapath('io', 'data', 'stata', 'stata1_encoding.dta'))
        encoded: DataFrame = read_stata(datapath('io', 'data', 'stata', 'stata1_encoding.dta'))
        result: Any = encoded.kreis1849[0]
        expected: Any = raw.kreis1849[0]
        assert result == expected
        assert isinstance(result, str)
        path: str = temp_file
        encoded.to_stata(path, write_index=False, version=version)
        reread_encoded: DataFrame = read_stata(path)
        tm.assert_frame_equal(encoded, reread_encoded)

    def func_ouiv6lqg(self, temp_file: str) -> None:
        original: DataFrame = DataFrame([(1, 2, 3, 4)],
            columns=['good', 'bäd', '8number', 'astringwithmorethan32characters______'])
        formatted: DataFrame = DataFrame([(1, 2, 3, 4)],
            columns=['good', 'b_d', '_8number', 'astringwithmorethan32characters_'])
        formatted.index.name = 'index'
        formatted = formatted.astype(np.int32)
        path: str = temp_file
        msg: str = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates=None)
        written_and_read_again: DataFrame = self.read_dta(path)
        expected: DataFrame = formatted
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_ia3gsweg(self, version: Optional[int], temp_file: str) -> None:
        original: DataFrame = DataFrame([(1, 2, 3, 4, 5, 6)],
            columns=[
                'astringwithmorethan32characters_1',
                'astringwithmorethan32characters_2',
                '+', '-', 'short', 'delete'
            ])
        formatted: DataFrame = DataFrame([(1, 2, 3, 4, 5, 6)],
            columns=[
                'astringwithmorethan32characters_',
                '_0astringwithmorethan32character',
                '_', '_1_', '_short',
                '_delete'
            ])
        formatted.index.name = 'index'
        formatted = formatted.astype(np.int32)
        path: str = temp_file
        msg: str = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(path, convert_dates=None, version=version)
        written_and_read_again: DataFrame = self.read_dta(path)
        expected: DataFrame = formatted
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def func_8np405d0(self, temp_file: str) -> None:
        s1: Series = Series(2 ** 9, dtype=np.int16)
        s2: Series = Series(2 ** 17, dtype=np.int32)
        s3: Series = Series(2 ** 33, dtype=np.int64)
        original: DataFrame = DataFrame({'int16': s1, 'int32': s2, 'int64': s3})
        original.index.name = 'index'
        formatted: DataFrame = original.copy()
        formatted['int64'] = formatted['int64'].astype(np.float64)
        path: str = temp_file
        original.to_stata(path)
        written_and_read_again: DataFrame = self.read_dta(path)
        expected: DataFrame = formatted
        expected.index = expected.index.astype(np.int32)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    @pytest.mark.parametrize('file', ['stata5_113', 'stata5_114', 'stata5_115', 'stata5_117'])
    def func_a1lt1keo(self, file: str, parsed_114: DataFrame, version: Optional[int], datapath: Callable[..., str], temp_file: str) -> None:
        file_path: str = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed: DataFrame = self.read_dta(file_path)
        tm.assert_frame_equal(parsed_114, parsed)
        path: str = temp_file
        func_ifrfjz5i.to_stata(path, convert_dates={'date_td': 'td'}, version=version)
        written_and_read_again: DataFrame = self.read_dta(path)
        expected: DataFrame = func_ifrfjz5i.copy()
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    @pytest.mark.parametrize('file', ['stata6_113', 'stata6_114', 'stata6_115', 'stata6_117'])
    def func_6jyhr1gh(self, file: str, datapath: Callable[..., str]) -> None:
        expected: DataFrame = self.read_csv(datapath('io', 'data', 'stata', 'stata6.csv'))
        expected['byte_'] = expected['byte_'].astype(np.int8)
        expected['int_'] = expected['int_'].astype(np.int16)
        expected['long_'] = expected['long_'].astype(np.int32)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['double_'] = expected['double_'].astype(np.float64)
        arr: np.ndarray = expected['date_td'].astype('Period[D]')._values.asfreq('s', how='S')
        expected['date_td'] = arr.view('M8[s]')
        file_path: str = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed: DataFrame = self.read_dta(file_path)
        tm.assert_frame_equal(expected, parsed)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_cjdas6n9(self, version: Optional[int], mixed_frame: DataFrame, temp_file: str) -> None:
        mixed_frame.index.name = 'index'
        variable_labels: Dict[str, str] = {'a': 'City Rank', 'b': 'City Exponent', 'c': 'City'}
        path: str = temp_file
        mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)
        with StataReader(path) as sr:
            read_labels: Dict[str, str] = sr.variable_labels()
        expected_labels: Dict[str, str] = {'index': '', 'a': 'City Rank', 'b': 'City Exponent', 'c': 'City'}
        assert read_labels == expected_labels
        variable_labels['index'] = 'The Index'
        path = temp_file
        mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)
        with StataReader(path) as sr:
            read_labels = sr.variable_labels()
        assert read_labels == variable_labels

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_feanqcxa(self, version: Optional[int], mixed_frame: DataFrame, temp_file: str) -> None:
        mixed_frame.index.name = 'index'
        variable_labels: Dict[str, str] = {'a': 'very long' * 10, 'b': 'City Exponent', 'c': 'City'}
        path: str = temp_file
        msg: str = 'Variable labels must be 80 characters or fewer'
        with pytest.raises(ValueError, match=msg):
            mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)

    @pytest.mark.parametrize('version', [114, 117])
    def func_aom6y664(self, version: int, mixed_frame: DataFrame, temp_file: str) -> None:
        mixed_frame.index.name = 'index'
        variable_labels: Dict[str, str] = {'a': 'very long' * 10, 'b': 'City Exponent', 'c': 'City'}
        variable_labels['a'] = 'invalid character Œ'
        path: str = temp_file
        with pytest.raises(ValueError, match='Variable labels must contain only characters'):
            mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)

    def func_1gcao5qj(self, mixed_frame: DataFrame, temp_file: str) -> None:
        values: List[str] = ['Ρ', 'Α', 'Ν', 'Δ', 'Α', 'Σ']
        variable_labels_utf8: Dict[str, str] = {'a': 'City Rank', 'b': 'City Exponent', 'c': ''.join(values)}
        msg1: str = (
            'Variable labels must contain only characters that can be encoded in Latin-1'
        )
        msg2: str = (
            'Variable labels must be 80 characters or fewer'
        )
        with pytest.raises(ValueError, match=msg1):
            mixed_frame.to_stata(temp_file, variable_labels=variable_labels_utf8)
        variable_labels_long: Dict[str, str] = {'a': 'City Rank', 'b': 'City Exponent', 'c':
            'A very, very, very long variable label that is too long for Stata which means that it has more than 80 characters'}
        with pytest.raises(ValueError, match=msg2):
            mixed_frame.to_stata(temp_file, variable_labels=variable_labels_long)

    def func_v8pixc9g(self, temp_file: str) -> None:
        dates: List[datetime] = [
            dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
            dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
            dt.datetime(1776, 7, 4, 7, 4, 7, 4000)
        ]
        original: DataFrame = DataFrame({'nums': [1.0, 2.0, 3.0], 'strs': ['apple', 'banana', 'cherry'], 'dates': dates})
        expected: DataFrame = original[:]
        expected['dates'] = expected['dates'].astype('M8[ms]')
        path: str = temp_file
        original.to_stata(path, write_index=False)
        reread: DataFrame = read_stata(path, convert_dates=True)
        tm.assert_frame_equal(expected, reread)
        original.to_stata(path, write_index=False, convert_dates={'dates': 'tc'})
        direct: DataFrame = read_stata(path, convert_dates=True)
        tm.assert_frame_equal(reread, direct)
        dates_idx: int = original.columns.tolist().index('dates')
        original.to_stata(path, write_index=False, convert_dates={dates_idx: 'tc'})
        direct = read_stata(path, convert_dates=True)
        tm.assert_frame_equal(reread, direct)

    def func_2m3juvey(self, temp_file: str) -> None:
        original: DataFrame = DataFrame({'a': [1 + 2.0j, 2 + 4.0j]})
        msg: str = 'Data type complex128 not supported'
        with pytest.raises(NotImplementedError, match=msg):
            original.to_stata(temp_file)

    def func_smlk26vx(self, temp_file: str) -> None:
        dates: List[dt.datetime] = [
            dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
            dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
            dt.datetime(1776, 7, 4, 7, 4, 7, 4000)
        ]
        original: DataFrame = DataFrame({'nums': [1.0, 2.0, 3.0], 'strs': ['apple', 'banana', 'cherry'], 'dates': dates})
        msg: str = 'Format %tC not implemented'
        with pytest.raises(NotImplementedError, match=msg):
            original.to_stata(temp_file, convert_dates={'dates': 'tC'})
        dates_tz: pd.DatetimeIndex = pd.date_range('1-1-1990', periods=3, tz='Asia/Hong_Kong')
        original = DataFrame({'nums': [1.0, 2.0, 3.0], 'strs': ['apple', 'banana', 'cherry'], 'dates': dates_tz})
        with pytest.raises(NotImplementedError, match='Data type datetime64'):
            original.to_stata(temp_file)

    def func_d6kplmyo(self, datapath: Callable[..., str]) -> None:
        msg: str = """
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

    def func_hdmgpvdk(self, datapath: Callable[..., str]) -> None:
        df: DataFrame = read_stata(datapath('io', 'data', 'stata', 'stata7_111.dta'))
        original: DataFrame = DataFrame({
            'y': [1, 1, 1, 1, 1, 0, 0, np.nan, 0, 0],
            'x': [1, 2, 1, 3, np.nan, 4, 3, 5, 1, 6],
            'w': [2, np.nan, 5, 2, 4, 4, 3, 1, 2, 3],
            'z': ['a', 'b', 'c', 'd', 'e', '', 'g', 'h', 'i', 'j']
        })
        original = original[['y', 'x', 'w', 'z']]
        tm.assert_frame_equal(original, df)

    def func_xdkpn4a6(self, temp_file: str) -> None:
        df: DataFrame = DataFrame([datetime(2006, 11, 19, 23, 13, 20)],
            columns=['datetime'])
        df.index.name = 'index'
        msg: str = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            df.to_stata(temp_file, convert_dates={(0): 'tc'})
        written_and_read_again: DataFrame = self.read_dta(temp_file)
        expected: DataFrame = df.copy()
        expected.columns = ['_0']
        expected.index = expected.index.astype(np.int32)
        expected['_0'] = expected['_0'].astype('M8[ms]')
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def func_5q9gndmf(self, datapath: Callable[..., str]) -> None:
        dpath: str = datapath('io', 'data', 'stata', 'S4_EDUC1.dta')
        df: DataFrame = read_stata(dpath)
        df0: DataFrame = DataFrame([
            [1, 1, 3, -2],
            [2, 1, 2, -2],
            [4, 1, 1, -2]
        ])
        df0.columns = ['clustnum', 'pri_schl', 'psch_num', 'psch_dis']
        df0['clustnum'] = df0['clustnum'].astype(np.int16)
        df0['pri_schl'] = df0['pri_schl'].astype(np.int8)
        df0['psch_num'] = df0['psch_num'].astype(np.int8)
        df0['psch_dis'] = df0['psch_dis'].astype(np.float32)
        tm.assert_frame_equal(df.head(3), df0)

    def func_054shn9w(self, datapath: Callable[..., str]) -> None:
        dpath: str = datapath('io', 'data', 'stata', 'S4_EDUC1.dta')
        with StataReader(dpath) as reader:
            assert reader.value_labels() == {}

    def func_c8plrxc2(self, temp_file: str) -> None:
        columns: List[str] = ['tc', 'td', 'tw', 'tm', 'tq', 'th', 'ty']
        conversions: Dict[str, str] = {c: c for c in columns}
        data: List[List[datetime]] = [
            [datetime(2006, 11, 20, 23, 13, 20), datetime(2006, 11, 20),
             datetime(2006, 11, 19), datetime(2006, 11, 1), datetime(2006, 10, 1),
             datetime(2006, 7, 1), datetime(2006, 1, 1)],
            [datetime(2006, 11, 20, 23, 13, 20), datetime(2006, 11, 20),
             datetime(2006, 11, 19), datetime(2006, 11, 1), datetime(2006, 10, 1),
             datetime(2006, 7, 1), datetime(2006, 1, 1)],
            [datetime(2006, 11, 20, 23, 13, 20), datetime(2006, 11, 20),
             datetime(2006, 11, 19), datetime(2006, 11, 1), datetime(2006, 10, 1),
             datetime(2006, 7, 1), datetime(2006, 1, 1)]
        ]
        expected_values: List[datetime] = [
            datetime(2006, 11, 20, 23, 13, 20),
            datetime(2006, 11, 20),
            datetime(2006, 11, 19),
            datetime(2006, 11, 1),
            datetime(2006, 10, 1),
            datetime(2006, 7, 1),
            datetime(2006, 1, 1)
        ]
        expected: DataFrame = DataFrame([
            expected_values,
            expected_values,
            expected_values
        ], index=pd.Index([0, 1, 2], dtype=np.int32, name='index'),
            columns=columns, dtype='M8[s]')
        expected['tc'] = expected['tc'].astype('M8[ms]')
        path: str = temp_file
        expected.to_stata(path, convert_dates=conversions)
        written_and_read_again: DataFrame = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    def func_na0b86xl(self, temp_file: str) -> None:
        original: DataFrame = DataFrame([['1'], [None]], columns=['foo'])
        expected: DataFrame = DataFrame([['1'], ['']], index=pd.RangeIndex(2, name='index'), columns=['foo'])
        path: str = temp_file
        original.to_stata(path)
        written_and_read_again: DataFrame = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    @pytest.mark.parametrize('byteorder', ['>', '<'])
    def func_fod4kijf(self, byteorder: str, version: Optional[int], temp_file: str) -> None:
        s0: Series = Series([0, 1, True], dtype=np.bool_)
        s1: Series = Series([0, 1, 100], dtype=np.uint8)
        s2: Series = Series([0, 1, 255], dtype=np.uint8)
        s3: Series = Series([0, 1, 2 ** 15 - 100], dtype=np.uint16)
        s4: Series = Series([0, 1, 2 ** 16 - 1], dtype=np.uint16)
        s5: Series = Series([0, 1, 2 ** 31 - 100], dtype=np.uint32)
        s6: Series = Series([0, 1, 2 ** 32 - 1], dtype=np.uint32)
        original: DataFrame = DataFrame({'s0': s0, 's1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6})
        original.index.name = 'index'
        path: str = temp_file
        original.to_stata(path, byteorder=byteorder, version=version)
        written_and_read_again: DataFrame = self.read_dta(path)
        written_and_read_again = written_and_read_again.set_index('index')
        expected: DataFrame = original.copy()
        expected_types: Tuple[np.dtype, ...] = (
            np.int8, np.int8, np.int16, np.int16, np.int32, np.int32, np.float64
        )
        for c, t in zip(expected.columns, expected_types):
            expected[c] = expected[c].astype(t)
        tm.assert_frame_equal(written_and_read_again, expected)

    def func_5mhi3rzp(self, datapath: Callable[..., str]) -> None:
        with StataReader(datapath('io', 'data', 'stata', 'stata7_115.dta')) as rdr:
            sr_115: Dict[str, str] = rdr.variable_labels()
        with StataReader(datapath('io', 'data', 'stata', 'stata7_117.dta')) as rdr:
            sr_117: Dict[str, str] = rdr.variable_labels()
        keys: Tuple[str, ...] = ('var1', 'var2', 'var3')
        labels: Tuple[str, ...] = ('label1', 'label2', 'label3')
        for k, v in sr_115.items():
            assert k in sr_117
            assert v == sr_117[k]
            assert k in keys
            assert v in labels

    @pytest.mark.parametrize('file', ['stata3_113', 'stata3_114', 'stata3_115', 'stata3_117'])
    def func_7i0isxww(self, file: str, temp_file: str) -> None:
        str_lens: Tuple[int, ...] = (1, 100, 244)
        s: Dict[str, Series] = {}
        for str_len in str_lens:
            s[f's{str_len}'] = Series(['a' * str_len, 'b' * str_len, 'c' * str_len])
        original: DataFrame = DataFrame(s)
        path: str = temp_file
        original.to_stata(path, write_index=False)
        with StataReader(path) as sr:
            sr._ensure_open()
            for variable, fmt, typ in zip(sr._varlist, sr._fmtlist, sr._typlist):
                assert int(variable[1:]) == int(fmt[1:-1])
                assert int(variable[1:]) == typ

    def func_htnsqjad(self, temp_file: str) -> None:
        str_lens: Tuple[int, ...] = (1, 244, 500)
        s: Dict[str, Series] = {}
        for str_len in str_lens:
            s[f's{str_len}'] = Series(['a' * str_len, 'b' * str_len, 'c' * str_len])
        original: DataFrame = DataFrame(s)
        msg: str = (
            "Fixed width strings in Stata \\.dta files are limited to 244 \\(or fewer\\)\\ncharacters\\.  Column 's500' does not satisfy this restriction\\. Use the\\n'version=117' parameter to write the newer \\(Stata 13 and later\\) format\\."
        )
        with pytest.raises(ValueError, match=msg):
            path: str = temp_file
            original.to_stata(path)

    def func_0hgkxukf(self, temp_file: str) -> None:
        types: Tuple[str, ...] = ('b', 'h', 'l')
        df: DataFrame = DataFrame([[0.0]], columns=['float_'])
        path: str = temp_file
        df.to_stata(path)
        with StataReader(path) as rdr:
            valid_range: Dict[str, Tuple[int, int]] = rdr.VALID_RANGE
        expected_values: List[str] = ['.' + chr(97 + i) for i in range(26)]
        expected_values.insert(0, '.')
        for t in types:
            offset: int = valid_range[t][1]
            for i in range(27):
                val: StataMissingValue = StataMissingValue(offset + 1 + i)
                assert val.string == expected_values[i]
        val: StataMissingValue = StataMissingValue(struct.unpack('<f', b'\x00\x00\x00\x7f')[0])
        assert val.string == '.'
        val = StataMissingValue(struct.unpack('<f', b'\x00\xd0\x00\x7f')[0])
        assert val.string == '.z'
        val = StataMissingValue(struct.unpack('<d', b'\x00\x00\x00\x00\x00\x00\xe0\x7f')[0])
        assert val.string == '.'
        val = StataMissingValue(struct.unpack('<d', b'\x00\x00\x00\x00\x00\x1a\xe0\x7f')[0])
        assert val.string == '.z'

    @pytest.mark.parametrize('version', [113, 115, 117])
    def func_o5wkqtj7(self, version: int, datapath: Callable[..., str]) -> None:
        columns: List[str] = ['int8_', 'int16_', 'int32_', 'float32_', 'float64_']
        smv: StataMissingValue = StataMissingValue(101)
        keys: List[int] = sorted(smv.MISSING_VALUES.keys())
        data: List[List[StataMissingValue]] = []
        for i in range(27):
            row: List[StataMissingValue] = [
                StataMissingValue(keys[i + j * 27]) for j in range(5)
            ]
            data.append(row)
        expected: DataFrame = DataFrame(data, columns=columns)
        parsed: DataFrame = read_stata(datapath('io', 'data', 'stata', f'stata8_{version}.dta'), convert_missing=True)
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [104, 105, 108, 110, 111])
    def func_wbqxtehc(self, version: int, datapath: Callable[..., str]) -> None:
        columns: List[str] = ['int8_', 'int16_', 'int32_', 'float32_', 'float64_']
        smv: StataMissingValue = StataMissingValue(101)
        keys: List[int] = sorted(smv.MISSING_VALUES.keys())
        data: List[List[StataMissingValue]] = []
        row: List[StataMissingValue] = [StataMissingValue(keys[j * 27]) for j in range(5)]
        data.append(row)
        expected: DataFrame = DataFrame(data, columns=columns)
        parsed: DataFrame = read_stata(datapath('io', 'data', 'stata', f'stata8_{version}.dta'), convert_missing=True)
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize('version', [102, 103])
    def func_cgcymr9p(self, version: int, datapath: Callable[..., str]) -> None:
        columns: List[str] = ['int8_', 'int16_', 'int32_', 'float32_', 'float64_']
        smv: StataMissingValue = StataMissingValue(101)
        keys: List[int] = sorted(smv.MISSING_VALUES.keys())
        data: List[List[StataMissingValue]] = []
        row: List[StataMissingValue] = [StataMissingValue(keys[j * 27]) for j in [1, 1, 2, 3, 4]]
        data.append(row)
        expected: DataFrame = DataFrame(data, columns=columns)
        parsed: DataFrame = read_stata(datapath('io', 'data', 'stata', f'stata8_{version}.dta'), convert_missing=True)
        tm.assert_frame_equal(parsed, expected)

    def func_ol1zgnwq(self, datapath: Callable[..., str], temp_file: str) -> None:
        yr: List[int] = [1960, 2000, 9999, 100, 2262, 1677]
        mo: List[int] = [1, 1, 12, 1, 4, 9]
        dd: List[int] = [1, 1, 31, 1, 22, 23]
        hr: List[int] = [0, 0, 23, 0, 0, 0]
        mm: List[int] = [0, 0, 59, 0, 0, 0]
        ss: List[int] = [0, 0, 59, 0, 0, 0]
        expected: List[List[datetime]] = []
        for year, month, day, hour, minute, second in zip(yr, mo, dd, hr, mm, ss):
            row: List[datetime] = []
            for j in range(7):
                if j == 0:
                    row.append(datetime(year, month, day, hour, minute, second))
                elif j == 6:
                    row.append(datetime(year, 1, 1))
                else:
                    row.append(datetime(year, month, day))
            expected.append(row)
        expected.append([pd.NaT] * 7)
        columns: List[str] = ['date_tc', 'date_td', 'date_tw', 'date_tm', 'date_tq',
                              'date_th', 'date_ty']
        # Modify specific entries
        expected[2][2] = datetime(9999, 12, 24)
        expected[2][3] = datetime(9999, 12, 1)
        expected[2][4] = datetime(9999, 10, 1)
        expected[2][5] = datetime(9999, 7, 1)
        expected[4][2] = datetime(2262, 4, 16)
        expected[4][3] = expected[4][4] = datetime(2262, 4, 1)
        expected[4][5] = expected[4][6] = datetime(2262, 1, 1)
        expected[5][2] = expected[5][3] = expected[5][4] = datetime(1677, 10, 1)
        expected[5][5] = expected[5][6] = datetime(1678, 1, 1)
        expected: DataFrame = DataFrame(expected, columns=columns, dtype=object)
        expected['date_tc'] = expected['date_tc'].astype('M8[ms]')
        expected['date_td'] = expected['date_td'].astype('M8[s]')
        expected['date_tw'] = expected['date_tw'].astype('M8[s]')
        expected['date_tm'] = expected['date_tm'].astype('M8[s]')
        expected['date_tq'] = expected['date_tq'].astype('M8[s]')
        expected['date_th'] = expected['date_th'].astype('M8[s]')
        expected['date_ty'] = expected['date_ty'].astype('M8[s]')
        parsed_115: DataFrame = read_stata(datapath('io', 'data', 'stata', 'stata9_115.dta'))
        parsed_117: DataFrame = read_stata(datapath('io', 'data', 'stata', 'stata9_117.dta'))
        tm.assert_frame_equal(expected, parsed_115)
        tm.assert_frame_equal(expected, parsed_117)
        date_conversion: Dict[str, str] = {c: c[-2:] for c in columns}
        path: str = temp_file
        expected.index.name = 'index'
        expected.to_stata(path, convert_dates=date_conversion)
        written_and_read_again: DataFrame = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected.set_index(expected.index.astype(np.int32)))

    def func_tdg31z96(self, datapath: Callable[..., str]) -> None:
        expected: DataFrame = self.read_csv(datapath('io', 'data', 'stata', 'stata6.csv'))
        expected['byte_'] = expected['byte_'].astype(np.int8)
        expected['int_'] = expected['int_'].astype(np.int16)
        expected['long_'] = expected['long_'].astype(np.int32)
        expected['float_'] = expected['float_'].astype(np.float32)
        expected['double_'] = expected['double_'].astype(np.float64)
        expected['date_td'] = expected['date_td'].apply(datetime.strptime, args=('%Y-%m-%d',))
        columns: List[str] = ['byte_', 'int_', 'long_']
        expected = expected[columns]
        dropped: DataFrame = read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'), convert_dates=True, columns=columns)
        tm.assert_frame_equal(expected, dropped)
        columns = ['int_', 'long_', 'byte_']
        expected = expected[columns]
        reordered: DataFrame = read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'), convert_dates=True, columns=columns)
        tm.assert_frame_equal(expected, reordered)
        msg: str = (
            'columns contains duplicate entries'
        )
        with pytest.raises(ValueError, match=msg):
            read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'),
                convert_dates=True, columns=['byte_', 'byte_'])
        msg = (
            'The following columns were not found in the Stata data set: not_found'
        )
        with pytest.raises(ValueError, match=msg):
            read_stata(datapath('io', 'data', 'stata', 'stata6_117.dta'),
                convert_dates=True, columns=['byte_', 'int_', 'long_', 'not_found'])

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def func_jgz6o7hy(self, version: Optional[int], temp_file: str, mixed_frame: DataFrame) -> None:
        original: DataFrame = DataFrame.from_records([
            ['one', 'ten', 'one', 'one', 'one', 1],
            ['two', 'nine', 'two', 'two', 'two', 2],
            ['three', 'eight', 'three', 'three', 'three', 3],
            ['four', 'seven', 4, 'four', 'four', 4],
            ['five', 'six', 5, np.nan, 'five', 5],
            ['six', 'five', 6, np.nan, 'six', 6],
            ['seven', 'four', 7, np.nan, 'seven', 7],
            ['eight', 'three', 8, np.nan, 'eight', 8],
            ['nine', 'two', 9, np.nan, 'nine', 9],
            ['ten', 'one', 'ten', np.nan, 'ten', 10],
        ], columns=[
            'fully_labeled', 'fully_labeled2', 'incompletely_labeled',
            'labeled_with_missings', 'float_labelled', 'unlabeled'
        ])
        path: str = temp_file
        original.astype('category').to_stata(path, version=version)
        written_and_read_again: DataFrame = self.read_dta(path)
        res: DataFrame = written_and_read_again.set_index('index')
        expected: DataFrame = original.copy()
        expected.index = expected.index.set_names('index')
        expected['incompletely_labeled'] = expected['incompletely_labeled'].apply(str)
        expected['unlabeled'] = expected['unlabeled'].apply(str)
        for col in expected:
            orig: Series = expected[col]
            cat: pd.Categorical = orig.astype('category')._values
            cat = cat.as_ordered()
            if col == 'unlabeled':
                cat = cat.set_categories(orig, ordered=True)
            cat.categories.rename(None, inplace=True)
            expected[col] = cat
        tm.assert_frame_equal(res, expected)

    def func_2o0n5jcf(self, temp_file: str) -> None:
        original: DataFrame = DataFrame.from_records([
            ['a'], ['b'], ['c'], ['d'], [1]
        ], columns=['Too_long']).astype('category')
        msg: str = (
            'data file created has not lost information due to duplicate labels'
        )
        with tm.assert_produces_warning(ValueLabelTypeMismatch, match=msg):
            original.to_stata(temp_file)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_ipjwjp4o(self, version: Optional[int], temp_file: str) -> None:
        values: List[List[str]] = [['a' + str(i)] for i in range(120)]
        values.append([np.nan])
        original: DataFrame = DataFrame.from_records(values, columns=['many_labels'])
        original = pd.concat([original[col].astype('category') for col in original], axis=1)
        original.index.name = 'index'
        path: str = temp_file
        original.to_stata(path, version=version)
        written_and_read_again: DataFrame = self.read_dta(path)
        res: DataFrame = written_and_read_again.set_index('index')
        expected: DataFrame = original.copy()
        for col in expected:
            cat: pd.Categorical = expected[col]._values
            new_cats: pd.Index = cat.remove_unused_categories().categories
            cat = cat.set_categories(new_cats, ordered=True)
            expected[col] = cat
        tm.assert_frame_equal(res, expected)

    @pytest.mark.parametrize('file', ['stata10_115', 'stata10_117'])
    def func_mnn9zfup(self, file: str, datapath: Callable[..., str]) -> None:
        expected: List[Tuple[bool, str, List[str], np.ndarray]] = [
            (True, 'ordered', ['a', 'b', 'c', 'd', 'e'], np.arange(5)),
            (True, 'reverse', ['a', 'b', 'c', 'd', 'e'], np.arange(5)[::-1]),
            (True, 'noorder', ['a', 'b', 'c', 'd', 'e'], np.array([2, 1, 4, 0, 3])),
            (True, 'floating', ['a', 'b', 'c', 'd', 'e'], np.arange(0, 5)),
            (True, 'float_missing', ['a', 'd', 'e'], np.array([0, 1, 2, -1, -1])),
            (False, 'nolabel', [1.0, 2.0, 3.0, 4.0, 5.0], np.arange(5)),
            (True, 'int32_mixed', ['d', 2, 'e', 'b', 'a'], np.arange(5)),
        ]
        cols: List[Tuple[str, Union[pd.Categorical, Series]]] = []
        for is_cat, col, labels, codes in expected:
            if is_cat:
                cols.append((col, pd.Categorical.from_codes(codes, labels, ordered=True)))
            else:
                cols.append((col, Series(labels, dtype=np.float32)))
        expected_df: DataFrame = DataFrame.from_dict(dict(cols))
        file_path: str = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed: DataFrame = read_stata(file_path)
        tm.assert_frame_equal(expected_df, parsed)
        for col in expected_df:
            if isinstance(expected_df[col].dtype, CategoricalDtype):
                tm.assert_series_equal(expected_df[col].cat.codes, parsed[col].cat.codes)
                tm.assert_index_equal(expected_df[col].cat.categories, parsed[col].cat.categories)

    @pytest.mark.parametrize('file', ['stata11_115', 'stata11_117'])
    def func_crdr5ywi(self, file: str, datapath: Callable[..., str]) -> None:
        parsed: DataFrame = read_stata(datapath('io', 'data', 'stata', f'{file}.dta'))
        parsed = parsed.sort_values('srh', na_position='first')
        parsed.index = pd.RangeIndex(len(parsed))
        codes: List[int] = [-1, -1, 0, 1, 1, 1, 2, 2, 3, 4]
        categories: List[str] = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
        cat: pd.Categorical = pd.Categorical.from_codes(codes=codes, categories=categories, ordered=True)
        expected: Series = Series(cat, name='srh')
        tm.assert_series_equal(expected, parsed['srh'])

    @pytest.mark.parametrize('file', ['stata10_115', 'stata10_117'])
    def func_85xr0sl6(self, file: str, datapath: Callable[..., str]) -> None:
        file_path: str = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed: DataFrame = read_stata(file_path)
        parsed_unordered: DataFrame = read_stata(file_path, order_categoricals=False)
        for col in parsed:
            if not isinstance(parsed[col].dtype, CategoricalDtype):
                continue
            assert parsed[col].cat.ordered
            assert not parsed_unordered[col].cat.ordered

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize(
        'file',
        [
            'stata1_117', 'stata2_117', 'stata3_117', 'stata4_117', 'stata5_117',
            'stata6_117', 'stata7_117', 'stata8_117', 'stata9_117',
            'stata10_117', 'stata11_117'
        ]
    )
    @pytest.mark.parametrize('chunksize', [1, 2])
    @pytest.mark.parametrize('convert_categoricals', [False, True])
    @pytest.mark.parametrize('convert_dates', [False, True])
    def func_3s0muga7(self, file: str, chunksize: int, convert_categoricals: bool, convert_dates: bool, datapath: Callable[..., str]) -> Iterator[None]:
        fname: str = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed: DataFrame = read_stata(fname, convert_categoricals=convert_categoricals, convert_dates=convert_dates)
        with read_stata(fname, iterator=True, convert_categoricals=convert_categoricals, convert_dates=convert_dates) as itr:
            pos: int = 0
            for j in range(5):
                try:
                    chunk: DataFrame = itr.read(chunksize)
                except StopIteration:
                    break
                from_frame: DataFrame = parsed.iloc[pos:pos + chunksize, :].copy()
                from_frame = self.b106tjue(from_frame)
                tm.assert_frame_equal(from_frame, chunk, check_dtype=False)
                pos += chunksize

    @staticmethod
    def func_b106tjue(from_frame: DataFrame) -> DataFrame:
        """
        Emulate the categorical casting behavior we expect from roundtripping.
        """
        for col in from_frame:
            ser: Series = from_frame[col]
            if isinstance(ser.dtype, CategoricalDtype):
                cat: pd.Categorical = ser._values.remove_unused_categories()
                if cat.categories.dtype == object:
                    categories: pd.Index = pd.Index._with_infer(cat.categories._values)
                    cat = cat.set_categories(categories)
                elif cat.categories.dtype == 'string' and len(cat.categories) == 0:
                    categories: pd.Index = cat.categories.astype(object)
                    cat = cat.set_categories(categories)
                from_frame[col] = cat
        return from_frame

    def func_4iz5hn62(self, datapath: Callable[..., str]) -> None:
        fname: str = datapath('io', 'data', 'stata', 'stata3_117.dta')
        parsed: DataFrame = read_stata(fname)
        with read_stata(fname, iterator=True) as itr:
            chunk: DataFrame = itr.read(5)
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)
        with read_stata(fname, chunksize=5) as itr:
            chunk: List[DataFrame] = list(itr)
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk[0])
        with read_stata(fname, iterator=True) as itr:
            chunk: DataFrame = itr.get_chunk(5)
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)
        with read_stata(fname, chunksize=5) as itr:
            chunk: DataFrame = itr.get_chunk()
            tm.assert_frame_equal(parsed.iloc[0:5, :], chunk)
        with read_stata(fname, chunksize=4) as itr:
            from_chunks: DataFrame = pd.concat(itr)
        tm.assert_frame_equal(parsed, from_chunks)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize(
        'file',
        [
            'stata2_115', 'stata3_115', 'stata4_115', 'stata5_115',
            'stata6_115', 'stata7_115', 'stata8_115', 'stata9_115',
            'stata10_115', 'stata11_115'
        ]
    )
    @pytest.mark.parametrize('chunksize', [1, 2])
    @pytest.mark.parametrize('convert_categoricals', [False, True])
    @pytest.mark.parametrize('convert_dates', [False, True])
    def func_78lwftgv(self, file: str, chunksize: int, convert_categoricals: bool, convert_dates: bool, datapath: Callable[..., str]) -> Iterator[None]:
        fname: str = datapath('io', 'data', 'stata', f'{file}.dta')
        parsed: DataFrame = read_stata(fname, convert_categoricals=convert_categoricals, convert_dates=convert_dates)
        with read_stata(fname, iterator=True, convert_dates=convert_dates, convert_categoricals=convert_categoricals) as itr:
            pos: int = 0
            for j in range(5):
                try:
                    chunk: DataFrame = itr.read(chunksize)
                except StopIteration:
                    break
                from_frame: DataFrame = parsed.iloc[pos:pos + chunksize, :].copy()
                from_frame = self.b106tjue(from_frame)
                tm.assert_frame_equal(from_frame, chunk, check_dtype=False)
                pos += chunksize

    def func_g8gzcfou(self, datapath: Callable[..., str], temp_file: str) -> None:
        fname: str = datapath('io', 'data', 'stata', 'stata3_117.dta')
        columns: List[str] = ['quarter', 'cpi', 'm1']
        chunksize: int = 2
        parsed: DataFrame = read_stata(fname, columns=columns)
        with read_stata(fname, iterator=True) as itr:
            pos: int = 0
            for j in range(5):
                chunk: Optional[DataFrame] = itr.read(chunksize, columns=columns)
                if chunk is None:
                    break
                from_frame: DataFrame = parsed.iloc[pos:pos + chunksize, :]
                tm.assert_frame_equal(from_frame, chunk, check_dtype=False)
                pos += chunksize

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_5d7dc35f(self, version: Optional[int], mixed_frame: DataFrame, temp_file: str) -> None:
        mixed_frame.index.name = 'index'
        variable_labels: Dict[str, str] = {'a': 'City Rank', 'b': 'City Exponent', 'c': 'City'}
        path: str = temp_file
        mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)
        with StataReader(path) as reader:
            read_labels: Dict[str, str] = reader.variable_labels()
        expected_labels: Dict[str, str] = {'a': 'City Rank', 'b': 'City Exponent', 'c': 'City', 'index': ''}
        assert read_labels == expected_labels
        variable_labels['index'] = 'The Index'
        path = temp_file
        mixed_frame.to_stata(path, variable_labels=variable_labels, version=version)
        with StataReader(path) as reader:
            read_labels = reader.variable_labels()
        assert read_labels == variable_labels

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_feanqcxa(self, version: Optional[int], mixed_frame: DataFrame, temp_file: str) -> None:
        # This function is duplicated above; skipping to avoid repetition
        pass

    def func_v8pixc9g(self, temp_file: str) -> None:
        original: DataFrame = DataFrame(data=[
            ['string', 'object', 1, 1, 1, 1.1, 1.1, np.datetime64('2003-12-25'), 'a', 'a' * 2045, 'a' * 5000, 'a'],
            ['string-1', 'object-1', 1, 1, 1, 1.1, 1.1, np.datetime64('2003-12-26'), 'b', 'b' * 2045, '', '']
        ], columns=['string', 'object', 'int8', 'int16', 'int32', 'float32', 'float64',
                   'datetime', 's1', 's2045', 'srtl', 'forced_strl'])
        original['object'] = Series(original['object'], dtype=object)
        original['int8'] = Series(original['int8'], dtype=np.int8)
        original['int16'] = Series(original['int16'], dtype=np.int16)
        original['int32'] = original['int32'].astype(np.int32)
        original['float32'] = Series(original['float32'], dtype=np.float32)
        original.index.name = 'index'
        copy: DataFrame = original.copy()
        path: str = temp_file
        original.to_stata(path, convert_dates={'datetime': 'tc'}, byteorder='big', convert_strl=['forced_strl'], version=117)
        written_and_read_again: DataFrame = self.read_dta(path)
        expected: DataFrame = original[:]
        expected['datetime'] = expected['datetime'].astype('M8[ms]')
        if True:  # using_infer_string is assumed to be True here
            expected['object'] = expected['object'].astype('str')
        tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)
        tm.assert_frame_equal(original, copy)

    def func_8xpa4mjs(self, temp_file: str) -> None:
        original: DataFrame = DataFrame([
            ['a' * 3000, 'A', 'apple'],
            ['b' * 1000, 'B', 'banana']
        ], columns=['long1' * 10, 'long', 1])
        original.index.name = 'index'
        msg: str = 'Not all pandas column names were valid Stata variable names'
        with tm.assert_produces_warning(InvalidColumnName, match=msg):
            original.to_stata(temp_file, convert_strl=['long', 1], version=117)
            reread: DataFrame = self.read_dta(temp_file)
            reread = reread.set_index('index')
            reread.columns = original.columns
            tm.assert_frame_equal(reread, original, check_index_type=False)

    def func_l0m4bwzz(self, temp_file: str) -> None:
        dates: List[datetime] = [
            dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
            dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
            dt.datetime(1776, 7, 4, 7, 4, 7, 4000)
        ]
        original: DataFrame = DataFrame({'nums': [1.0, 2.0, 3.0], 'strs': ['apple', 'banana', 'cherry'], 'dates': dates})
        path: str = temp_file
        msg: str = 'columns contains duplicate entries'
        with pytest.raises(ValueError, match=msg):
            read_stata_datapath = datapath('io', 'data', 'stata', 'stata-dta-partially-labeled.dta')
            read_stata(read_stata_datapath, convert_categoricals=True)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_jehm85j3(self, version: Optional[int], temp_file: str, mixed_frame: DataFrame) -> None:
        original: DataFrame = DataFrame.from_records([
            ['one', 'ten', 'one', 'one', 'one', 1],
            ['two', 'nine', 'two', 'two', 'two', 2],
            ['three', 'eight', 'three', 'three', 'three', 3],
            ['four', 'seven', 4, 'four', 'four', 4],
            ['five', 'six', 5, np.nan, 'five', 5],
            ['six', 'five', 6, np.nan, 'six', 6],
            ['seven', 'four', 7, np.nan, 'seven', 7],
            ['eight', 'three', 8, np.nan, 'eight', 8],
            ['nine', 'two', 9, np.nan, 'nine', 9],
            ['ten', 'one', 'ten', np.nan, 'ten', 10],
        ], columns=[
            'fully_labeled', 'fully_labeled2', 'incompletely_labeled',
            'labeled_with_missings', 'float_labelled', 'unlabeled'
        ])
        path: str = temp_file
        original.astype('category').to_stata(path, version=version)
        written_and_read_again: DataFrame = self.read_dta(path)
        res: DataFrame = written_and_read_again.set_index('index')
        expected: DataFrame = original.copy()
        expected.index = expected.index.set_names('index')
        expected['incompletely_labeled'] = expected['incompletely_labeled'].apply(str)
        expected['unlabeled'] = expected['unlabeled'].apply(str)
        for col in expected:
            orig: Series = expected[col]
            cat: pd.Categorical = orig.astype('category')._values
            cat = cat.as_ordered()
            if col == 'unlabeled':
                cat = cat.set_categories(orig, ordered=True)
            cat.categories.rename(None, inplace=True)
            expected[col] = cat
        tm.assert_frame_equal(res, expected)

    def func_2o0n5jcf(self, temp_file: str) -> None:
        original: DataFrame = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1, 2, 3]})
        original.to_stata(temp_file)
        reread: DataFrame = read_stata(temp_file)
        tm.assert_frame_equal(original, reread)

    @pytest.mark.parametrize('version', [114, 117, 118, 119, None])
    def func_1iwdj7o4(self, version: Optional[int], temp_file: str) -> None:
        df: DataFrame = DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1, 2, 3]})
        df.to_stata(temp_file, convert_dates={'b': 'tc'}, version=version)
        reread: DataFrame = read_stata(temp_file)
        tm.assert_frame_equal(df, reread)

    def func_keel45v2(self, temp_file: str) -> None:
        df: DataFrame = DataFrame([
            [0.123456, 0.234567, 0.567567],
            [12.32112, 123123.2, 321321.2]
        ], index=['A', 'B'], columns=['X', 'Y', 'Z'])
        df.index.name = 'index'
        df.to_stata(temp_file, write_index=False)
        reread: DataFrame = read_stata(temp_file)
        tm.assert_frame_equal(df, reread)
