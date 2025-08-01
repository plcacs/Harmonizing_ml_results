#!/usr/bin/env python3
from __future__ import annotations
import contextlib
from datetime import datetime
import io
import os
from pathlib import Path
from typing import Any, Callable, Iterator, List, Tuple, Union

import numpy as np
import pytest
from pandas.compat._constants import IS64, WASM
from pandas.errors import EmptyDataError
import pandas as pd
import pandas._testing as tm
from pandas.io.sas.sas7bdat import SAS7BDATReader
from _pytest.fixtures import FixtureRequest

@pytest.fixture
def dirpath(datapath: Callable[..., str]) -> str:
    return datapath('io', 'sas', 'data')

@pytest.fixture(params=[(1, range(1, 16)), (2, [16])])
def data_test_ix(request: FixtureRequest, dirpath: str) -> Tuple[pd.DataFrame, Union[range, List[int]]]:
    i, test_ix = request.param  # type: int, Union[range, List[int]]
    fname: str = os.path.join(dirpath, f'test_sas7bdat_{i}.csv')
    df: pd.DataFrame = pd.read_csv(fname)
    epoch: datetime = datetime(1960, 1, 1)
    t1 = pd.to_timedelta(df['Column4'], unit='D')
    df['Column4'] = (epoch + t1).astype('M8[s]')
    t2 = pd.to_timedelta(df['Column12'], unit='D')
    df['Column12'] = (epoch + t2).astype('M8[s]')
    for k in range(df.shape[1]):
        col = df.iloc[:, k]
        if col.dtype == np.int64:
            df.isetitem(k, df.iloc[:, k].astype(np.float64))
    return (df, test_ix)

class TestSAS7BDAT:
    @pytest.mark.slow
    def test_from_file(self, dirpath: str, data_test_ix: Tuple[pd.DataFrame, Union[range, List[int]]]) -> None:
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname: str = os.path.join(dirpath, f'test{k}.sas7bdat')
            df: pd.DataFrame = pd.read_sas(fname, encoding='utf-8')
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    def test_from_buffer(self, dirpath: str, data_test_ix: Tuple[pd.DataFrame, Union[range, List[int]]]) -> None:
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname: str = os.path.join(dirpath, f'test{k}.sas7bdat')
            with open(fname, 'rb') as f:
                byts: bytes = f.read()
            buf: io.BytesIO = io.BytesIO(byts)
            with pd.read_sas(buf, format='sas7bdat', iterator=True, encoding='utf-8') as rdr:
                df: pd.DataFrame = rdr.read()
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    def test_from_iterator(self, dirpath: str, data_test_ix: Tuple[pd.DataFrame, Union[range, List[int]]]) -> None:
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname: str = os.path.join(dirpath, f'test{k}.sas7bdat')
            with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
                df: pd.DataFrame = rdr.read(2)
                tm.assert_frame_equal(df, expected.iloc[0:2, :])
                df = rdr.read(3)
                tm.assert_frame_equal(df, expected.iloc[2:5, :])

    @pytest.mark.slow
    def test_path_pathlib(self, dirpath: str, data_test_ix: Tuple[pd.DataFrame, Union[range, List[int]]]) -> None:
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname: Path = Path(os.path.join(dirpath, f'test{k}.sas7bdat'))
            df: pd.DataFrame = pd.read_sas(fname, encoding='utf-8')
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    @pytest.mark.parametrize('chunksize', (3, 5, 10, 11))
    @pytest.mark.parametrize('k', range(1, 17))
    def test_iterator_loop(self, dirpath: str, k: int, chunksize: int) -> None:
        fname: str = os.path.join(dirpath, f'test{k}.sas7bdat')
        with pd.read_sas(fname, chunksize=chunksize, encoding='utf-8') as rdr:
            y: int = 0
            for x in rdr:
                y += x.shape[0]
        assert y == rdr.row_count

    def test_iterator_read_too_much(self, dirpath: str) -> None:
        fname: str = os.path.join(dirpath, 'test1.sas7bdat')
        with pd.read_sas(fname, format='sas7bdat', iterator=True, encoding='utf-8') as rdr:
            d1: pd.DataFrame = rdr.read(rdr.row_count + 20)
        with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
            d2: pd.DataFrame = rdr.read(rdr.row_count + 20)
        tm.assert_frame_equal(d1, d2)

def test_encoding_options(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'test1.sas7bdat')
    df1: pd.DataFrame = pd.read_sas(fname)
    df2: pd.DataFrame = pd.read_sas(fname, encoding='utf-8')
    for col in df1.columns:
        try:
            df1[col] = df1[col].str.decode('utf-8')
        except AttributeError:
            pass
    tm.assert_frame_equal(df1, df2)
    with contextlib.closing(SAS7BDATReader(fname, convert_header_text=False)) as rdr:
        df3: pd.DataFrame = rdr.read()
    for x, y in zip(df1.columns, df3.columns):
        assert x == y.decode()

def test_encoding_infer(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'test1.sas7bdat')
    with pd.read_sas(fname, encoding='infer', iterator=True) as df1_reader:
        assert df1_reader.inferred_encoding == 'cp1252'
        df1: pd.DataFrame = df1_reader.read()
    with pd.read_sas(fname, encoding='cp1252', iterator=True) as df2_reader:
        df2: pd.DataFrame = df2_reader.read()
    tm.assert_frame_equal(df1, df2)

def test_productsales(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'productsales.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname, encoding='utf-8')
    fname = datapath('io', 'sas', 'data', 'productsales.csv')
    df0: pd.DataFrame = pd.read_csv(fname, parse_dates=['MONTH'])
    vn: List[str] = ['ACTUAL', 'PREDICT', 'QUARTER', 'YEAR']
    df0[vn] = df0[vn].astype(np.float64)
    df0['MONTH'] = df0['MONTH'].astype('M8[s]')
    tm.assert_frame_equal(df, df0)

def test_12659(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'test_12659.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname)
    fname = datapath('io', 'sas', 'data', 'test_12659.csv')
    df0: pd.DataFrame = pd.read_csv(fname)
    df0 = df0.astype(np.float64)
    tm.assert_frame_equal(df, df0)

def test_airline(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'airline.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname)
    fname = datapath('io', 'sas', 'data', 'airline.csv')
    df0: pd.DataFrame = pd.read_csv(fname)
    df0 = df0.astype(np.float64)
    tm.assert_frame_equal(df, df0)

@pytest.mark.skipif(WASM, reason='Pyodide/WASM has 32-bitness')
def test_date_time(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'datetime.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname)
    fname = datapath('io', 'sas', 'data', 'datetime.csv')
    df0: pd.DataFrame = pd.read_csv(fname, parse_dates=['Date1', 'Date2', 'DateTime', 'DateTimeHi', 'Taiw'])
    df[df.columns[3]] = df.iloc[:, 3].dt.round('us')
    df0['Date1'] = df0['Date1'].astype('M8[s]')
    df0['Date2'] = df0['Date2'].astype('M8[s]')
    df0['DateTime'] = df0['DateTime'].astype('M8[ms]')
    df0['Taiw'] = df0['Taiw'].astype('M8[s]')
    res = df0['DateTimeHi'].astype('M8[us]').dt.round('ms')
    df0['DateTimeHi'] = res.astype('M8[ms]')
    if not IS64:
        df0.loc[0, 'DateTimeHi'] += np.timedelta64(1, 'ms')
        df0.loc[[2, 3], 'DateTimeHi'] -= np.timedelta64(1, 'ms')
    tm.assert_frame_equal(df, df0)

@pytest.mark.parametrize('column', ['WGT', 'CYL'])
def test_compact_numerical_values(datapath: Callable[..., str], column: str) -> None:
    fname: str = datapath('io', 'sas', 'data', 'cars.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname, encoding='latin-1')
    result: pd.Series = df[column]
    expected: pd.Series = df[column].round()
    tm.assert_series_equal(result, expected, check_exact=True)

def test_many_columns(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'many_columns.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname, encoding='latin-1')
    fname = datapath('io', 'sas', 'data', 'many_columns.csv')
    df0: pd.DataFrame = pd.read_csv(fname, encoding='latin-1')
    tm.assert_frame_equal(df, df0)

def test_inconsistent_number_of_rows(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'load_log.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname, encoding='latin-1')
    assert len(df) == 2097

def test_zero_variables(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'zero_variables.sas7bdat')
    with pytest.raises(EmptyDataError, match='No columns to parse from file'):
        pd.read_sas(fname)

@pytest.mark.parametrize('encoding', [None, 'utf8'])
def test_zero_rows(datapath: Callable[..., str], encoding: Union[None, str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'zero_rows.sas7bdat')
    result: pd.DataFrame = pd.read_sas(fname, encoding=encoding)
    str_value: Union[bytes, str] = b'a' if encoding is None else 'a'
    expected: pd.DataFrame = pd.DataFrame([{'char_field': str_value, 'num_field': 1.0}]).iloc[:0]
    tm.assert_frame_equal(result, expected)

def test_corrupt_read(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'corrupt.sas7bdat')
    msg: str = "'SAS7BDATReader' object has no attribute 'row_count'"
    with pytest.raises(AttributeError, match=msg):
        pd.read_sas(fname)

@pytest.mark.xfail(WASM, reason='failing with currently set tolerances on WASM')
def test_max_sas_date(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'max_sas_date.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname, encoding='iso-8859-1')
    expected: pd.DataFrame = pd.DataFrame({
        'text': ['max', 'normal'],
        'dt_as_float': [253717747199.999, 1880323199.999],
        'dt_as_dt': np.array([datetime(9999, 12, 29, 23, 59, 59, 999000),
                              datetime(2019, 8, 1, 23, 59, 59, 999000)], dtype='M8[ms]'),
        'date_as_float': [2936547.0, 21762.0],
        'date_as_date': np.array([datetime(9999, 12, 29), datetime(2019, 8, 1)], dtype='M8[s]')
    }, columns=['text', 'dt_as_float', 'dt_as_dt', 'date_as_float', 'date_as_date'])
    if not IS64:
        expected.loc[:, 'dt_as_dt'] -= np.timedelta64(1, 'ms')
    tm.assert_frame_equal(df, expected)

@pytest.mark.xfail(WASM, reason='failing with currently set tolerances on WASM')
def test_max_sas_date_iterator(datapath: Callable[..., str]) -> None:
    col_order: List[str] = ['text', 'dt_as_float', 'dt_as_dt', 'date_as_float', 'date_as_date']
    fname: str = datapath('io', 'sas', 'data', 'max_sas_date.sas7bdat')
    results: List[pd.DataFrame] = []
    for df in pd.read_sas(fname, encoding='iso-8859-1', chunksize=1):  # type: pd.DataFrame
        df.reset_index(inplace=True, drop=True)
        results.append(df)
    expected: List[pd.DataFrame] = [
        pd.DataFrame({
            'text': ['max'],
            'dt_as_float': [253717747199.999],
            'dt_as_dt': np.array([datetime(9999, 12, 29, 23, 59, 59, 999000)], dtype='M8[ms]'),
            'date_as_float': [2936547.0],
            'date_as_date': np.array([datetime(9999, 12, 29)], dtype='M8[s]')
        }, columns=col_order),
        pd.DataFrame({
            'text': ['normal'],
            'dt_as_float': [1880323199.999],
            'dt_as_dt': np.array(['2019-08-01 23:59:59.999'], dtype='M8[ms]'),
            'date_as_float': [21762.0],
            'date_as_date': np.array(['2019-08-01'], dtype='M8[s]')
        }, columns=col_order)
    ]
    if not IS64:
        expected[0].loc[0, 'dt_as_dt'] -= np.timedelta64(1, 'ms')
        expected[1].loc[0, 'dt_as_dt'] -= np.timedelta64(1, 'ms')
    tm.assert_frame_equal(results[0], expected[0])
    tm.assert_frame_equal(results[1], expected[1])

@pytest.mark.skipif(WASM, reason='Pyodide/WASM has 32-bitness')
def test_null_date(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'dates_null.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname, encoding='utf-8')
    expected: pd.DataFrame = pd.DataFrame({
        'datecol': np.array([datetime(9999, 12, 29), np.datetime64('NaT')], dtype='M8[s]'),
        'datetimecol': np.array([datetime(9999, 12, 29, 23, 59, 59, 999000), np.datetime64('NaT')], dtype='M8[ms]')
    })
    if not IS64:
        expected.loc[0, 'datetimecol'] -= np.timedelta64(1, 'ms')
    tm.assert_frame_equal(df, expected)

def test_meta2_page(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', 'test_meta2_page.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname)
    assert len(df) == 1000

@pytest.mark.parametrize(
    'test_file, override_offset, override_value, expected_msg',
    [
        ('test2.sas7bdat', 65536 + 55229, 128 | 15, 'Out of bounds'),
        ('test2.sas7bdat', 65536 + 55229, 16, 'unknown control byte'),
        ('test3.sas7bdat', 118170, 184, 'Out of bounds')
    ]
)
def test_rle_rdc_exceptions(datapath: Callable[..., str],
                            test_file: str,
                            override_offset: int,
                            override_value: int,
                            expected_msg: str) -> None:
    """Errors in RLE/RDC decompression should propagate."""
    with open(datapath('io', 'sas', 'data', test_file), 'rb') as fd:
        data: bytearray = bytearray(fd.read())
    data[override_offset] = override_value
    with pytest.raises(Exception, match=expected_msg):
        pd.read_sas(io.BytesIO(data), format='sas7bdat')

def test_0x40_control_byte(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', '0x40controlbyte.sas7bdat')
    df: pd.DataFrame = pd.read_sas(fname, encoding='ascii')
    fname = datapath('io', 'sas', 'data', '0x40controlbyte.csv')
    df0: pd.DataFrame = pd.read_csv(fname, dtype='str')
    tm.assert_frame_equal(df, df0)

def test_0x00_control_byte(datapath: Callable[..., str]) -> None:
    fname: str = datapath('io', 'sas', 'data', '0x00controlbyte.sas7bdat.bz2')
    df: pd.DataFrame = next(pd.read_sas(fname, chunksize=11000))
    assert df.shape == (11000, 20)