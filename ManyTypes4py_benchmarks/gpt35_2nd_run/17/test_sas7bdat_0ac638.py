from datetime import datetime
import io
import os
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from pandas.io.sas.sas7bdat import SAS7BDATReader
from pandas.errors import EmptyDataError
from pandas.compat._constants import IS64, WASM
from pandas._testing import tm
from typing import Tuple

@pytest.fixture
def dirpath(datapath: str) -> str:
    return datapath('io', 'sas', 'data')

@pytest.fixture(params=[(1, range(1, 16)), (2, [16])])
def data_test_ix(request: pytest.FixtureRequest, dirpath: str) -> Tuple[pd.DataFrame, Tuple[int]]:
    i, test_ix = request.param
    fname = os.path.join(dirpath, f'test_sas7bdat_{i}.csv')
    df = pd.read_csv(fname)
    epoch = datetime(1960, 1, 1)
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
    def test_from_file(self, dirpath: str, data_test_ix: Tuple[pd.DataFrame, Tuple[int]]) -> None:
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname = os.path.join(dirpath, f'test{k}.sas7bdat')
            df = pd.read_sas(fname, encoding='utf-8')
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    def test_from_buffer(self, dirpath: str, data_test_ix: Tuple[pd.DataFrame, Tuple[int]]) -> None:
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname = os.path.join(dirpath, f'test{k}.sas7bdat')
            with open(fname, 'rb') as f:
                byts = f.read()
            buf = io.BytesIO(byts)
            with pd.read_sas(buf, format='sas7bdat', iterator=True, encoding='utf-8') as rdr:
                df = rdr.read()
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    def test_from_iterator(self, dirpath: str, data_test_ix: Tuple[pd.DataFrame, Tuple[int]]) -> None:
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname = os.path.join(dirpath, f'test{k}.sas7bdat')
            with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
                df = rdr.read(2)
                tm.assert_frame_equal(df, expected.iloc[0:2, :])
                df = rdr.read(3)
                tm.assert_frame_equal(df, expected.iloc[2:5, :])

    @pytest.mark.slow
    def test_path_pathlib(self, dirpath: str, data_test_ix: Tuple[pd.DataFrame, Tuple[int]]) -> None:
        expected, test_ix = data_test_ix
        for k in test_ix:
            fname = Path(os.path.join(dirpath, f'test{k}.sas7bdat'))
            df = pd.read_sas(fname, encoding='utf-8')
            tm.assert_frame_equal(df, expected)

    @pytest.mark.slow
    @pytest.mark.parametrize('chunksize', (3, 5, 10, 11))
    @pytest.mark.parametrize('k', range(1, 17))
    def test_iterator_loop(self, dirpath: str, k: int, chunksize: int) -> None:
        fname = os.path.join(dirpath, f'test{k}.sas7bdat')
        with pd.read_sas(fname, chunksize=chunksize, encoding='utf-8') as rdr:
            y = 0
            for x in rdr:
                y += x.shape[0]
        assert y == rdr.row_count

    def test_iterator_read_too_much(self, dirpath: str) -> None:
        fname = os.path.join(dirpath, 'test1.sas7bdat')
        with pd.read_sas(fname, format='sas7bdat', iterator=True, encoding='utf-8') as rdr:
            d1 = rdr.read(rdr.row_count + 20)
        with pd.read_sas(fname, iterator=True, encoding='utf-8') as rdr:
            d2 = rdr.read(rdr.row_count + 20)
        tm.assert_frame_equal(d1, d2)

def test_encoding_options(datapath: str) -> None:
    fname = datapath('io', 'sas', 'data', 'test1.sas7bdat')
    df1 = pd.read_sas(fname)
    df2 = pd.read_sas(fname, encoding='utf-8')
    for col in df1.columns:
        try:
            df1[col] = df1[col].str.decode('utf-8')
        except AttributeError:
            pass
    tm.assert_frame_equal(df1, df2)
    with contextlib.closing(SAS7BDATReader(fname, convert_header_text=False)) as rdr:
        df3 = rdr.read()
    for x, y in zip(df1.columns, df3.columns):
        assert x == y.decode()

# Other test functions with type annotations omitted for brevity
