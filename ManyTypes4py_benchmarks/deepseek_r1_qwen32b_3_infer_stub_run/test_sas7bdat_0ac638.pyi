from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pandas.io.sas.sas7bdat import SAS7BDATReader

@pytest.fixture
def dirpath(datapath: Callable[[str], Path]) -> Path:
    ...

@pytest.fixture
def data_test_ix(request: Any, dirpath: Path) -> Tuple[pd.DataFrame, List[int]]:
    ...

class TestSAS7BDAT:
    def test_from_file(self, dirpath: Path, data_test_ix: Tuple[pd.DataFrame, List[int]]) -> None:
        ...

    def test_from_buffer(self, dirpath: Path, data_test_ix: Tuple[pd.DataFrame, List[int]]) -> None:
        ...

    def test_from_iterator(self, dirpath: Path, data_test_ix: Tuple[pd.DataFrame, List[int]]) -> None:
        ...

    def test_path_pathlib(self, dirpath: Path, data_test_ix: Tuple[pd.DataFrame, List[int]]) -> None:
        ...

    @pytest.mark.parametrize('chunksize', (3, 5, 10, 11))
    @pytest.mark.parametrize('k', range(1, 17))
    def test_iterator_loop(self, dirpath: Path, k: int, chunksize: int) -> None:
        ...

    def test_iterator_read_too_much(self, dirpath: Path) -> None:
        ...

def test_encoding_options(datapath: Callable[[str], Path]) -> None:
    ...

def test_encoding_infer(datapath: Callable[[str], Path]) -> None:
    ...

def test_productsales(datapath: Callable[[str], Path]) -> None:
    ...

def test_12659(datapath: Callable[[str], Path]) -> None:
    ...

def test_airline(datapath: Callable[[str], Path]) -> None:
    ...

@pytest.mark.skipif(WASM, reason='Pyodide/WASM has 32-bitness')
def test_date_time(datapath: Callable[[str], Path]) -> None:
    ...

@pytest.mark.parametrize('column', ['WGT', 'CYL'])
def test_compact_numerical_values(datapath: Callable[[str], Path], column: str) -> None:
    ...

def test_many_columns(datapath: Callable[[str], Path]) -> None:
    ...

def test_inconsistent_number_of_rows(datapath: Callable[[str], Path]) -> None:
    ...

def test_zero_variables(datapath: Callable[[str], Path]) -> None:
    ...

@pytest.mark.parametrize('encoding', [None, 'utf8'])
def test_zero_rows(datapath: Callable[[str], Path], encoding: Optional[str]) -> None:
    ...

def test_corrupt_read(datapath: Callable[[str], Path]) -> None:
    ...

@pytest.mark.xfail(WASM, reason='failing with currently set tolerances on WASM')
def test_max_sas_date(datapath: Callable[[str], Path]) -> None:
    ...

@pytest.mark.xfail(WASM, reason='failing with currently set tolerances on WASM')
def test_max_sas_date_iterator(datapath: Callable[[str], Path]) -> None:
    ...

@pytest.mark.skipif(WASM, reason='Pyodide/WASM has 32-bitness')
def test_null_date(datapath: Callable[[str], Path]) -> None:
    ...

def test_meta2_page(datapath: Callable[[str], Path]) -> None:
    ...

@pytest.mark.parametrize('test_file, override_offset, override_value, expected_msg', [('test2.sas7bdat', 65536 + 55229, 128 | 15, 'Out of bounds'), ('test2.sas7bdat', 65536 + 55229, 16, 'unknown control byte'), ('test3.sas7bdat', 118170, 184, 'Out of bounds')])
def test_rle_rdc_exceptions(datapath: Callable[[str], Path], test_file: str, override_offset: int, override_value: int, expected_msg: str) -> None:
    ...

def test_0x40_control_byte(datapath: Callable[[str], Path]) -> None:
    ...

def test_0x00_control_byte(datapath: Callable[[str], Path]) -> None:
    ...