from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import PyperclipException, PyperclipWindowsException
import pandas as pd
from pandas import NA, DataFrame, Series, get_option, read_clipboard
import pandas._testing as tm
from pandas.io.clipboard import CheckedCall, _stringifyText, init_qt_clipboard
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

def build_kwargs(sep: Union[str, Literal['default'], None], excel: Union[bool, None, Literal['default']]) -> Dict[str, Any]: ...

@pytest.fixture(params=['delims', 'utf8', 'utf16', 'string', 'long', 'nonascii', 'colwidth', 'mixed', 'float', 'int'])
def df(request: Any) -> DataFrame: ...

@pytest.fixture
def mock_ctypes(monkeypatch: Any) -> Iterator[None]: ...

@pytest.mark.usefixtures('mock_ctypes')
def test_checked_call_with_bad_call(monkeypatch: Any) -> None: ...

@pytest.mark.usefixtures('mock_ctypes')
def test_checked_call_with_valid_call(monkeypatch: Any) -> None: ...

@pytest.mark.parametrize('text', ['String_test', True, 1, 1.0, 1j])
def test_stringify_text(text: Any) -> None: ...

@pytest.fixture
def set_pyqt_clipboard(monkeypatch: Any) -> Iterator[None]: ...

@pytest.fixture
def clipboard(qapp: Any) -> Iterator[Any]: ...

@pytest.mark.single_cpu
@pytest.mark.clipboard
@pytest.mark.usefixtures('set_pyqt_clipboard')
@pytest.mark.usefixtures('clipboard')
class TestClipboard:
    @pytest.mark.parametrize('sep', [None, '\t', ',', '|'])
    @pytest.mark.parametrize('encoding', [None, 'UTF-8', 'utf-8', 'utf8'])
    def test_round_trip_frame_sep(self, df: DataFrame, sep: Optional[str], encoding: Optional[str]) -> None: ...

    def test_round_trip_frame_string(self, df: DataFrame) -> None: ...

    def test_excel_sep_warning(self, df: DataFrame) -> None: ...

    def test_copy_delim_warning(self, df: DataFrame) -> None: ...

    @pytest.mark.parametrize('sep', ['\t', None, 'default'])
    @pytest.mark.parametrize('excel', [True, None, 'default'])
    def test_clipboard_copy_tabs_default(self, sep: Union[str, None, Literal['default']], excel: Union[bool, None, Literal['default']], df: DataFrame, clipboard: Any) -> None: ...

    @pytest.mark.parametrize('sep', [None, 'default'])
    def test_clipboard_copy_strings(self, sep: Union[str, None, Literal['default']], df: DataFrame) -> None: ...

    def test_read_clipboard_infer_excel(self, clipboard: Any) -> None: ...

    def test_infer_excel_with_nulls(self, clipboard: Any) -> None: ...

    @pytest.mark.parametrize('multiindex', [('\n'.join(['\t\t\tcol1\tcol2', 'A\t0\tTrue\t1\tred', 'A\t1\tTrue\t\tblue', 'B\t0\tFalse\t2\tgreen']), [['A', 'A', 'B'], [0, 1, 0], [True, True, False]]), ('\n'.join(['\t\tcol1\tcol2', 'A\t0\t1\tred', 'A\t1\t\tblue', 'B\t0\t2\tgreen']), [['A', 'A', 'B'], [0, 1, 0]])])
    def test_infer_excel_with_multiindex(self, clipboard: Any, multiindex: Tuple[str, List[List[Any]]]) -> None: ...

    def test_invalid_encoding(self, df: DataFrame) -> None: ...

    @pytest.mark.parametrize('data', ['👍...', 'Ωœ∑`...', 'abcd...'])
    def test_raw_roundtrip(self, data: str) -> None: ...

    @pytest.mark.parametrize('engine', ['c', 'python'])
    def test_read_clipboard_dtype_backend(self, clipboard: Any, string_storage: str, dtype_backend: str, engine: str, using_infer_string: bool) -> None: ...

    def test_invalid_dtype_backend(self) -> None: ...