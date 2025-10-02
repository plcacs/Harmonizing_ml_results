from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import PyperclipException, PyperclipWindowsException
import pandas as pd
from pandas import NA, DataFrame, Series, get_option, read_clipboard
import pandas._testing as tm
from pandas.io.clipboard import CheckedCall, _stringifyText, init_qt_clipboard
from typing import Any, Dict, Generator, List, Tuple

def build_kwargs(sep: str, excel: str) -> Dict[str, str]:
    kwargs: Dict[str, str] = {}
    if excel != 'default':
        kwargs['excel'] = excel
    if sep != 'default':
        kwargs['sep'] = sep
    return kwargs

@pytest.fixture(params=['delims', 'utf8', 'utf16', 'string', 'long', 'nonascii', 'colwidth', 'mixed', 'float', 'int'])
def df(request: Any) -> DataFrame:
    data_type = request.param
    if data_type == 'delims':
        return DataFrame({'a': ['"a,\t"b|c', 'd\tef`'], 'b': ["hi'j", "k''lm"]})
    elif data_type == 'utf8':
        return DataFrame({'a': ['µasd', 'Ωœ∑`'], 'b': ['øπ∆˚¬', 'œ∑`®']})
    # Remaining code for df fixture omitted for brevity

@pytest.fixture
def mock_ctypes(monkeypatch: Any) -> Generator:
    def _mock_win_error() -> str:
        return 'Window Error'
    with monkeypatch.context() as m:
        m.setattr('ctypes.WinError', _mock_win_error, raising=False)
        yield

# Remaining code for fixtures and tests omitted for brevity
