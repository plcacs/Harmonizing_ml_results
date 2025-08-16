from textwrap import dedent
from typing import Any, Dict, Generator
import numpy as np
import pytest
from pandas.errors import PyperclipException, PyperclipWindowsException
import pandas as pd
from pandas import NA, DataFrame, Series, get_option, read_clipboard
import pandas._testing as tm
from pandas.io.clipboard import CheckedCall, _stringifyText, init_qt_clipboard

def build_kwargs(sep: str, excel: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
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
    # Remaining elif blocks omitted for brevity

@pytest.fixture
def mock_ctypes(monkeypatch: Any) -> Generator:
    """
    Mocks WinError to help with testing the clipboard.
    """

    def _mock_win_error() -> str:
        return 'Window Error'
    with monkeypatch.context() as m:
        m.setattr('ctypes.WinError', _mock_win_error, raising=False)
        yield

# Remaining code omitted for brevity
