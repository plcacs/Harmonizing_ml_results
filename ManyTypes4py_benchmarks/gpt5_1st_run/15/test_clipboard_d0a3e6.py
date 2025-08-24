from textwrap import dedent
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Literal

import numpy as np
import pytest
from pandas.errors import PyperclipException, PyperclipWindowsException
import pandas as pd
from pandas import NA, DataFrame, Series, get_option, read_clipboard
import pandas._testing as tm
from pandas.io.clipboard import CheckedCall, _stringifyText, init_qt_clipboard


def build_kwargs(sep: Union[str, None, Literal['default']], excel: Union[bool, None, Literal['default']]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if excel != 'default':
        kwargs['excel'] = excel
    if sep != 'default':
        kwargs['sep'] = sep
    return kwargs


@pytest.fixture(params=['delims', 'utf8', 'utf16', 'string', 'long', 'nonascii', 'colwidth', 'mixed', 'float', 'int'])
def df(request: pytest.FixtureRequest) -> DataFrame:
    data_type = request.param
    if data_type == 'delims':
        return DataFrame({'a': ['"a,\t"b|c', 'd\tef`'], 'b': ["hi'j", "k''lm"]})
    elif data_type == 'utf8':
        return DataFrame({'a': ['µasd', 'Ωœ∑`'], 'b': ['øπ∆˚¬', 'œ∑`®']})
    elif data_type == 'utf16':
        return DataFrame({'a': ['👍👍', '👍👍'], 'b': ['abc', 'def']})
    elif data_type == 'string':
        return DataFrame(np.array([f'i-{i}' for i in range(15)]).reshape(5, 3), columns=list('abc'))
    elif data_type == 'long':
        max_rows: int = get_option('display.max_rows')
        return DataFrame(np.random.default_rng(2).integers(0, 10, size=(max_rows + 1, 3)), columns=list('abc'))
    elif data_type == 'nonascii':
        return DataFrame({'en': 'in English'.split(), 'es': 'en español'.split()})
    elif data_type == 'colwidth':
        _cw: int = get_option('display.max_colwidth') + 1
        return DataFrame(np.array(['x' * _cw for _ in range(15)]).reshape(5, 3), columns=list('abc'))
    elif data_type == 'mixed':
        return DataFrame({'a': np.arange(1.0, 6.0) + 0.01, 'b': np.arange(1, 6).astype(np.int64), 'c': list('abcde')})
    elif data_type == 'float':
        return DataFrame(np.random.default_rng(2).random((5, 3)), columns=list('abc'))
    elif data_type == 'int':
        return DataFrame(np.random.default_rng(2).integers(0, 10, (5, 3)), columns=list('abc'))
    else:
        raise ValueError


@pytest.fixture
def mock_ctypes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    Mocks WinError to help with testing the clipboard.
    """

    def _mock_win_error() -> str:
        return 'Window Error'
    with monkeypatch.context() as m:
        m.setattr('ctypes.WinError', _mock_win_error, raising=False)
        yield


@pytest.mark.usefixtures('mock_ctypes')
def test_checked_call_with_bad_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Give CheckCall a function that returns a falsey value and
    mock get_errno so it returns false so an exception is raised.
    """

    def _return_false() -> bool:
        return False
    monkeypatch.setattr('pandas.io.clipboard.get_errno', lambda: True)
    msg = f'Error calling {_return_false.__name__} \\(Window Error\\)'
    with pytest.raises(PyperclipWindowsException, match=msg):
        CheckedCall(_return_false)()


@pytest.mark.usefixtures('mock_ctypes')
def test_checked_call_with_valid_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Give CheckCall a function that returns a truthy value and
    mock get_errno so it returns true so an exception is not raised.
    The function should return the results from _return_true.
    """

    def _return_true() -> bool:
        return True
    monkeypatch.setattr('pandas.io.clipboard.get_errno', lambda: False)
    checked_call = CheckedCall(_return_true)
    assert checked_call() is True


@pytest.mark.parametrize('text', ['String_test', True, 1, 1.0, 1j])
def test_stringify_text(text: object) -> None:
    valid_types = (str, int, float, bool)
    if isinstance(text, valid_types):
        result = _stringifyText(text)  # type: ignore[arg-type]
        assert result == str(text)
    else:
        msg = f'only str, int, float, and bool values can be copied to the clipboard, not {type(text).__name__}'
        with pytest.raises(PyperclipException, match=msg):
            _stringifyText(text)  # type: ignore[arg-type]


@pytest.fixture
def set_pyqt_clipboard(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    qt_cut, qt_paste = init_qt_clipboard()
    with monkeypatch.context() as m:
        m.setattr(pd.io.clipboard, 'clipboard_set', qt_cut)
        m.setattr(pd.io.clipboard, 'clipboard_get', qt_paste)
        yield


@pytest.fixture
def clipboard(qapp: Any) -> Iterator[Any]:
    clip = qapp.clipboard()
    yield clip
    clip.clear()


@pytest.mark.single_cpu
@pytest.mark.clipboard
@pytest.mark.usefixtures('set_pyqt_clipboard')
@pytest.mark.usefixtures('clipboard')
class TestClipboard:

    @pytest.mark.parametrize('sep', [None, '\t', ',', '|'])
    @pytest.mark.parametrize('encoding', [None, 'UTF-8', 'utf-8', 'utf8'])
    def test_round_trip_frame_sep(self, df: DataFrame, sep: Optional[str], encoding: Optional[str]) -> None:
        df.to_clipboard(excel=None, sep=sep, encoding=encoding)
        result = read_clipboard(sep=sep or '\t', index_col=0, encoding=encoding)
        tm.assert_frame_equal(df, result)

    def test_round_trip_frame_string(self, df: DataFrame) -> None:
        df.to_clipboard(excel=False, sep=None)
        result = read_clipboard()
        assert df.to_string() == result.to_string()
        assert df.shape == result.shape

    def test_excel_sep_warning(self, df: DataFrame) -> None:
        with tm.assert_produces_warning(UserWarning, match='to_clipboard in excel mode requires a single character separator.', check_stacklevel=False):
            df.to_clipboard(excel=True, sep='\\t')

    def test_copy_delim_warning(self, df: DataFrame) -> None:
        with tm.assert_produces_warning(UserWarning, match='ignores the sep argument'):
            df.to_clipboard(excel=False, sep='\t')

    @pytest.mark.parametrize('sep', ['\t', None, 'default'])
    @pytest.mark.parametrize('excel', [True, None, 'default'])
    def test_clipboard_copy_tabs_default(self, sep: Union[str, None, Literal['default']], excel: Union[bool, None, Literal['default']], df: DataFrame, clipboard: Any) -> None:
        kwargs = build_kwargs(sep, excel)
        df.to_clipboard(**kwargs)
        assert clipboard.text() == df.to_csv(sep='\t')

    @pytest.mark.parametrize('sep', [None, 'default'])
    def test_clipboard_copy_strings(self, sep: Union[None, Literal['default']], df: DataFrame) -> None:
        kwargs = build_kwargs(sep, False)
        df.to_clipboard(**kwargs)
        result = read_clipboard(sep='\\s+')
        assert result.to_string() == df.to_string()
        assert df.shape == result.shape

    def test_read_clipboard_infer_excel(self, clipboard: Any) -> None:
        clip_kwargs: Dict[str, Any] = {'engine': 'python'}
        text = dedent('\n            John James\tCharlie Mingus\n            1\t2\n            4\tHarry Carney\n            '.strip())
        clipboard.setText(text)
        df = read_clipboard(**clip_kwargs)
        assert df.iloc[1, 1] == 'Harry Carney'
        text = dedent('\n            a\t b\n            1  2\n            3  4\n            '.strip())
        clipboard.setText(text)
        res = read_clipboard(**clip_kwargs)
        text = dedent('\n            a  b\n            1  2\n            3  4\n            '.strip())
        clipboard.setText(text)
        exp = read_clipboard(**clip_kwargs)
        tm.assert_frame_equal(res, exp)

    def test_infer_excel_with_nulls(self, clipboard: Any) -> None:
        text = 'col1\tcol2\n1\tred\n\tblue\n2\tgreen'
        clipboard.setText(text)
        df = read_clipboard()
        df_expected = DataFrame(data={'col1': [1, None, 2], 'col2': ['red', 'blue', 'green']})
        tm.assert_frame_equal(df, df_expected)

    @pytest.mark.parametrize('multiindex', [('\n'.join(['\t\t\tcol1\tcol2', 'A\t0\tTrue\t1\tred', 'A\t1\tTrue\t\tblue', 'B\t0\tFalse\t2\tgreen']), [['A', 'A', 'B'], [0, 1, 0], [True, True, False]]), ('\n'.join(['\t\tcol1\tcol2', 'A\t0\t1\tred', 'A\t1\t\tblue', 'B\t0\t2\tgreen']), [['A', 'A', 'B'], [0, 1, 0]])])
    def test_infer_excel_with_multiindex(self, clipboard: Any, multiindex: Tuple[str, List[List[Any]]]) -> None:
        clipboard.setText(multiindex[0])
        df = read_clipboard()
        df_expected = DataFrame(data={'col1': [1, None, 2], 'col2': ['red', 'blue', 'green']}, index=multiindex[1])
        tm.assert_frame_equal(df, df_expected)

    def test_invalid_encoding(self, df: DataFrame) -> None:
        msg = 'clipboard only supports utf-8 encoding'
        with pytest.raises(ValueError, match=msg):
            df.to_clipboard(encoding='ascii')
        with pytest.raises(NotImplementedError, match=msg):
            read_clipboard(encoding='ascii')

    @pytest.mark.parametrize('data', ['👍...', 'Ωœ∑`...', 'abcd...'])
    def test_raw_roundtrip(self, data: str) -> None:
        df = DataFrame({'data': [data]})
        df.to_clipboard()
        result = read_clipboard()
        tm.assert_frame_equal(df, result)

    @pytest.mark.parametrize('engine', ['c', 'python'])
    def test_read_clipboard_dtype_backend(self, clipboard: Any, string_storage: str, dtype_backend: str, engine: Literal['c', 'python'], using_infer_string: bool) -> None:
        if dtype_backend == 'pyarrow':
            pa = pytest.importorskip('pyarrow')
            if engine == 'c' and string_storage == 'pyarrow':
                string_dtype = pd.ArrowDtype(pa.large_string())
            else:
                string_dtype = pd.ArrowDtype(pa.string())
        else:
            string_dtype = pd.StringDtype(string_storage)
        text = 'a,b,c,d,e,f,g,h,i\nx,1,4.0,x,2,4.0,,True,False\ny,2,5.0,,,,,False,'
        clipboard.setText(text)
        with pd.option_context('mode.string_storage', string_storage):
            result = read_clipboard(sep=',', dtype_backend=dtype_backend, engine=engine)
        expected = DataFrame({'a': Series(['x', 'y'], dtype=string_dtype), 'b': Series([1, 2], dtype='Int64'), 'c': Series([4.0, 5.0], dtype='Float64'), 'd': Series(['x', None], dtype=string_dtype), 'e': Series([2, NA], dtype='Int64'), 'f': Series([4.0, NA], dtype='Float64'), 'g': Series([NA, NA], dtype='Int64'), 'h': Series([True, False], dtype='boolean'), 'i': Series([False, NA], dtype='boolean')})
        if dtype_backend == 'pyarrow':
            from pandas.arrays import ArrowExtensionArray
            expected = DataFrame({col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True)) for col in expected.columns})
            expected['g'] = ArrowExtensionArray(pa.array([None, None]))
        if using_infer_string:
            expected.columns = expected.columns.astype(pd.StringDtype(string_storage, na_value=np.nan))
        tm.assert_frame_equal(result, expected)

    def test_invalid_dtype_backend(self) -> None:
        msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
        with pytest.raises(ValueError, match=msg):
            read_clipboard(dtype_backend='numpy')