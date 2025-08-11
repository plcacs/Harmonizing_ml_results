from contextlib import ExitStack, contextmanager
import pytest
import terminaltables
from faust.utils.terminal import tables
from mode.utils.mocks import Mock, patch
TABLE_DATA = [('foo', 1.0, 3.33, 6.66, 9.99), ('bar', 2.0, 2.34, 4.23, 3.33)]

def fh(isatty: bool=True) -> Mock:
    fh = Mock()
    fh.isatty.return_value = isatty
    return fh

@contextmanager
def mock_stdout(isatty: bool=True) -> typing.Generator:
    with patch('sys.stdout') as stdout:
        stdout.isatty.return_value = isatty
        yield

@contextmanager
def mock_logging(isatty: bool=True) -> typing.Generator:
    with patch('mode.utils.logging.LOG_ISATTY', isatty):
        yield

@pytest.mark.parametrize('target,contexts,kwargs,expected_tty', [(None, [mock_stdout(isatty=True)], {}, True), (None, [mock_stdout(isatty=True)], {'kw': 1}, True), (None, [mock_stdout(isatty=False)], {}, False), (None, [mock_stdout(isatty=False)], {'kw': 2}, False), (fh(isatty=True), [], {}, True), (fh(isatty=True), [], {'kw': 1}, True), (fh(isatty=False), [], {}, False), (fh(isatty=False), [], {'kw': 1}, False)])
def test_table(target: Union[bool, str], contexts: Any, kwargs: Any, expected_tty: Union[bool, str, typing.Callable[dict, bool], None]) -> None:
    with ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)
        with patch('faust.utils.terminal.tables._get_best_table_type') as _g:
            table = tables.table(TABLE_DATA, title='Title', target=target, **kwargs)
            _g.assert_called_with(expected_tty)
            _g.return_value.assert_called_with(TABLE_DATA, title='Title', **kwargs)
            assert table is _g.return_value.return_value

@pytest.mark.parametrize('tty,contexts,headers,expected_tty,expected_data', [(None, [mock_logging(isatty=True)], None, True, TABLE_DATA), (None, [mock_logging(isatty=True)], ['foo'], True, [['foo']] + TABLE_DATA), (None, [mock_logging(isatty=False)], None, False, TABLE_DATA), (None, [mock_logging(isatty=False)], ['f'], False, [['f']] + TABLE_DATA), (True, [], None, True, TABLE_DATA), (True, [], ['foo'], True, [['foo']] + TABLE_DATA), (False, [], None, False, TABLE_DATA), (False, [], ['foo'], False, [['foo']] + TABLE_DATA)])
def test_logtable(tty: Union[bool, None, bytes, typing.TextIO], contexts: str, headers: Union[str, None, dict], expected_tty: Union[int, str, None], expected_data: Union[int, str, None]) -> None:
    with ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)
        with patch('faust.utils.terminal.tables.table') as table:
            ret = tables.logtable(TABLE_DATA, title='Title', target=None, tty=tty, headers=headers)
            table.assert_called_with(expected_data, title='Title', target=None, tty=expected_tty)
            assert ret is table().table

@pytest.mark.parametrize('tty,expected_table_type', [(True, terminaltables.SingleTable), (False, terminaltables.AsciiTable)])
def test_get_best_table_type(tty: Union[str, list[str]], expected_table_type: Union[str, list[str]]) -> None:
    assert tables._get_best_table_type(tty) is expected_table_type

def test_table__default_tty() -> None:
    with patch('faust.utils.terminal.tables._get_best_table_type') as g:
        with patch('faust.utils.terminal.tables.isatty') as i:
            i.return_value = None
            tables.table({}, tty=None, title='foo')
            g.assert_called_once_with(False)