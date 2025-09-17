from __future__ import annotations
import os
from typing import Any, Callable, Dict, List, Tuple, Union
import pytest
from pandas.compat import HAS_PYARROW
from pandas.compat._optional import VERSIONS
from pandas import read_csv, read_table
import pandas._testing as tm


class BaseParser:
    engine: Union[str, None] = None
    low_memory: bool = True
    float_precision_choices: List[Any] = []

    def update_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = kwargs.copy()
        kwargs.update({'engine': self.engine, 'low_memory': self.low_memory})
        return kwargs

    def read_csv(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = self.update_kwargs(kwargs)
        return read_csv(*args, **kwargs)

    def read_csv_check_warnings(
        self,
        warn_type: Any,
        warn_msg: str,
        *args: Any,
        raise_on_extra_warnings: bool = True,
        check_stacklevel: bool = True,
        **kwargs: Any,
    ) -> Any:
        kwargs = self.update_kwargs(kwargs)
        with tm.assert_produces_warning(
            warn_type,
            match=warn_msg,
            raise_on_extra_warnings=raise_on_extra_warnings,
            check_stacklevel=check_stacklevel,
        ):
            return read_csv(*args, **kwargs)

    def read_table(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = self.update_kwargs(kwargs)
        return read_table(*args, **kwargs)

    def read_table_check_warnings(
        self,
        warn_type: Any,
        warn_msg: str,
        *args: Any,
        raise_on_extra_warnings: bool = True,
        **kwargs: Any,
    ) -> Any:
        kwargs = self.update_kwargs(kwargs)
        with tm.assert_produces_warning(warn_type, match=warn_msg, raise_on_extra_warnings=raise_on_extra_warnings):
            return read_table(*args, **kwargs)


class CParser(BaseParser):
    engine: str = 'c'
    float_precision_choices: List[Any] = [None, 'high', 'round_trip']


class CParserHighMemory(CParser):
    low_memory: bool = False


class CParserLowMemory(CParser):
    low_memory: bool = True


class PythonParser(BaseParser):
    engine: str = 'python'
    float_precision_choices: List[Any] = [None]


class PyArrowParser(BaseParser):
    engine: str = 'pyarrow'
    float_precision_choices: List[Any] = [None]


@pytest.fixture
def csv_dir_path(datapath: Callable[..., str]) -> str:
    """
    The directory path to the data files needed for parser tests.
    """
    return datapath('io', 'parser', 'data')


@pytest.fixture
def csv1(datapath: Callable[..., str]) -> str:
    """
    The path to the data file "test1.csv" needed for parser tests.
    """
    return os.path.join(datapath('io', 'data', 'csv'), 'test1.csv')


_cParserHighMemory = CParserHighMemory
_cParserLowMemory = CParserLowMemory
_pythonParser = PythonParser
_pyarrowParser = PyArrowParser
_py_parsers_only: List[Any] = [_pythonParser]
_c_parsers_only: List[Any] = [_cParserHighMemory, _cParserLowMemory]
_pyarrow_parsers_only: List[Any] = [
    pytest.param(
        _pyarrowParser, marks=[pytest.mark.single_cpu, pytest.mark.skipif(not HAS_PYARROW, reason='pyarrow is not installed')]
    )
]
_all_parsers: List[Any] = [*_c_parsers_only, *_py_parsers_only, *_pyarrow_parsers_only]
_py_parser_ids: List[str] = ['python']
_c_parser_ids: List[str] = ['c_high', 'c_low']
_pyarrow_parsers_ids: List[str] = ['pyarrow']
_all_parser_ids: List[str] = [*_c_parser_ids, *_py_parser_ids, *_pyarrow_parsers_ids]


@pytest.fixture(params=_all_parsers, ids=_all_parser_ids)
def all_parsers(request: pytest.FixtureRequest) -> BaseParser:
    """
    Fixture all of the CSV parsers.
    """
    parser: BaseParser = request.param()
    if parser.engine == 'pyarrow':
        pytest.importorskip('pyarrow', VERSIONS['pyarrow'])
        import pyarrow
        pyarrow.set_cpu_count(1)
    return parser


@pytest.fixture(params=_c_parsers_only, ids=_c_parser_ids)
def c_parser_only(request: pytest.FixtureRequest) -> BaseParser:
    """
    Fixture all of the CSV parsers using the C engine.
    """
    return request.param()


@pytest.fixture(params=_py_parsers_only, ids=_py_parser_ids)
def python_parser_only(request: pytest.FixtureRequest) -> BaseParser:
    """
    Fixture all of the CSV parsers using the Python engine.
    """
    return request.param()


@pytest.fixture(params=_pyarrow_parsers_only, ids=_pyarrow_parsers_ids)
def pyarrow_parser_only(request: pytest.FixtureRequest) -> BaseParser:
    """
    Fixture all of the CSV parsers using the Pyarrow engine.
    """
    return request.param()


def _get_all_parser_float_precision_combinations() -> Dict[str, List[Any]]:
    """
    Return all allowable parser and float precision
    combinations and corresponding ids.
    """
    params: List[Any] = []
    ids: List[str] = []
    for parser, parser_id in zip(_all_parsers, _all_parser_ids):
        if hasattr(parser, 'values'):
            parser = parser.values[0]
        for precision in parser.float_precision_choices:
            mark = (
                [pytest.mark.single_cpu, pytest.mark.skipif(not HAS_PYARROW, reason='pyarrow is not installed')]
                if parser.engine == 'pyarrow'
                else []
            )
            param = pytest.param((parser(), precision), marks=mark)
            params.append(param)
            ids.append(f'{parser_id}-{precision}')
    return {'params': params, 'ids': ids}


@pytest.fixture(
    params=_get_all_parser_float_precision_combinations()['params'],
    ids=_get_all_parser_float_precision_combinations()['ids'],
)
def all_parsers_all_precisions(request: pytest.FixtureRequest) -> Tuple[BaseParser, Any]:
    """
    Fixture for all allowable combinations of parser
    and float precision
    """
    return request.param


_utf_values: List[int] = [8, 16, 32]
_encoding_seps: List[str] = ['', '-', '_']
_encoding_prefixes: List[str] = ['utf', 'UTF']
_encoding_fmts: List[str] = [f'{prefix}{sep}{{0}}' for sep in _encoding_seps for prefix in _encoding_prefixes]


@pytest.fixture(params=_utf_values)
def utf_value(request: pytest.FixtureRequest) -> int:
    """
    Fixture for all possible integer values for a UTF encoding.
    """
    return request.param


@pytest.fixture(params=_encoding_fmts)
def encoding_fmt(request: pytest.FixtureRequest) -> str:
    """
    Fixture for all possible string formats of a UTF encoding.
    """
    return request.param


@pytest.fixture(
    params=[
        ('-1,0', -1.0),
        ('-1,2e0', -1.2),
        ('-1e0', -1.0),
        ('+1e0', 1.0),
        ('+1e+0', 1.0),
        ('+1e-1', 0.1),
        ('+,1e1', 1.0),
        ('+1,e0', 1.0),
        ('-,1e1', -1.0),
        ('-1,e0', -1.0),
        ('0,1', 0.1),
        ('1,', 1.0),
        (',1', 0.1),
        ('-,1', -0.1),
        ('1_,', 1.0),
        ('1_234,56', 1234.56),
        ('1_234,56e0', 1234.56),
        ('_', '_'),
        ('-_', '-_'),
        ('-_1', '-_1'),
        ('-_1e0', '-_1e0'),
        ('_1', '_1'),
        ('_1,', '_1,'),
        ('_1,_', '_1,_'),
        ('_1e0', '_1e0'),
        ('1,2e_1', '1,2e_1'),
        ('1,2e1_0', '1,2e1_0'),
        ('1,_2', '1,_2'),
        (',1__2', ',1__2'),
        (',1e', ',1e'),
        ('-,1e', '-,1e'),
        ('1_000,000_000', '1_000,000_000'),
        ('1,e1_2', '1,e1_2'),
        ('e11,2', 'e11,2'),
        ('1e11,2', '1e11,2'),
        ('1,2,2', '1,2,2'),
        ('1,2_1', '1,2_1'),
        ('1,2e-10e1', '1,2e-10e1'),
        ('--1,2', '--1,2'),
        ('1a_2,1', '1a_2,1'),
        ('1,2E-1', 0.12),
        ('1,2E1', 12.0),
    ]
)
def numeric_decimal(request: pytest.FixtureRequest) -> Tuple[str, Any]:
    """
    Fixture for all numeric formats which should get recognized. The first entry
    represents the value to read while the second represents the expected result.
    """
    return request.param


@pytest.fixture
def pyarrow_xfail(request: pytest.FixtureRequest) -> None:
    """
    Fixture that xfails a test if the engine is pyarrow.

    Use if failure is do to unsupported keywords or inconsistent results.
    """
    if 'all_parsers' in request.fixturenames:
        parser: BaseParser = request.getfixturevalue('all_parsers')
    elif 'all_parsers_all_precisions' in request.fixturenames:
        parser = request.getfixturevalue('all_parsers_all_precisions')[0]
    else:
        return
    if parser.engine == 'pyarrow':
        mark = pytest.mark.xfail(reason="pyarrow doesn't support this.")
        request.applymarker(mark)


@pytest.fixture
def pyarrow_skip(request: pytest.FixtureRequest) -> None:
    """
    Fixture that skips a test if the engine is pyarrow.

    Use if failure is do a parsing failure from pyarrow.csv.read_csv
    """
    if 'all_parsers' in request.fixturenames:
        parser: BaseParser = request.getfixturevalue('all_parsers')
    elif 'all_parsers_all_precisions' in request.fixturenames:
        parser = request.getfixturevalue('all_parsers_all_precisions')[0]
    else:
        return
    if parser.engine == 'pyarrow':
        pytest.skip(reason='https://github.com/apache/arrow/issues/38676')