import io
import pytest
from typing import Any, Optional, List, Tuple, Union
from scripts import validate_docstrings

class BadDocstrings:
    def private_classes(self) -> None:
        ...

    def prefix_pandas(self) -> None:
        ...

    def redundant_import(self, paramx: Optional[Any] = None, paramy: Optional[Any] = None) -> None:
        ...

    def unused_import(self) -> None:
        ...

    def missing_whitespace_around_arithmetic_operator(self) -> None:
        ...

    def indentation_is_not_a_multiple_of_four(self) -> None:
        ...

    def missing_whitespace_after_comma(self) -> None:
        ...

    def write_array_like_with_hyphen_not_underscore(self) -> None:
        ...

    def leftover_files(self) -> None:
        ...

class TestValidator:
    def _import_path(self, klass: Optional[str] = None, func: Optional[str] = None) -> str:
        ...

    def test_bad_class(self, capsys: Any) -> None:
        ...

    @pytest.mark.parametrize('klass,func,msgs', [('BadDocstrings', 'private_classes', ('Private classes (NDFrame) should not be mentioned in public docstrings',)), ('BadDocstrings', 'prefix_pandas', ('pandas.Series.rename in `See Also` section does not need `pandas` prefix',)), ('BadDocstrings', 'redundant_import', ('Do not import numpy, as it is imported automatically',)), ('BadDocstrings', 'redundant_import', ('Do not import pandas, as it is imported automatically',)), ('BadDocstrings', 'unused_import', ("flake8 error: line 1, col 1: F401 'pandas as pdf' imported but unused",)), ('BadDocstrings', 'missing_whitespace_around_arithmetic_operator', ('flake8 error: line 1, col 2: E226 missing whitespace around arithmetic operator',)), ('BadDocstrings', 'indentation_is_not_a_multiple_of_four', ('flake8 error: line 2, col 3: E111 indentation is not a multiple of 4',)), ('BadDocstrings', 'missing_whitespace_after_comma', ("flake8 error: line 1, col 33: E231 missing whitespace after ','",)), ('BadDocstrings', 'write_array_like_with_hyphen_not_underscore', ("Use 'array-like' rather than 'array_like' in docstrings",))])
    def test_bad_docstrings(self, capsys: Any, klass: str, func: str, msgs: Tuple[str, ...]) -> None:
        ...

    def test_validate_all_ignore_deprecated(self, monkeypatch: Any) -> None:
        ...

    def test_validate_all_ignore_errors(self, monkeypatch: Any) -> None:
        ...

class TestApiItems:
    @property
    def api_doc(self) -> io.StringIO:
        ...

    @pytest.mark.parametrize('idx,name', [(0, 'itertools.cycle'), (1, 'itertools.count'), (2, 'itertools.chain'), (3, 'random.seed'), (4, 'random.randint')])
    def test_item_name(self, idx: int, name: str) -> None:
        ...

    @pytest.mark.parametrize('idx,func', [(0, 'cycle'), (1, 'count'), (2, 'chain'), (3, 'seed'), (4, 'randint')])
    def test_item_function(self, idx: int, func: str) -> None:
        ...

    @pytest.mark.parametrize('idx,section', [(0, 'Itertools'), (1, 'Itertools'), (2, 'Itertools'), (3, 'Random'), (4, 'Random')])
    def test_item_section(self, idx: int, section: str) -> None:
        ...

    @pytest.mark.parametrize('idx,subsection', [(0, 'Infinite'), (1, 'Infinite'), (2, 'Finite'), (3, 'All'), (4, 'All')])
    def test_item_subsection(self, idx: int, subsection: str) -> None:
        ...

class TestPandasDocstringClass:
    @pytest.mark.parametrize('name', ['pandas.Series.str.isdecimal', 'pandas.Series.str.islower'])
    def test_encode_content_write_to_file(self, name: str) -> None:
        ...

class TestMainFunction:
    def test_exit_status_for_main(self, monkeypatch: Any) -> None:
        ...

    def test_exit_status_errors_for_validate_all(self, monkeypatch: Any) -> None:
        ...

    def test_no_exit_status_noerrors_for_validate_all(self, monkeypatch: Any) -> None:
        ...

    def test_exit_status_for_validate_all_json(self, monkeypatch: Any) -> None:
        ...

    def test_errors_param_filters_errors(self, monkeypatch: Any) -> None:
        ...