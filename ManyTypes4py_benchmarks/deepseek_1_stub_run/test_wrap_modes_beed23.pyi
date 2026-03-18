```python
import pytest
from hypothesis import given
from hypothesis import strategies as st
import isort
from isort import wrap_modes

def test_wrap_mode_interface() -> None: ...

def test_auto_saved() -> None: ...

def test_backslash_grid() -> None: ...

@pytest.mark.parametrize('include_trailing_comma', (False, True))
@pytest.mark.parametrize('line_length', (18, 19))
@pytest.mark.parametrize('multi_line_output', (4, 5))
def test_vertical_grid_size_near_line_length(
    multi_line_output: int,
    line_length: int,
    include_trailing_comma: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_backslash_grid(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_grid(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_hanging_indent(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@pytest.mark.parametrize('include_trailing_comma', (True, False))
def test_hanging_indent__with_include_trailing_comma__expect_same_result(
    include_trailing_comma: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_hanging_indent_with_parentheses(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_noqa(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_vertical(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_vertical_grid(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_vertical_grid_grouped(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_vertical_hanging_indent(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_vertical_hanging_indent_bracket(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...

@given(
    statement=st.text(),
    imports=st.lists(st.text()),
    white_space=st.text(),
    indent=st.text(),
    line_length=st.integers(),
    comments=st.lists(st.text()),
    line_separator=st.text(),
    comment_prefix=st.text(),
    include_trailing_comma=st.booleans(),
    remove_comments=st.booleans()
)
def test_fuzz_vertical_prefix_from_module_import(
    statement: str,
    imports: list[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: list[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None: ...
```