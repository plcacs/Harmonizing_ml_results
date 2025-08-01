from typing import List

import pytest
from hypothesis import given, reject
from hypothesis import strategies as st
import isort
from isort import wrap_modes


def test_wrap_mode_interface() -> None:
    assert wrap_modes._wrap_mode_interface('statement', [], '', '', 80, [], '', '', True, True) == ''


def test_auto_saved() -> None:
    """hypothesis_auto tests cases that have been saved to ensure they run each test cycle"""
    assert wrap_modes.noqa(**{
        'comment_prefix': '-\U000bf82c\x0c\U0004608f\x10%',
        'comments': [],
        'imports': [],
        'include_trailing_comma': False,
        'indent': '0\x19',
        'line_length': -19659,
        'line_separator': '\x15\x0b\U00086494\x1d\U000e00a2\U000ee216\U0006708a\x03\x1f',
        'remove_comments': False,
        'statement': '\U00092452',
        'white_space': '\U000a7322\U000c20e3-\U0010eae4\x07\x14\U0007d486'
    }) == '\U00092452-\U000bf82c\x0c\U0004608f\x10% NOQA'
    assert wrap_modes.noqa(**{
        'comment_prefix': '\x12\x07\U0009e994🁣"\U000ae787\x0e',
        'comments': ['\x00\U0001ae99\U0005c3e7\U0004d08e', '\x1e', '', ''],
        'imports': ['*'],
        'include_trailing_comma': True,
        'indent': '',
        'line_length': 31492,
        'line_separator': '\U00071610\U0005bfbc',
        'remove_comments': False,
        'statement': '',
        'white_space': '\x08\x01ⷓ\x16%\U0006cd8c'
    }) == '*\x12\x07\U0009e994🁣"\U000ae787\x0e \x00\U0001ae99\U0005c3e7\U0004d08e \x1e  '
    assert wrap_modes.noqa(**{
        'comment_prefix': '  #',
        'comments': ['NOQA', 'THERE'],
        'imports': [],
        'include_trailing_comma': False,
        'indent': '0\x19',
        'line_length': -19659,
        'line_separator': '\n',
        'remove_comments': False,
        'statement': 'hi',
        'white_space': ' '
    }) == 'hi  # NOQA THERE'


def test_backslash_grid() -> None:
    """Tests the backslash_grid grid wrap mode, ensuring it matches formatting expectations.
    See: https://github.com/PyCQA/isort/issues/1434
    """
    code_in = (
        '\nfrom kopf.engines import loggers, posting\n'
        'from kopf.reactor import causation, daemons, effects, handling, lifecycles, registries\n'
        'from kopf.storage import finalizers, states\n'
        'from kopf.structs import (bodies, configuration, containers, diffs,\n'
        '                          handlers as handlers_, patches, resources)\n'
    )
    expected = (
        '\nfrom kopf.engines import loggers, posting\n'
        'from kopf.reactor import causation, daemons, effects, handling, lifecycles, registries\n'
        'from kopf.storage import finalizers, states\n'
        'from kopf.structs import bodies, configuration, containers, diffs, \\\n'
        '                         handlers as handlers_, patches, resources\n'
    )
    assert isort.code(code_in, multi_line_output=11, line_length=88, combine_as_imports=True) == expected


@pytest.mark.parametrize('include_trailing_comma', (False, True))
@pytest.mark.parametrize('line_length', (18, 19))
@pytest.mark.parametrize('multi_line_output', (4, 5))
def test_vertical_grid_size_near_line_length(
    multi_line_output: int,
    line_length: int,
    include_trailing_comma: bool
) -> None:
    separator: str = ' '
    if multi_line_output == 4 and line_length < 19 + int(include_trailing_comma) or (
        multi_line_output != 4 and line_length < 18 + int(include_trailing_comma)
    ):
        separator = '\n    '
    test_input: str = f'from foo import (\n    aaaa, bbb,{separator}ccc'
    if include_trailing_comma:
        test_input += ','
    if multi_line_output != 4:
        test_input += '\n'
    test_input += ')\n'
    assert isort.code(test_input, multi_line_output=multi_line_output, line_length=line_length, include_trailing_comma=include_trailing_comma) == test_input


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.backslash_grid(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.grid(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.hanging_indent(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


@pytest.mark.parametrize('include_trailing_comma', (True, False))
def test_hanging_indent__with_include_trailing_comma__expect_same_result(include_trailing_comma: bool) -> None:
    result: str = isort.wrap_modes.hanging_indent(
        statement='from datetime import ',
        imports=['datetime', 'time', 'timedelta', 'timezone', 'tzinfo'],
        white_space=' ',
        indent='    ',
        line_length=50,
        comments=[],
        line_separator='\n',
        comment_prefix=' #',
        include_trailing_comma=include_trailing_comma,
        remove_comments=False
    )
    assert result == 'from datetime import datetime, time, timedelta, \\\n    timezone, tzinfo'


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.hanging_indent_with_parentheses(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.noqa(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.vertical(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.vertical_grid(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.vertical_grid_grouped(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.vertical_hanging_indent(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.vertical_hanging_indent_bracket(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()


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
    imports: List[str],
    white_space: str,
    indent: str,
    line_length: int,
    comments: List[str],
    line_separator: str,
    comment_prefix: str,
    include_trailing_comma: bool,
    remove_comments: bool
) -> None:
    try:
        isort.wrap_modes.vertical_prefix_from_module_import(
            statement=statement,
            imports=imports,
            white_space=white_space,
            indent=indent,
            line_length=line_length,
            comments=comments,
            line_separator=line_separator,
            comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma,
            remove_comments=remove_comments
        )
    except ValueError:
        reject()