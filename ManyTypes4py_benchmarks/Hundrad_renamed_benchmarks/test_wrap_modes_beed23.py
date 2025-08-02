import pytest
from hypothesis import given, reject
from hypothesis import strategies as st
import isort
from isort import wrap_modes


def func_wq6qlqhg():
    assert wrap_modes._wrap_mode_interface('statement', [], '', '', 80, [],
        '', '', True, True) == ''


def func_vydlghlr():
    """hypothesis_auto tests cases that have been saved to ensure they run each test cycle"""
    assert wrap_modes.noqa(**{'comment_prefix':
        '-\U000bf82c\x0c\U0004608f\x10%', 'comments': [], 'imports': [],
        'include_trailing_comma': False, 'indent': '0\x19', 'line_length': 
        -19659, 'line_separator':
        '\x15\x0b\U00086494\x1d\U000e00a2\U000ee216\U0006708a\x03\x1f',
        'remove_comments': False, 'statement': '\U00092452', 'white_space':
        '\U000a7322\U000c20e3-\U0010eae4\x07\x14\U0007d486'}
        ) == '\U00092452-\U000bf82c\x0c\U0004608f\x10% NOQA'
    assert wrap_modes.noqa(**{'comment_prefix':
        '\x12\x07\U0009e994🁣"\U000ae787\x0e', 'comments': [
        '\x00\U0001ae99\U0005c3e7\U0004d08e', '\x1e', '', ''], 'imports': [
        '*'], 'include_trailing_comma': True, 'indent': '', 'line_length': 
        31492, 'line_separator': '\U00071610\U0005bfbc', 'remove_comments':
        False, 'statement': '', 'white_space': '\x08\x01ⷓ\x16%\U0006cd8c'}
        ) == '*\x12\x07\U0009e994🁣"\U000ae787\x0e \x00\U0001ae99\U0005c3e7\U0004d08e \x1e  '
    assert wrap_modes.noqa(**{'comment_prefix': '  #', 'comments': ['NOQA',
        'THERE'], 'imports': [], 'include_trailing_comma': False, 'indent':
        '0\x19', 'line_length': -19659, 'line_separator': '\n',
        'remove_comments': False, 'statement': 'hi', 'white_space': ' '}
        ) == 'hi  # NOQA THERE'


def func_z0cwy5io():
    """Tests the backslash_grid grid wrap mode, ensuring it matches formatting expectations.
    See: https://github.com/PyCQA/isort/issues/1434
    """
    assert isort.code("""
from kopf.engines import loggers, posting
from kopf.reactor import causation, daemons, effects, handling, lifecycles, registries
from kopf.storage import finalizers, states
from kopf.structs import (bodies, configuration, containers, diffs,
                          handlers as handlers_, patches, resources)
""", multi_line_output=11, line_length=88, combine_as_imports=True) == """
from kopf.engines import loggers, posting
from kopf.reactor import causation, daemons, effects, handling, lifecycles, registries
from kopf.storage import finalizers, states
from kopf.structs import bodies, configuration, containers, diffs, \\
                         handlers as handlers_, patches, resources
"""


@pytest.mark.parametrize('include_trailing_comma', (False, True))
@pytest.mark.parametrize('line_length', (18, 19))
@pytest.mark.parametrize('multi_line_output', (4, 5))
def func_h44o31aa(multi_line_output, line_length, include_trailing_comma):
    separator = ' '
    if multi_line_output == 4 and line_length < 19 + int(include_trailing_comma
        ) or multi_line_output != 4 and line_length < 18 + int(
        include_trailing_comma):
        separator = '\n    '
    test_input = f"""from foo import (
    aaaa, bbb,{separator}ccc"""
    if include_trailing_comma:
        test_input += ','
    if multi_line_output != 4:
        test_input += '\n'
    test_input += ')\n'
    assert isort.code(test_input, multi_line_output=multi_line_output,
        line_length=line_length, include_trailing_comma=include_trailing_comma
        ) == test_input


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_dgfy6ccl(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.backslash_grid(statement=statement, imports=
            imports, white_space=white_space, indent=indent, line_length=
            line_length, comments=comments, line_separator=line_separator,
            comment_prefix=comment_prefix, include_trailing_comma=
            include_trailing_comma, remove_comments=remove_comments)
    except ValueError:
        reject()


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_jz6w7x2j(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.grid(statement=statement, imports=imports,
            white_space=white_space, indent=indent, line_length=line_length,
            comments=comments, line_separator=line_separator,
            comment_prefix=comment_prefix, include_trailing_comma=
            include_trailing_comma, remove_comments=remove_comments)
    except ValueError:
        reject()


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_5j44d2e8(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.hanging_indent(statement=statement, imports=
            imports, white_space=white_space, indent=indent, line_length=
            line_length, comments=comments, line_separator=line_separator,
            comment_prefix=comment_prefix, include_trailing_comma=
            include_trailing_comma, remove_comments=remove_comments)
    except ValueError:
        reject()


@pytest.mark.parametrize('include_trailing_comma', (True, False))
def func_cints0ka(include_trailing_comma):
    result = isort.wrap_modes.hanging_indent(statement=
        'from datetime import ', imports=['datetime', 'time', 'timedelta',
        'timezone', 'tzinfo'], white_space=' ', indent='    ', line_length=
        50, comments=[], line_separator='\n', comment_prefix=' #',
        include_trailing_comma=include_trailing_comma, remove_comments=False)
    assert result == """from datetime import datetime, time, timedelta, \\
    timezone, tzinfo"""


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_eznlr940(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.hanging_indent_with_parentheses(statement=
            statement, imports=imports, white_space=white_space, indent=
            indent, line_length=line_length, comments=comments,
            line_separator=line_separator, comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma, remove_comments=
            remove_comments)
    except ValueError:
        reject()


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_r0s0ze8c(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.noqa(statement=statement, imports=imports,
            white_space=white_space, indent=indent, line_length=line_length,
            comments=comments, line_separator=line_separator,
            comment_prefix=comment_prefix, include_trailing_comma=
            include_trailing_comma, remove_comments=remove_comments)
    except ValueError:
        reject()


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_c3mh7mps(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.vertical(statement=statement, imports=imports,
            white_space=white_space, indent=indent, line_length=line_length,
            comments=comments, line_separator=line_separator,
            comment_prefix=comment_prefix, include_trailing_comma=
            include_trailing_comma, remove_comments=remove_comments)
    except ValueError:
        reject()


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_b1w78faq(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.vertical_grid(statement=statement, imports=imports,
            white_space=white_space, indent=indent, line_length=line_length,
            comments=comments, line_separator=line_separator,
            comment_prefix=comment_prefix, include_trailing_comma=
            include_trailing_comma, remove_comments=remove_comments)
    except ValueError:
        reject()


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_vw911pwr(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.vertical_grid_grouped(statement=statement, imports
            =imports, white_space=white_space, indent=indent, line_length=
            line_length, comments=comments, line_separator=line_separator,
            comment_prefix=comment_prefix, include_trailing_comma=
            include_trailing_comma, remove_comments=remove_comments)
    except ValueError:
        reject()


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_7zem6sio(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.vertical_hanging_indent(statement=statement,
            imports=imports, white_space=white_space, indent=indent,
            line_length=line_length, comments=comments, line_separator=
            line_separator, comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma, remove_comments=
            remove_comments)
    except ValueError:
        reject()


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_jjjpd9ho(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.vertical_hanging_indent_bracket(statement=
            statement, imports=imports, white_space=white_space, indent=
            indent, line_length=line_length, comments=comments,
            line_separator=line_separator, comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma, remove_comments=
            remove_comments)
    except ValueError:
        reject()


@given(statement=st.text(), imports=st.lists(st.text()), white_space=st.
    text(), indent=st.text(), line_length=st.integers(), comments=st.lists(
    st.text()), line_separator=st.text(), comment_prefix=st.text(),
    include_trailing_comma=st.booleans(), remove_comments=st.booleans())
def func_lkr01xk3(statement, imports, white_space, indent, line_length,
    comments, line_separator, comment_prefix, include_trailing_comma,
    remove_comments):
    try:
        isort.wrap_modes.vertical_prefix_from_module_import(statement=
            statement, imports=imports, white_space=white_space, indent=
            indent, line_length=line_length, comments=comments,
            line_separator=line_separator, comment_prefix=comment_prefix,
            include_trailing_comma=include_trailing_comma, remove_comments=
            remove_comments)
    except ValueError:
        reject()
