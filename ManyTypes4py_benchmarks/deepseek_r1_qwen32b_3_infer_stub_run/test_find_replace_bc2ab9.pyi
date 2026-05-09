from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    overload,
)
import numpy as np
import pandas as pd
from pandas import Series
import pytest

def test_contains(any_string_dtype: str) -> None:
    ...

def test_contains_object_mixed() -> None:
    ...

def test_contains_na_kwarg_for_object_category() -> None:
    ...

def test_contains_na_kwarg_for_nullable_string_dtype(
    nullable_string_dtype: str,
    na: Any,
    expected: Any,
    regex: bool,
) -> None:
    ...

def test_contains_moar(any_string_dtype: str) -> None:
    ...

def test_contains_nan(any_string_dtype: str) -> None:
    ...

def test_startswith_endswith_validate_na(any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('pat', ['foo', ('foo', 'baz')])
@pytest.mark.parametrize('dtype', ['object', 'category'])
@pytest.mark.parametrize('null_value', [None, np.nan, pd.NA])
@pytest.mark.parametrize('na', [True, False])
def test_startswith(
    pat: Union[str, tuple[str, str]],
    dtype: str,
    null_value: Any,
    na: bool,
    using_infer_string: bool,
) -> None:
    ...

@pytest.mark.parametrize('na', [None, True, False])
def test_startswith_string_dtype(
    any_string_dtype: str,
    na: Optional[bool],
) -> None:
    ...

@pytest.mark.parametrize('pat', ['foo', ('foo', 'baz')])
@pytest.mark.parametrize('dtype', ['object', 'category'])
@pytest.mark.parametrize('null_value', [None, np.nan, pd.NA])
@pytest.mark.parametrize('na', [True, False])
def test_endswith(
    pat: Union[str, tuple[str, str]],
    dtype: str,
    null_value: Any,
    na: bool,
    using_infer_string: bool,
) -> None:
    ...

@pytest.mark.parametrize('na', [None, True, False])
def test_endswith_string_dtype(
    any_string_dtype: str,
    na: Optional[bool],
) -> None:
    ...

def test_replace_dict_invalid(any_string_dtype: str) -> None:
    ...

def test_replace_dict(any_string_dtype: str) -> None:
    ...

def test_replace(any_string_dtype: str) -> None:
    ...

def test_replace_max_replacements(any_string_dtype: str) -> None:
    ...

def test_replace_mixed_object() -> None:
    ...

def test_replace_unicode(any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('repl', [None, 3, {'a': 'b'}])
@pytest.mark.parametrize('data', [['a', 'b', None], ['a', 'b', 'c', 'ad']])
def test_replace_wrong_repl_type_raises(
    any_string_dtype: str,
    index_or_series: Any,
    repl: Any,
    data: List[Any],
) -> None:
    ...

def test_replace_callable(any_string_dtype: str) -> None:
    ...

def test_replace_callable_raises(
    any_string_dtype: str,
    repl: Callable[..., Any],
) -> None:
    ...

def test_replace_callable_named_groups(any_string_dtype: str) -> None:
    ...

def test_replace_compiled_regex(any_string_dtype: str) -> None:
    ...

def test_replace_compiled_regex_mixed_object() -> None:
    ...

def test_replace_compiled_regex_unicode(any_string_dtype: str) -> None:
    ...

def test_replace_compiled_regex_raises(any_string_dtype: str) -> None:
    ...

def test_replace_compiled_regex_callable(any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('regex,expected_val', [(True, 'bao'), (False, 'foo')])
def test_replace_literal(
    regex: bool,
    expected_val: str,
    any_string_dtype: str,
) -> None:
    ...

def test_replace_literal_callable_raises(any_string_dtype: str) -> None:
    ...

def test_replace_literal_compiled_raises(any_string_dtype: str) -> None:
    ...

def test_replace_moar(any_string_dtype: str) -> None:
    ...

def test_replace_not_case_sensitive_not_regex(any_string_dtype: str) -> None:
    ...

def test_replace_regex(any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('regex', [True, False])
def test_replace_regex_single_character(
    regex: bool,
    any_string_dtype: str,
) -> None:
    ...

def test_match(any_string_dtype: str) -> None:
    ...

def test_match_mixed_object() -> None:
    ...

def test_match_na_kwarg(any_string_dtype: str) -> None:
    ...

def test_match_case_kwarg(any_string_dtype: str) -> None:
    ...

def test_fullmatch(any_string_dtype: str) -> None:
    ...

def test_fullmatch_dollar_literal(any_string_dtype: str) -> None:
    ...

def test_fullmatch_na_kwarg(any_string_dtype: str) -> None:
    ...

def test_fullmatch_case_kwarg(any_string_dtype: str) -> None:
    ...

def test_findall(any_string_dtype: str) -> None:
    ...

def test_findall_mixed_object() -> None:
    ...

def test_find(any_string_dtype: str) -> None:
    ...

def test_find_bad_arg_raises(any_string_dtype: str) -> None:
    ...

def test_find_nan(any_string_dtype: str) -> None:
    ...

@pytest.mark.parametrize('infer_string', [False, pytest.param(True, marks=td.skip_if_no('pyarrow'))])
def test_translate(
    index_or_series: Any,
    any_string_dtype: str,
    infer_string: bool,
) -> None:
    ...

def test_translate_mixed_object() -> None:
    ...

def test_flags_kwarg(any_string_dtype: str) -> None:
    ...