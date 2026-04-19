from typing import Any, Callable, Optional, Sequence, Union
from hypothesis.strategies import SearchStrategy
from hypothesis.strategies._internal.core import DataObject

A_FEW: int = ...
Y: int = ...
two: int = ...
lambda_without_source: Callable[[Any], bool] = ...


def test_filter_rewriting_ints(
    data: DataObject,
    strategy: SearchStrategy[int],
    predicate: Callable[[int], object],
    start: Optional[int],
    end: Optional[int],
) -> None: ...


def test_filter_rewriting_floats(
    data: DataObject,
    strategy: SearchStrategy[float],
    predicate: Callable[[float], object],
    min_value: float,
    max_value: float,
) -> None: ...


def test_rewrite_unsatisfiable_filter(
    s: SearchStrategy[Union[int, float]],
    pred: Callable[[Any], object],
) -> None: ...


def test_erroring_rewrite_unsatisfiable_filter(
    s: SearchStrategy[Union[int, float]],
    pred: Callable[[Any], object],
) -> None: ...


def test_misc_sat_filter_rewrites(
    data: DataObject,
    strategy: SearchStrategy[float],
    predicate: Callable[[float], object],
) -> None: ...


def test_misc_unsat_filter_rewrites(
    data: DataObject,
    strategy: SearchStrategy[float],
    predicate: Callable[[float], object],
) -> None: ...


def test_unhandled_operator(x: int) -> None: ...


def test_rewriting_does_not_compare_decimal_snan() -> None: ...


def test_applying_noop_filter_returns_self(
    strategy: SearchStrategy[Union[int, float]],
) -> None: ...


def mod2(x: Union[int, float]) -> Union[int, float]: ...


def test_rewrite_filter_chains_with_some_unhandled(
    data: DataObject,
    predicates: Sequence[Callable[[Any], object]],
    s: SearchStrategy[Union[int, float]],
) -> None: ...


class NotAFunction:
    def __call__(self, bar: Any) -> bool: ...


def test_rewriting_partially_understood_filters(
    data: DataObject,
    start: Optional[int],
    end: Optional[int],
    predicate: Callable[[Any], object],
) -> None: ...


def test_sequence_filter_rewriting(
    strategy: SearchStrategy[Any],
    predicate: Callable[[Any], object],
) -> None: ...


def test_warns_on_suspicious_string_methods(method: Callable[[str], Any]) -> None: ...


def test_bumps_min_size_and_filters_for_content_str_methods(
    method: Callable[[str], bool],
) -> None: ...


def test_isidentifier_filter_properly_rewritten(
    al: Optional[str],
    data: DataObject,
) -> None: ...


def test_isidentifer_filter_unsatisfiable(al: str) -> None: ...


def test_filter_floats_can_skip_subnormals(
    op: Callable[..., bool],
    attr: str,
    value: float,
    expected: float,
) -> None: ...


def test_filter_rewriting_text_partial_len(
    data: DataObject,
    strategy: SearchStrategy[str],
    predicate: Callable[[str], bool],
    start: Union[int, float],
    end: Union[int, float],
) -> None: ...


def test_can_rewrite_multiple_length_filters_if_not_lambdas(
    data: DataObject,
) -> None: ...


def test_filter_rewriting_text_lambda_len(
    data: DataObject,
    strategy: SearchStrategy[Any],
    predicate: Callable[[Any], object],
    start: Union[int, float],
    end: Union[int, float],
) -> None: ...


def test_filter_rewriting_lambda_len_unique_elements(
    data: DataObject,
    strategy: SearchStrategy[Any],
    predicate: Callable[[Any], object],
    start: int,
    end: int,
) -> None: ...


def test_does_not_rewrite_unsatisfiable_len_filter(
    predicate: Callable[[Any], object],
) -> None: ...


def test_regex_filter_rewriting(
    data: DataObject,
    strategy: SearchStrategy[Union[str, bytes]],
    pattern: Union[str, bytes],
    method: str,
) -> None: ...


def test_error_on_method_which_requires_multiple_args_(_: Any) -> None: ...


def test_dates_filter_rewriting() -> None: ...