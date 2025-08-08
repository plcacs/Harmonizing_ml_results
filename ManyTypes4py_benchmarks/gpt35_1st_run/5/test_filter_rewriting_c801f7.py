from typing import Any, Callable, Optional, Tuple

def test_filter_rewriting_ints(data: Any, strategy: Any, predicate: Callable[[int], bool], start: Optional[int], end: Optional[int]) -> None:
def test_filter_rewriting_floats(data: Any, strategy: Any, predicate: Callable[[float], bool], min_value: Optional[float], max_value: Optional[float]) -> None:
def test_rewrite_unsatisfiable_filter(s: Any, pred: Callable[[Any], bool]) -> None:
def test_erroring_rewrite_unsatisfiable_filter(s: Any, pred: Callable[[Any], bool]) -> None:
def test_misc_sat_filter_rewrites(data: Any, strategy: Any, predicate: Callable[[Any], bool]) -> None:
def test_misc_unsat_filter_rewrites(data: Any, strategy: Any, predicate: Callable[[Any], bool]) -> None:
def test_unhandled_operator(x: int) -> None:
def test_rewriting_does_not_compare_decimal_snan() -> None:
def test_applying_noop_filter_returns_self(strategy: Any) -> None:
def test_rewrite_filter_chains_with_some_unhandled(data: Any, predicates: Tuple[Callable[[Any], bool], ...], s: Any) -> None:
def test_rewriting_partially_understood_filters(data: Any, start: Optional[int], end: Optional[int], predicate: Callable[[Any], bool]) -> None:
def test_sequence_filter_rewriting(strategy: Any, predicate: Callable[[Any], bool]) -> None:
def test_warns_on_suspicious_string_methods(method: Callable[[str], str]) -> None:
def test_bumps_min_size_and_filters_for_content_str_methods(method: Callable[[str], bool]) -> None:
def test_isidentifier_filter_properly_rewritten(al: Optional[str], data: Any) -> None:
def test_isidentifer_filter_unsatisfiable(al: str) -> None:
def test_filter_floats_can_skip_subnormals(op: Callable[[float, float], bool], attr: str, value: float, expected: float) -> None:
def test_filter_rewriting_text_partial_len(data: Any, strategy: Any, predicate: Callable[[str], bool], start: Optional[int], end: Optional[int]) -> None:
def test_can_rewrite_multiple_length_filters_if_not_lambdas(data: Any) -> None:
def test_filter_rewriting_text_lambda_len(data: Any, strategy: Any, predicate: Callable[[str], bool], start: Optional[int], end: Optional[int]) -> None:
def test_filter_rewriting_lambda_len_unique_elements(data: Any, strategy: Any, predicate: Callable[[str], bool], start: Optional[int], end: Optional[int]) -> None:
def test_does_not_rewrite_unsatisfiable_len_filter(predicate: Callable[[str], bool]) -> None:
def test_regex_filter_rewriting(data: Any, strategy: Any, pattern: str, method: str) -> None:
def test_error_on_method_which_requires_multiple_args(_: Any) -> None:
def test_dates_filter_rewriting() -> None:
