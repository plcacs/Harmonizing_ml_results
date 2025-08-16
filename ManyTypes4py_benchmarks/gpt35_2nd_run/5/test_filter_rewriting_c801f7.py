from typing import Any, Callable, Optional, Tuple

def test_filter_rewriting_ints(data: Any, strategy: IntegersStrategy, predicate: Callable, start: Optional[int], end: Optional[int]) -> None:
def test_filter_rewriting_floats(data: Any, strategy: FloatStrategy, predicate: Callable, min_value: float, max_value: float) -> None:
def test_rewrite_unsatisfiable_filter(s: Union[IntegersStrategy, FloatStrategy], pred: Callable) -> None:
def test_erroring_rewrite_unsatisfiable_filter(s: Union[IntegersStrategy, FloatStrategy], pred: Callable) -> None:
def test_misc_sat_filter_rewrites(data: Any, strategy: Any, predicate: Callable) -> None:
def test_misc_unsat_filter_rewrites(data: Any, strategy: Any, predicate: Callable) -> None:
def test_unhandled_operator(x: int) -> None:
def test_rewriting_does_not_compare_decimal_snan() -> None:
def test_applying_noop_filter_returns_self(strategy: Any) -> None:
def mod2(x: int) -> int:
def test_rewrite_filter_chains_with_some_unhandled(data: Any, predicates: Tuple[Callable, ...], s: Any) -> None:
def test_rewriting_partially_understood_filters(data: Any, start: Optional[int], end: Optional[int], predicate: Callable) -> None:
def test_sequence_filter_rewriting(strategy: Any, predicate: Callable) -> None:
def test_warns_on_suspicious_string_methods(method: Callable) -> None:
def test_bumps_min_size_and_filters_for_content_str_methods(method: Callable) -> None:
def test_filter_floats_can_skip_subnormals(op: Callable, attr: str, value: float, expected: float) -> None:
def test_filter_rewriting_text_partial_len(data: Any, strategy: TextStrategy, predicate: Callable, start: Optional[int], end: Optional[int]) -> None:
def test_filter_rewriting_text_lambda_len(data: Any, strategy: Any, predicate: Callable, start: Optional[int], end: Optional[int]) -> None:
def test_filter_rewriting_lambda_len_unique_elements(data: Any, strategy: Any, predicate: Callable, start: Optional[int], end: Optional[int]) -> None:
def test_does_not_rewrite_unsatisfiable_len_filter(predicate: Callable) -> None:
def test_regex_filter_rewriting(data: Any, strategy: Any, pattern: str, method: str) -> None:
def test_error_on_method_which_requires_multiple_args(_: Any) -> None:
def test_dates_filter_rewriting() -> None:
