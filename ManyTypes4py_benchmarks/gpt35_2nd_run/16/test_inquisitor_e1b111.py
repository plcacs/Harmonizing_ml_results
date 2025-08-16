from typing import List, Union, Text, Tuple, Any

def fails_with_output(expected: Text, error: Any, **kw: Any) -> Any:
    def _inner(f: Any) -> Any:
        def _new() -> None:
            ...
        return _new
    return _inner

def test_inquisitor_comments_basic_fail_if_either(a: bool, b: bool, c: List[None], d: bool, e: bool) -> None:
    ...

def test_inquisitor_comments_basic_fail_if_not_all(a: Text, b: Text, c: Text) -> None:
    ...

def test_inquisitor_no_together_comment_if_single_argument(a: Text, b: Text) -> None:
    ...

def ints_with_forced_draw(draw: Any) -> int:
    ...

def test_inquisitor_doesnt_break_on_varying_forced_nodes(n1: int, n2: int) -> None:
    ...

def test_issue_3755_regression(start_date: Any, data: Any) -> None:
    ...
