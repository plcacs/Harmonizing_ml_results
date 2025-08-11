import traceback
import pytest
from hypothesis import given, settings, strategies as st
from tests.common.utils import fails_with

def fails_with_output(expected: Union[bool, dict[str, typing.Any], None], error: AssertionError=AssertionError, **kw):

    def _inner(f: Any):

        def _new() -> None:
            with pytest.raises(error) as err:
                settings(print_blob=False, derandomize=True, **kw)(f)()
            if not hasattr(err.value, '__notes__'):
                traceback.print_exception(err.value)
                raise Exception('err.value does not have __notes__, something has gone deeply wrong in the internals')
            got = '\n'.join(err.value.__notes__).strip() + '\n'
            assert got == expected.strip() + '\n'
        return _new
    return _inner

@fails_with_output('\nFalsifying example: test_inquisitor_comments_basic_fail_if_either(\n    # The test always failed when commented parts were varied together.\n    a=False,  # or any other generated value\n    b=True,\n    c=[],  # or any other generated value\n    d=True,\n    e=False,  # or any other generated value\n)\n')
@given(st.booleans(), st.booleans(), st.lists(st.none()), st.booleans(), st.booleans())
def test_inquisitor_comments_basic_fail_if_either(a: Union[int, float, str], b: Union[int, dict], c: Union[int, float, str], d: Union[int, dict], e: Union[int, float, str]) -> None:
    assert not (b and d)

@fails_with_output("\nFalsifying example: test_inquisitor_comments_basic_fail_if_not_all(\n    # The test sometimes passed when commented parts were varied together.\n    a='',  # or any other generated value\n    b='',  # or any other generated value\n    c='',  # or any other generated value\n)\n")
@given(st.text(), st.text(), st.text())
def test_inquisitor_comments_basic_fail_if_not_all(a: Union[int, bytes, complex], b: Union[int, bytes, complex], c: Union[int, bytes, complex]) -> None:
    condition = a and b and c
    assert condition

@fails_with_output("\nFalsifying example: test_inquisitor_no_together_comment_if_single_argument(\n    a='',\n    b='',  # or any other generated value\n)\n")
@given(st.text(), st.text())
def test_inquisitor_no_together_comment_if_single_argument(a: Union[int, float, str], b: Union[int, float, str]) -> None:
    assert a

@st.composite
def ints_with_forced_draw(draw: typing.Callable):
    data = draw(st.data())
    n = draw(st.integers())
    data.conjecture_data.draw_boolean(forced=True)
    return n

@fails_with_output('\nFalsifying example: test_inquisitor_doesnt_break_on_varying_forced_nodes(\n    n1=100,\n    n2=0,  # or any other generated value\n)\n')
@given(st.integers(), ints_with_forced_draw())
def test_inquisitor_doesnt_break_on_varying_forced_nodes(n1: tests.basilisp.helpers.CompileFn, n2: tests.basilisp.helpers.CompileFn) -> None:
    assert n1 < 100

@fails_with(ZeroDivisionError)
@settings(database=None)
@given(start_date=st.datetimes(), data=st.data())
def test_issue_3755_regression(start_date: Union[str, int, None], data: Union[str, int, None]) -> None:
    data.draw(st.datetimes(min_value=start_date))
    raise ZeroDivisionError