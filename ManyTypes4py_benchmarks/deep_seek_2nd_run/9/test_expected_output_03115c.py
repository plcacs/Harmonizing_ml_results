"""
'Golden master' tests for the ghostwriter.

To update the recorded outputs, run `pytest --hypothesis-update-outputs ...`.
"""
import ast
import base64
import builtins
import collections.abc
import operator
import pathlib
import re
import subprocess
import sys
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import numpy
import numpy.typing
import pytest
from example_code.future_annotations import add_custom_classes, invalid_types, merge_dicts
import hypothesis
from hypothesis.extra import ghostwriter
from hypothesis.utils.conventions import not_set

T = TypeVar('T')

@pytest.fixture
def update_recorded_outputs(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption('--hypothesis-update-outputs')  # type: ignore

def get_recorded(name: str, actual: str = '') -> str:
    file_ = pathlib.Path(__file__).parent / 'recorded' / f'{name}.txt'
    if actual:
        file_.write_text(actual, encoding='utf-8')
    return file_.read_text(encoding='utf-8')

def timsort(seq: Sequence[T]) -> List[T]:
    return sorted(seq)

def with_docstring(a: Sequence[int], b: Optional[Union[List, Tuple]], c: Optional[str], d: Type[int] = int, e: Callable[[Any], str] = lambda x: f'xx{x}xx') -> None:
    """Demonstrates parsing params from the docstring

    :param a: sphinx docstring style
    :type a: sequence of integers

    b (list, tuple, or None): Google docstring style

    c : {"foo", "bar", or None}
        Numpy docstring style
    """

class A_Class:
    @classmethod
    def a_classmethod(cls, arg: Any) -> None:
        pass

    @staticmethod
    def a_staticmethod(arg: Any) -> None:
        pass

def add(a: Any, b: Any) -> Any:
    return a + b

def divide(a: float, b: float) -> float:
    """This is a RST-style docstring for `divide`.

    :raises ZeroDivisionError: if b == 0
    """
    return a / b

def optional_parameter(a: int, b: Optional[int]) -> int:
    return optional_union_parameter(a, b)

def optional_union_parameter(a: int, b: Optional[int]) -> int:
    return a if b is None else a + b

if sys.version_info[:2] >= (3, 10):
    def union_sequence_parameter(items: Union[Sequence[int], Sequence[float]]) -> Union[int, float]:
        return sum(items)
else:
    def union_sequence_parameter(items: Union[Sequence[int], Sequence[float]]) -> Union[int, float]:
        return sum(items)

def sequence_from_collections(items: Sequence[T]) -> T:
    return min(items)

if sys.version_info[:2] >= (3, 10):
    def various_numpy_annotations(
        f: numpy.typing.NDArray[numpy.float64],
        fc: numpy.typing.NDArray[numpy.complex128],
        union: Union[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.complex128]]
    ) -> None:
        pass
else:
    various_numpy_annotations = add

@pytest.mark.parametrize('data', [
    ('fuzz_sorted', lambda: ghostwriter.fuzz(sorted)),
    ('fuzz_sorted_with_annotations', lambda: ghostwriter.fuzz(sorted, annotate=True)),
    ('fuzz_with_docstring', lambda: ghostwriter.fuzz(with_docstring)),
    ('fuzz_classmethod', lambda: ghostwriter.fuzz(A_Class.a_classmethod)),
    ('fuzz_staticmethod', lambda: ghostwriter.fuzz(A_Class.a_staticmethod)),
    ('fuzz_ufunc', lambda: ghostwriter.fuzz(numpy.add)),
    ('magic_gufunc', lambda: ghostwriter.magic(numpy.matmul)),
    ('optional_parameter', lambda: ghostwriter.magic(optional_parameter)),
    ('optional_union_parameter', lambda: ghostwriter.magic(optional_union_parameter)),
    ('union_sequence_parameter', lambda: ghostwriter.magic(union_sequence_parameter)),
    ('sequence_from_collections', lambda: ghostwriter.magic(sequence_from_collections)),
    pytest.param(('add_custom_classes', lambda: ghostwriter.magic(add_custom_classes)), marks=pytest.mark.skipif('sys.version_info[:2] < (3, 10)')),
    pytest.param(('merge_dicts', lambda: ghostwriter.magic(merge_dicts)), marks=pytest.mark.skipif('sys.version_info[:2] < (3, 10)')),
    pytest.param(('invalid_types', lambda: ghostwriter.magic(invalid_types)), marks=pytest.mark.skipif('sys.version_info[:2] < (3, 10)')),
    ('magic_base64_roundtrip', lambda: ghostwriter.magic(base64.b64encode)),
    ('magic_base64_roundtrip_with_annotations', lambda: ghostwriter.magic(base64.b64encode, annotate=True)),
    ('re_compile', lambda: ghostwriter.fuzz(re.compile)),
    ('re_compile_except', lambda: ghostwriter.fuzz(re.compile, except_=re.error).replace('re.PatternError', 're.error')),
    ('re_compile_unittest', lambda: ghostwriter.fuzz(re.compile, style='unittest')),
    pytest.param(('base64_magic', lambda: ghostwriter.magic(base64)), marks=pytest.mark.skipif('sys.version_info[:2] >= (3, 10)')),
    ('sorted_idempotent', lambda: ghostwriter.idempotent(sorted)),
    ('timsort_idempotent', lambda: ghostwriter.idempotent(timsort)),
    ('timsort_idempotent_asserts', lambda: ghostwriter.idempotent(timsort, except_=AssertionError)),
    pytest.param(('eval_equivalent', lambda: ghostwriter.equivalent(eval, ast.literal_eval)), marks=[pytest.mark.skipif(sys.version_info[:2] >= (3, 13), reason='kw')]),
    ('sorted_self_equivalent', lambda: ghostwriter.equivalent(sorted, sorted, sorted)),
    ('sorted_self_equivalent_with_annotations', lambda: ghostwriter.equivalent(sorted, sorted, sorted, annotate=True)),
    ('addition_op_magic', lambda: ghostwriter.magic(add)),
    ('multiplication_magic', lambda: ghostwriter.magic(operator.mul)),
    ('matmul_magic', lambda: ghostwriter.magic(operator.matmul)),
    ('addition_op_multimagic', lambda: ghostwriter.magic(add, operator.add, numpy.add)),
    ('division_fuzz_error_handler', lambda: ghostwriter.fuzz(divide)),
    ('division_binop_error_handler', lambda: ghostwriter.binary_operation(divide, identity=1)),
    ('division_roundtrip_error_handler', lambda: ghostwriter.roundtrip(divide, operator.mul)),
    ('division_roundtrip_error_handler_without_annotations', lambda: ghostwriter.roundtrip(divide, operator.mul, annotate=False)),
    ('division_roundtrip_arithmeticerror_handler', lambda: ghostwriter.roundtrip(divide, operator.mul, except_=ArithmeticError)),
    ('division_roundtrip_typeerror_handler', lambda: ghostwriter.roundtrip(divide, operator.mul, except_=TypeError)),
    ('division_operator', lambda: ghostwriter.binary_operation(operator.truediv, associative=False, commutative=False)),
    ('division_operator_with_annotations', lambda: ghostwriter.binary_operation(operator.truediv, associative=False, commutative=False, annotate=True)),
    ('multiplication_operator', lambda: ghostwriter.binary_operation(operator.mul, identity=1, distributes_over=operator.add)),
    ('multiplication_operator_unittest', lambda: ghostwriter.binary_operation(operator.mul, identity=1, distributes_over=operator.add, style='unittest')),
    ('sorted_self_error_equivalent_simple', lambda: ghostwriter.equivalent(sorted, sorted, allow_same_errors=True)),
    ('sorted_self_error_equivalent_threefuncs', lambda: ghostwriter.equivalent(sorted, sorted, sorted, allow_same_errors=True)),
    ('sorted_self_error_equivalent_1error', lambda: ghostwriter.equivalent(sorted, sorted, allow_same_errors=True, except_=ValueError)),
    ('sorted_self_error_equivalent_2error_unittest', lambda: ghostwriter.equivalent(sorted, sorted, allow_same_errors=True, except_=(TypeError, ValueError), style='unittest')),
    ('magic_class', lambda: ghostwriter.magic(A_Class)),
    pytest.param(('magic_builtins', lambda: ghostwriter.magic(builtins)), marks=[pytest.mark.skipif(sys.version_info[:2] != (3, 10), reason='often small changes')]),
    pytest.param(('magic_numpy', lambda: ghostwriter.magic(various_numpy_annotations, annotate=False)), marks=pytest.mark.skipif(various_numpy_annotations is add, reason='<=3.9'))
], ids=lambda x: x[0])
def test_ghostwriter_example_outputs(update_recorded_outputs: bool, data: Tuple[str, Callable[[], str]]) -> None:
    name, get_actual = data
    actual = get_actual()
    expected = get_recorded(name, actual * update_recorded_outputs)
    assert actual == expected
    exec(expected, {})

def test_ghostwriter_on_hypothesis(update_recorded_outputs: bool) -> None:
    actual = ghostwriter.magic(hypothesis).replace('Strategy[+Ex]', 'Strategy')
    expected = get_recorded('hypothesis_module_magic', actual * update_recorded_outputs)
    if sys.version_info[:2] == (3, 10):
        assert actual == expected
    exec(expected, {'not_set': not_set})

def test_ghostwriter_suggests_submodules_for_empty_toplevel(tmp_path: pathlib.Path, update_recorded_outputs: bool) -> None:
    foo = tmp_path / 'foo'
    foo.mkdir()
    (foo / '__init__.py').write_text('from . import bar\n', encoding='utf-8')
    (foo / 'bar.py').write_text('def baz(x: int): ...\n', encoding='utf-8')
    proc = subprocess.run(['hypothesis', 'write', 'foo'], check=True, capture_output=True, encoding='utf-8', cwd=tmp_path)
    actual = proc.stdout.replace(re.search("from '(.+)foo/", proc.stdout).group(1), '')
    expected = get_recorded('nothing_found', actual * update_recorded_outputs)
    assert actual == expected
    exec(expected, {})
