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
from typing import Optional, Union

import numpy
import numpy.typing
import pytest
from example_code.future_annotations import (
    add_custom_classes,
    invalid_types,
    merge_dicts,
)

import hypothesis
from hypothesis.extra import ghostwriter
from hypothesis.utils.conventions import not_set


@pytest.fixture
def update_recorded_outputs(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--hypothesis-update-outputs")


def get_recorded(name: str, actual: str = "") -> str:
    file_ = pathlib.Path(__file__).parent / "recorded" / f"{name}.txt"
    if actual:
        file_.write_text(actual, encoding="utf-8")
    return file_.read_text(encoding="utf-8")


def timsort(seq: Sequence[int]) -> Sequence[int]:
    return sorted(seq)


def with_docstring(a: Sequence[int], b: Optional[Union[list, tuple]], c: Optional[Literal["foo", "bar"]], d: type = int, e: Callable[[str], str] = lambda x: f"xx{x}xx") -> None:
    """Demonstrates parsing params from the docstring

    :param a: sphinx docstring style
    :type a: sequence of integers

    b (list, tuple, or None): Google docstring style

    c : {"foo", "bar", or None}
        Numpy docstring style
    """


class A_Class:
    @classmethod
    def a_classmethod(cls, arg: int) -> None:
        pass

    @staticmethod
    def a_staticmethod(arg: int) -> None:
        pass


def add(a: float, b: float) -> float:
    return a + b


def divide(a: int, b: int) -> float:
    """This is a RST-style docstring for `divide`.

    :raises ZeroDivisionError: if b == 0
    """
    return a / b


def optional_parameter(a: float, b: Optional[float]) -> float:
    return optional_union_parameter(a, b)


def optional_union_parameter(a: float, b: Optional[Union[float, int]]) -> float:
    return a if b is None else a + b


if sys.version_info[:2] >= (3, 10):
    def union_sequence_parameter(items: Sequence[float | int]) -> float:
        return sum(items)
else:
    def union_sequence_parameter(items: Sequence[Union[float, int]]) -> float:
        return sum(items)


def sequence_from_collections(items: collections.abc.Sequence[int]) -> int:
    return min(items)


if sys.version_info[:2] >= (3, 10):
    def various_numpy_annotations(
        f: numpy.typing.NDArray[numpy.float64],
        fc: numpy.typing.NDArray[numpy.float64 | numpy.complex128],
        union: Optional[numpy.typing.NDArray[numpy.float64 | numpy.complex128]],
    ) -> None:
        pass
else:
    various_numpy_annotations = add


@pytest.mark.parametrize(
    "data",
    [
        ("fuzz_sorted", lambda: ghostwriter.fuzz(sorted)),
        # ... (rest of the parametrize decorator remains the same)
    ],
    ids=lambda x: x[0],
)
def test_ghostwriter_example_outputs(update_recorded_outputs: bool, data: tuple[str, Callable[[], str]]) -> None:
    name, get_actual = data
    actual = get_actual()
    expected = get_recorded(name, actual * update_recorded_outputs)
    assert actual == expected
    exec(expected, {})


def test_ghostwriter_on_hypothesis(update_recorded_outputs: bool) -> None:
    actual = ghostwriter.magic(hypothesis).replace("Strategy[+Ex]", "Strategy")
    expected = get_recorded("hypothesis_module_magic", actual * update_recorded_outputs)
    if sys.version_info[:2] == (3, 10):
        assert actual == expected
    exec(expected, {"not_set": not_set})


def test_ghostwriter_suggests_submodules_for_empty_toplevel(
    tmp_path: pathlib.Path, update_recorded_outputs: bool
) -> None:
    foo = tmp_path / "foo"
    foo.mkdir()
    (foo / "__init__.py").write_text("from . import bar\n", encoding="utf-8")
    (foo / "bar.py").write_text("def baz(x: int): ...\n", encoding="utf-8")

    proc = subprocess.run(
        ["hypothesis", "write", "foo"],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
    )
    actual = proc.stdout.replace(re.search(r"from '(.+)foo/", proc.stdout).group(1), "")

    expected = get_recorded("nothing_found", actual * update_recorded_outputs)
    assert actual == expected
    exec(expected, {})
