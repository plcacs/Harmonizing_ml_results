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
from example_code.future_annotations import add_custom_classes, invalid_types, merge_dicts
import hypothesis
from hypothesis.extra import ghostwriter
from hypothesis.utils.conventions import not_set

def timsort(seq: Sequence) -> Sequence:
    return sorted(seq)

def with_docstring(a: Sequence[int], b: Union[list, tuple, None], c: Optional[str], d: Optional[int] = int, e: Optional[Callable] = lambda x: f'xx{x}xx') -> None:
    """Demonstrates parsing params from the docstring

    :param a: sphinx docstring style
    :type a: sequence of integers

    b (list, tuple, or None): Google docstring style

    c : {"foo", "bar", or None}
        Numpy docstring style
    """

class A_Class:

    @classmethod
    def a_classmethod(cls, arg) -> None:
        pass

    @staticmethod
    def a_staticmethod(arg) -> None:
        pass

def add(a: int, b: int) -> int:
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

def union_sequence_parameter(items: Sequence) -> int:
    return sum(items)

def sequence_from_collections(items: Sequence) -> int:
    return min(items)

def various_numpy_annotations(f: Any, fc: Any, union: Any) -> None:
    pass
