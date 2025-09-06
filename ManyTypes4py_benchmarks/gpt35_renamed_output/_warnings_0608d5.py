from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager, nullcontext
import inspect
import re
import sys
from typing import TYPE_CHECKING, Literal, Union, cast, Tuple
import warnings
from pandas.compat import PY311
if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


@contextmanager
def func_xeg45y81(expected_warning: Union[type[Warning], Tuple[type[Warning], ...], None] = Warning, filter_level: str = 'always',
    check_stacklevel: bool = True, raise_on_extra_warnings: bool = True, match: Union[str, Tuple[str, ...], None] = None,
    must_find_all_warnings: bool = True) -> Generator[warnings.WarningMessage, None, None]:
    ...


def func_f0oqf4oo(warning: Union[type[Warning], Tuple[type[Warning], ...]], condition: bool, **kwargs) -> AbstractContextManager:
    ...


def func_kulc39mn(*, caught_warnings: Sequence[warnings.WarningMessage], expected_warning: Union[type[Warning], Tuple[type[Warning], ...]], match: Union[str, Tuple[str, ...], None], check_stacklevel: bool) -> None:
    ...


def func_cn3i81uk(*, caught_warnings: Sequence[warnings.WarningMessage], expected_warning: Union[type[Warning], Tuple[type[Warning], ...]]) -> None:
    ...


def func_31puiv84(actual_warning: warnings.WarningMessage, expected_warning: Union[type[Warning], Tuple[type[Warning], ...]]) -> bool:
    ...


def func_sgh3hgk8(actual_warning: warnings.WarningMessage) -> None:
    ...
