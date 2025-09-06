from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager, nullcontext
import inspect
import re
import sys
import warnings
from typing import Any, Generator, List, Optional, Sequence, Tuple, Union, cast, Literal, Type
from pandas.compat import PY311
if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

@contextmanager
def func_xeg45y81(
    expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...], Literal[False], None] = Warning,
    filter_level: str = 'always',
    check_stacklevel: bool = True,
    raise_on_extra_warnings: bool = True,
    match: Union[str, Tuple[str, ...], None] = None,
    must_find_all_warnings: bool = True
) -> Generator[List[warnings.WarningMessage], None, None]:
    """
    Context manager for running code expected to either raise a specific warning,
    multiple specific warnings, or not raise any warnings. Verifies that the code
    raises the expected warning(s), and that it does not raise any other unexpected
    warnings. It is basically a wrapper around ``warnings.catch_warnings``.
    """
    __tracebackhide__ = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter(filter_level)
        try:
            yield w
        finally:
            if expected_warning:
                if isinstance(expected_warning, tuple) and must_find_all_warnings:
                    match_tuple: Tuple[Optional[str], ...] = match if isinstance(match, tuple) else (match,) * len(expected_warning)  # type: ignore
                    for warning_type, warning_match in zip(expected_warning, match_tuple):
                        _assert_caught_expected_warnings(
                            caught_warnings=w,
                            expected_warning=warning_type,
                            match=warning_match,
                            check_stacklevel=check_stacklevel,
                        )
                else:
                    expected_warning_cast = cast(Union[Type[Warning], Tuple[Type[Warning], ...]], expected_warning)
                    match_str: Optional[str] = '|'.join(m for m in match if m) if isinstance(match, tuple) and match is not None else match
                    _assert_caught_expected_warnings(
                        caught_warnings=w,
                        expected_warning=expected_warning_cast,
                        match=match_str,
                        check_stacklevel=check_stacklevel,
                    )
            if raise_on_extra_warnings:
                _assert_caught_no_extra_warnings(
                    caught_warnings=w,
                    expected_warning=expected_warning,
                )

def func_f0oqf4oo(
    warning: Union[Type[Warning], Tuple[Type[Warning], ...], Literal[False], None],
    condition: bool,
    **kwargs: Any
) -> AbstractContextManager[Any]:
    """
    Return a context manager that possibly checks a warning based on the condition
    """
    if condition:
        return func_xeg45y81(warning, **kwargs)
    else:
        return nullcontext()

def func_kulc39mn(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...]],
    match: Optional[str],
    check_stacklevel: bool
) -> None:
    """Assert that there was the expected warning among the caught warnings."""
    saw_warning: bool = False
    matched_message: bool = False
    unmatched_messages: List[Any] = []
    warning_name: Union[str, Tuple[str, ...]] = (
        tuple(x.__name__ for x in expected_warning)
        if isinstance(expected_warning, tuple)
        else expected_warning.__name__
    )
    for actual_warning in caught_warnings:
        if issubclass(actual_warning.category, expected_warning):  # type: ignore
            saw_warning = True
            if check_stacklevel:
                _assert_raised_with_correct_stacklevel(actual_warning)
            if match is not None:
                if re.search(match, str(actual_warning.message)):
                    matched_message = True
                else:
                    unmatched_messages.append(actual_warning.message)
    if not saw_warning:
        raise AssertionError(f"Did not see expected warning of class {warning_name!r}")
    if match and not matched_message:
        raise AssertionError(
            f"Did not see warning {warning_name!r} matching '{match}'. The emitted warning messages are {unmatched_messages}"
        )

def func_cn3i81uk(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...], Literal[False], None]
) -> None:
    """Assert that no extra warnings apart from the expected ones are caught."""
    extra_warnings: List[Tuple[str, Any, str, int]] = []
    for actual_warning in caught_warnings:
        if _is_unexpected_warning(actual_warning, expected_warning):
            if actual_warning.category == ResourceWarning:
                if "unclosed <ssl.SSLSocket" in str(actual_warning.message):
                    continue
                if any("matplotlib" in mod for mod in sys.modules):
                    continue
            if PY311 and actual_warning.category == EncodingWarning:  # type: ignore
                continue
            extra_warnings.append(
                (
                    actual_warning.category.__name__,
                    actual_warning.message,
                    actual_warning.filename,
                    actual_warning.lineno,
                )
            )
    if extra_warnings:
        raise AssertionError(f"Caused unexpected warning(s): {extra_warnings!r}")

def func_31puiv84(
    actual_warning: warnings.WarningMessage,
    expected_warning: Union[Type[Warning], Literal[False], None]
) -> bool:
    """Check if the actual warning issued is unexpected."""
    if actual_warning and not expected_warning:
        return True
    expected_warning_cast = cast(Type[Warning], expected_warning)
    return bool(not issubclass(actual_warning.category, expected_warning_cast))  # type: ignore

def func_sgh3hgk8(actual_warning: warnings.WarningMessage) -> None:
    frame = inspect.currentframe()
    for _ in range(4):
        frame = frame.f_back  # type: ignore
    try:
        caller_filename: str = inspect.getfile(frame)  # type: ignore
    finally:
        del frame
    msg = (
        f"Warning not set with correct stacklevel. File where warning is raised: {actual_warning.filename} != {caller_filename}. Warning message: {actual_warning.message}"
    )
    assert actual_warning.filename == caller_filename, msg

_assert_caught_expected_warnings = func_kulc39mn
_assert_caught_no_extra_warnings = func_cn3i81uk
_is_unexpected_warning = func_31puiv84
_assert_raised_with_correct_stacklevel = func_sgh3hgk8