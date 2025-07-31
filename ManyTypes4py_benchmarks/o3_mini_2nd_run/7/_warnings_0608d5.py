from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager, nullcontext
import inspect
import re
import sys
import warnings
from typing import Any, Generator, Iterator, Literal, Optional, Tuple, Type, Union

from pandas.compat import PY311

if False:
    from collections.abc import Generator, Sequence  # TYPE_CHECKING

# Type alias for expected warning parameter.
ExpectedWarningType = Union[Type[Warning], Tuple[Type[Warning], ...], Literal[False], None]
MatchType = Optional[Union[str, Tuple[Optional[str], ...]]]

@contextmanager
def assert_produces_warning(
    expected_warning: ExpectedWarningType = Warning,
    filter_level: Optional[str] = 'always',
    check_stacklevel: bool = True,
    raise_on_extra_warnings: bool = True,
    match: MatchType = None,
    must_find_all_warnings: bool = True
) -> Iterator[list[warnings.WarningMessage]]:
    __tracebackhide__ = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter(filter_level)
        try:
            yield w
        finally:
            if expected_warning:
                if isinstance(expected_warning, tuple) and must_find_all_warnings:
                    resolved_match: Tuple[Optional[str], ...] = match if isinstance(match, tuple) else (match,) * len(expected_warning)  # type: ignore
                    for warning_type, warning_match in zip(expected_warning, resolved_match):
                        _assert_caught_expected_warnings(
                            caught_warnings=w,
                            expected_warning=warning_type,
                            match=warning_match,
                            check_stacklevel=check_stacklevel
                        )
                else:
                    # If not a tuple or must_find_all_warnings is False
                    expected_warning_cast = cast(Union[Type[Warning], Tuple[Type[Warning], ...]], expected_warning)
                    combined_match: Optional[str]
                    if isinstance(match, tuple):
                        combined_match = '|'.join(m for m in match if m)  # type: ignore
                    else:
                        combined_match = match
                    _assert_caught_expected_warnings(
                        caught_warnings=w,
                        expected_warning=expected_warning_cast,
                        match=combined_match,
                        check_stacklevel=check_stacklevel
                    )
            if raise_on_extra_warnings:
                _assert_caught_no_extra_warnings(
                    caught_warnings=w,
                    expected_warning=expected_warning
                )

def maybe_produces_warning(
    warning: ExpectedWarningType,
    condition: bool,
    **kwargs: Any
) -> AbstractContextManager[Any]:
    if condition:
        return assert_produces_warning(warning, **kwargs)
    else:
        return nullcontext()

def _assert_caught_expected_warnings(
    *,
    caught_warnings: list[warnings.WarningMessage],
    expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...]],
    match: MatchType,
    check_stacklevel: bool
) -> None:
    saw_warning = False
    matched_message = False
    unmatched_messages: list[Any] = []
    if isinstance(expected_warning, tuple):
        warning_name = tuple(x.__name__ for x in expected_warning)
    else:
        warning_name = expected_warning.__name__
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
    if match and (not matched_message):
        raise AssertionError(
            f"Did not see warning {warning_name!r} matching '{match}'. The emitted warning messages are {unmatched_messages}"
        )

def _assert_caught_no_extra_warnings(
    *,
    caught_warnings: list[warnings.WarningMessage],
    expected_warning: ExpectedWarningType
) -> None:
    extra_warnings: list[Tuple[str, Any, str, int]] = []
    for actual_warning in caught_warnings:
        if _is_unexpected_warning(actual_warning, expected_warning):
            if actual_warning.category == ResourceWarning:
                if 'unclosed <ssl.SSLSocket' in str(actual_warning.message):
                    continue
                if any(('matplotlib' in mod for mod in sys.modules)):
                    continue
            if PY311 and actual_warning.category == EncodingWarning:
                continue
            extra_warnings.append((
                actual_warning.category.__name__,
                actual_warning.message,
                actual_warning.filename,
                actual_warning.lineno
            ))
    if extra_warnings:
        raise AssertionError(f"Caused unexpected warning(s): {extra_warnings!r}")

def _is_unexpected_warning(
    actual_warning: warnings.WarningMessage,
    expected_warning: ExpectedWarningType
) -> bool:
    if actual_warning and (not expected_warning):
        return True
    expected_warning_cast = cast(Type[Warning], expected_warning)
    return bool(not issubclass(actual_warning.category, expected_warning_cast))

def _assert_raised_with_correct_stacklevel(
    actual_warning: warnings.WarningMessage
) -> None:
    frame = inspect.currentframe()
    for _ in range(4):
        frame = frame.f_back  # type: ignore
    try:
        caller_filename = inspect.getfile(frame)  # type: ignore
    finally:
        del frame
    msg = (f"Warning not set with correct stacklevel. File where warning is raised: {actual_warning.filename} != {caller_filename}. "
           f"Warning message: {actual_warning.message}")
    assert actual_warning.filename == caller_filename, msg

def cast(typ: Any, obj: Any) -> Any:
    return obj  # This is a dummy cast for type hints purposes.
