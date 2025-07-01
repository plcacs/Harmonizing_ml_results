from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager, nullcontext
import inspect
import re
import sys
from typing import TYPE_CHECKING, Literal, Union, cast, Optional, Tuple, List
import warnings
from pandas.compat import PY311
if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

@contextmanager
def assert_produces_warning(
    expected_warning: Union[type[Warning], Tuple[type[Warning], ...], Literal[False], None] = Warning,
    filter_level: Optional[str] = 'always',
    check_stacklevel: bool = True,
    raise_on_extra_warnings: bool = True,
    match: Optional[Union[str, Tuple[str, ...]]] = None,
    must_find_all_warnings: bool = True
) -> Generator[List[warnings.WarningMessage], None, None]:
    __tracebackhide__ = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter(filter_level)
        try:
            yield w
        finally:
            if expected_warning:
                if isinstance(expected_warning, tuple) and must_find_all_warnings:
                    match = match if isinstance(match, tuple) else (match,) * len(expected_warning)
                    for warning_type, warning_match in zip(expected_warning, match):
                        _assert_caught_expected_warnings(
                            caught_warnings=w,
                            expected_warning=warning_type,
                            match=warning_match,
                            check_stacklevel=check_stacklevel
                        )
                else:
                    expected_warning = cast(Union[type[Warning], Tuple[type[Warning], ...]], expected_warning)
                    match = '|'.join((m for m in match if m)) if isinstance(match, tuple) else match
                    _assert_caught_expected_warnings(
                        caught_warnings=w,
                        expected_warning=expected_warning,
                        match=match,
                        check_stacklevel=check_stacklevel
                    )
            if raise_on_extra_warnings:
                _assert_caught_no_extra_warnings(caught_warnings=w, expected_warning=expected_warning)

def maybe_produces_warning(
    warning: Union[type[Warning], Tuple[type[Warning], ...], Literal[False], None],
    condition: bool,
    **kwargs
) -> AbstractContextManager:
    if condition:
        return assert_produces_warning(warning, **kwargs)
    else:
        return nullcontext()

def _assert_caught_expected_warnings(
    *,
    caught_warnings: List[warnings.WarningMessage],
    expected_warning: Union[type[Warning], Tuple[type[Warning], ...]],
    match: Optional[str],
    check_stacklevel: bool
) -> None:
    saw_warning = False
    matched_message = False
    unmatched_messages = []
    warning_name = tuple((x.__name__ for x in expected_warning)) if isinstance(expected_warning, tuple) else expected_warning.__name__
    for actual_warning in caught_warnings:
        if issubclass(actual_warning.category, expected_warning):
            saw_warning = True
            if check_stacklevel:
                _assert_raised_with_correct_stacklevel(actual_warning)
            if match is not None:
                if re.search(match, str(actual_warning.message)):
                    matched_message = True
                else:
                    unmatched_messages.append(actual_warning.message)
    if not saw_warning:
        raise AssertionError(f'Did not see expected warning of class {warning_name!r}')
    if match and (not matched_message):
        raise AssertionError(f"Did not see warning {warning_name!r} matching '{match}'. The emitted warning messages are {unmatched_messages}")

def _assert_caught_no_extra_warnings(
    *,
    caught_warnings: List[warnings.WarningMessage],
    expected_warning: Union[type[Warning], Tuple[type[Warning], ...], Literal[False], None]
) -> None:
    extra_warnings = []
    for actual_warning in caught_warnings:
        if _is_unexpected_warning(actual_warning, expected_warning):
            if actual_warning.category == ResourceWarning:
                if 'unclosed <ssl.SSLSocket' in str(actual_warning.message):
                    continue
                if any(('matplotlib' in mod for mod in sys.modules)):
                    continue
            if PY311 and actual_warning.category == EncodingWarning:
                continue
            extra_warnings.append((actual_warning.category.__name__, actual_warning.message, actual_warning.filename, actual_warning.lineno))
    if extra_warnings:
        raise AssertionError(f'Caused unexpected warning(s): {extra_warnings!r}')

def _is_unexpected_warning(
    actual_warning: warnings.WarningMessage,
    expected_warning: Union[type[Warning], Tuple[type[Warning], ...], Literal[False], None]
) -> bool:
    if actual_warning and (not expected_warning):
        return True
    expected_warning = cast(type[Warning], expected_warning)
    return bool(not issubclass(actual_warning.category, expected_warning))

def _assert_raised_with_correct_stacklevel(actual_warning: warnings.WarningMessage) -> None:
    frame = inspect.currentframe()
    for _ in range(4):
        frame = frame.f_back
    try:
        caller_filename = inspect.getfile(frame)
    finally:
        del frame
    msg = f'Warning not set with correct stacklevel. File where warning is raised: {actual_warning.filename} != {caller_filename}. Warning message: {actual_warning.message}'
    assert actual_warning.filename == caller_filename, msg
