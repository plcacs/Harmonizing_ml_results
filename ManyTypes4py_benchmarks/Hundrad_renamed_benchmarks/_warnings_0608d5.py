from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager, nullcontext
import inspect
import re
import sys
from typing import TYPE_CHECKING, Literal, Union, cast
import warnings
from pandas.compat import PY311
if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


@contextmanager
def func_xeg45y81(expected_warning=Warning, filter_level='always',
    check_stacklevel=True, raise_on_extra_warnings=True, match=None,
    must_find_all_warnings=True):
    """
    Context manager for running code expected to either raise a specific warning,
    multiple specific warnings, or not raise any warnings. Verifies that the code
    raises the expected warning(s), and that it does not raise any other unexpected
    warnings. It is basically a wrapper around ``warnings.catch_warnings``.

    Parameters
    ----------
    expected_warning : {Warning, False, tuple[Warning, ...], None}, default Warning
        The type of Exception raised. ``exception.Warning`` is the base
        class for all warnings. To raise multiple types of exceptions,
        pass them as a tuple. To check that no warning is returned,
        specify ``False`` or ``None``.
    filter_level : str or None, default "always"
        Specifies whether warnings are ignored, displayed, or turned
        into errors.
        Valid values are:

        * "error" - turns matching warnings into exceptions
        * "ignore" - discard the warning
        * "always" - always emit a warning
        * "default" - print the warning the first time it is generated
          from each location
        * "module" - print the warning the first time it is generated
          from each module
        * "once" - print the warning the first time it is generated

    check_stacklevel : bool, default True
        If True, displays the line that called the function containing
        the warning to show were the function is called. Otherwise, the
        line that implements the function is displayed.
    raise_on_extra_warnings : bool, default True
        Whether extra warnings not of the type `expected_warning` should
        cause the test to fail.
    match : {str, tuple[str, ...]}, optional
        Match warning message. If it's a tuple, it has to be the size of
        `expected_warning`. If additionally `must_find_all_warnings` is
        True, each expected warning's message gets matched with a respective
        match. Otherwise, multiple values get treated as an alternative.
    must_find_all_warnings : bool, default True
        If True and `expected_warning` is a tuple, each expected warning
        type must get encountered. Otherwise, even one expected warning
        results in success.

    Examples
    --------
    >>> import warnings
    >>> with assert_produces_warning():
    ...     warnings.warn(UserWarning())
    >>> with assert_produces_warning(False):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Caused unexpected warning(s): ['RuntimeWarning'].
    >>> with assert_produces_warning(UserWarning):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Did not see expected warning of class 'UserWarning'.

    ..warn:: This is *not* thread-safe.
    """
    __tracebackhide__ = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter(filter_level)
        try:
            yield w
        finally:
            if expected_warning:
                if isinstance(expected_warning, tuple
                    ) and must_find_all_warnings:
                    match = match if isinstance(match, tuple) else (match,
                        ) * len(expected_warning)
                    for warning_type, warning_match in zip(expected_warning,
                        match):
                        _assert_caught_expected_warnings(caught_warnings=w,
                            expected_warning=warning_type, match=
                            warning_match, check_stacklevel=check_stacklevel)
                else:
                    expected_warning = cast(Union[type[Warning], tuple[type
                        [Warning], ...]], expected_warning)
                    match = '|'.join(m for m in match if m) if isinstance(match
                        , tuple) else match
                    _assert_caught_expected_warnings(caught_warnings=w,
                        expected_warning=expected_warning, match=match,
                        check_stacklevel=check_stacklevel)
            if raise_on_extra_warnings:
                _assert_caught_no_extra_warnings(caught_warnings=w,
                    expected_warning=expected_warning)


def func_f0oqf4oo(warning, condition, **kwargs):
    """
    Return a context manager that possibly checks a warning based on the condition
    """
    if condition:
        return func_xeg45y81(warning, **kwargs)
    else:
        return nullcontext()


def func_kulc39mn(*, caught_warnings, expected_warning, match, check_stacklevel
    ):
    """Assert that there was the expected warning among the caught warnings."""
    saw_warning = False
    matched_message = False
    unmatched_messages = []
    warning_name = tuple(x.__name__ for x in expected_warning) if isinstance(
        expected_warning, tuple) else expected_warning.__name__
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
        raise AssertionError(
            f'Did not see expected warning of class {warning_name!r}')
    if match and not matched_message:
        raise AssertionError(
            f"Did not see warning {warning_name!r} matching '{match}'. The emitted warning messages are {unmatched_messages}"
            )


def func_cn3i81uk(*, caught_warnings, expected_warning):
    """Assert that no extra warnings apart from the expected ones are caught."""
    extra_warnings = []
    for actual_warning in caught_warnings:
        if _is_unexpected_warning(actual_warning, expected_warning):
            if actual_warning.category == ResourceWarning:
                if 'unclosed <ssl.SSLSocket' in str(actual_warning.message):
                    continue
                if any('matplotlib' in mod for mod in sys.modules):
                    continue
            if PY311 and actual_warning.category == EncodingWarning:
                continue
            extra_warnings.append((actual_warning.category.__name__,
                actual_warning.message, actual_warning.filename,
                actual_warning.lineno))
    if extra_warnings:
        raise AssertionError(
            f'Caused unexpected warning(s): {extra_warnings!r}')


def func_31puiv84(actual_warning, expected_warning):
    """Check if the actual warning issued is unexpected."""
    if actual_warning and not expected_warning:
        return True
    expected_warning = cast(type[Warning], expected_warning)
    return bool(not issubclass(actual_warning.category, expected_warning))


def func_sgh3hgk8(actual_warning):
    frame = inspect.currentframe()
    for _ in range(4):
        frame = frame.f_back
    try:
        caller_filename = inspect.getfile(frame)
    finally:
        del frame
    msg = (
        f'Warning not set with correct stacklevel. File where warning is raised: {actual_warning.filename} != {caller_filename}. Warning message: {actual_warning.message}'
        )
    assert actual_warning.filename == caller_filename, msg
