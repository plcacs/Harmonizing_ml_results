# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Set, Tuple

import pytest

from hypothesis.extra._patching import (
    FAIL_MSG,
    HEADER,
    get_patch_for,
    indent,
    make_patch,
)
from hypothesis.internal.compat import WINDOWS

from .callables import WHERE, Cases, covered, fn, undef_name
from .toplevel import WHERE_TOP, fn_top

SIMPLE: Tuple[Callable[..., Any], str, str] = (
    fn,
    ("fn(\n    x=1,\n)", FAIL_MSG),
    indent('@example(x=1).via("discovered failure")', prefix="+"),
)
CASES: Tuple[Callable[..., Any], str, str] = (
    Cases.mth,
    ('mth(\n    n=100,\n    label="a long label which forces a newline",\n)', FAIL_MSG),
    indent(
        '@example(n=100, label="a long label which forces a newline")'
        '.via(\n    "discovered failure"\n)',
        prefix="+    ",
    ),
)
TOPLEVEL: Tuple[Callable[..., Any], str, str] = (
    fn_top,
    ("fn_top(\n    x=1,\n)", FAIL_MSG),
    indent('@hypothesis.example(x=1).via("discovered failure")', prefix="+"),
)
COVERING: Tuple[Callable[..., Any], str, str] = (
    covered,
    ("covered(\n    x=0,\n)", "covering example"),
    indent('@example(x=1).via("covering example")', prefix="-")
    + "\n"
    + indent('@example(x=0).via("covering example")', prefix="+"),
)
UNDEF_NAME: Tuple[Callable[..., Any], str, str] = (
    undef_name,
    ("undef_name(\n    array=array([100], dtype=int8),\n)", FAIL_MSG),
    '+@example(array=np.array([100], dtype=np.int8)).via("discovered failure")',
)


def strip_trailing_whitespace(s: str) -> str:
    """Patches have whitespace-only lines; strip that out."""
    return re.sub(r" +$", "", s, flags=re.MULTILINE)


@pytest.mark.parametrize(
    "tst, example, expected",
    [
        pytest.param(*SIMPLE, id="simple"),
        pytest.param(*CASES, id="cases"),
    ],
)
def test_adds_simple_patch(
    tst: Callable[..., Any], example: str, expected: str
) -> None:
    where: str
    before: str
    after: str
    where, before, after = get_patch_for(tst, [example])
    assert Path(where) == WHERE
    added: Set[str] = set(after.splitlines()) - set(before.splitlines())
    assert added == {line.lstrip("+") for line in expected.splitlines()}


SIMPLE_PATCH_BODY: str = f'''\
--- ./{WHERE}
+++ ./{WHERE}
@@ -21,6 +21,7 @@
 
 
 @given(st.integers())
{{0}}
 def fn(x):
     """A trivial test function."""

'''
CASES_PATCH_BODY: str = f'''\
--- ./{WHERE}
+++ ./{WHERE}
@@ -28,6 +28,9 @@
 class Cases:
     @example(n=0, label="whatever")
     @given(st.integers(), st.text())
{{0}}
     def mth(self, n, label):
         """Indented method with existing example decorator."""

'''
TOPLEVEL_PATCH_BODY: str = f'''\
--- ./{WHERE_TOP}
+++ ./{WHERE_TOP}
@@ -19,5 +19,6 @@
 
 
 @hypothesis.given(st.integers())
{{0}}
 def fn_top(x):
     """A trivial test function."""
'''
COVERING_PATCH_BODY: str = f'''\
--- ./{WHERE}
+++ ./{WHERE}
@@ -34,7 +34,7 @@

 @given(st.integers())
 @example(x=2).via("not a literal when repeated " * 2)
{{0}}
 def covered(x):
     """A test function with a removable explicit example."""

'''
UNDEF_NAME_PATCH_BODY: str = f"""\
--- ./{WHERE}
+++ ./{WHERE}
@@ -40,6 +40,7 @@


 @given(npst.arrays(np.int8, 1))
{{0}}
 def undef_name(array):
     assert sum(array) < 100

"""


@pytest.mark.parametrize(
    "tst, example, expected, body, remove",
    [
        pytest.param(*SIMPLE, SIMPLE_PATCH_BODY, (), id="simple"),
        pytest.param(*CASES, CASES_PATCH_BODY, (), id="cases"),
        pytest.param(*TOPLEVEL, TOPLEVEL_PATCH_BODY, (), id="toplevel"),
        pytest.param(
            *COVERING, COVERING_PATCH_BODY, ("covering example",), id="covering"
        ),
        pytest.param(*UNDEF_NAME, UNDEF_NAME_PATCH_BODY, (), id="undef_name"),
    ],
)
def test_make_full_patch(
    tst: Callable[..., Any],
    example: str,
    expected: str,
    body: str,
    remove: Tuple[str, ...],
) -> None:
    when: datetime = datetime.now()
    msg: str = "a message from the test"
    expected_full: str = HEADER.format(when=when, msg=msg) + body.format(expected)

    triples: Optional[Tuple[str, str, str]] = get_patch_for(tst, [example], strip_via=remove)
    assert triples is not None
    patch: str = make_patch([triples], when=when, msg=msg)
    stripped: str = strip_trailing_whitespace(patch)

    assert stripped.splitlines() == expected_full.splitlines()


@pytest.mark.parametrize("n", [0, 1, 2])
def test_invalid_syntax_cases_dropped(n: int) -> None:
    tst: Callable[..., Any]
    ex: str
    via: str
    expected: str
    tst, (ex, via), expected = SIMPLE
    example_ls: list[Tuple[str, str]] = [(ex.replace("x=1", f"x={x}"), via) for x in range(n)]
    example_ls.insert(-1, ("fn(\n    x=<__main__.Cls object at 0x>,\n)", FAIL_MSG))

    got: Optional[Tuple[str, str, str]] = get_patch_for(tst, example_ls)
    if n == 0:
        assert got is None, "no valid examples, and hence no patch"
        return
    where: str
    _, _, after: str
    where, _, after = got  # type: ignore

    assert Path(where) == WHERE
    assert after.count("@example(x=") == n


def test_no_example_for_data_strategy() -> None:
    assert get_patch_for(fn, [("fn(data=data(...))", "msg")]) is None
    assert get_patch_for(fn, [("fn(123, data(...))", "msg")]) is None

    assert get_patch_for(fn, [("fn(data='data(...)')", "msg")]) is not None
    assert get_patch_for(fn, [("fn(Foo(data=data(...)))", "msg")]) is not None


def test_deduplicates_examples() -> None:
    tst: Callable[..., Any]
    example: str
    expected: str
    tst, example, expected = SIMPLE
    where: str
    before: str
    after: str
    where, before, after = get_patch_for(tst, [example, example])
    assert Path(where) == WHERE
    assert after.count(expected.lstrip("+")) == 1


def test_irretrievable_callable() -> None:
    # Check that we return None instead of raising an exception
    old_module: str = fn.__module__
    try:
        fn.__module__ = "this.does.not.exist"
        triple: Optional[Tuple[str, str, str]] = get_patch_for(fn, [(SIMPLE[1], FAIL_MSG)])
    finally:
        fn.__module__ = old_module
    assert triple is None


TESTSCRIPT_DUMPS_PATCH: str = """
from hypothesis import Phase, given, settings, strategies as st

@settings(phases=list(Phase))
@given(st.integers(0, 10), st.integers(0, 10))
def test_failing_pbt(x, y):
    assert not x
"""
ADDED_LINES: str = """
+@example(
+    x=1,
+    y=0,  # or any other generated value
+).via("discovered failure")
"""


@pytest.mark.skipif(WINDOWS, reason="backslash support is tricky")
def test_pytest_reports_patch_file_location(pytester: Any) -> None:
    script: Path = pytester.makepyfile(TESTSCRIPT_DUMPS_PATCH)
    result: Any = pytester.runpytest(script)
    result.assert_outcomes(failed=1)

    fname_pat: str = r"\.hypothesis/patches/\d{4}-\d\d-\d\d--[0-9a-f]{8}.patch"
    pattern: str = f"`git apply ({fname_pat})` to add failing examples to your code\\."
    print(f"{pattern=}")
    print(f"result.stdout=\n{indent(str(result.stdout), '    ')}")
    match: Optional[re.Match[str]] = re.search(pattern, str(result.stdout))
    assert match is not None
    fname: str = match.group(1)
    patch: str = Path(pytester.path).joinpath(fname).read_text(encoding="utf-8")
    print(patch)
    assert ADDED_LINES in patch
