#!/usr/bin/env python3

import asyncio
import inspect
import io
import logging
import multiprocessing
import os
import re
import sys
import textwrap
import types
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr
from dataclasses import fields, replace
from io import BytesIO
from pathlib import Path, WindowsPath
from platform import system
from tempfile import TemporaryDirectory
from typing import Any, Optional, TypeVar, Union, Dict, List, Set, Tuple
from unittest.mock import MagicMock, patch

import click
import pytest
from click import unstyle
from click.testing import CliRunner
from pathspec import PathSpec

import black
import black.files
from black import Feature, TargetVersion
from black import re_compile_maybe_verbose as compile_pattern
from black.cache import FileData, get_cache_dir, get_cache_file
from black.debug import DebugVisitor
from black.mode import Mode, Preview
from black.output import color_diff, diff
from black.parsing import ASTSafetyError
from black.report import Report
from black.strings import lines_with_leading_tabs_expanded

# Import other test classes
from tests.util import (
    DATA_DIR,
    DEFAULT_MODE,
    DETERMINISTIC_HEADER,
    PROJECT_ROOT,
    PY36_VERSIONS,
    THIS_DIR,
    BlackBaseTestCase,
    assert_format,
    change_directory,
    dump_to_stderr,
    ff,
    fs,
    get_case_path,
    read_data,
    read_data_from_file,
)

THIS_FILE: Path = Path(__file__)
EMPTY_CONFIG: Path = THIS_DIR / "data" / "empty_pyproject.toml"
PY36_ARGS: List[str] = [f"--target-version={version.name.lower()}" for version in PY36_VERSIONS]
DEFAULT_EXCLUDE: Pattern[str] = black.re_compile_maybe_verbose(black.const.DEFAULT_EXCLUDES)
DEFAULT_INCLUDE: Pattern[str] = black.re_compile_maybe_verbose(black.const.DEFAULT_INCLUDES)
T = TypeVar("T")
R = TypeVar("R")

# Match the time output in a diff, but nothing else
DIFF_TIME: Pattern[str] = re.compile(r"\t[\d\-:+\. ]+")


@contextmanager
def cache_dir(exists: bool = True) -> Iterator[Path]:
    with TemporaryDirectory() as workspace:
        cache_dir = Path(workspace)
        if not exists:
            cache_dir = cache_dir / "new"
        with patch("black.cache.CACHE_DIR", cache_dir):
            yield cache_dir


@contextmanager
def event_loop() -> Iterator[None]:
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        loop.close()


class FakeContext(click.Context):
    """A fake click Context for when calling functions that need it."""

    def __init__(self) -> None:
        self.default_map: Dict[str, Any] = {}
        self.params: Dict[str, Any] = {}
        self.command: click.Command = black.main
        # Dummy root, since most of the tests don't care about it
        self.obj: Dict[str, Any] = {"root": PROJECT_ROOT}


class FakeParameter(click.Parameter):
    """A fake click Parameter for when calling functions that need it."""

    def __init__(self) -> None:
        pass


class BlackRunner(CliRunner):
    """Make sure STDOUT and STDERR are kept separate when testing Black via its CLI."""

    def __init__(self) -> None:
        super().__init__(mix_stderr=False)


def invokeBlack(args: List[str], exit_code: int = 0, ignore_config: bool = True) -> None:
    runner = BlackRunner()
    if ignore_config:
        args = ["--verbose", "--config", str(THIS_DIR / "empty.toml"), *args]
    result = runner.invoke(black.main, args, catch_exceptions=False)
    assert result.stdout_bytes is not None
    assert result.stderr_bytes is not None
    msg = (
        f"Failed with args: {args}\n"
        f"stdout: {result.stdout_bytes.decode()!r}\n"
        f"stderr: {result.stderr_bytes.decode()!r}\n"
        f"exception: {result.exception}"
    )
    assert result.exit_code == exit_code, msg


class BlackTestCase(BlackBaseTestCase):
    invokeBlack = staticmethod(invokeBlack)

    def test_empty_ff(self) -> None:
        expected = ""
        tmp_file = Path(black.dump_to_file())
        try:
            self.assertFalse(ff(tmp_file, write_back=black.WriteBack.YES))
            actual = tmp_file.read_text(encoding="utf-8")
        finally:
            os.unlink(tmp_file)
        self.assertFormatEqual(expected, actual)

    @patch("black.dump_to_file", dump_to_stderr)
    def test_one_empty_line(self) -> None:
        for nl in ["\n", "\r\n"]:
            source = expected = nl
            assert_format(source, expected)

    def test_one_empty_line_ff(self) -> None:
        for nl in ["\n", "\r\n"]:
            expected = nl
            tmp_file = Path(black.dump_to_file(nl))
            if system() == "Windows":
                with open(tmp_file, "wb") as f:
                    f.write(nl.encode("utf-8"))
            try:
                self.assertFalse(ff(tmp_file, write_back=black.WriteBack.YES))
                with open(tmp_file, "rb") as f:
                    actual = f.read().decode("utf-8")
            finally:
                os.unlink(tmp_file)
            self.assertFormatEqual(expected, actual)

    def test_piping(self) -> None:
        _, source, expected = read_data_from_file(
            PROJECT_ROOT / "src/black/__init__.py"
        )
        result = BlackRunner().invoke(
            black.main,
            [
                "-",
                "--fast",
                f"--line-length={black.DEFAULT_LINE_LENGTH}",
                f"--config={EMPTY_CONFIG}",
            ],
            input=BytesIO(source.encode("utf-8")),
        )
        self.assertEqual(result.exit_code, 0)
        self.assertFormatEqual(expected, result.output)
        if source != result.output:
            black.assert_equivalent(source, result.output)
            black.assert_stable(source, result.output, DEFAULT_MODE)

    # ... (rest of the test methods continue with similar type annotations)
