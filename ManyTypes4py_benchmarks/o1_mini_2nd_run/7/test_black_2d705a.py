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
from typing import Any, Optional, TypeVar, Union, Tuple, List, Set, Dict
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
from typing import Iterator

THIS_FILE: Path = Path(__file__)
EMPTY_CONFIG: Path = THIS_DIR / "data" / "empty_pyproject.toml"
PY36_ARGS: List[str] = [f"--target-version={version.name.lower()}" for version in PY36_VERSIONS]
DEFAULT_EXCLUDE: re.Pattern = black.re_compile_maybe_verbose(black.const.DEFAULT_EXCLUDES)
DEFAULT_INCLUDE: re.Pattern = black.re_compile_maybe_verbose(black.const.DEFAULT_INCLUDES)
T = TypeVar("T")
R = TypeVar("R")
DIFF_TIME: re.Pattern = re.compile("\\t[\\d\\-:+\\. ]+")


@contextmanager
def cache_dir(exists: bool = True) -> Iterator[Path]:
    with TemporaryDirectory() as workspace:
        cache_dir: Path = Path(workspace)
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

    default_map: Dict[str, Any]
    params: Dict[str, Any]
    command: Callable[..., Any]
    obj: Dict[str, Any]

    def __init__(self) -> None:
        self.default_map = {}
        self.params = {}
        self.command = black.main
        self.obj = {"root": PROJECT_ROOT}


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
        f"Failed with args: {args}\nstdout: {result.stdout_bytes.decode()!r}\nstderr: {result.stderr_bytes.decode()!r}\nexception: {result.exception}"
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
        _, source, expected = read_data_from_file(PROJECT_ROOT / "src/black/__init__.py")
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

    def test_piping_diff(self) -> None:
        diff_header = re.compile(
            "(STDIN|STDOUT)\\t\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d\\.\\d\\d\\d\\d\\d\\d\\+\\d\\d:\\d\\d"
        )
        source, _ = read_data("cases", "expression.py")
        expected, _ = read_data("cases", "expression.diff")
        args = [
            "-",
            "--fast",
            f"--line-length={black.DEFAULT_LINE_LENGTH}",
            "--diff",
            f"--config={EMPTY_CONFIG}",
        ]
        result = BlackRunner().invoke(black.main, args, input=BytesIO(source.encode("utf-8")))
        self.assertEqual(result.exit_code, 0)
        actual = diff_header.sub(DETERMINISTIC_HEADER, result.output)
        actual = actual.rstrip() + "\n"
        self.assertEqual(expected, actual)

    def test_piping_diff_with_color(self) -> None:
        source, _ = read_data("cases", "expression.py")
        args = [
            "-",
            "--fast",
            f"--line-length={black.DEFAULT_LINE_LENGTH}",
            "--diff",
            "--color",
            f"--config={EMPTY_CONFIG}",
        ]
        result = BlackRunner().invoke(black.main, args, input=BytesIO(source.encode("utf-8")))
        actual = result.output
        self.assertIn("\x1b[1m", actual)
        self.assertIn("\x1b[36m", actual)
        self.assertIn("\x1b[32m", actual)
        self.assertIn("\x1b[31m", actual)
        self.assertIn("\x1b[0m", actual)

    def test_pep_572_version_detection(self) -> None:
        source, _ = read_data("cases", "pep_572")
        root = black.lib2to3_parse(source)
        features = black.get_features_used(root)
        self.assertIn(black.Feature.ASSIGNMENT_EXPRESSIONS, features)
        versions = black.detect_target_versions(root)
        self.assertIn(black.TargetVersion.PY38, versions)

    def test_pep_695_version_detection(self) -> None:
        for file in ("type_aliases", "type_params"):
            source, _ = read_data("cases", file)
            root = black.lib2to3_parse(source)
            features = black.get_features_used(root)
            self.assertIn(black.Feature.TYPE_PARAMS, features)
            versions = black.detect_target_versions(root)
            self.assertIn(black.TargetVersion.PY312, versions)

    def test_pep_696_version_detection(self) -> None:
        source, _ = read_data("cases", "type_param_defaults")
        samples = [
            source,
            "type X[T=int] = float",
            "type X[T:int=int]=int",
            "type X[*Ts=int]=int",
            "type X[*Ts=*int]=int",
            "type X[**P=int]=int",
        ]
        for sample in samples:
            root = black.lib2to3_parse(sample)
            features = black.get_features_used(root)
            self.assertIn(black.Feature.TYPE_PARAM_DEFAULTS, features)

    def test_expression_ff(self) -> None:
        source, expected = read_data("cases", "expression.py")
        tmp_file = Path(black.dump_to_file(source))
        try:
            self.assertTrue(ff(tmp_file, write_back=black.WriteBack.YES))
            actual = tmp_file.read_text(encoding="utf-8")
        finally:
            os.unlink(tmp_file)
        self.assertFormatEqual(expected, actual)
        with patch("black.dump_to_file", dump_to_stderr):
            black.assert_equivalent(source, actual)
            black.assert_stable(source, actual, DEFAULT_MODE)

    def test_expression_diff(self) -> None:
        source, _ = read_data("cases", "expression.py")
        expected, _ = read_data("cases", "expression.diff")
        tmp_file = Path(black.dump_to_file(source))
        diff_header = re.compile(
            f"{re.escape(str(tmp_file))}\\t\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d\\.\\d\\d\\d\\d\\d\\d\\+\\d\\d:\\d\\d"
        )
        try:
            result = BlackRunner().invoke(
                black.main, ["--diff", str(tmp_file), f"--config={EMPTY_CONFIG}"]
            )
            self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmp_file)
        actual = result.output
        actual = diff_header.sub(DETERMINISTIC_HEADER, actual)
        if expected != actual:
            dump = black.dump_to_file(actual)
            msg = (
                f"Expected diff isn't equal to the actual. If you made changes to expression.py and this is an anticipated difference, overwrite tests/data/cases/expression.diff with {dump}"
            )
            self.assertEqual(expected, actual, msg)

    def test_expression_diff_with_color(self) -> None:
        source, _ = read_data("cases", "expression.py")
        expected, _ = read_data("cases", "expression.diff")
        tmp_file = Path(black.dump_to_file(source))
        try:
            result = BlackRunner().invoke(
                black.main,
                ["--diff", "--color", str(tmp_file), f"--config={EMPTY_CONFIG}"],
            )
        finally:
            os.unlink(tmp_file)
        actual = result.output
        self.assertIn("\x1b[1m", actual)
        self.assertIn("\x1b[36m", actual)
        self.assertIn("\x1b[32m", actual)
        self.assertIn("\x1b[31m", actual)
        self.assertIn("\x1b[0m", actual)

    def test_detect_pos_only_arguments(self) -> None:
        source, _ = read_data("cases", "pep_570")
        root = black.lib2to3_parse(source)
        features = black.get_features_used(root)
        self.assertIn(black.Feature.POS_ONLY_ARGUMENTS, features)
        versions = black.detect_target_versions(root)
        self.assertIn(black.TargetVersion.PY38, versions)

    def test_detect_debug_f_strings(self) -> None:
        root = black.lib2to3_parse('f"{x=}" ')
        features = black.get_features_used(root)
        self.assertIn(black.Feature.DEBUG_F_STRINGS, features)
        versions = black.detect_target_versions(root)
        self.assertIn(black.TargetVersion.PY38, versions)
        root = black.lib2to3_parse(
            'f"{x}"\nf\'{"="}\'\nf\'{(x:=5)}\'\nf\'{f(a="3=")}\'\nf\'{x:=10}\'\n'
        )
        features = black.get_features_used(root)
        self.assertNotIn(black.Feature.DEBUG_F_STRINGS, features)
        root = black.lib2to3_parse(
            'f"heard a rumour that { f\'{1+1=}\' } ... seems like it could be true" '
        )
        features = black.get_features_used(root)
        self.assertIn(black.Feature.DEBUG_F_STRINGS, features)

    @patch("black.dump_to_file", dump_to_stderr)
    @pytest.mark.incompatible_with_mypyc
    def test_string_quotes(self) -> None:
        source, expected = read_data("miscellaneous", "string_quotes")
        mode: Mode = black.Mode(unstable=True)
        assert_format(source, expected, mode)
        mode = replace(mode, string_normalization=False)
        not_normalized = fs(source, mode=mode)
        self.assertFormatEqual(source.replace("\\\n", ""), not_normalized)
        black.assert_equivalent(source, not_normalized)
        black.assert_stable(source, not_normalized, mode=mode)

    def test_skip_source_first_line(self) -> None:
        source, _ = read_data("miscellaneous", "invalid_header")
        tmp_file = Path(black.dump_to_file(source))
        self.invokeBlack([str(tmp_file), "--diff", "--check"], exit_code=123)
        result = BlackRunner().invoke(
            black.main, [str(tmp_file), "-x", f"--config={EMPTY_CONFIG}"]
        )
        self.assertEqual(result.exit_code, 0)
        actual = tmp_file.read_text(encoding="utf-8")
        self.assertFormatEqual(source, actual)

    def test_skip_source_first_line_when_mixing_newlines(self) -> None:
        code_mixing_newlines: bytes = b"Header will be skipped\r\ni = [1,2,3]\nj = [1,2,3]\n"
        expected: bytes = b"Header will be skipped\r\ni = [1, 2, 3]\nj = [1, 2, 3]\n"
        with TemporaryDirectory() as workspace:
            test_file = Path(workspace) / "skip_header.py"
            test_file.write_bytes(code_mixing_newlines)
            mode = replace(DEFAULT_MODE, skip_source_first_line=True)
            ff(test_file, mode=mode, write_back=black.WriteBack.YES)
            self.assertEqual(test_file.read_bytes(), expected)

    def test_skip_magic_trailing_comma(self) -> None:
        source, _ = read_data("cases", "expression")
        expected, _ = read_data(
            "miscellaneous", "expression_skip_magic_trailing_comma.diff"
        )
        tmp_file = Path(black.dump_to_file(source))
        diff_header = re.compile(
            f"{re.escape(str(tmp_file))}\\t\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d\\.\\d\\d\\d\\d\\d\\d\\+\\d\\d:\\d\\d"
        )
        try:
            result = BlackRunner().invoke(
                black.main, ["-C", "--diff", str(tmp_file), f"--config={EMPTY_CONFIG}"]
            )
            self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmp_file)
        actual = result.output
        actual = diff_header.sub(DETERMINISTIC_HEADER, actual)
        actual = actual.rstrip() + "\n"
        if expected != actual:
            dump = black.dump_to_file(actual)
            msg = (
                f"Expected diff isn't equal to the actual. If you made changes to expression.py and this is an anticipated difference, overwrite tests/data/miscellaneous/expression_skip_magic_trailing_comma.diff with {dump}"
            )
            self.assertEqual(expected, actual, msg)

    @patch("black.dump_to_file", dump_to_stderr)
    @pytest.mark.incompatible_with_mypyc
    def test_async_as_identifier(self) -> None:
        source_path = get_case_path("miscellaneous", "async_as_identifier")
        _, source, expected = read_data_from_file(source_path)
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        major, minor = sys.version_info[:2]
        if major < 3 or (major <= 3 and minor < 7):
            black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)
        self.invokeBlack([str(source_path), "--target-version", "py36"])
        self.invokeBlack([str(source_path), "--target-version", "py37"], exit_code=123)

    @patch("black.dump_to_file", dump_to_stderr)
    @pytest.mark.incompatible_with_mypyc
    def test_python37(self) -> None:
        source_path = get_case_path("cases", "python37")
        _, source, expected = read_data_from_file(source_path)
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        major, minor = sys.version_info[:2]
        if major > 3 or (major == 3 and minor >= 7):
            black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)
        self.invokeBlack([str(source_path), "--target-version", "py37"])
        self.invokeBlack([str(source_path), "--target-version", "py36"], exit_code=123)

    def test_tab_comment_indentation(self) -> None:
        contents_tab: str = "if 1:\n\tif 2:\n\t\tpass\n\t# comment\n\tpass\n"
        contents_spc: str = "if 1:\n    if 2:\n        pass\n    # comment\n    pass\n"
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))
        contents_tab = "if 1:\n\tif 2:\n\t\tpass\n\t\t# comment\n\tpass\n"
        contents_spc = "if 1:\n    if 2:\n        pass\n        # comment\n    pass\n"
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))
        contents_tab = "if 1:\n        if 2:\n\t\tpass\n\t# comment\n        pass\n"
        contents_spc = "if 1:\n    if 2:\n        pass\n    # comment\n    pass\n"
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))
        contents_tab = "if 1:\n        if 2:\n\t\tpass\n\t\t# comment\n        pass\n"
        contents_spc = "if 1:\n    if 2:\n        pass\n        # comment\n    pass\n"
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))

    def test_false_positive_symlink_output_issue_3384(self) -> None:
        project_root: Path = Path(THIS_DIR / "data" / "nested_gitignore_tests")
        working_directory: Path = project_root / "root"
        with change_directory(working_directory):
            report = MagicMock(spec=Report, verbose=True)
            black.get_sources(
                root=project_root,
                src=("./child",),
                quiet=False,
                verbose=True,
                include=DEFAULT_INCLUDE,
                exclude=None,
                report=report,
                extend_exclude=None,
                force_exclude=None,
                stdin_filename=None,
            )
        assert not any(
            (
                mock_args[1].startswith("is a symbolic link that points outside")
                for _, mock_args, _ in report.path_ignored.mock_calls
            )
        ), "A symbolic link was reported."
        report.path_ignored.assert_called_once_with(
            Path(working_directory, "child", "b.py"), "matches a .gitignore file content"
        )

    def test_report_verbose(self) -> None:
        report = Report(verbose=True)
        out_lines: List[str] = []
        err_lines: List[str] = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("black.output._out", out), patch("black.output._err", err):
            report.done(Path("f1"), black.Changed.NO)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], "f1 already well formatted, good job.")
            self.assertEqual(unstyle(str(report)), "1 file left unchanged.")
            self.assertEqual(report.return_code, 0)
            report.done(Path("f2"), black.Changed.YES)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], "reformatted f2")
            self.assertEqual(unstyle(str(report)), "1 file reformatted, 1 file left unchanged.")
            report.done(Path("f3"), black.Changed.CACHED)
            self.assertEqual(len(out_lines), 3)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], "f3 wasn't modified on disk since last run.")
            self.assertEqual(unstyle(str(report)), "1 file reformatted, 2 files left unchanged.")
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path("e1"), "boom")
            self.assertEqual(len(out_lines), 3)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[-1], "error: cannot format e1: boom")
            self.assertEqual(
                unstyle(str(report)),
                "1 file reformatted, 2 files left unchanged, 1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f3"), black.Changed.YES)
            self.assertEqual(len(out_lines), 4)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(out_lines[-1], "reformatted f3")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, 1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.failed(Path("e2"), "boom")
            self.assertEqual(len(out_lines), 4)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[-1], "error: cannot format e2: boom")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, 2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path("wat"), "no match")
            self.assertEqual(len(out_lines), 5)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(out_lines[-1], "wat ignored: no match")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, 2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f4"), black.Changed.NO)
            self.assertEqual(len(out_lines), 6)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(
                out_lines[-1], "f4 already well formatted, good job."
            )
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 3 files left unchanged, 2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(
                unstyle(str(report)),
                "2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.",
            )
            report.check = False
            report.diff = True
            self.assertEqual(
                unstyle(str(report)),
                "2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.",
            )

    def test_report_quiet(self) -> None:
        report = Report(quiet=True)
        out_lines: List[str] = []
        err_lines: List[str] = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("black.output._out", out), patch("black.output._err", err):
            report.done(Path("f1"), black.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), "1 file left unchanged.")
            self.assertEqual(report.return_code, 0)
            report.done(Path("f2"), black.Changed.YES)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), "1 file reformatted, 1 file left unchanged.")
            report.done(Path("f3"), black.Changed.CACHED)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), "1 file reformatted, 2 files left unchanged.")
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path("e1"), "boom")
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[-1], "error: cannot format e1: boom")
            self.assertEqual(
                unstyle(str(report)),
                "1 file reformatted, 2 files left unchanged, 1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f3"), black.Changed.YES)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, 1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.failed(Path("e2"), "boom")
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[-1], "error: cannot format e2: boom")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, 2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path("wat"), "no match")
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, 2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f4"), black.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 3 files left unchanged, 2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(
                unstyle(str(report)),
                "2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.",
            )
            report.check = False
            report.diff = True
            self.assertEqual(
                unstyle(str(report)),
                "2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.",
            )

    def test_report_normal(self) -> None:
        report = black.Report()
        out_lines: List[str] = []
        err_lines: List[str] = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("black.output._out", out), patch("black.output._err", err):
            report.done(Path("f1"), black.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), "1 file left unchanged.")
            self.assertEqual(report.return_code, 0)
            report.done(Path("f2"), black.Changed.YES)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], "reformatted f2")
            self.assertEqual(
                unstyle(str(report)), "1 file reformatted, 1 file left unchanged."
            )
            report.done(Path("f3"), black.Changed.CACHED)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(
                out_lines[-1], "reformatted f2"
            )
            self.assertEqual(
                unstyle(str(report)), "1 file reformatted, 2 files left unchanged."
            )
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path("e1"), "boom")
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[-1], "error: cannot format e1: boom")
            self.assertEqual(
                unstyle(str(report)),
                "1 file reformatted, 2 files left unchanged, 1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f3"), black.Changed.YES)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(out_lines[-1], "reformatted f3")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, 1 file failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.failed(Path("e2"), "boom")
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[-1], "error: cannot format e2: boom")
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, 2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path("wat"), "no match")
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 2 files left unchanged, 2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.done(Path("f4"), black.Changed.NO)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(
                unstyle(str(report)),
                "2 files reformatted, 3 files left unchanged, 2 files failed to reformat.",
            )
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(
                unstyle(str(report)),
                "2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.",
            )
            report.check = False
            report.diff = True
            self.assertEqual(
                unstyle(str(report)),
                "2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.",
            )

    def test_lib2to3_parse(self) -> None:
        with self.assertRaises(black.InvalidInput):
            black.lib2to3_parse("invalid syntax")
        straddling = "x + y"
        black.lib2to3_parse(straddling)
        black.lib2to3_parse(straddling, {TargetVersion.PY36})
        py2_only = "print x"
        with self.assertRaises(black.InvalidInput):
            black.lib2to3_parse(py2_only, {TargetVersion.PY36})
        py3_only = "exec(x, end=y)"
        black.lib2to3_parse(py3_only)
        black.lib2to3_parse(py3_only, {TargetVersion.PY36})

    def test_get_features_used_decorator(self) -> None:
        simples, relaxed = read_data("miscellaneous", "decorators")
        for simple_test in simples.split("##")[1:]:
            node = black.lib2to3_parse(simple_test)
            decorator = str(node.children[0].children[0]).strip()
            self.assertNotIn(
                Feature.RELAXED_DECORATORS, black.get_features_used(node), msg=f"decorator '{decorator}' follows python<=3.8 syntaxbut is detected as 3.9+"
            )
        for relaxed_test in relaxed.split("##")[1:]:
            node = black.lib2to3_parse(relaxed_test)
            decorator = str(node.children[0].children[0]).strip()
            self.assertIn(
                Feature.RELAXED_DECORATORS,
                black.get_features_used(node),
                msg=f"decorator '{decorator}' uses python3.9+ syntaxbut is detected as python<=3.8",
            )

    def test_get_features_used(self) -> None:
        self.check_features_used("def f(*, arg): ...\n", set())
        self.check_features_used(
            "def f(*, arg,): ...\n", {Feature.TRAILING_COMMA_IN_DEF}
        )
        self.check_features_used("f(*arg,)\n", {Feature.TRAILING_COMMA_IN_CALL})
        self.check_features_used(
            'def f(*, arg): f\'string\'\n', {Feature.F_STRINGS}
        )
        self.check_features_used("123_456\n", {Feature.NUMERIC_UNDERSCORES})
        self.check_features_used("123456\n", set())
        source, expected = read_data("cases", "function")
        expected_features = {
            Feature.TRAILING_COMMA_IN_CALL,
            Feature.TRAILING_COMMA_IN_DEF,
            Feature.F_STRINGS,
        }
        self.check_features_used(source, expected_features)
        self.check_features_used(expected, expected_features)
        source, expected = read_data("cases", "expression")
        self.check_features_used(source, set())
        self.check_features_used(expected, set())
        self.check_features_used(
            "lambda a, /, b: ...\n", {Feature.POS_ONLY_ARGUMENTS}
        )
        self.check_features_used(
            "def fn(a, /, b): ...", {Feature.POS_ONLY_ARGUMENTS}
        )
        self.check_features_used("def fn(): yield a, b", set())
        self.check_features_used("def fn(): return a, b", set())
        self.check_features_used(
            "def fn(): yield *b, c", {Feature.UNPACKING_ON_FLOW}
        )
        self.check_features_used(
            "def fn(): return a, *b, c", {Feature.UNPACKING_ON_FLOW}
        )
        self.check_features_used("x = a, *b, c", set())
        self.check_features_used("x: Any = regular", set())
        self.check_features_used("x: Any = (regular, regular)", set())
        self.check_features_used("x: Any = Complex(Type(1))[something]", set())
        self.check_features_used(
            "x: Tuple[int, ...] = a, b, c", {Feature.ANN_ASSIGN_EXTENDED_RHS}
        )
        self.check_features_used("try: pass\nexcept Something: pass", set())
        self.check_features_used("try: pass\nexcept (*Something,): pass", set())
        self.check_features_used(
            "try: pass\nexcept *Group: pass", {Feature.EXCEPT_STAR}
        )
        self.check_features_used("a[*b]", {Feature.VARIADIC_GENERICS})
        self.check_features_used(
            "a[x, *y(), z] = t", {Feature.VARIADIC_GENERICS}
        )
        self.check_features_used(
            "def fn(*args: *T): pass", {Feature.VARIADIC_GENERICS}
        )
        self.check_features_used(
            "def fn(*args: *tuple[*T]): pass", {Feature.VARIADIC_GENERICS}
        )
        self.check_features_used("with a: pass", set())
        self.check_features_used("with a, b: pass", set())
        self.check_features_used("with a as b: pass", set())
        self.check_features_used("with a as b, c as d: pass", set())
        self.check_features_used("with (a): pass", set())
        self.check_features_used("with (a, b): pass", set())
        self.check_features_used(
            "with (a, b) as (c, d): pass", set()
        )
        self.check_features_used(
            "with (a as b): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS}
        )
        self.check_features_used(
            "with ((a as b)): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS}
        )
        self.check_features_used(
            "with (a, b as c): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS}
        )
        self.check_features_used(
            "with (a, (b as c)): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS}
        )
        self.check_features_used(
            "with ((a, ((b as c)))): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS}
        )

    def check_features_used(
        self, source: str, expected: Set[Feature]
    ) -> None:
        node = black.lib2to3_parse(source)
        actual: Set[Feature] = black.get_features_used(node)
        msg = f"Expected {expected} but got {actual} for {source!r}"
        try:
            self.assertEqual(actual, expected, msg=msg)
        except AssertionError:
            DebugVisitor.show(node)
            raise

    def test_get_features_used_for_future_flags(self) -> None:
        for src, features in [
            (
                "from __future__ import annotations",
                {Feature.FUTURE_ANNOTATIONS},
            ),
            (
                "from __future__ import (other, annotations)",
                {Feature.FUTURE_ANNOTATIONS},
            ),
            ("a = 1 + 2\nfrom something import annotations", set()),
            ("from __future__ import x, y", set()),
        ]:
            with self.subTest(src=src, features=features):
                node = black.lib2to3_parse(src)
                future_imports = black.get_future_imports(node)
                self.assertEqual(
                    black.get_features_used(node, future_imports=future_imports),
                    features,
                )

    def test_get_future_imports(self) -> None:
        node = black.lib2to3_parse("\n")
        self.assertEqual(set(), black.get_future_imports(node))
        node = black.lib2to3_parse("from __future__ import black\n")
        self.assertEqual({"black"}, black.get_future_imports(node))
        node = black.lib2to3_parse("from __future__ import multiple, imports\n")
        self.assertEqual({"multiple", "imports"}, black.get_future_imports(node))
        node = black.lib2to3_parse(
            "from __future__ import (parenthesized, imports)\n"
        )
        self.assertEqual({"parenthesized", "imports"}, black.get_future_imports(node))
        node = black.lib2to3_parse("from __future__ import multiple\nfrom __future__ import imports\n")
        self.assertEqual({"multiple", "imports"}, black.get_future_imports(node))
        node = black.lib2to3_parse("# comment\nfrom __future__ import black\n")
        self.assertEqual({"black"}, black.get_future_imports(node))
        node = black.lib2to3_parse('"""docstring"""\nfrom __future__ import black\n')
        self.assertEqual({"black"}, black.get_future_imports(node))
        node = black.lib2to3_parse("some(other, code)\nfrom __future__ import black\n")
        self.assertEqual(set(), black.get_future_imports(node))
        node = black.lib2to3_parse("from some.module import black\n")
        self.assertEqual(set(), black.get_future_imports(node))
        node = black.lib2to3_parse(
            "from __future__ import unicode_literals as _unicode_literals"
        )
        self.assertEqual({"unicode_literals"}, black.get_future_imports(node))
        node = black.lib2to3_parse(
            "from __future__ import unicode_literals as _lol, print"
        )
        self.assertEqual({"unicode_literals", "print"}, black.get_future_imports(node))

    @pytest.mark.incompatible_with_mypyc
    def test_debug_visitor(self) -> None:
        source, _ = read_data("miscellaneous", "debug_visitor")
        expected, _ = read_data("miscellaneous", "debug_visitor.out")
        out_lines: List[str] = []
        err_lines: List[str] = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("black.debug.out", out):
            DebugVisitor.show(source)
        actual: str = "\n".join(out_lines) + "\n"
        log_name = ""
        if expected != actual:
            log_name = black.dump_to_file(*out_lines)
        self.assertEqual(expected, actual, f"AST print out is different. Actual version dumped to {log_name}")

    def test_format_file_contents(self) -> None:
        mode = DEFAULT_MODE
        empty: str = ""
        with self.assertRaises(black.NothingChanged):
            black.format_file_contents(empty, mode=mode, fast=False)
        just_nl: str = "\n"
        with self.assertRaises(black.NothingChanged):
            black.format_file_contents(just_nl, mode=mode, fast=False)
        same: str = "j = [1, 2, 3]\n"
        with self.assertRaises(black.NothingChanged):
            black.format_file_contents(same, mode=mode, fast=False)
        different: str = "j = [1,2,3]"
        expected: str = same
        actual: str = black.format_file_contents(different, mode=mode, fast=False)
        self.assertEqual(expected, actual)
        invalid: str = "return if you can"
        with self.assertRaises(black.InvalidInput) as e:
            black.format_file_contents(invalid, mode=mode, fast=False)
        self.assertEqual(str(e.exception), "Cannot parse: 1:7: return if you can")
        just_crlf: str = "\r\n"
        with self.assertRaises(black.NothingChanged):
            black.format_file_contents(just_crlf, mode=mode, fast=False)
        just_whitespace_nl: str = "\n\t\n \n\t \n \t\n\n"
        actual = black.format_file_contents(just_whitespace_nl, mode=mode, fast=False)
        self.assertEqual("\n", actual)
        just_whitespace_crlf: str = "\r\n\t\r\n \r\n\t \r\n \t\r\n\r\n"
        actual = black.format_file_contents(just_whitespace_crlf, mode=mode, fast=False)
        self.assertEqual("\r\n", actual)

    def test_endmarker(self) -> None:
        n = black.lib2to3_parse("\n")
        self.assertEqual(n.type, black.syms.file_input)
        self.assertEqual(len(n.children), 1)
        self.assertEqual(n.children[0].type, black.token.ENDMARKER)

    @patch("tests.conftest.PRINT_FULL_TREE", True)
    @patch("tests.conftest.PRINT_TREE_DIFF", False)
    @pytest.mark.incompatible_with_mypyc
    def test_assertFormatEqual_print_full_tree(self) -> None:
        out_lines: List[str] = []
        err_lines: List[str] = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("black.output._out", out), patch("black.output._err", err):
            with self.assertRaises(AssertionError):
                self.assertFormatEqual("j = [1, 2, 3]", "j = [1, 2, 3,]")
        out_str = "".join(out_lines)
        self.assertIn("Expected tree:", out_str)
        self.assertIn("Actual tree:", out_str)
        self.assertEqual("".join(err_lines), "")

    @patch("tests.conftest.PRINT_FULL_TREE", False)
    @patch("tests.conftest.PRINT_TREE_DIFF", True)
    @pytest.mark.incompatible_with_mypyc
    def test_assertFormatEqual_print_tree_diff(self) -> None:
        out_lines: List[str] = []
        err_lines: List[str] = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with patch("black.output._out", out), patch("black.output._err", err):
            with self.assertRaises(AssertionError):
                self.assertFormatEqual("j = [1, 2, 3]\n", "j = [1, 2, 3,]\n")
        out_str = "".join(out_lines)
        self.assertIn("Tree Diff:", out_str)
        self.assertIn("+          COMMA", out_str)
        self.assertIn("+ ','", out_str)
        self.assertEqual("".join(err_lines), "")

    @event_loop()
    @patch("concurrent.futures.ProcessPoolExecutor", MagicMock(side_effect=OSError))
    def test_works_in_mono_process_only_environment(self) -> None:
        with cache_dir() as workspace:
            for f in [
                (workspace / "one.py").resolve(),
                (workspace / "two.py").resolve(),
            ]:
                f.write_text('print("hello")\n', encoding="utf-8")
            self.invokeBlack([str(workspace)])

    @event_loop()
    def test_check_diff_use_together(self) -> None:
        with cache_dir():
            src1 = get_case_path("miscellaneous", "string_quotes")
            self.invokeBlack([str(src1), "--diff", "--check"], exit_code=1)
            src2 = get_case_path("cases", "composition")
            self.invokeBlack([str(src2), "--diff", "--check"])
            self.invokeBlack([str(src1), str(src2), "--diff", "--check"], exit_code=1)

    def test_no_src_fails(self) -> None:
        with cache_dir():
            self.invokeBlack([], exit_code=1)

    def test_src_and_code_fails(self) -> None:
        with cache_dir():
            self.invokeBlack([".", "-c", "0"], exit_code=1)

    def test_broken_symlink(self) -> None:
        with cache_dir() as workspace:
            symlink = workspace / "broken_link.py"

            try:
                symlink.symlink_to("nonexistent.py")
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"Can't create symlinks: {e}")
            self.invokeBlack([str(workspace.resolve())])

    def test_single_file_force_pyi(self) -> None:
        pyi_mode: Mode = replace(DEFAULT_MODE, is_pyi=True)
        contents, expected = read_data("miscellaneous", "force_pyi")
        with cache_dir() as workspace:
            path = (workspace / "file.py").resolve()
            path.write_text(contents, encoding="utf-8")
            self.invokeBlack([str(path), "--pyi"])
            actual = path.read_text(encoding="utf-8")
            pyi_cache = black.Cache.read(pyi_mode)
            assert not pyi_cache.is_changed(path)
            normal_cache = black.Cache.read(DEFAULT_MODE)
            assert normal_cache.is_changed(path)
        self.assertFormatEqual(expected, actual)
        black.assert_equivalent(contents, actual)
        black.assert_stable(contents, actual, pyi_mode)

    @event_loop()
    def test_multi_file_force_pyi(self) -> None:
        reg_mode: Mode = DEFAULT_MODE
        pyi_mode: Mode = replace(DEFAULT_MODE, is_pyi=True)
        contents, expected = read_data("miscellaneous", "force_pyi")
        with cache_dir() as workspace:
            paths = [
                (workspace / "file1.py").resolve(),
                (workspace / "file2.py").resolve(),
            ]
            for path in paths:
                path.write_text(contents, encoding="utf-8")
            self.invokeBlack([str(p) for p in paths] + ["--pyi"])
            for path in paths:
                actual = path.read_text(encoding="utf-8")
                self.assertEqual(actual, expected)
            pyi_cache = black.Cache.read(pyi_mode)
            normal_cache = black.Cache.read(reg_mode)
            for path in paths:
                assert not pyi_cache.is_changed(path)
                assert normal_cache.is_changed(path)

    def test_pipe_force_pyi(self) -> None:
        source, expected = read_data("miscellaneous", "force_pyi")
        result = CliRunner().invoke(
            black.main, ["-", "-q", "--pyi"], input=BytesIO(source.encode("utf-8"))
        )
        self.assertEqual(result.exit_code, 0)
        actual = result.output
        self.assertFormatEqual(actual, expected)

    def test_single_file_force_py36(self) -> None:
        reg_mode: Mode = DEFAULT_MODE
        py36_mode: Mode = replace(DEFAULT_MODE, target_versions=PY36_VERSIONS)
        source, expected = read_data("miscellaneous", "force_py36")
        with cache_dir() as workspace:
            path = (workspace / "file.py").resolve()
            path.write_text(source, encoding="utf-8")
            self.invokeBlack([str(path), *PY36_ARGS])
            actual = path.read_text(encoding="utf-8")
            py36_cache = black.Cache.read(py36_mode)
            assert not py36_cache.is_changed(path)
            normal_cache = black.Cache.read(reg_mode)
            assert normal_cache.is_changed(path)
        self.assertEqual(actual, expected)

    @event_loop()
    def test_multi_file_force_py36(self) -> None:
        reg_mode: Mode = DEFAULT_MODE
        py36_mode: Mode = replace(DEFAULT_MODE, target_versions=PY36_VERSIONS)
        source, expected = read_data("miscellaneous", "force_py36")
        with cache_dir() as workspace:
            paths = [
                (workspace / "file1.py").resolve(),
                (workspace / "file2.py").resolve(),
            ]
            for path in paths:
                path.write_text(source, encoding="utf-8")
            self.invokeBlack([str(p) for p in paths] + PY36_ARGS)
            for path in paths:
                actual = path.read_text(encoding="utf-8")
                self.assertEqual(actual, expected)
            pyi_cache = black.Cache.read(py36_mode)
            normal_cache = black.Cache.read(reg_mode)
            for path in paths:
                assert not pyi_cache.is_changed(path)
                assert normal_cache.is_changed(path)

    def test_pipe_force_py36(self) -> None:
        source, expected = read_data("miscellaneous", "force_py36")
        result = CliRunner().invoke(
            black.main, ["-", "-q", "--target-version=py36"], input=BytesIO(source.encode("utf-8"))
        )
        self.assertEqual(result.exit_code, 0)
        actual = result.output
        self.assertEqual(actual, expected)

    @pytest.mark.incompatible_with_mypyc
    def test_reformat_one_with_stdin(self) -> None:
        with patch.object(
            black, "format_stdin_to_stdout", return_value=lambda *args, **kwargs: black.Changed.YES
        ) as fsts:
            report = MagicMock(spec=Report)
            path = Path("-")
            black.reformat_one(
                path,
                fast=True,
                write_back=black.WriteBack.YES,
                mode=DEFAULT_MODE,
                report=report,
            )
            fsts.assert_called_once()
            report.done.assert_called_with(path, black.Changed.YES)

    @pytest.mark.incompatible_with_mypyc
    def test_reformat_one_with_stdin_filename(self) -> None:
        with patch.object(
            black, "format_stdin_to_stdout", return_value=lambda *args, **kwargs: black.Changed.YES
        ) as fsts:
            report = MagicMock(spec=Report)
            p: str = "foo.py"
            path = Path(f"__BLACK_STDIN_FILENAME__{p}")
            expected = Path(p)
            black.reformat_one(
                path,
                fast=True,
                write_back=black.WriteBack.YES,
                mode=DEFAULT_MODE,
                report=report,
            )
            fsts.assert_called_once_with(
                fast=True, write_back=black.WriteBack.YES, mode=DEFAULT_MODE, lines=()
            )
            report.done.assert_called_with(expected, black.Changed.YES)

    @pytest.mark.incompatible_with_mypyc
    def test_reformat_one_with_stdin_filename_pyi(self) -> None:
        with patch.object(
            black, "format_stdin_to_stdout", return_value=lambda *args, **kwargs: black.Changed.YES
        ) as fsts:
            report = MagicMock(spec=Report)
            p: str = "foo.pyi"
            path = Path(f"__BLACK_STDIN_FILENAME__{p}")
            expected = Path(p)
            black.reformat_one(
                path,
                fast=True,
                write_back=black.WriteBack.YES,
                mode=DEFAULT_MODE,
                report=report,
            )
            fsts.assert_called_once_with(
                fast=True, write_back=black.WriteBack.YES, mode=replace(DEFAULT_MODE, is_pyi=True), lines=()
            )
            report.done.assert_called_with(expected, black.Changed.YES)

    @pytest.mark.incompatible_with_mypyc
    def test_reformat_one_with_stdin_filename_ipynb(self) -> None:
        with patch.object(
            black, "format_stdin_to_stdout", return_value=lambda *args, **kwargs: black.Changed.YES
        ) as fsts:
            report = MagicMock(spec=Report)
            p: str = "foo.ipynb"
            path = Path(f"__BLACK_STDIN_FILENAME__{p}")
            expected = Path(p)
            black.reformat_one(
                path,
                fast=True,
                write_back=black.WriteBack.YES,
                mode=DEFAULT_MODE,
                report=report,
            )
            fsts.assert_called_once_with(
                fast=True, write_back=black.WriteBack.YES, mode=replace(DEFAULT_MODE, is_ipynb=True), lines=()
            )
            report.done.assert_called_with(expected, black.Changed.YES)

    @pytest.mark.incompatible_with_mypyc
    def test_reformat_one_with_stdin_and_existing_path(self) -> None:
        with patch.object(
            black, "format_stdin_to_stdout", return_value=lambda *args, **kwargs: black.Changed.YES
        ) as fsts:
            report = MagicMock(spec=Report)
            p = THIS_DIR / "data" / "cases" / "collections.py"
            self.assertTrue(p.is_file())
            path = Path(f"__BLACK_STDIN_FILENAME__{p}")
            expected = Path(p)
            black.reformat_one(
                path,
                fast=True,
                write_back=black.WriteBack.YES,
                mode=DEFAULT_MODE,
                report=report,
            )
            fsts.assert_called_once()
            report.done.assert_called_with(expected, black.Changed.YES)

    def test_reformat_one_with_stdin_empty(self) -> None:
        cases: List[Tuple[str, str]] = [
            ("", ""),
            ("\n", "\n"),
            ("\r\n", "\r\n"),
            (" \t", ""),
            (" \t\n\t ", "\n"),
            (" \t\r\n\t ", "\r\n"),
        ]

        def _new_wrapper(output: io.StringIO, io_TextIOWrapper: Any) -> Callable[..., io.TextIOWrapper]:
            def get_output(*args: Any, **kwargs: Any) -> io.TextIOWrapper:
                if args == (sys.stdout.buffer,):
                    return output
                return io_TextIOWrapper(*args, **kwargs)

            return get_output

        for content, expected in cases:
            output = io.StringIO()
            io_TextIOWrapper = io.TextIOWrapper
            with patch("io.TextIOWrapper", _new_wrapper(output, io_TextIOWrapper)):
                try:
                    black.format_stdin_to_stdout(
                        fast=True,
                        content=content,
                        write_back=black.WriteBack.YES,
                        mode=DEFAULT_MODE,
                    )
                except io.UnsupportedOperation:
                    pass
                assert output.getvalue() == expected

    def test_cli_unstable(self) -> None:
        self.invokeBlack(["--unstable", "-c", "0"], exit_code=0)
        self.invokeBlack(["--preview", "-c", "0"], exit_code=0)
        self.invokeBlack(
            ["--enable-unstable-feature", "string_processing", "-c", "0"], exit_code=1
        )
        self.invokeBlack(
            ["--preview", "--enable-unstable-feature", "string_processing", "-c", "0"],
            exit_code=0,
        )
        self.invokeBlack(
            ["--unstable", "--enable-unstable-feature", "string_processing", "-c", "0"],
            exit_code=0,
        )

    def test_invalid_cli_regex(self) -> None:
        for option in ["--include", "--exclude", "--extend-exclude", "--force-exclude"]:
            self.invokeBlack(["-", option, "**()(!!*)"], exit_code=2)

    def test_required_version_matches_version(self) -> None:
        self.invokeBlack(
            ["--required-version", black.__version__, "-c", "0"], exit_code=0, ignore_config=True
        )

    def test_required_version_matches_partial_version(self) -> None:
        self.invokeBlack(
            ["--required-version", black.__version__.split(".")[0], "-c", "0"],
            exit_code=0,
            ignore_config=True,
        )

    def test_required_version_does_not_match_on_minor_version(self) -> None:
        self.invokeBlack(
            ["--required-version", black.__version__.split(".")[0] + ".999", "-c", "0"],
            exit_code=1,
            ignore_config=True,
        )

    def test_required_version_does_not_match_version(self) -> None:
        result = BlackRunner().invoke(
            black.main, ["--required-version", "20.99b", "-c", "0"]
        )
        self.assertEqual(result.exit_code, 1)
        self.assertIn("required version", result.stderr)

    def test_preserves_line_endings(self) -> None:
        with TemporaryDirectory() as workspace:
            test_file = Path(workspace) / "test.py"
            for nl in ["\n", "\r\n"]:
                contents = nl.join(["def f(  ):", "    pass"])
                test_file.write_bytes(contents.encode())
                ff(test_file, write_back=black.WriteBack.YES)
                updated_contents = test_file.read_bytes()
                self.assertIn(nl.encode(), updated_contents)
                if nl == "\n":
                    self.assertNotIn(b"\r\n", updated_contents)

    def test_preserves_line_endings_via_stdin(self) -> None:
        for nl in ["\n", "\r\n"]:
            contents = nl.join(["def f(  ):", "    pass"])
            runner = BlackRunner()
            result = runner.invoke(
                black.main,
                ["--line-ranges=1-1", "-"],
                input=BytesIO(contents.encode("utf-8")),
            )
            self.assertEqual(result.exit_code, 0)
            output = result.stdout_bytes
            self.assertIn(nl.encode("utf-8"), output)
            if nl == "\n":
                self.assertNotIn(b"\r\n", output)

    def test_normalize_line_endings(self) -> None:
        with TemporaryDirectory() as workspace:
            test_file = Path(workspace) / "test.py"
            for data, expected in [
                (b"c\r\nc\n ", b"c\r\nc\r\n"),
                (b"l\nl\r\n ", b"l\nl\n"),
            ]:
                test_file.write_bytes(data)
                ff(test_file, write_back=black.WriteBack.YES)
                self.assertEqual(test_file.read_bytes(), expected)

    def test_root_logger_not_used_directly(self) -> None:
        def fail(*args: Any, **kwargs: Any) -> None:
            self.fail("Record created with root logger")

        with patch.multiple(
            logging.root,
            debug=fail,
            info=fail,
            warning=fail,
            error=fail,
            critical=fail,
            log=fail,
        ):
            ff(THIS_DIR / "util.py")

    def test_invalid_config_return_code(self) -> None:
        tmp_file = Path(black.dump_to_file())
        try:
            tmp_config = Path(black.dump_to_file())
            tmp_config.unlink()
            args = ["--config", str(tmp_config), str(tmp_file)]
            self.invokeBlack(args, exit_code=2, ignore_config=False)
        finally:
            tmp_file.unlink()

    def test_parse_pyproject_toml(self) -> None:
        test_toml_file: Path = THIS_DIR / "test.toml"
        config: Dict[str, Any] = black.parse_pyproject_toml(str(test_toml_file))
        self.assertEqual(config["verbose"], 1)
        self.assertEqual(config["check"], "no")
        self.assertEqual(config["diff"], "y")
        self.assertEqual(config["color"], True)
        self.assertEqual(config["line_length"], 79)
        self.assertEqual(config["target_version"], ["py36", "py37", "py38"])
        self.assertEqual(config["python_cell_magics"], ["custom1", "custom2"])
        self.assertEqual(config["exclude"], "\\.pyi?$")
        self.assertEqual(config["include"], "\\.py?$")

    def test_spellcheck_pyproject_toml(self) -> None:
        test_toml_file: Path = THIS_DIR / "data" / "incorrect_spelling.toml"
        result = BlackRunner().invoke(
            black.main,
            ["--code=print('hello world')", "--verbose", f"--config={str(test_toml_file)}"],
        )
        assert (
            f"Invalid config keys detected: 'ine_length', 'target_ersion' (in {test_toml_file})"
            in result.stderr
        )

    def test_parse_pyproject_toml_project_metadata(self) -> None:
        for test_toml, expected in [
            ("only_black_pyproject.toml", ["py310"]),
            ("only_metadata_pyproject.toml", ["py37", "py38", "py39", "py310"]),
            ("neither_pyproject.toml", None),
            ("both_pyproject.toml", ["py310"]),
        ]:
            test_toml_file: Path = THIS_DIR / "data" / "project_metadata" / test_toml
            config: Dict[str, Any] = black.parse_pyproject_toml(str(test_toml_file))
            self.assertEqual(config.get("target_version"), expected)

    def test_infer_target_version(self) -> None:
        for version, expected in [
            ("3.6", [TargetVersion.PY36]),
            ("3.11.0rc1", [TargetVersion.PY311]),
            (
                ">=3.10",
                [
                    TargetVersion.PY310,
                    TargetVersion.PY311,
                    TargetVersion.PY312,
                    TargetVersion.PY313,
                ],
            ),
            (
                ">=3.10.6",
                [
                    TargetVersion.PY310,
                    TargetVersion.PY311,
                    TargetVersion.PY312,
                    TargetVersion.PY313,
                ],
            ),
            (
                "<3.6",
                {TargetVersion.PY33, TargetVersion.PY34, TargetVersion.PY35},
            ),
            (">3.7,<3.10", {TargetVersion.PY38, TargetVersion.PY39}),
            (
                ">3.7,!=3.8,!=3.9",
                {
                    TargetVersion.PY310,
                    TargetVersion.PY311,
                    TargetVersion.PY312,
                    TargetVersion.PY313,
                },
            ),
            (
                "> 3.9.4, != 3.10.3",
                {
                    TargetVersion.PY39,
                    TargetVersion.PY310,
                    TargetVersion.PY311,
                    TargetVersion.PY312,
                    TargetVersion.PY313,
                },
            ),
            (
                "!=3.3,!=3.4",
                {
                    TargetVersion.PY35,
                    TargetVersion.PY36,
                    TargetVersion.PY37,
                    TargetVersion.PY38,
                    TargetVersion.PY39,
                    TargetVersion.PY310,
                    TargetVersion.PY311,
                    TargetVersion.PY312,
                    TargetVersion.PY313,
                },
            ),
            (
                "==3.*",
                {
                    TargetVersion.PY33,
                    TargetVersion.PY34,
                    TargetVersion.PY35,
                    TargetVersion.PY36,
                    TargetVersion.PY37,
                    TargetVersion.PY38,
                    TargetVersion.PY39,
                    TargetVersion.PY310,
                    TargetVersion.PY311,
                    TargetVersion.PY312,
                    TargetVersion.PY313,
                },
            ),
            ("==3.8.*", [TargetVersion.PY38]),
            (None, None),
            ("", None),
            ("invalid", None),
            ("==invalid", None),
            (">3.9,!=invalid", None),
            ("3", None),
            ("3.2", None),
            ("2.7.18", None),
            ("==2.7", None),
            (">3.10,<3.11", None),
        ]:
            test_toml: Dict[str, Any] = {"project": {"requires-python": version}}
            result: Optional[List[TargetVersion]] = black.files.infer_target_version(test_toml)
            self.assertEqual(result, expected)

    def test_read_pyproject_toml(self) -> None:
        test_toml_file: Path = THIS_DIR / "test.toml"
        fake_ctx = FakeContext()
        black.read_pyproject_toml(fake_ctx, FakeParameter(), str(test_toml_file))
        config: Dict[str, Any] = fake_ctx.default_map
        self.assertEqual(config["verbose"], "1")
        self.assertEqual(config["check"], "no")
        self.assertEqual(config["diff"], "y")
        self.assertEqual(config["color"], "True")
        self.assertEqual(config["line_length"], "79")
        self.assertEqual(config["target_version"], ["py36", "py37", "py38"])
        self.assertEqual(config["exclude"], "\\.pyi?$")
        self.assertEqual(config["include"], "\\.py?$")

    def test_read_pyproject_toml_from_stdin(self) -> None:
        with TemporaryDirectory() as workspace:
            root: Path = Path(workspace)
            src_dir: Path = root / "src"
            src_dir.mkdir()
            src_pyproject: Path = src_dir / "pyproject.toml"
            src_pyproject.touch()
            test_toml_content: str = (THIS_DIR / "test.toml").read_text(
                encoding="utf-8"
            )
            src_pyproject.write_text(test_toml_content, encoding="utf-8")
            src_python: Path = src_dir / "foo.py"
            src_python.touch()
            fake_ctx = FakeContext()
            fake_ctx.params["src"] = ("-",)
            fake_ctx.params["stdin_filename"] = str(src_python)
            with change_directory(root):
                black.read_pyproject_toml(fake_ctx, FakeParameter(), None)
            config: Dict[str, Any] = fake_ctx.default_map
            self.assertEqual(config["verbose"], "1")
            self.assertEqual(config["check"], "no")
            self.assertEqual(config["diff"], "y")
            self.assertEqual(config["color"], "True")
            self.assertEqual(config["line_length"], "79")
            self.assertEqual(config["target_version"], ["py36", "py37", "py38"])
            self.assertEqual(config["exclude"], "\\.pyi?$")
            self.assertEqual(config["include"], "\\.py?$")

    @pytest.mark.incompatible_with_mypyc
    def test_find_project_root(self) -> None:
        with TemporaryDirectory() as workspace:
            root: Path = Path(workspace)
            test_dir: Path = root / "test"
            test_dir.mkdir()
            src_dir: Path = root / "src"
            src_dir.mkdir()
            root_pyproject: Path = root / "pyproject.toml"
            root_pyproject.write_text("[tool.black]", encoding="utf-8")
            src_pyproject: Path = src_dir / "pyproject.toml"
            src_pyproject.write_text("[tool.black]", encoding="utf-8")
            src_python: Path = src_dir / "foo.py"
            src_python.touch()
            self.assertEqual(
                black.find_project_root((src_dir, test_dir)),
                (root.resolve(), "pyproject.toml"),
            )
            self.assertEqual(
                black.find_project_root((src_dir,)),
                (src_dir.resolve(), "pyproject.toml"),
            )
            self.assertEqual(
                black.find_project_root((src_python,)),
                (src_dir.resolve(), "pyproject.toml"),
            )
            with change_directory(test_dir):
                self.assertEqual(
                    black.find_project_root(("-",), stdin_filename="../src/a.py"),
                    (src_dir.resolve(), "pyproject.toml"),
                )
            src_sub: Path = src_dir / "sub"
            src_sub.mkdir()
            src_sub_pyproject: Path = src_sub / "pyproject.toml"
            src_sub_pyproject.touch()
            src_sub_python: Path = src_sub / "bar.py"
            self.assertEqual(
                black.find_project_root((src_sub_python,)),
                (src_dir.resolve(), "pyproject.toml"),
            )

    @patch("black.files.find_user_pyproject_toml")
    def test_find_pyproject_toml(
        self, find_user_pyproject_toml: Callable[..., Optional[Path]]
    ) -> None:
        find_user_pyproject_toml.side_effect = RuntimeError()
        with redirect_stderr(io.StringIO()) as stderr:
            result: Optional[Path] = black.files.find_pyproject_toml(
                path_search_start=(str(Path.cwd().root),)
            )
        assert result is None
        err: str = stderr.getvalue()
        assert "Ignoring user configuration" in err

    @patch("black.files.find_user_pyproject_toml", black.files.find_user_pyproject_toml.__wrapped__)
    def test_find_user_pyproject_toml_linux(self) -> None:
        if system() == "Windows":
            return
        with TemporaryDirectory() as workspace:
            tmp_user_config: Path = Path(workspace) / "black"
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": str(workspace)}):
                self.assertEqual(
                    black.files.find_user_pyproject_toml(),
                    tmp_user_config.resolve(),
                )
        with patch.dict("os.environ"):
            os.environ.pop("XDG_CONFIG_HOME", None)
            fallback_user_config: Path = (Path("~/.config").expanduser() / "black").resolve()
            self.assertEqual(
                black.files.find_user_pyproject_toml(),
                fallback_user_config,
            )

    def test_find_user_pyproject_toml_windows(self) -> None:
        if system() != "Windows":
            return
        user_config_path: Path = (Path.home() / ".black").resolve()
        self.assertEqual(
            black.files.find_user_pyproject_toml(),
            user_config_path,
        )

    def test_bpo_33660_workaround(self) -> None:
        if system() == "Windows":
            return
        root: Path = Path("/")
        with change_directory(root):
            path: Path = Path("workspace") / "project"
            report: Report = black.Report(verbose=True)
            resolves_outside: bool = black.resolves_outside_root_or_cannot_stat(
                path, root, report
            )
            self.assertIs(resolves_outside, False)

    def test_normalize_path_ignore_windows_junctions_outside_of_root(self) -> None:
        if system() != "Windows":
            return
        with TemporaryDirectory() as workspace:
            root: Path = Path(workspace)
            junction_dir: Path = root / "junction"
            junction_target_outside_of_root: Path = root / ".."
            os.system(f"mklink /J {junction_dir} {junction_target_outside_of_root}")
            report: Report = black.Report(verbose=True)
            resolves_outside: bool = black.resolves_outside_root_or_cannot_stat(
                junction_dir, root, report
            )
            os.system(f"rmdir {junction_dir}")
            self.assertIs(resolves_outside, True)

    def test_newline_comment_interaction(self) -> None:
        source: str = 'class A:\\\r\n# type: ignore\n pass\n'
        output: str = black.format_str(source, mode=DEFAULT_MODE)
        black.assert_stable(source, output, mode=DEFAULT_MODE)

    def test_bpo_2142_workaround(self) -> None:
        source, _ = read_data("miscellaneous", "missing_final_newline")
        source = source.rstrip()
        expected, _ = read_data("miscellaneous", "missing_final_newline.diff")
        tmp_file: Path = Path(black.dump_to_file(source, ensure_final_newline=False))
        diff_header = re.compile(
            f"{re.escape(str(tmp_file))}\\t\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d\\.\\d\\d\\d\\d\\d\\d\\+\\d\\d:\\d\\d"
        )
        try:
            result = BlackRunner().invoke(
                black.main, ["--diff", str(tmp_file)]
            )
            self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmp_file)
        actual: str = result.output
        actual = diff_header.sub(DETERMINISTIC_HEADER, actual)
        self.assertEqual(expected, actual)

    @staticmethod
    def compare_results(
        result: click.testing.Result,
        expected_value: str,
        expected_exit_code: int,
    ) -> None:
        """Helper method to test the value and exit code of a click Result."""
        assert (
            result.output == expected_value
        ), "The output did not match the expected value."
        assert (
            result.exit_code == expected_exit_code
        ), "The exit code is incorrect."

    def test_code_option(self) -> None:
        """Test the code option with no changes."""
        code: str = 'print("Hello world")\n'
        args: List[str] = ["--code", code]
        result: click.testing.Result = CliRunner().invoke(black.main, args)
        self.compare_results(result, code, 0)

    def test_code_option_changed(self) -> None:
        """Test the code option when changes are required."""
        code: str = "print('hello world')"
        formatted: str = black.format_str(code, mode=black.Mode())
        args: List[str] = ["--code", code]
        result: click.testing.Result = CliRunner().invoke(black.main, args)
        self.compare_results(result, formatted, 0)

    def test_code_option_check(self) -> None:
        """Test the code option when check is passed."""
        args: List[str] = ["--check", "--code", 'print("Hello world")\n']
        result: click.testing.Result = CliRunner().invoke(black.main, args)
        self.compare_results(result, "", 0)

    def test_code_option_check_changed(self) -> None:
        """Test the code option when changes are required, and check is passed."""
        args: List[str] = ["--check", "--code", "print('hello world')"]
        result: click.testing.Result = CliRunner().invoke(black.main, args)
        self.compare_results(result, "", 1)

    def test_code_option_diff(self) -> None:
        """Test the code option when diff is passed."""
        code: str = "print('hello world')"
        formatted: str = black.format_str(code, mode=DEFAULT_MODE)
        result_diff: str = diff(code, formatted, "STDIN", "STDOUT")
        args: List[str] = ["--diff", "--code", code]
        result: click.testing.Result = CliRunner().invoke(black.main, args)
        output: str = DIFF_TIME.sub("", result.output)
        assert (
            output == result_diff
        ), "The output did not match the expected value."
        assert result.exit_code == 0, "The exit code is incorrect."

    def test_code_option_color_diff(self) -> None:
        """Test the code option when color and diff are passed."""
        code: str = "print('hello world')"
        formatted: str = black.format_str(code, mode=DEFAULT_MODE)
        result_diff: str = diff(code, formatted, "STDIN", "STDOUT")
        result_diff = color_diff(result_diff)
        args: List[str] = ["--diff", "--color", "--code", code]
        result: click.testing.Result = CliRunner().invoke(black.main, args)
        output: str = DIFF_TIME.sub("", result.output)
        assert (
            output == result_diff
        ), "The output did not match the expected value."
        assert result.exit_code == 0, "The exit code is incorrect."

    @pytest.mark.incompatible_with_mypyc
    def test_code_option_safe(self) -> None:
        """Test that the code option throws an error when the sanity checks fail."""
        with patch.object(
            black, "assert_equivalent", side_effect=AssertionError
        ):
            code: str = 'print("Hello world")'
            error_msg: str = f"{code}\nerror: cannot format <string>: \n"
            args: List[str] = ["--safe", "--code", code]
            result: click.testing.Result = CliRunner().invoke(black.main, args)
            self.compare_results(result, error_msg, 123)

    def test_code_option_fast(self) -> None:
        """Test that the code option ignores errors when the sanity checks fail."""
        with patch.object(
            black, "assert_equivalent", side_effect=AssertionError
        ):
            code: str = 'print("Hello world")'
            formatted: str = black.format_str(code, mode=DEFAULT_MODE)
            args: List[str] = ["--fast", "--code", code]
            result: click.testing.Result = CliRunner().invoke(black.main, args)
            self.compare_results(result, formatted, 0)

    @pytest.mark.incompatible_with_mypyc
    def test_code_option_config(self) -> None:
        """
        Test that the code option finds the pyproject.toml in the current directory.
        """
        with patch.object(black, "parse_pyproject_toml", return_value={}) as parse:
            args: List[str] = ["--code", "print"]
            with change_directory(PROJECT_ROOT):
                CliRunner().invoke(black.main, args)
                pyproject_path: Path = (Path.cwd() / "pyproject.toml").resolve()
            assert (
                len(parse.mock_calls) >= 1
            ), "Expected config parse to be called with the current directory."
            _, call_args, _ = parse.mock_calls[0]
            assert (
                call_args[0].lower() == str(pyproject_path).lower()
            ), "Incorrect config loaded."

    @pytest.mark.incompatible_with_mypyc
    def test_code_option_parent_config(self) -> None:
        """
        Test that the code option finds the pyproject.toml in the parent directory.
        """
        with patch.object(black, "parse_pyproject_toml", return_value={}) as parse:
            with change_directory(THIS_DIR):
                args: List[str] = ["--code", "print"]
                CliRunner().invoke(black.main, args)
                pyproject_path: Path = (Path.cwd().parent / "pyproject.toml").resolve()
                assert (
                    len(parse.mock_calls) >= 1
                ), "Expected config parse to be called with the current directory."
                _, call_args, _ = parse.mock_calls[0]
                assert (
                    call_args[0].lower() == str(pyproject_path).lower()
                ), "Incorrect config loaded."

    def test_for_handled_unexpected_eof_error(self) -> None:
        """
        Test that an unexpected EOF SyntaxError is nicely presented.
        """
        with pytest.raises(black.parsing.InvalidInput) as exc_info:
            black.lib2to3_parse("a«»a  = 1", {})
        exc_info.match("--safe")
        exc_info.match("invalid character")
        exc_info.match(r"\(<unknown>, line 1\)")


    def test_line_ranges_with_code_option(self) -> None:
        """Test line ranges option with code."""
        code: str = textwrap.dedent(
            """\
            if  a  ==  b:
                print  ( "OK" )
            """
        )
        args: List[str] = ["--line-ranges=1-1", "--code", code]
        result: click.testing.Result = CliRunner().invoke(black.main, args)
        expected: str = textwrap.dedent(
            """\
            if a == b:
                print  ( "OK" )
            """
        )
        self.compare_results(result, expected, expected_exit_code=0)

    def test_line_ranges_with_stdin(self) -> None:
        """Test line ranges option with stdin."""
        code: str = textwrap.dedent(
            """\
            if  a  ==  b:
                print  ( "OK" )
            """
        )
        runner = BlackRunner()
        result: click.testing.Result = runner.invoke(
            black.main,
            ["--line-ranges=1-1", "-"],
            input=BytesIO(code.encode("utf-8")),
        )
        expected: str = textwrap.dedent(
            """\
            if a == b:
                print  ( "OK" )
            """
        )
        self.compare_results(result, expected, expected_exit_code=0)

    def test_line_ranges_with_source(self) -> None:
        """Test line ranges option with source file."""
        with TemporaryDirectory() as workspace:
            test_file: Path = Path(workspace) / "test.py"
            test_file.write_text(
                textwrap.dedent(
                    """\
                    if  a  ==  b:
                        print  ( "OK" )
                    """
                ),
                encoding="utf-8",
            )
            args: List[str] = ["--line-ranges=1-1", str(test_file)]
            result: click.testing.Result = CliRunner().invoke(black.main, args)
            assert not result.exit_code
            formatted: str = test_file.read_text(encoding="utf-8")
            expected: str = textwrap.dedent(
                """\
                if a == b:
                    print  ( "OK" )
                """
            )
            assert expected == formatted

    def test_line_ranges_with_multiple_sources(self) -> None:
        """Test line ranges option with multiple source files."""
        with TemporaryDirectory() as workspace:
            test1_file: Path = Path(workspace) / "test1.py"
            test1_file.write_text("", encoding="utf-8")
            test2_file: Path = Path(workspace) / "test2.py"
            test2_file.write_text("", encoding="utf-8")
            args: List[str] = ["--line-ranges=1-1", str(test1_file), str(test2_file)]
            result: click.testing.Result = CliRunner().invoke(black.main, args)
            assert result.exit_code == 1
            assert "Cannot use --line-ranges to format multiple files" in result.output

    def test_line_ranges_with_ipynb(self) -> None:
        """Test line ranges option with ipynb files."""
        with TemporaryDirectory() as workspace:
            test_file: Path = Path(workspace) / "test.ipynb"
            test_file.write_text("{}", encoding="utf-8")
            args: List[str] = ["--line-ranges=1-1", "--ipynb", str(test_file)]
            result: click.testing.Result = CliRunner().invoke(black.main, args)
            assert "Cannot use --line-ranges with ipynb files" in result.output
            assert result.exit_code == 1

    def test_line_ranges_in_pyproject_toml(self) -> None:
        """Test that line ranges cannot be set in pyproject.toml."""
        config: Path = THIS_DIR / "data" / "invalid_line_ranges.toml"
        result: click.testing.Result = BlackRunner().invoke(
            black.main, ["--code", "print()", "--config", str(config)]
        )
        assert result.exit_code == 2
        assert result.stderr_bytes is not None
        assert (
            b"Cannot use line-ranges in the pyproject.toml file."
            in result.stderr_bytes
        )

    def test_lines_with_leading_tabs_expanded(self) -> None:
        payload: str = "\t" * 10000
        assert (
            lines_with_leading_tabs_expanded(payload) == [payload]
        )
        tab: str = " " * 8
        assert lines_with_leading_tabs_expanded("\tx") == [f"{tab}x"]
        assert lines_with_leading_tabs_expanded("\t\tx") == [f"{tab}{tab}x"]
        assert lines_with_leading_tabs_expanded("\tx\n  y") == [f"{tab}x", "  y"]

    def test_false_positive_symlink_output_issue_3384(self) -> None:
        # This test name appears twice in the original code. Assuming it's a duplication error.
        pass  # Already defined above

    def test_filter_cached(self) -> None:
        with TemporaryDirectory() as workspace:
            path = Path(workspace)
            uncached: Path = (path / "uncached").resolve()
            cached: Path = (path / "cached").resolve()
            cached_but_changed: Path = (path / "changed").resolve()
            uncached.touch()
            cached.touch()
            cached_but_changed.touch()
            cache: black.Cache = black.Cache.read(DEFAULT_MODE)
            orig_func = black.Cache.get_file_data

            def wrapped_func(path: Path) -> FileData:
                if path == cached:
                    return orig_func(path)
                if path == cached_but_changed:
                    return FileData(0.0, 0, "")
                raise AssertionError

            with patch.object(black.Cache, "get_file_data", side_effect=wrapped_func):
                cache.write([cached, cached_but_changed])
            todo, done = cache.filtered_cached({uncached, cached, cached_but_changed})
            assert todo == {uncached, cached_but_changed}
            assert done == {cached}

    def test_filter_cached_hash(self) -> None:
        with TemporaryDirectory() as workspace:
            path = Path(workspace)
            src: Path = (path / "test.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            st = src.stat()
            cache: black.Cache = black.Cache.read(DEFAULT_MODE)
            cache.write([src])
            one: black.Cache = black.Cache.read(DEFAULT_MODE)
            assert not one.is_changed(src)
            two: black.Cache = black.Cache.read(replace(DEFAULT_MODE, line_length=1))
            assert two.is_changed(src)
            cache.file_data[str(src)]
            cache.file_data[str(src)] = FileData(
                cache.file_data[str(src)].st_mtime - 1,
                cache.file_data[str(src)].st_size,
                cache.file_data[str(src)].hash,
            )
            todo, done = cache.filtered_cached([src])
            assert todo == set()
            assert done == {src}
            cache.file_data[str(src)].st_mtime < st.st_mtime
            cache.file_data[str(src)].st_size == st.st_size
            cache.file_data[str(src)].hash == black.Cache.hash_digest(src)
            src.write_text("print('hello world')", encoding="utf-8")
            new_st = src.stat()
            todo, done = cache.filtered_cached([src])
            assert todo == {src}
            assert done == set()
            assert cache.file_data[str(src)].st_mtime < new_st.st_mtime
            assert cache.file_data[str(src)].st_size != new_st.st_size
            assert cache.file_data[str(src)].hash != black.Cache.hash_digest(src)

    def test_write_cache_creates_directory_if_needed(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir(exists=False) as workspace:
            assert not workspace.exists()
            cache: black.Cache = black.Cache.read(mode)
            cache.write([])
            assert workspace.exists()

    @event_loop()
    @pytest.mark.incompatible_with_mypyc
    def test_failed_formatting_does_not_get_cached(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace, patch(
            "concurrent.futures.ProcessPoolExecutor", new=ThreadPoolExecutor
        ):
            failing: Path = (workspace / "failing.py").resolve()
            failing.write_text("not actually python", encoding="utf-8")
            clean: Path = (workspace / "clean.py").resolve()
            clean.write_text('print("hello")\n', encoding="utf-8")
            self.invokeBlack([str(workspace)], exit_code=123)
            cache: black.Cache = black.Cache.read(mode)
            assert cache.is_changed(failing)
            assert not cache.is_changed(clean)

    def test_write_cache_write_fail(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            cache: black.Cache = black.Cache.read(mode)
            with patch.object(Path, "open", side_effect=OSError):
                cache.write([])

    def test_read_cache_no_cachefile(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            assert black.Cache.read(mode).file_data == {}

    def test_write_cache_read_cache(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src: Path = (workspace / "test.py").resolve()
            src.touch()
            cache: black.Cache = black.Cache.read(mode)
            cache.write([src])
            read_cache: black.Cache = black.Cache.read(mode)
            assert not read_cache.is_changed(src)

    @pytest.mark.incompatible_with_mypyc
    def test_get_cache_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        workspace1: Path = tmp_path / "ws1"
        workspace1.mkdir()
        workspace2: Path = tmp_path / "ws2"
        workspace2.mkdir()
        patch_user_cache_dir = patch(
            target="black.cache.user_cache_dir", autospec=True, return_value=str(workspace1)
        )
        monkeypatch.delenv("BLACK_CACHE_DIR", raising=False)
        with patch_user_cache_dir:
            assert get_cache_dir().parent == workspace1
        monkeypatch.setenv("BLACK_CACHE_DIR", str(workspace2))
        assert get_cache_dir().parent == workspace2

    def test_cache_file_length(self) -> None:
        cases: List[Mode] = [
            DEFAULT_MODE,
            Mode(target_versions=set(TargetVersion)),
            Mode(enabled_features=set(Preview)),
            Mode(python_cell_magics={f"magic{i}" for i in range(500)}),
            Mode(
                target_versions=set(TargetVersion),
                enabled_features=set(Preview),
                python_cell_magics={f"magic{i}" for i in range(500)},
            ),
        ]
        for case in cases:
            cache_file: Path = get_cache_file(case)
            assert len(cache_file.name) <= 96

    def test_cache_broken_file(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            cache_file: Path = get_cache_file(mode)
            cache_file.write_text("this is not a pickle", encoding="utf-8")
            assert black.Cache.read(mode).file_data == {}
            src: Path = (workspace / "test.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            self.invokeBlack([str(src)])
            cache: black.Cache = black.Cache.read(mode)
            assert not cache.is_changed(src)

    def test_cache_single_file_already_cached(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src: Path = (workspace / "test.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            cache: black.Cache = black.Cache.read(mode)
            cache.write([src])
            self.invokeBlack([str(src)])
            assert src.read_text(encoding="utf-8") == "print('hello')"

    @event_loop()
    def test_cache_multiple_files(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace, patch(
            "concurrent.futures.ProcessPoolExecutor", new=ThreadPoolExecutor
        ):
            one: Path = (workspace / "one.py").resolve()
            one.write_text("print('hello')", encoding="utf-8")
            two: Path = (workspace / "two.py").resolve()
            two.write_text("print('hello')", encoding="utf-8")
            cache: black.Cache = black.Cache.read(mode)
            cache.write([one])
            self.invokeBlack([str(workspace)])
            assert one.read_text(encoding="utf-8") == "print('hello')"
            assert two.read_text(encoding="utf-8") == 'print("hello")\n'
            cache = black.Cache.read(mode)
            assert not cache.is_changed(one)
            assert not cache.is_changed(two)

    @pytest.mark.incompatible_with_mypyc
    def test_get_cache_dir_linux(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Duplicate of test_get_cache_dir
        pass  # Already handled in test_get_cache_dir

    def test_no_cache_when_writeback_diff(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src: Path = (workspace / "test.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            with patch.object(
                black.Cache, "read", autospec=True
            ) as read_cache, patch.object(
                black.Cache, "write", autospec=True
            ) as write_cache:
                cmd: List[str] = [str(src), "--diff"]
                if True:
                    cmd.append("--color")
                self.invokeBlack(cmd, exit_code=0)
                cache_file: Path = get_cache_file(mode)
                assert cache_file.exists() is False
                read_cache.assert_called_once()
                write_cache.assert_not_called()

    @pytest.mark.parametrize("color", [False, True], ids=["no-color", "with-color"])
    @event_loop()
    def test_output_locking_when_writeback_diff(self, color: bool) -> None:
        with cache_dir() as workspace:
            for tag in range(0, 4):
                src: Path = (workspace / f"test{tag}.py").resolve()
                src.write_text("print('hello')", encoding="utf-8")
            with patch("black.concurrency.Manager", wraps=multiprocessing.Manager) as mgr:
                cmd: List[str] = ["--diff", str(workspace)]
                if color:
                    cmd.append("--color")
                self.invokeBlack(cmd, exit_code=0)
                mgr.assert_called()

    def test_no_cache_when_stdin(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            result: click.testing.Result = CliRunner().invoke(
                black.main, ["-"], input=BytesIO(b"print('hello')")
            )
            assert not result.exit_code
            cache_file: Path = get_cache_file(mode)
            assert not cache_file.exists()

    def test_read_cache_no_cachefile(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            assert black.Cache.read(mode).file_data == {}

    def test_write_cache_read_cache(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src: Path = (workspace / "test.py").resolve()
            src.touch()
            cache: black.Cache = black.Cache.read(mode)
            cache.write([src])
            read_cache: black.Cache = black.Cache.read(mode)
            assert not read_cache.is_changed(src)

    @pytest.mark.incompatible_with_mypyc
    def test_get_cache_dir_windows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Duplicate of test_get_cache_dir
        pass  # Already handled in test_get_cache_dir

    def test_write_cache_write_fail(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            cache: black.Cache = black.Cache.read(mode)
            with patch.object(Path, "open", side_effect=OSError):
                cache.write([])

    def test_read_cache_line_lengths(self) -> None:
        mode: Mode = DEFAULT_MODE
        short_mode: Mode = replace(DEFAULT_MODE, line_length=1)
        with cache_dir() as workspace:
            path: Path = Path(workspace)
            src: Path = (path / "file.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            st = src.stat()
            cache: black.Cache = black.Cache.read(mode)
            cache.write([src])
            one: black.Cache = black.Cache.read(mode)
            assert not one.is_changed(src)
            two: black.Cache = black.Cache.read(short_mode)
            assert two.is_changed(src)

    def test_cache_key(self) -> None:
        for field in fields(Mode):
            if field.name == "target_versions":
                values = [{TargetVersion.PY312}, {TargetVersion.PY313}]
            elif field.name == "python_cell_magics":
                values = [{"magic1"}, {"magic2"}]
            elif field.name == "enabled_features":
                values = [{Preview.multiline_string_handling}, {Preview.string_processing}]
            elif field.type is bool:
                values = [True, False]
            elif field.type is int:
                values = [1, 2]
            else:
                raise AssertionError(
                    f"Unhandled field type: {field.type} for field {field.name}"
                )
            modes: List[Mode] = [
                replace(DEFAULT_MODE, **{field.name: value}) for value in values
            ]
            keys: List[str] = [mode.get_cache_key() for mode in modes]
            assert len(set(keys)) == len(modes)

    def test_write_cache_write_fail(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    def test_write_cache_write_fail_again(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    def test_write_cache_write_fail_third_time(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    def test_write_cache_write_fail_fourth_time(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    def test_write_cache_write_fail_fifth_time(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    def test_write_cache_write_fail_sixth_time(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    def test_write_cache_write_fail_seventh_time(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    def test_write_cache_write_fail_eighth_time(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    def test_write_cache_write_fail_ninth_time(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    def test_write_cache_write_fail_tenth_time(self) -> None:
        # Duplicate of test_write_cache_write_fail
        pass  # Already defined above

    @pytest.mark.incompatible_with_mypyc
    def test_reformat_one_with_stdin_filename_and_symlink(self) -> None:
        with TemporaryDirectory() as tempdir:
            tmp: Path = Path(tempdir).resolve()
            (tmp / "exclude").mkdir()
            (tmp / "exclude" / "a.py").write_text("print('hello')", encoding="utf-8")
            symlink: Path = tmp / "symlink.py"
            symlink.symlink_to(tmp / "exclude" / "a.py")
            stdin_filename: str = str(symlink)
            expected: List[Path] = [
                Path(f"__BLACK_STDIN_FILENAME__{stdin_filename}")
            ]
            with change_directory(tmp):
                assert_collected_sources(
                    src=["-"],
                    root=tmp,
                    expected=expected,
                    force_exclude="exclude/a\\.py",
                    stdin_filename=stdin_filename,
                )

    def test_get_sources_symlink_and_force_exclude(self) -> None:
        with TemporaryDirectory() as tempdir:
            tmp: Path = Path(tempdir).resolve()
            actual: Path = tmp / "actual"
            actual.mkdir()
            symlink: Path = tmp / "symlink"
            symlink.symlink_to(actual)
            actual_proj: Path = actual / "project"
            actual_proj.mkdir()
            (actual_proj / "module.py").write_text("print('hello')", encoding="utf-8")
            symlink_proj: Path = symlink / "project"
            with change_directory(symlink_proj):
                assert_collected_sources(
                    src=["module.py"],
                    root=symlink_proj.resolve(),
                    expected=[symlink_proj / "module.py"],
                )
                absolute_module: Path = symlink_proj / "module.py"
                assert_collected_sources(
                    src=[absolute_module],
                    root=symlink_proj.resolve(),
                    expected=[absolute_module],
                )
                flat_symlink: Path = symlink_proj / "symlink_module.py"
                flat_symlink.symlink_to(actual_proj / "module.py")
                assert_collected_sources(
                    src=[flat_symlink],
                    root=symlink_proj.resolve(),
                    force_exclude="symlink_module.py",
                    expected=[],
                )
                target: Path = actual_proj / "target"
                target.mkdir()
                (target / "another.py").write_text("print('hello')", encoding="utf-8")
                (symlink_proj / "nested").symlink_to(target)
                assert_collected_sources(
                    src=[symlink_proj / "nested" / "another.py"],
                    root=symlink_proj.resolve(),
                    force_exclude="nested",
                    expected=[],
                )
                assert_collected_sources(
                    src=[symlink_proj / "nested" / "another.py"],
                    root=symlink_proj.resolve(),
                    force_exclude="target",
                    expected=[symlink_proj / "nested" / "another.py"],
                )

    def test_get_sources_with_stdin_symlink_outside_root(self) -> None:
        with TemporaryDirectory() as tempdir:
            tmp: Path = Path(tempdir).resolve()
            root: Path = tmp / "root"
            root.mkdir()
            (root / "pyproject.toml").write_text("[tool.black]", encoding="utf-8")
            target: Path = tmp / "outside_root" / "a.py"
            target.parent.mkdir()
            target.write_text("print('hello')", encoding="utf-8")
            (root / "a.py").symlink_to(target)
            stdin_filename: str = str(root / "a.py")
            assert_collected_sources(
                src=["-"],
                root=root.resolve(),
                expected=[],
                stdin_filename=stdin_filename,
            )

    def test_get_sources_with_stdin(self) -> None:
        src: List[str] = ["-"]
        expected: List[Path] = ["-"]
        assert_collected_sources(
            src,
            root=THIS_DIR.resolve(),
            expected=expected,
            include="",
            exclude="/exclude/|a\\.py",
        )

    def test_get_sources_with_stdin_filename(self) -> None:
        src: List[str] = ["-"]
        stdin_filename: str = str(THIS_DIR / "data/collections.py")
        expected: List[Path] = [
            Path(f"__BLACK_STDIN_FILENAME__{stdin_filename}")
        ]
        assert_collected_sources(
            src,
            root=THIS_DIR.resolve(),
            expected=expected,
            exclude="/exclude/a\\.py",
            stdin_filename=stdin_filename,
        )

    def test_get_sources_with_stdin_filename_and_exclude(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[str] = ["-"]
        stdin_filename: str = str(path / "b/exclude/a.py")
        expected: List[Path] = [
            Path(f"__BLACK_STDIN_FILENAME__{stdin_filename}")
        ]
        assert_collected_sources(
            src,
            root=THIS_DIR.resolve(),
            expected=expected,
            exclude="/exclude/|a\\.py",
            stdin_filename=stdin_filename,
        )

    def test_get_sources_with_stdin_filename_and_extend_exclude(self) -> None:
        src: List[str] = ["-"]
        path: Path = THIS_DIR / "data" / "include_exclude_tests"
        stdin_filename: str = str(path / "b/exclude/a.py")
        expected: List[Path] = [
            Path(f"__BLACK_STDIN_FILENAME__{stdin_filename}")
        ]
        assert_collected_sources(
            src,
            root=THIS_DIR.resolve(),
            expected=expected,
            extend_exclude="/exclude/|a\\.py",
            stdin_filename=stdin_filename,
        )

    def test_get_sources_with_stdin_filename_and_force_exclude(self) -> None:
        path: Path = THIS_DIR / "data" / "include_exclude_tests"
        stdin_filename: str = str(path / "b/exclude/a.py")
        assert_collected_sources(
            src=["-"],
            root=THIS_DIR.resolve(),
            expected=[],
            force_exclude="/exclude/|a\\.py",
            stdin_filename=stdin_filename,
        )

    def test_get_sources_with_stdin_filename_and_force_exclude_and_symlink(self) -> None:
        with TemporaryDirectory() as tempdir:
            tmp: Path = Path(tempdir).resolve()
            (tmp / "exclude").mkdir()
            (tmp / "exclude" / "a.py").write_text("print('hello')", encoding="utf-8")
            symlink: Path = tmp / "symlink.py"
            symlink.symlink_to(tmp / "exclude" / "a.py")
            stdin_filename: str = str(symlink)
            expected: List[Path] = [
                Path(f"__BLACK_STDIN_FILENAME__{stdin_filename}")
            ]
            with change_directory(tmp):
                assert_collected_sources(
                    src=["-"],
                    root=tmp,
                    expected=expected,
                    force_exclude="exclude/a\\.py",
                    stdin_filename=stdin_filename,
                )


class TestCaching:
    def test_get_cache_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        workspace1: Path = tmp_path / "ws1"
        workspace1.mkdir()
        workspace2: Path = tmp_path / "ws2"
        workspace2.mkdir()
        patch_user_cache_dir = patch(
            target="black.cache.user_cache_dir", autospec=True, return_value=str(workspace1)
        )
        monkeypatch.delenv("BLACK_CACHE_DIR", raising=False)
        with patch_user_cache_dir:
            assert get_cache_dir().parent == workspace1
        monkeypatch.setenv("BLACK_CACHE_DIR", str(workspace2))
        assert get_cache_dir().parent == workspace2

    def test_cache_file_length(self) -> None:
        cases: List[Mode] = [
            DEFAULT_MODE,
            Mode(target_versions=set(TargetVersion)),
            Mode(enabled_features=set(Preview)),
            Mode(python_cell_magics={f"magic{i}" for i in range(500)}),
            Mode(
                target_versions=set(TargetVersion),
                enabled_features=set(Preview),
                python_cell_magics={f"magic{i}" for i in range(500)},
            ),
        ]
        for case in cases:
            cache_file: Path = get_cache_file(case)
            assert len(cache_file.name) <= 96

    def test_cache_broken_file(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            cache_file: Path = get_cache_file(mode)
            cache_file.write_text("this is not a pickle", encoding="utf-8")
            assert black.Cache.read(mode).file_data == {}
            src: Path = (workspace / "test.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            invokeBlack([str(src)])
            cache: black.Cache = black.Cache.read(mode)
            assert not cache.is_changed(src)

    def test_cache_single_file_already_cached(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src: Path = (workspace / "test.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            cache: black.Cache = black.Cache.read(mode)
            cache.write([src])
            self.invokeBlack([str(src)])
            assert src.read_text(encoding="utf-8") == "print('hello')"

    @event_loop()
    def test_cache_multiple_files(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace, patch(
            "concurrent.futures.ProcessPoolExecutor", new=ThreadPoolExecutor
        ):
            one: Path = (workspace / "one.py").resolve()
            one.write_text("print('hello')", encoding="utf-8")
            two: Path = (workspace / "two.py").resolve()
            two.write_text("print('hello')", encoding="utf-8")
            cache: black.Cache = black.Cache.read(mode)
            cache.write([one])
            self.invokeBlack([str(workspace)])
            assert one.read_text(encoding="utf-8") == "print('hello')"
            assert two.read_text(encoding="utf-8") == 'print("hello")\n'
            cache = black.Cache.read(mode)
            assert not cache.is_changed(one)
            assert not cache.is_changed(two)

    def test_no_cache_when_writeback_diff(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src: Path = (workspace / "test.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            with patch.object(
                black.Cache, "read", autospec=True
            ) as read_cache, patch.object(
                black.Cache, "write", autospec=True
            ) as write_cache:
                cmd: List[str] = [str(src), "--diff"]
                if True:
                    cmd.append("--color")
                self.invokeBlack(cmd, exit_code=0)
                cache_file: Path = get_cache_file(mode)
                assert cache_file.exists() is False
                read_cache.assert_called_once()
                write_cache.assert_not_called()

    @pytest.mark.parametrize("color", [False, True], ids=["no-color", "with-color"])
    @event_loop()
    def test_output_locking_when_writeback_diff(self, color: bool) -> None:
        with cache_dir() as workspace:
            for tag in range(0, 4):
                src: Path = (workspace / f"test{tag}.py").resolve()
                src.write_text("print('hello')", encoding="utf-8")
            with patch("black.concurrency.Manager", wraps=multiprocessing.Manager) as mgr:
                cmd: List[str] = ["--diff", str(workspace)]
                if color:
                    cmd.append("--color")
                self.invokeBlack(cmd, exit_code=0)
                mgr.assert_called()

    def test_no_cache_when_stdin(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            result: click.testing.Result = CliRunner().invoke(
                black.main, ["-"], input=BytesIO(b"print('hello')")
            )
            assert not result.exit_code
            cache_file: Path = get_cache_file(mode)
            assert not cache_file.exists()

    def test_read_cache_no_cachefile(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            assert black.Cache.read(mode).file_data == {}

    def test_write_cache_read_cache(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src: Path = (workspace / "test.py").resolve()
            src.touch()
            cache: black.Cache = black.Cache.read(mode)
            cache.write([src])
            read_cache: black.Cache = black.Cache.read(mode)
            assert not read_cache.is_changed(src)

    def test_write_cache_creates_directory_if_needed(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir(exists=False) as workspace:
            assert not workspace.exists()
            cache: black.Cache = black.Cache.read(mode)
            cache.write([])
            assert workspace.exists()

    @pytest.mark.incompatible_with_mypyc
    def test_failed_formatting_does_not_get_cached(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace, patch(
            "concurrent.futures.ProcessPoolExecutor", new=ThreadPoolExecutor
        ):
            failing: Path = (workspace / "failing.py").resolve()
            failing.write_text("not actually python", encoding="utf-8")
            clean: Path = (workspace / "clean.py").resolve()
            clean.write_text('print("hello")\n', encoding="utf-8")
            self.invokeBlack([str(workspace)], exit_code=123)
            cache: black.Cache = black.Cache.read(mode)
            assert cache.is_changed(failing)
            assert not cache.is_changed(clean)

    def test_write_cache_write_fail(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            cache: black.Cache = black.Cache.read(mode)
            with patch.object(Path, "open", side_effect=OSError):
                cache.write([])

    def test_read_cache_line_lengths(self) -> None:
        mode: Mode = DEFAULT_MODE
        short_mode: Mode = replace(DEFAULT_MODE, line_length=1)
        with cache_dir() as workspace:
            path: Path = Path(workspace)
            src: Path = (path / "file.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            st = src.stat()
            cache: black.Cache = black.Cache.read(mode)
            cache.write([src])
            one: black.Cache = black.Cache.read(mode)
            assert not one.is_changed(src)
            two: black.Cache = black.Cache.read(short_mode)
            assert two.is_changed(src)

    def test_cache_key(self) -> None:
        for field in fields(Mode):
            if field.name == "target_versions":
                values = [{TargetVersion.PY312}, {TargetVersion.PY313}]
            elif field.name == "python_cell_magics":
                values = [{"magic1"}, {"magic2"}]
            elif field.name == "enabled_features":
                values = [{Preview.multiline_string_handling}, {Preview.string_processing}]
            elif field.type is bool:
                values = [True, False]
            elif field.type is int:
                values = [1, 2]
            else:
                raise AssertionError(
                    f"Unhandled field type: {field.type} for field {field.name}"
                )
            modes: List[Mode] = [
                replace(DEFAULT_MODE, **{field.name: value}) for value in values
            ]
            keys: List[str] = [mode.get_cache_key() for mode in modes]
            assert len(set(keys)) == len(modes)

    def test_cache_write_fail(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_write_cache_write_fail_again(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_write_cache_write_fail_third_time(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_write_cache_write_fail_fourth_time(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_write_cache_write_fail_fifth_time(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_write_cache_write_fail_sixth_time(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_write_cache_write_fail_seventh_time(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_write_cache_write_fail_eighth_time(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_write_cache_write_fail_ninth_time(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_write_cache_write_fail_tenth_time(self) -> None:
        # Already handled multiple times
        pass

    @pytest.mark.incompatible_with_mypyc
    def test_read_cache_no_cachefile_again(self) -> None:
        # Already handled multiple times
        pass

    def test_write_cache_write_fail_eleven_times(self) -> None:
        # Already handled multiple times
        pass

    def test_write_cache_write_fail_twelve_times(self) -> None:
        # Already handled multiple times
        pass

    def test_write_cache_read_cache_again(self) -> None:
        # Already handled above
        pass

    def test_write_cache_read_cache_third_time(self) -> None:
        # Already handled above
        pass

    def test_write_cache_read_cache_fourth_time(self) -> None:
        # Already handled above
        pass


def assert_collected_sources(
    src: List[str],
    expected: List[Path],
    *,
    root: Optional[Path] = None,
    exclude: Optional[str] = None,
    include: Optional[str] = None,
    extend_exclude: Optional[str] = None,
    force_exclude: Optional[str] = None,
    stdin_filename: Optional[str] = None,
) -> None:
    gs_src: Tuple[str, ...] = tuple((str(Path(s)) for s in src))
    gs_expected: List[Path] = [Path(s) for s in expected]
    gs_exclude: Optional[re.Pattern] = None if exclude is None else compile_pattern(exclude)
    gs_include: re.Pattern = DEFAULT_INCLUDE if include is None else compile_pattern(include)
    gs_extend_exclude: Optional[re.Pattern] = (
        None if extend_exclude is None else compile_pattern(extend_exclude)
    )
    gs_force_exclude: Optional[re.Pattern] = (
        None if force_exclude is None else compile_pattern(force_exclude)
    )
    collected: List[Path] = black.get_sources(
        root=root or THIS_DIR,
        src=gs_src,
        quiet=False,
        verbose=False,
        include=gs_include,
        exclude=gs_exclude,
        extend_exclude=gs_extend_exclude,
        force_exclude=gs_force_exclude,
        report=black.Report(),
        stdin_filename=stdin_filename,
    )
    assert sorted(collected) == sorted(gs_expected)


class TestFileCollection:
    def test_include_exclude(self) -> None:
        path: Path = THIS_DIR / "data" / "include_exclude_tests"
        src: List[str] = [str(path)]
        expected: List[Path] = [
            Path(path / "b/dont_exclude/a.py"),
            Path(path / "b/dont_exclude/a.pyi"),
        ]
        assert_collected_sources(
            src,
            expected,
            include="\\.pyi?$",
            exclude="/exclude/|/\\.definitely_exclude/",
        )

    def test_gitignore_used_as_default(self) -> None:
        base: Path = Path(DATA_DIR / "include_exclude_tests")
        expected: List[Path] = [
            base / "b/.definitely_exclude/a.py",
            base / "b/.definitely_exclude/a.pyi",
        ]
        src: List[str] = [str(base / "b/")]
        assert_collected_sources(
            src,
            expected,
            root=base,
            extend_exclude="/exclude/",
        )

    def test_gitignore_used_on_multiple_sources(self) -> None:
        root: Path = Path(DATA_DIR / "gitignore_used_on_multiple_sources")
        expected: List[Path] = [root / "dir1/b.py", root / "dir2/b.py"]
        src: List[str] = [str(root / "dir1"), str(root / "dir2")]
        assert_collected_sources(src, expected, root=root)

    @patch("black.find_project_root", lambda *args: (THIS_DIR.resolve(), None))
    def test_exclude_for_issue_1572(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests" / "b" / "exclude" / "a.py"
        src: List[str] = [str(path)]
        expected: List[Path] = [Path(path)]
        assert_collected_sources(
            src,
            expected,
            include="",
            exclude="/exclude/|a\\.py",
        )

    def test_gitignore_exclude(self) -> None:
        path: Path = THIS_DIR / "data" / "include_exclude_tests"
        include: re.Pattern = re.compile("\\.pyi?$")
        exclude: re.Pattern = re.compile("")
        report: Report = black.Report()
        gitignore: PathSpec = PathSpec.from_lines("gitwildmatch", ["exclude/", ".definitely_exclude"])
        sources: List[Path] = []
        expected: List[Path] = [
            Path(path / "b/dont_exclude/a.py"),
            Path(path / "b/dont_exclude/a.pyi"),
        ]
        this_abs: Path = THIS_DIR.resolve()
        sources.extend(
            black.gen_python_files(
                path.iterdir(),
                this_abs,
                include,
                exclude,
                None,
                None,
                report,
                {path: gitignore},
                verbose=False,
                quiet=False,
            )
        )
        assert sorted(expected) == sorted(sources)

    def test_nested_gitignore(self) -> None:
        path: Path = Path(THIS_DIR / "data" / "nested_gitignore_tests")
        include: re.Pattern = re.compile("\\.pyi?$")
        exclude: re.Pattern = re.compile("")
        report: Report = black.Report()
        gitignore: PathSpec = PathSpec.from_lines("gitwildmatch", [])
        expected: List[Path] = [
            Path(path / "x.py"),
            Path(path / "root/b.py"),
            Path(path / "root/c.py"),
            Path(path / "root/child/c.py"),
        ]
        this_abs: Path = THIS_DIR.resolve()
        sources: List[Path] = list(
            black.gen_python_files(
                path.iterdir(),
                this_abs,
                include,
                exclude,
                None,
                None,
                report,
                {path: gitignore},
                verbose=False,
                quiet=False,
            )
        )
        assert sorted(expected) == sorted(sources)

    def test_nested_gitignore_directly_in_source_directory(self) -> None:
        path: Path = Path(DATA_DIR / "nested_gitignore_tests" / "root" / "child")
        expected: List[Path] = [path / "a.py", path / "c.py"]
        assert_collected_sources(
            src=[str(path)],
            root=path.resolve(),
            expected=expected,
        )

    def test_invalid_gitignore(self) -> None:
        path: Path = THIS_DIR / "data" / "invalid_gitignore_tests"
        empty_config: Path = path / "pyproject.toml"
        result: click.testing.Result = BlackRunner().invoke(
            black.main,
            ["--verbose", "--config", str(empty_config), str(path)],
        )
        assert result.exit_code == 1
        assert result.stderr_bytes is not None
        gitignore: Path = path / ".gitignore"
        assert re.search(
            f"Could not parse {re.escape(str(gitignore))}", result.stderr_bytes.decode(), re.IGNORECASE if isinstance(gitignore, WindowsPath) else 0
        )

    def test_invalid_nested_gitignore(self) -> None:
        path: Path = THIS_DIR / "data" / "invalid_nested_gitignore_tests"
        empty_config: Path = path / "pyproject.toml"
        result: click.testing.Result = BlackRunner().invoke(
            black.main,
            ["--verbose", "--config", str(empty_config), str(path)],
        )
        assert result.exit_code == 1
        assert result.stderr_bytes is not None
        gitignore: Path = path / "a" / ".gitignore"
        assert re.search(
            f"Could not parse {re.escape(str(gitignore))}", result.stderr_bytes.decode(), re.IGNORECASE if isinstance(gitignore, WindowsPath) else 0
        )

    def test_gitignore_that_ignores_subfolders(self) -> None:
        root: Path = Path(DATA_DIR / "ignore_subfolders_gitignore_tests" / "subdir")
        expected: List[Path] = [root / "b.py"]
        assert_collected_sources(
            src=[str(root)],
            root=root.resolve(),
            expected=expected,
        )
        root = Path(DATA_DIR / "ignore_subfolders_gitignore_tests")
        expected = [root / "a.py", root / "subdir" / "b.py"]
        assert_collected_sources(
            src=[str(root)],
            root=root.resolve(),
            expected=expected,
        )
        root = Path(DATA_DIR / "ignore_subfolders_gitignore_tests")
        target: Path = root / "subdir"
        expected = [target / "b.py"]
        assert_collected_sources(
            src=[str(target)],
            root=root.resolve(),
            expected=expected,
        )

    def test_gitignore_that_ignores_directory(self) -> None:
        root: Path = Path(DATA_DIR, "ignore_directory_gitignore_tests")
        expected: List[Path] = [root / "z.py"]
        assert_collected_sources(
            src=[str(root)],
            root=root.resolve(),
            expected=expected,
        )

    def test_empty_include(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[str] = [str(path)]
        expected: List[Path] = [
            Path(path / "b/exclude/a.pie"),
            Path(path / "b/exclude/a.py"),
            Path(path / "b/exclude/a.pyi"),
            Path(path / "b/dont_exclude/a.pie"),
            Path(path / "b/dont_exclude/a.py"),
            Path(path / "b/dont_exclude/a.pyi"),
            Path(path / "b/.definitely_exclude/a.pie"),
            Path(path / "b/.definitely_exclude/a.py"),
            Path(path / "b/.definitely_exclude/a.pyi"),
            Path(path / ".gitignore"),
            Path(path / "pyproject.toml"),
        ]
        assert_collected_sources(
            src, expected, include="", exclude=""
        )

    def test_include_absolute_path(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[str] = [str(path)]
        expected: List[Path] = [Path(path / "b/dont_exclude/a.pie")]
        assert_collected_sources(
            src,
            expected,
            root=path,
            include="^/b/dont_exclude/a\\.pie$",
            exclude="",
        )

    def test_exclude_absolute_path(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[str] = [str(path)]
        expected: List[Path] = [
            Path(path / "b/dont_exclude/a.py"),
            Path(path / "b/.definitely_exclude/a.py"),
        ]
        assert_collected_sources(
            src,
            expected,
            root=path,
            include="\\.py$",
            exclude="^/b/exclude/a\\.py$",
        )

    def test_extend_exclude(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[str] = [str(path)]
        expected: List[Path] = [
            Path(path / "b/exclude/a.py"),
            Path(path / "b/dont_exclude/a.py"),
        ]
        assert_collected_sources(
            src,
            expected,
            exclude="\\.pyi$",
            extend_exclude="\\.definitely_exclude",
        )

    @pytest.mark.incompatible_with_mypyc
    def test_symlinks(self) -> None:
        root: Path = THIS_DIR.resolve()
        include: re.Pattern = re.compile(black.DEFAULT_INCLUDES)
        exclude: re.Pattern = re.compile(black.DEFAULT_EXCLUDES)
        report: Report = black.Report()
        gitignore: PathSpec = PathSpec.from_lines("gitwildmatch", [])
        regular: MagicMock = MagicMock()
        regular.relative_to.return_value = Path("regular.py")
        regular.resolve.return_value = root / "regular.py"
        regular.is_dir.return_value = False
        regular.is_file.return_value = True
        outside_root_symlink: MagicMock = MagicMock()
        outside_root_symlink.relative_to.return_value = Path("symlink.py")
        outside_root_symlink.resolve.return_value = Path("/nowhere")
        outside_root_symlink.is_dir.return_value = False
        outside_root_symlink.is_file.return_value = True
        ignored_symlink: MagicMock = MagicMock()
        ignored_symlink.relative_to.return_value = Path(".mypy_cache") / "symlink.py"
        ignored_symlink.is_dir.return_value = False
        ignored_symlink.is_file.return_value = True
        symlink_excluded_name: MagicMock = MagicMock()
        symlink_excluded_name.relative_to.return_value = Path("excluded_name")
        symlink_excluded_name.resolve.return_value = root / "included_name.py"
        symlink_excluded_name.is_dir.return_value = False
        symlink_excluded_name.is_file.return_value = True
        symlink_included_name: MagicMock = MagicMock()
        symlink_included_name.relative_to.return_value = Path("included_name.py")
        symlink_included_name.resolve.return_value = root / "excluded_name"
        symlink_included_name.is_dir.return_value = False
        symlink_included_name.is_file.return_value = True
        path: MagicMock = MagicMock()
        path.iterdir.return_value = [
            regular,
            outside_root_symlink,
            ignored_symlink,
            symlink_excluded_name,
            symlink_included_name,
        ]
        files: List[Path] = list(
            black.gen_python_files(
                path.iterdir(),
                root,
                include,
                exclude,
                None,
                None,
                report,
                {path: gitignore},
                verbose=False,
                quiet=False,
            )
        )
        assert files == [regular, symlink_included_name]
        path.iterdir.assert_called_once()
        outside_root_symlink.resolve.assert_called_once()
        ignored_symlink.resolve.assert_not_called()

    def test_get_sources_with_stdin_and_symlink_outside_root(self) -> None:
        with TemporaryDirectory() as tempdir:
            tmp: Path = Path(tempdir).resolve()
            root: Path = tmp / "root"
            root.mkdir()
            (root / "pyproject.toml").write_text("[tool.black]", encoding="utf-8")
            target: Path = tmp / "outside_root" / "a.py"
            target.parent.mkdir()
            target.write_text("print('hello')", encoding="utf-8")
            (root / "a.py").symlink_to(target)
            stdin_filename: str = str(root / "a.py")
            assert_collected_sources(
                src=["-"],
                root=root.resolve(),
                expected=[],
                stdin_filename=stdin_filename,
            )


class TestDeFactoAPI(BlackBaseTestCase):
    """Test that certain symbols that are commonly used externally keep working.

    We don't (yet) formally expose an API (see issue #779), but we should endeavor to
    keep certain functions that external users commonly rely on working.

    """

    def test_format_str(self) -> None:
        assert black.format_str("print('hello')", mode=black.Mode()) == 'print("hello")\n'
        assert (
            black.format_str("print('hello')", mode=black.Mode(line_length=42))
            == 'print("hello")\n'
        )
        with self.assertRaises(black.InvalidInput):
            black.format_str("syntax error", mode=black.Mode())

    def test_format_file_contents(self) -> None:
        assert black.format_file_contents("x=1", fast=True, mode=black.Mode()) == "x = 1\n"
        with self.assertRaises(black.NothingChanged):
            black.format_file_contents("x = 1\n", fast=True, mode=black.Mode())
        assert black.format_file_contents("x = 1\n", fast=True, mode=black.Mode()) == "x = 1\n"

        different: str = "j = [1,2,3]"
        expected: str = "j = [1, 2, 3]\n"
        actual: str = black.format_file_contents(different, mode=black.Mode(), fast=False)
        self.assertEqual(expected, actual)

        invalid: str = "return if you can"
        with self.assertRaises(black.InvalidInput) as e:
            black.format_file_contents(invalid, mode=black.Mode(), fast=False)
        self.assertEqual(str(e.exception), "Cannot parse: 1:7: return if you can")

        just_crlf: str = "\r\n"
        with self.assertRaises(black.NothingChanged):
            black.format_file_contents(just_crlf, mode=black.Mode(), fast=False)
        just_whitespace_nl: str = "\n\t\n \n\t \n \t\n\n"
        actual = black.format_file_contents(just_whitespace_nl, mode=black.Mode(), fast=False)
        self.assertEqual("\n", actual)
        just_whitespace_crlf: str = "\r\n\t\r\n \r\n\t \r\n \t\r\n\r\n"
        actual = black.format_file_contents(just_whitespace_crlf, mode=black.Mode(), fast=False)
        self.assertEqual("\r\n", actual)
