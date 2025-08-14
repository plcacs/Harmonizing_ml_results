#!/usr/bin/env python3
from __future__ import annotations

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
from contextlib import redirect_stderr
from dataclasses import fields, replace
from io import BytesIO
from pathlib import Path, WindowsPath
from platform import system
from tempfile import TemporaryDirectory
from typing import Any, Optional, TypeVar, Union, List

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
from black.output import color_diff, diff, lines_with_leading_tabs_expanded
from black.parsing import ASTSafetyError
from black.report import Report

T = TypeVar("T")
R = TypeVar("R")


def assert_collected_sources(
    src: Sequence[Union[str, Path]],
    expected: Sequence[Union[str, Path]],
    *,
    root: Optional[Path] = None,
    exclude: Optional[str] = None,
    include: Optional[str] = None,
    extend_exclude: Optional[str] = None,
    force_exclude: Optional[str] = None,
    stdin_filename: Optional[str] = None,
) -> None:
    gs_src: tuple[str, ...] = tuple(str(Path(s)) for s in src)
    gs_expected: List[Path] = [Path(s) for s in expected]
    gs_exclude: Optional[re.Pattern[str]] = None if exclude is None else compile_pattern(exclude)
    gs_include: re.Pattern[str] = DEFAULT_INCLUDE if include is None else compile_pattern(include)
    gs_extend_exclude: Optional[re.Pattern[str]] = None if extend_exclude is None else compile_pattern(extend_exclude)
    gs_force_exclude: Optional[re.Pattern[str]] = None if force_exclude is None else compile_pattern(force_exclude)
    collected: List[Path] = black.get_sources(
        root=root or THIS_DIR,
        src=gs_src,
        quiet=False,
        verbose=False,
        include=gs_include,
        exclude=gs_exclude,
        extend_exclude=gs_extend_exclude,
        force_exclude=gs_force_exclude,
        report=Report(),
        stdin_filename=stdin_filename,
    )
    assert sorted(collected) == sorted(gs_expected)


class BlackTestCase(BlackBaseTestCase):
    def test_empty_ff(self) -> None:
        self.assert_format_roundtrip("")

    def test_one_empty_line(self) -> None:
        self.assert_format_roundtrip("\n")

    def test_line_breaks(self) -> None:
        self.assert_format_roundtrip("a = 1\n", fast=True)

    def check_features_used(self, source: str, expected: set[Feature]) -> None:
        node = black.lib2to3_parse(source)
        actual = black.get_features_used(node)
        msg = f"Expected {expected} but got {actual} for {source!r}"
        try:
            self.assertEqual(actual, expected, msg=msg)
        except AssertionError:
            DebugVisitor.show(node)
            raise

    def test_get_features_used(self) -> None:
        self.check_features_used("def f(*, arg): ...\n", set())
        self.check_features_used("def f(*, arg,): ...\n", {Feature.TRAILING_COMMA_IN_DEF})
        self.check_features_used("f(*arg,)\n", {Feature.TRAILING_COMMA_IN_CALL})
        self.check_features_used("def f(*, arg): f'string'\n", {Feature.F_STRINGS})
        self.check_features_used("123_456\n", {Feature.NUMERIC_UNDERSCORES})
        self.check_features_used("123456\n", set())
        source, expected = read_data("cases", "function")
        expected_features: set[Feature] = {Feature.TRAILING_COMMA_IN_CALL, Feature.TRAILING_COMMA_IN_DEF, Feature.F_STRINGS}
        self.check_features_used(source, expected_features)
        self.check_features_used(expected, expected_features)
        source, expected = read_data("cases", "expression")
        self.check_features_used(source, set())
        self.check_features_used(expected, set())
        self.check_features_used("lambda a, /, b: ...\n", {Feature.POS_ONLY_ARGUMENTS})
        self.check_features_used("def fn(a, /, b): ...", {Feature.POS_ONLY_ARGUMENTS})
        self.check_features_used("def fn(): yield a, b", set())
        self.check_features_used("def fn(): return a, b", set())
        self.check_features_used("def fn(): yield *b, c", {Feature.UNPACKING_ON_FLOW})
        self.check_features_used("def fn(): return a, *b, c", {Feature.UNPACKING_ON_FLOW})
        self.check_features_used("x = a, *b, c", set())
        self.check_features_used("x: Any = regular", set())
        self.check_features_used("x: Any = (regular, regular)", set())
        self.check_features_used("x: Any = Complex(Type(1))[something]", set())
        self.check_features_used("x: Tuple[int, ...] = a, b, c", {Feature.ANN_ASSIGN_EXTENDED_RHS})
        self.check_features_used("try: pass\nexcept Something: pass", set())
        self.check_features_used("try: pass\nexcept (*Something,): pass", set())
        self.check_features_used("try: pass\nexcept *Group: pass", {Feature.EXCEPT_STAR})
        self.check_features_used("a[*b]", {Feature.VARIADIC_GENERICS})
        self.check_features_used("a[x, *y(), z] = t", {Feature.VARIADIC_GENERICS})
        self.check_features_used("def fn(*args: *T): pass", {Feature.VARIADIC_GENERICS})
        self.check_features_used("def fn(*args: *tuple[*T]): pass", {Feature.VARIADIC_GENERICS})
        self.check_features_used("with a: pass", set())
        self.check_features_used("with a, b: pass", set())
        self.check_features_used("with a as b: pass", set())
        self.check_features_used("with a as b, c as d: pass", set())
        self.check_features_used("with (a): pass", set())
        self.check_features_used("with (a, b): pass", set())
        self.check_features_used("with (a, b) as (c, d): pass", set())
        self.check_features_used("with (a as b): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS})
        self.check_features_used("with ((a as b)): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS})
        self.check_features_used("with (a, b as c): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS})
        self.check_features_used("with (a, (b as c)): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS})
        self.check_features_used("with ((a, ((b as c)))): pass", {Feature.PARENTHESIZED_CONTEXT_MANAGERS})

    def test_debug_visitor(self) -> None:
        source, _ = read_data("miscellaneous", "debug_visitor")
        expected, _ = read_data("miscellaneous", "debug_visitor.out")
        out_lines: List[str] = []
        err_lines: List[str] = []

        def out(msg: str, **kwargs: Any) -> None:
            out_lines.append(msg)

        def err(msg: str, **kwargs: Any) -> None:
            err_lines.append(msg)

        with pytest.raises(AssertionError):
            with pytest.raises(Exception):
                DebugVisitor.show(source)
        actual = "\n".join(out_lines) + "\n"
        log_name: str = ""
        if expected != actual:
            log_name = black.dump_to_file(*out_lines)
        self.assertEqual(
            expected,
            actual,
            f"AST print out is different. Actual version dumped to {log_name}",
        )


class TestCaching:
    def test_get_cache_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        workspace1: Path = tmp_path / "ws1"
        workspace1.mkdir()
        workspace2: Path = tmp_path / "ws2"
        workspace2.mkdir()

        patch_user_cache_dir: Any = pytest.helpers.patch(
            target="black.cache.user_cache_dir",
            autospec=True,
            return_value=str(workspace1),
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
            cache = black.Cache.read(mode)
            assert not cache.is_changed(src)

    def test_cache_single_file_already_cached(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src: Path = (workspace / "test.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            cache = black.Cache.read(mode)
            cache.write([src])
            invokeBlack([str(src)])
            assert src.read_text(encoding="utf-8") == "print('hello')"

    @pytest.mark.asyncio
    async def test_cache_multiple_files(self) -> None:
        mode: Mode = DEFAULT_MODE
        with (
            cache_dir() as workspace,
            patch("concurrent.futures.ProcessPoolExecutor", new=ThreadPoolExecutor),
        ):
            one: Path = (workspace / "one.py").resolve()
            one.write_text("print('hello')", encoding="utf-8")
            two: Path = (workspace / "two.py").resolve()
            two.write_text("print('hello')", encoding="utf-8")
            cache = black.Cache.read(mode)
            cache.write([one])
            invokeBlack([str(workspace)])
            assert one.read_text(encoding="utf-8") == "print('hello')"
            assert two.read_text(encoding="utf-8") == 'print("hello")\n'
            cache = black.Cache.read(mode)
            assert not cache.is_changed(one)
            assert not cache.is_changed(two)

    @pytest.mark.parametrize("color", [False, True], ids=["no-color", "with-color"])
    def test_no_cache_when_writeback_diff(self, color: bool) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir() as workspace:
            src: Path = (workspace / "test.py").resolve()
            src.write_text("print('hello')", encoding="utf-8")
            with (
                patch.object(black.Cache, "read") as read_cache,
                patch.object(black.Cache, "write") as write_cache,
            ):
                cmd: List[str] = [str(src), "--diff"]
                if color:
                    cmd.append("--color")
                invokeBlack(cmd)
                cache_file: Path = get_cache_file(mode)
                assert cache_file.exists() is False
                read_cache.assert_called_once()
                write_cache.assert_not_called()

    @pytest.mark.parametrize("color", [False, True], ids=["no-color", "with-color"])
    @pytest.mark.asyncio
    async def test_output_locking_when_writeback_diff(self, color: bool) -> None:
        with cache_dir() as workspace:
            for tag in range(0, 4):
                src: Path = (workspace / f"test{tag}.py").resolve()
                src.write_text("print('hello')", encoding="utf-8")
            with patch("black.concurrency.Manager", wraps=multiprocessing.Manager) as mgr:
                cmd: List[str] = ["--diff", str(workspace)]
                if color:
                    cmd.append("--color")
                invokeBlack(cmd, exit_code=0)
                mgr.assert_called()

    def test_no_cache_when_stdin(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            result = CliRunner().invoke(
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
            write_cache = black.Cache.read(mode)
            write_cache.write([src])
            read_cache = black.Cache.read(mode)
            assert not read_cache.is_changed(src)

    def test_filter_cached(self) -> None:
        with TemporaryDirectory() as workspace:
            path = Path(workspace)
            uncached: Path = (path / "uncached").resolve()
            cached: Path = (path / "cached").resolve()
            cached_but_changed: Path = (path / "changed").resolve()
            uncached.touch()
            cached.touch()
            cached_but_changed.touch()
            cache = black.Cache.read(DEFAULT_MODE)

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
            cache = black.Cache.read(DEFAULT_MODE)
            cache.write([src])
            cached_file_data = cache.file_data[str(src)]
            todo, done = cache.filtered_cached([src])
            assert todo == set()
            assert done == {src}
            assert cached_file_data.st_mtime == st.st_mtime
            cached_file_data = cache.file_data[str(src)] = FileData(
                cached_file_data.st_mtime - 1,
                cached_file_data.st_size,
                cached_file_data.hash,
            )
            todo, done = cache.filtered_cached([src])
            assert todo == set()
            assert done == {src}
            assert cached_file_data.st_mtime < st.st_mtime
            assert cached_file_data.st_size == st.st_size
            assert cached_file_data.hash == black.Cache.hash_digest(src)
            src.write_text("print('hello world')", encoding="utf-8")
            new_st = src.stat()
            todo, done = cache.filtered_cached([src])
            assert todo == {src}
            assert done == set()
            assert cached_file_data.st_mtime < new_st.st_mtime
            assert cached_file_data.st_size != new_st.st_size
            assert cached_file_data.hash != black.Cache.hash_digest(src)

    def test_write_cache_creates_directory_if_needed(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir(exists=False) as workspace:
            assert not workspace.exists()
            cache = black.Cache.read(mode)
            cache.write([])
            assert workspace.exists()

    @pytest.mark.asyncio
    async def test_failed_formatting_does_not_get_cached(self) -> None:
        mode: Mode = DEFAULT_MODE
        with (
            cache_dir() as workspace,
            patch("concurrent.futures.ProcessPoolExecutor", new=ThreadPoolExecutor),
        ):
            failing: Path = (workspace / "failing.py").resolve()
            failing.write_text("not actually python", encoding="utf-8")
            clean: Path = (workspace / "clean.py").resolve()
            clean.write_text('print("hello")\n', encoding="utf-8")
            invokeBlack([str(workspace)], exit_code=123)
            cache = black.Cache.read(mode)
            assert cache.is_changed(failing)
            assert not cache.is_changed(clean)

    def test_write_cache_write_fail(self) -> None:
        mode: Mode = DEFAULT_MODE
        with cache_dir():
            cache = black.Cache.read(mode)
            with patch.object(Path, "open") as mock:
                mock.side_effect = OSError
                cache.write([])

    def test_read_cache_line_lengths(self) -> None:
        mode: Mode = DEFAULT_MODE
        short_mode: Mode = replace(DEFAULT_MODE, line_length=1)
        with cache_dir() as workspace:
            path: Path = (workspace / "file.py").resolve()
            path.touch()
            cache = black.Cache.read(mode)
            cache.write([path])
            one = black.Cache.read(mode)
            assert not one.is_changed(path)
            two = black.Cache.read(short_mode)
            assert two.is_changed(path)

    def test_cache_key(self) -> None:
        for field in fields(Mode):
            values: List[Any]
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
                raise AssertionError(f"Unhandled field type: {field.type} for field {field.name}")
            modes = [replace(DEFAULT_MODE, **{field.name: value}) for value in values]
            keys = [mode.get_cache_key() for mode in modes]
            assert len(set(keys)) == len(modes)


class TestFileCollection:
    def test_include_exclude(self) -> None:
        path: Path = THIS_DIR / "data" / "include_exclude_tests"
        src: List[Path] = [path]
        expected: List[Path] = [
            Path(path / "b/dont_exclude/a.py"),
            Path(path / "b/dont_exclude/a.pyi"),
        ]
        assert_collected_sources(
            src,
            expected,
            include=r"\.pyi?$",
            exclude=r"/exclude/|/\.definitely_exclude/",
        )

    def test_gitignore_used_as_default(self) -> None:
        base: Path = Path(DATA_DIR / "include_exclude_tests")
        expected: List[Path] = [
            base / "b/.definitely_exclude/a.py",
            base / "b/.definitely_exclude/a.pyi",
        ]
        src: List[Path] = [base / "b/"]
        assert_collected_sources(src, expected, root=base, extend_exclude=r"/exclude/")

    def test_gitignore_used_on_multiple_sources(self) -> None:
        root: Path = Path(DATA_DIR / "gitignore_used_on_multiple_sources")
        expected: List[Path] = [
            root / "dir1" / "b.py",
            root / "dir2" / "b.py",
        ]
        src: List[Path] = [root / "dir1", root / "dir2"]
        assert_collected_sources(src, expected, root=root)

    @patch("black.find_project_root", lambda *args: (THIS_DIR.resolve(), None))
    def test_exclude_for_issue_1572(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[Path] = [path / "b/exclude/a.py"]
        expected: List[Path] = [path / "b/exclude/a.py"]
        assert_collected_sources(src, expected, include="", exclude=r"/exclude/|a\.py")

    def test_gitignore_exclude(self) -> None:
        path: Path = THIS_DIR / "data" / "include_exclude_tests"
        include: re.Pattern[str] = re.compile(r"\.pyi?$")
        exclude: re.Pattern[str] = re.compile(r"")
        report: Report = Report()
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
        include: re.Pattern[str] = re.compile(r"\.pyi?$")
        exclude: re.Pattern[str] = re.compile(r"")
        root_gitignore: Any = black.files.get_gitignore(path)
        report: Report = Report()
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
                {path: root_gitignore},
                verbose=False,
                quiet=False,
            )
        )
        assert sorted(expected) == sorted(sources)

    def test_nested_gitignore_directly_in_source_directory(self) -> None:
        path: Path = Path(DATA_DIR / "nested_gitignore_tests")
        src: Path = Path(path / "root" / "child")
        expected: List[Path] = [src / "a.py", src / "c.py"]
        assert_collected_sources([src], expected)

    def test_invalid_gitignore(self) -> None:
        path: Path = THIS_DIR / "data" / "invalid_gitignore_tests"
        empty_config: Path = path / "pyproject.toml"
        result = BlackRunner().invoke(
            black.main, ["--verbose", "--config", str(empty_config), str(path)]
        )
        assert result.exit_code == 1
        assert result.stderr_bytes is not None
        gitignore: Path = path / ".gitignore"
        assert re.search(
            f"Could not parse {gitignore}".replace("\\", "\\\\"),
            result.stderr_bytes.decode(),
            re.IGNORECASE if isinstance(gitignore, WindowsPath) else 0,
        )

    def test_invalid_nested_gitignore(self) -> None:
        path: Path = THIS_DIR / "data" / "invalid_nested_gitignore_tests"
        empty_config: Path = path / "pyproject.toml"
        result = BlackRunner().invoke(
            black.main, ["--verbose", "--config", str(empty_config), str(path)]
        )
        assert result.exit_code == 1
        assert result.stderr_bytes is not None
        gitignore: Path = path / "a" / ".gitignore"
        assert re.search(
            f"Could not parse {gitignore}".replace("\\", "\\\\"),
            result.stderr_bytes.decode(),
            re.IGNORECASE if isinstance(gitignore, WindowsPath) else 0,
        )

    def test_gitignore_that_ignores_subfolders(self) -> None:
        root: Path = Path(DATA_DIR / "ignore_subfolders_gitignore_tests" / "subdir")
        expected: List[Path] = [root / "b.py"]
        assert_collected_sources([root], expected, root=root)
        root = Path(DATA_DIR / "ignore_subfolders_gitignore_tests")
        expected = [root / "a.py", root / "subdir" / "b.py"]
        assert_collected_sources([root], expected, root=root)
        root = Path(DATA_DIR / "ignore_subfolders_gitignore_tests")
        target: Path = root / "subdir"
        expected = [target / "b.py"]
        assert_collected_sources([target], expected, root=root)

    def test_gitignore_that_ignores_directory(self) -> None:
        root: Path = Path(DATA_DIR, "ignore_directory_gitignore_tests")
        expected: List[Path] = [root / "z.py"]
        assert_collected_sources([root], expected, root=root)

    def test_empty_include(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[Path] = [path]
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
        assert_collected_sources(src, expected, include="", exclude="")

    def test_include_absolute_path(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[Path] = [path]
        expected: List[Path] = [Path(path / "b/dont_exclude/a.pie")]
        assert_collected_sources(
            src, expected, root=path, include=r"^/b/dont_exclude/a\.pie$", exclude=""
        )

    def test_exclude_absolute_path(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[Path] = [path]
        expected: List[Path] = [
            Path(path / "b/dont_exclude/a.py"),
            Path(path / "b/.definitely_exclude/a.py"),
        ]
        assert_collected_sources(
            src, expected, root=path, include=r"\.py$", exclude=r"^/b/exclude/a\.py$"
        )

    def test_extend_exclude(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[Path] = [path]
        expected: List[Path] = [
            Path(path / "b/exclude/a.py"),
            Path(path / "b/dont_exclude/a.py"),
        ]
        assert_collected_sources(
            src,
            expected,
            exclude=r"\.pyi$",
            extend_exclude=r"\.definitely_exclude"
        )

    @pytest.mark.asyncio
    async def test_symlinks(self) -> None:
        root: Path = THIS_DIR.resolve()
        include: re.Pattern[str] = re.compile(black.DEFAULT_INCLUDES)
        exclude: re.Pattern[str] = re.compile(black.DEFAULT_EXCLUDES)
        report: Report = Report()
        gitignore: PathSpec = PathSpec.from_lines("gitwildmatch", [])
        regular: Any = MagicMock()
        regular.relative_to.return_value = Path("regular.py")
        regular.resolve.return_value = root / "regular.py"
        regular.is_dir.return_value = False
        regular.is_file.return_value = True
        outside_root_symlink: Any = MagicMock()
        outside_root_symlink.relative_to.return_value = Path("symlink.py")
        outside_root_symlink.resolve.return_value = Path("/nowhere")
        outside_root_symlink.is_dir.return_value = False
        outside_root_symlink.is_file.return_value = True
        ignored_symlink: Any = MagicMock()
        ignored_symlink.relative_to.return_value = Path(".mypy_cache") / "symlink.py"
        ignored_symlink.is_dir.return_value = False
        ignored_symlink.is_file.return_value = True
        symlink_excluded_name: Any = MagicMock()
        symlink_excluded_name.relative_to.return_value = Path("excluded_name")
        symlink_excluded_name.resolve.return_value = root / "included_name.py"
        symlink_excluded_name.is_dir.return_value = False
        symlink_excluded_name.is_file.return_value = True
        symlink_included_name: Any = MagicMock()
        symlink_included_name.relative_to.return_value = Path("included_name.py")
        symlink_included_name.resolve.return_value = root / "excluded_name"
        symlink_included_name.is_dir.return_value = False
        symlink_included_name.is_file.return_value = True
        path_obj: Any = MagicMock()
        path_obj.iterdir.return_value = [
            regular,
            outside_root_symlink,
            ignored_symlink,
            symlink_excluded_name,
            symlink_included_name,
        ]
        files: List[Any] = list(
            black.gen_python_files(
                path_obj.iterdir(),
                root,
                include,
                exclude,
                None,
                None,
                report,
                {path_obj: gitignore},
                verbose=False,
                quiet=False,
            )
        )
        assert files == [regular, symlink_included_name]
        path_obj.iterdir.assert_called_once()
        outside_root_symlink.resolve.assert_called_once()
        ignored_symlink.resolve.assert_not_called()

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
                    expected=["module.py"],
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
                    force_exclude=r"/symlink_module.py",
                    expected=[],
                )
                target: Path = actual_proj / "target"
                target.mkdir()
                (target / "another.py").write_text("print('hello')", encoding="utf-8")
                (symlink_proj / "nested").symlink_to(target)
                assert_collected_sources(
                    src=[symlink_proj / "nested" / "another.py"],
                    root=symlink_proj.resolve(),
                    force_exclude=r"nested",
                    expected=[],
                )
                assert_collected_sources(
                    src=[symlink_proj / "nested" / "another.py"],
                    root=symlink_proj.resolve(),
                    force_exclude=r"target",
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
                root=root,
                expected=[],
                stdin_filename=stdin_filename,
            )

    def test_get_sources_with_stdin(self) -> None:
        src: List[str] = ["-"]
        expected: List[str] = ["-"]
        assert_collected_sources(
            src,
            root=THIS_DIR.resolve(),
            expected=expected,
            include="",
            exclude=r"/exclude/|a\.py",
        )

    def test_get_sources_with_stdin_filename(self) -> None:
        src: List[str] = ["-"]
        stdin_filename: str = str(THIS_DIR / "data/collections.py")
        expected: List[str] = [f"__BLACK_STDIN_FILENAME__{stdin_filename}"]
        assert_collected_sources(
            src,
            root=THIS_DIR.resolve(),
            expected=expected,
            exclude=r"/exclude/a\.py",
            stdin_filename=stdin_filename,
        )

    def test_get_sources_with_stdin_filename_and_exclude(self) -> None:
        path: Path = DATA_DIR / "include_exclude_tests"
        src: List[str] = ["-"]
        stdin_filename: str = str(path / "b/exclude/a.py")
        expected: List[str] = [f"__BLACK_STDIN_FILENAME__{stdin_filename}"]
        assert_collected_sources(
            src,
            root=THIS_DIR.resolve(),
            expected=expected,
            exclude=r"/exclude/|a\.py",
            stdin_filename=stdin_filename,
        )

    def test_get_sources_with_stdin_filename_and_extend_exclude(self) -> None:
        src: List[str] = ["-"]
        path: Path = THIS_DIR / "data" / "include_exclude_tests"
        stdin_filename: str = str(path / "b/exclude/a.py")
        expected: List[str] = [f"__BLACK_STDIN_FILENAME__{stdin_filename}"]
        assert_collected_sources(
            src,
            root=THIS_DIR.resolve(),
            expected=expected,
            extend_exclude=r"/exclude/|a\.py",
            stdin_filename=stdin_filename,
        )

    def test_get_sources_with_stdin_filename_and_force_exclude(self) -> None:
        path: Path = THIS_DIR / "data" / "include_exclude_tests"
        stdin_filename: str = str(path / "b/exclude/a.py")
        assert_collected_sources(
            src=["-"],
            root=THIS_DIR.resolve(),
            expected=[],
            force_exclude=r"/exclude/|a\.py",
            stdin_filename=stdin_filename,
        )

    def test_get_sources_with_stdin_filename_and_force_exclude_and_symlink(self) -> None:
        with TemporaryDirectory() as tempdir:
            tmp: Path = Path(tempdir).resolve()
            (tmp / "exclude").mkdir()
            (tmp / "exclude" / "a.py").write_text("print('hello')", encoding="utf-8")
            (tmp / "symlink.py").symlink_to(tmp / "exclude" / "a.py")
            stdin_filename: str = str(tmp / "symlink.py")
            expected: List[str] = [f"__BLACK_STDIN_FILENAME__{stdin_filename}"]
            with change_directory(tmp):
                assert_collected_sources(
                    src=["-"],
                    root=tmp,
                    expected=expected,
                    force_exclude=r"exclude/a\.py",
                    stdin_filename=stdin_filename,
                )


class TestDeFactoAPI:
    def test_format_str(self) -> None:
        assert black.format_str("print('hello')", mode=black.Mode()) == 'print("hello")\n'
        assert black.format_str("print('hello')", mode=black.Mode(line_length=42)) == 'print("hello")\n'
        with pytest.raises(black.InvalidInput):
            black.format_str("syntax error", mode=black.Mode())

    def test_format_file_contents(self) -> None:
        assert black.format_file_contents("x=1", fast=True, mode=black.Mode()) == "x = 1\n"
        with pytest.raises(black.NothingChanged):
            black.format_file_contents("x = 1\n", fast=True, mode=black.Mode())


class TestASTSafety(BlackBaseTestCase):
    def check_ast_equivalence(self, source: str, dest: str, *, should_fail: bool = False) -> None:
        source = textwrap.dedent(source)
        dest = textwrap.dedent(dest)
        black.parse_ast(source)
        black.parse_ast(dest)
        if should_fail:
            with self.assertRaises(ASTSafetyError):
                black.assert_equivalent(source, dest)
        else:
            black.assert_equivalent(source, dest)

    def test_assert_equivalent_basic(self) -> None:
        self.check_ast_equivalence("{}", "None", should_fail=True)
        self.check_ast_equivalence("1+2", "1    +   2")
        self.check_ast_equivalence("hi # comment", "hi")

    def test_assert_equivalent_del(self) -> None:
        self.check_ast_equivalence("del (a, b)", "del a, b")

    def test_assert_equivalent_strings(self) -> None:
        self.check_ast_equivalence('x = "x"', 'x = " x "', should_fail=True)
        self.check_ast_equivalence(
            '''
            """docstring  """
            ''',
            '''
            """docstring"""
            ''',
        )
        self.check_ast_equivalence(
            '''
            """docstring  """
            ''',
            '''
            """ddocstring"""
            ''',
            should_fail=True,
        )
        self.check_ast_equivalence(
            '''
            class A:
                """

                docstring


                """
            ''',
            '''
            class A:
                """docstring"""
            ''',
        )
        self.check_ast_equivalence(
            """
            def f():
                " docstring  "
            """,
            '''
            def f():
                """docstring"""
            ''',
        )
        self.check_ast_equivalence(
            """
            async def f():
                "   docstring  "
            """,
            '''
            async def f():
                """docstring"""
            ''',
        )
        self.check_ast_equivalence(
            """
            if __name__ == "__main__":
                "  docstring-like  "
            """,
            '''
            if __name__ == "__main__":
                """docstring-like"""
            ''',
        )
        self.check_ast_equivalence(r'def f(): r" \n "', r'def f(): "\\n"')
        self.check_ast_equivalence('try: pass\nexcept: " x "', 'try: pass\nexcept: "x"')
        self.check_ast_equivalence(
            'def foo(): return " x "', 'def foo(): return "x"', should_fail=True
        )

    def test_assert_equivalent_fstring(self) -> None:
        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 12):
            pytest.skip("relies on 3.12+ syntax")
        self.check_ast_equivalence(
            """print(f"{"|".join([a,b,c])}")""",
            """print(f"{" | ".join([a,b,c])}")""",
            should_fail=True,
        )
        self.check_ast_equivalence(
            """print(f"{"|".join(['a','b','c'])}")""",
            """print(f"{" | ".join(['a','b','c'])}")""",
            should_fail=True,
        )

    def test_equivalency_ast_parse_failure_includes_error(self) -> None:
        with pytest.raises(ASTSafetyError) as err:
            black.assert_equivalent("a«»a  = 1", "a«»a  = 1")
        err.match("--safe")
        err.match("invalid character")
        err.match(r"\(<unknown>, line 1\)")


try:
    with open(black.__file__, encoding="utf-8") as _bf:
        black_source_lines: List[str] = _bf.readlines()
except UnicodeDecodeError:
    if not black.COMPILED:
        raise


def tracefunc(
    frame: types.FrameType, event: str, arg: Any
) -> Callable[[types.FrameType, str, Any], Any]:
    if event != "call":
        return tracefunc
    stack: int = len(inspect.stack()) - 19
    stack *= 2
    filename: str = frame.f_code.co_filename
    lineno: int = frame.f_lineno
    func_sig_lineno: int = lineno - 1
    funcname: str = black_source_lines[func_sig_lineno].strip()
    while funcname.startswith("@"):
        func_sig_lineno += 1
        funcname = black_source_lines[func_sig_lineno].strip()
    if "black/__init__.py" in filename:
        print(f"{' ' * stack}{lineno}:{funcname}")
    return tracefunc


def _new_wrapper(
    output: io.StringIO, io_TextIOWrapper: type[io.TextIOWrapper]
) -> Callable[[Any, Any], Union[io.StringIO, io.TextIOWrapper]]:
    def get_output(*args: Any, **kwargs: Any) -> Union[io.StringIO, io.TextIOWrapper]:
        if args == (sys.stdout.buffer,):
            return output
        return io_TextIOWrapper(*args, **kwargs)
    return get_output