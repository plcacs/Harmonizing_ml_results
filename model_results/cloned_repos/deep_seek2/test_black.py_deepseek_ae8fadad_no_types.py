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
from typing import Any, Optional, TypeVar, Union
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
from tests.util import DATA_DIR, DEFAULT_MODE, DETERMINISTIC_HEADER, PROJECT_ROOT, PY36_VERSIONS, THIS_DIR, BlackBaseTestCase, assert_format, change_directory, dump_to_stderr, ff, fs, get_case_path, read_data, read_data_from_file
THIS_FILE: Path = Path(__file__)
EMPTY_CONFIG: Path = THIS_DIR / 'data' / 'empty_pyproject.toml'
PY36_ARGS: list[str] = [f'--target-version={version.name.lower()}' for version in PY36_VERSIONS]
DEFAULT_EXCLUDE: Pattern[str] = black.re_compile_maybe_verbose(black.const.DEFAULT_EXCLUDES)
DEFAULT_INCLUDE: Pattern[str] = black.re_compile_maybe_verbose(black.const.DEFAULT_INCLUDES)
T = TypeVar('T')
R = TypeVar('R')
DIFF_TIME: Pattern[str] = re.compile('\\t[\\d\\-:+\\. ]+')

@contextmanager
def cache_dir(exists=True):
    with TemporaryDirectory() as workspace:
        cache_dir = Path(workspace)
        if not exists:
            cache_dir = cache_dir / 'new'
        with patch('black.cache.CACHE_DIR', cache_dir):
            yield cache_dir

@contextmanager
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        loop.close()

class FakeContext(click.Context):
    """A fake click Context for when calling functions that need it."""

    def __init__(self):
        self.default_map: dict[str, Any] = {}
        self.params: dict[str, Any] = {}
        self.command: click.Command = black.main
        self.obj: dict[str, Any] = {'root': PROJECT_ROOT}

class FakeParameter(click.Parameter):
    """A fake click Parameter for when calling functions that need it."""

    def __init__(self):
        pass

class BlackRunner(CliRunner):
    """Make sure STDOUT and STDERR are kept separate when testing Black via its CLI."""

    def __init__(self):
        super().__init__(mix_stderr=False)

def invokeBlack(args, exit_code=0, ignore_config=True):
    runner = BlackRunner()
    if ignore_config:
        args = ['--verbose', '--config', str(THIS_DIR / 'empty.toml'), *args]
    result = runner.invoke(black.main, args, catch_exceptions=False)
    assert result.stdout_bytes is not None
    assert result.stderr_bytes is not None
    msg = f'Failed with args: {args}\nstdout: {result.stdout_bytes.decode()!r}\nstderr: {result.stderr_bytes.decode()!r}\nexception: {result.exception}'
    assert result.exit_code == exit_code, msg

class BlackTestCase(BlackBaseTestCase):
    invokeBlack = staticmethod(invokeBlack)

    def test_empty_ff(self):
        expected = ''
        tmp_file = Path(black.dump_to_file())
        try:
            self.assertFalse(ff(tmp_file, write_back=black.WriteBack.YES))
            actual = tmp_file.read_text(encoding='utf-8')
        finally:
            os.unlink(tmp_file)
        self.assertFormatEqual(expected, actual)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_one_empty_line(self):
        for nl in ['\n', '\r\n']:
            source = expected = nl
            assert_format(source, expected)

    def test_one_empty_line_ff(self):
        for nl in ['\n', '\r\n']:
            expected = nl
            tmp_file = Path(black.dump_to_file(nl))
            if system() == 'Windows':
                with open(tmp_file, 'wb') as f:
                    f.write(nl.encode('utf-8'))
            try:
                self.assertFalse(ff(tmp_file, write_back=black.WriteBack.YES))
                with open(tmp_file, 'rb') as f:
                    actual = f.read().decode('utf-8')
            finally:
                os.unlink(tmp_file)
            self.assertFormatEqual(expected, actual)

    def test_piping(self):
        _, source, expected = read_data_from_file(PROJECT_ROOT / 'src/black/__init__.py')
        result = BlackRunner().invoke(black.main, ['-', '--fast', f'--line-length={black.DEFAULT_LINE_LENGTH}', f'--config={EMPTY_CONFIG}'], input=BytesIO(source.encode('utf-8')))
        self.assertEqual(result.exit_code, 0)
        self.assertFormatEqual(expected, result.output)
        if source != result.output:
            black.assert_equivalent(source, result.output)
            black.assert_stable(source, result.output, DEFAULT_MODE)

    def test_piping_diff(self):
        diff_header = re.compile('(STDIN|STDOUT)\\t\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d\\.\\d\\d\\d\\d\\d\\d\\+\\d\\d:\\d\\d')
        source, _ = read_data('cases', 'expression.py')
        expected, _ = read_data('cases', 'expression.diff')
        args = ['-', '--fast', f'--line-length={black.DEFAULT_LINE_LENGTH}', '--diff', f'--config={EMPTY_CONFIG}']
        result = BlackRunner().invoke(black.main, args, input=BytesIO(source.encode('utf-8')))
        self.assertEqual(result.exit_code, 0)
        actual = diff_header.sub(DETERMINISTIC_HEADER, result.output)
        actual = actual.rstrip() + '\n'
        self.assertEqual(expected, actual)

    def test_piping_diff_with_color(self):
        source, _ = read_data('cases', 'expression.py')
        args = ['-', '--fast', f'--line-length={black.DEFAULT_LINE_LENGTH}', '--diff', '--color', f'--config={EMPTY_CONFIG}']
        result = BlackRunner().invoke(black.main, args, input=BytesIO(source.encode('utf-8')))
        actual = result.output
        self.assertIn('\x1b[1m', actual)
        self.assertIn('\x1b[36m', actual)
        self.assertIn('\x1b[32m', actual)
        self.assertIn('\x1b[31m', actual)
        self.assertIn('\x1b[0m', actual)

    def test_pep_572_version_detection(self):
        source, _ = read_data('cases', 'pep_572')
        root = black.lib2to3_parse(source)
        features = black.get_features_used(root)
        self.assertIn(black.Feature.ASSIGNMENT_EXPRESSIONS, features)
        versions = black.detect_target_versions(root)
        self.assertIn(black.TargetVersion.PY38, versions)

    def test_pep_695_version_detection(self):
        for file in ('type_aliases', 'type_params'):
            source, _ = read_data('cases', file)
            root = black.lib2to3_parse(source)
            features = black.get_features_used(root)
            self.assertIn(black.Feature.TYPE_PARAMS, features)
            versions = black.detect_target_versions(root)
            self.assertIn(black.TargetVersion.PY312, versions)

    def test_pep_696_version_detection(self):
        source, _ = read_data('cases', 'type_param_defaults')
        samples = [source, 'type X[T=int] = float', 'type X[T:int=int]=int', 'type X[*Ts=int]=int', 'type X[*Ts=*int]=int', 'type X[**P=int]=int']
        for sample in samples:
            root = black.lib2to3_parse(sample)
            features = black.get_features_used(root)
            self.assertIn(black.Feature.TYPE_PARAM_DEFAULTS, features)

    def test_expression_ff(self):
        source, expected = read_data('cases', 'expression.py')
        tmp_file = Path(black.dump_to_file(source))
        try:
            self.assertTrue(ff(tmp_file, write_back=black.WriteBack.YES))
            actual = tmp_file.read_text(encoding='utf-8')
        finally:
            os.unlink(tmp_file)
        self.assertFormatEqual(expected, actual)
        with patch('black.dump_to_file', dump_to_stderr):
            black.assert_equivalent(source, actual)
            black.assert_stable(source, actual, DEFAULT_MODE)

    def test_expression_diff(self):
        source, _ = read_data('cases', 'expression.py')
        expected, _ = read_data('cases', 'expression.diff')
        tmp_file = Path(black.dump_to_file(source))
        diff_header = re.compile(f'{re.escape(str(tmp_file))}\\t\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d\\.\\d\\d\\d\\d\\d\\d\\+\\d\\d:\\d\\d')
        try:
            result = BlackRunner().invoke(black.main, ['--diff', str(tmp_file), f'--config={EMPTY_CONFIG}'])
            self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmp_file)
        actual = result.output
        actual = diff_header.sub(DETERMINISTIC_HEADER, actual)
        if expected != actual:
            dump = black.dump_to_file(actual)
            msg = f"Expected diff isn't equal to the actual. If you made changes to expression.py and this is an anticipated difference, overwrite tests/data/cases/expression.diff with {dump}"
            self.assertEqual(expected, actual, msg)

    def test_expression_diff_with_color(self):
        source, _ = read_data('cases', 'expression.py')
        expected, _ = read_data('cases', 'expression.diff')
        tmp_file = Path(black.dump_to_file(source))
        try:
            result = BlackRunner().invoke(black.main, ['--diff', '--color', str(tmp_file), f'--config={EMPTY_CONFIG}'])
        finally:
            os.unlink(tmp_file)
        actual = result.output
        self.assertIn('\x1b[1m', actual)
        self.assertIn('\x1b[36m', actual)
        self.assertIn('\x1b[32m', actual)
        self.assertIn('\x1b[31m', actual)
        self.assertIn('\x1b[0m', actual)

    def test_detect_pos_only_arguments(self):
        source, _ = read_data('cases', 'pep_570')
        root = black.lib2to3_parse(source)
        features = black.get_features_used(root)
        self.assertIn(black.Feature.POS_ONLY_ARGUMENTS, features)
        versions = black.detect_target_versions(root)
        self.assertIn(black.TargetVersion.PY38, versions)

    def test_detect_debug_f_strings(self):
        root = black.lib2to3_parse('f"{x=}" ')
        features = black.get_features_used(root)
        self.assertIn(black.Feature.DEBUG_F_STRINGS, features)
        versions = black.detect_target_versions(root)
        self.assertIn(black.TargetVersion.PY38, versions)
        root = black.lib2to3_parse('f"{x}"\nf\'{"="}\'\nf\'{(x:=5)}\'\nf\'{f(a="3=")}\'\nf\'{x:=10}\'\n')
        features = black.get_features_used(root)
        self.assertNotIn(black.Feature.DEBUG_F_STRINGS, features)
        root = black.lib2to3_parse('f"heard a rumour that { f\'{1+1=}\' } ... seems like it could be true" ')
        features = black.get_features_used(root)
        self.assertIn(black.Feature.DEBUG_F_STRINGS, features)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_string_quotes(self):
        source, expected = read_data('miscellaneous', 'string_quotes')
        mode = black.Mode(unstable=True)
        assert_format(source, expected, mode)
        mode = replace(mode, string_normalization=False)
        not_normalized = fs(source, mode=mode)
        self.assertFormatEqual(source.replace('\\\n', ''), not_normalized)
        black.assert_equivalent(source, not_normalized)
        black.assert_stable(source, not_normalized, mode=mode)

    def test_skip_source_first_line(self):
        source, _ = read_data('miscellaneous', 'invalid_header')
        tmp_file = Path(black.dump_to_file(source))
        self.invokeBlack([str(tmp_file), '--diff', '--check'], exit_code=123)
        result = BlackRunner().invoke(black.main, [str(tmp_file), '-x', f'--config={EMPTY_CONFIG}'])
        self.assertEqual(result.exit_code, 0)
        actual = tmp_file.read_text(encoding='utf-8')
        self.assertFormatEqual(source, actual)

    def test_skip_source_first_line_when_mixing_newlines(self):
        code_mixing_newlines = b'Header will be skipped\r\ni = [1,2,3]\nj = [1,2,3]\n'
        expected = b'Header will be skipped\r\ni = [1, 2, 3]\nj = [1, 2, 3]\n'
        with TemporaryDirectory() as workspace:
            test_file = Path(workspace) / 'skip_header.py'
            test_file.write_bytes(code_mixing_newlines)
            mode = replace(DEFAULT_MODE, skip_source_first_line=True)
            ff(test_file, mode=mode, write_back=black.WriteBack.YES)
            self.assertEqual(test_file.read_bytes(), expected)

    def test_skip_magic_trailing_comma(self):
        source, _ = read_data('cases', 'expression')
        expected, _ = read_data('miscellaneous', 'expression_skip_magic_trailing_comma.diff')
        tmp_file = Path(black.dump_to_file(source))
        diff_header = re.compile(f'{re.escape(str(tmp_file))}\\t\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d\\.\\d\\d\\d\\d\\d\\d\\+\\d\\d:\\d\\d')
        try:
            result = BlackRunner().invoke(black.main, ['-C', '--diff', str(tmp_file), f'--config={EMPTY_CONFIG}'])
            self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmp_file)
        actual = result.output
        actual = diff_header.sub(DETERMINISTIC_HEADER, actual)
        actual = actual.rstrip() + '\n'
        if expected != actual:
            dump = black.dump_to_file(actual)
            msg = f"Expected diff isn't equal to the actual. If you made changes to expression.py and this is an anticipated difference, overwrite tests/data/miscellaneous/expression_skip_magic_trailing_comma.diff with {dump}"
            self.assertEqual(expected, actual, msg)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_async_as_identifier(self):
        source_path = get_case_path('miscellaneous', 'async_as_identifier')
        _, source, expected = read_data_from_file(source_path)
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        major, minor = sys.version_info[:2]
        if major < 3 or (major <= 3 and minor < 7):
            black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)
        self.invokeBlack([str(source_path), '--target-version', 'py36'])
        self.invokeBlack([str(source_path), '--target-version', 'py37'], exit_code=123)

    @patch('black.dump_to_file', dump_to_stderr)
    def test_python37(self):
        source_path = get_case_path('cases', 'python37')
        _, source, expected = read_data_from_file(source_path)
        actual = fs(source)
        self.assertFormatEqual(expected, actual)
        major, minor = sys.version_info[:2]
        if major > 3 or (major == 3 and minor >= 7):
            black.assert_equivalent(source, actual)
        black.assert_stable(source, actual, DEFAULT_MODE)
        self.invokeBlack([str(source_path), '--target-version', 'py37'])
        self.invokeBlack([str(source_path), '--target-version', 'py36'], exit_code=123)

    def test_tab_comment_indentation(self):
        contents_tab = 'if 1:\n\tif 2:\n\t\tpass\n\t# comment\n\tpass\n'
        contents_spc = 'if 1:\n    if 2:\n        pass\n    # comment\n    pass\n'
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))
        contents_tab = 'if 1:\n\tif 2:\n\t\tpass\n\t\t# comment\n\tpass\n'
        contents_spc = 'if 1:\n    if 2:\n        pass\n        # comment\n    pass\n'
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))
        contents_tab = 'if 1:\n        if 2:\n\t\tpass\n\t# comment\n        pass\n'
        contents_spc = 'if 1:\n    if 2:\n        pass\n    # comment\n    pass\n'
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))
        contents_tab = 'if 1:\n        if 2:\n\t\tpass\n\t\t# comment\n        pass\n'
        contents_spc = 'if 1:\n    if 2:\n        pass\n        # comment\n    pass\n'
        self.assertFormatEqual(contents_spc, fs(contents_spc))
        self.assertFormatEqual(contents_spc, fs(contents_tab))

    def test_false_positive_symlink_output_issue_3384(self):
        project_root = Path(THIS_DIR / 'data' / 'nested_gitignore_tests')
        working_directory = project_root / 'root'
        with change_directory(working_directory):
            report = MagicMock(verbose=True)
            black.get_sources(root=project_root, src=('./child',), quiet=False, verbose=True, include=DEFAULT_INCLUDE, exclude=None, report=report, extend_exclude=None, force_exclude=None, stdin_filename=None)
        assert not any((mock_args[1].startswith('is a symbolic link that points outside') for _, mock_args, _ in report.path_ignored.mock_calls)), 'A symbolic link was reported.'
        report.path_ignored.assert_called_once_with(Path(working_directory, 'child', 'b.py'), 'matches a .gitignore file content')

    def test_report_verbose(self):
        report = Report(verbose=True)
        out_lines = []
        err_lines = []

        def out(msg, **kwargs: Any):
            out_lines.append(msg)

        def err(msg, **kwargs: Any):
            err_lines.append(msg)
        with patch('black.output._out', out), patch('black.output._err', err):
            report.done(Path('f1'), black.Changed.NO)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], 'f1 already well formatted, good job.')
            self.assertEqual(unstyle(str(report)), '1 file left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.done(Path('f2'), black.Changed.YES)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], 'reformatted f2')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 1 file left unchanged.')
            report.done(Path('f3'), black.Changed.CACHED)
            self.assertEqual(len(out_lines), 3)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], "f3 wasn't modified on disk since last run.")
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path('e1'), 'boom')
            self.assertEqual(len(out_lines), 3)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[-1], 'error: cannot format e1: boom')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f3'), black.Changed.YES)
            self.assertEqual(len(out_lines), 4)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(out_lines[-1], 'reformatted f3')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.failed(Path('e2'), 'boom')
            self.assertEqual(len(out_lines), 4)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[-1], 'error: cannot format e2: boom')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path('wat'), 'no match')
            self.assertEqual(len(out_lines), 5)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(out_lines[-1], 'wat ignored: no match')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f4'), black.Changed.NO)
            self.assertEqual(len(out_lines), 6)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(out_lines[-1], 'f4 already well formatted, good job.')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 3 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')
            report.check = False
            report.diff = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')

    def test_report_quiet(self):
        report = Report(quiet=True)
        out_lines = []
        err_lines = []

        def out(msg, **kwargs: Any):
            out_lines.append(msg)

        def err(msg, **kwargs: Any):
            err_lines.append(msg)
        with patch('black.output._out', out), patch('black.output._err', err):
            report.done(Path('f1'), black.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), '1 file left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.done(Path('f2'), black.Changed.YES)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 1 file left unchanged.')
            report.done(Path('f3'), black.Changed.CACHED)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path('e1'), 'boom')
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[-1], 'error: cannot format e1: boom')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f3'), black.Changed.YES)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.failed(Path('e2'), 'boom')
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[-1], 'error: cannot format e2: boom')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path('wat'), 'no match')
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f4'), black.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 3 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')
            report.check = False
            report.diff = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')

    def test_report_normal(self):
        report = black.Report()
        out_lines = []
        err_lines = []

        def out(msg, **kwargs: Any):
            out_lines.append(msg)

        def err(msg, **kwargs: Any):
            err_lines.append(msg)
        with patch('black.output._out', out), patch('black.output._err', err):
            report.done(Path('f1'), black.Changed.NO)
            self.assertEqual(len(out_lines), 0)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(unstyle(str(report)), '1 file left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.done(Path('f2'), black.Changed.YES)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], 'reformatted f2')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 1 file left unchanged.')
            report.done(Path('f3'), black.Changed.CACHED)
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 0)
            self.assertEqual(out_lines[-1], 'reformatted f2')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged.')
            self.assertEqual(report.return_code, 0)
            report.check = True
            self.assertEqual(report.return_code, 1)
            report.check = False
            report.failed(Path('e1'), 'boom')
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(err_lines[-1], 'error: cannot format e1: boom')
            self.assertEqual(unstyle(str(report)), '1 file reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f3'), black.Changed.YES)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 1)
            self.assertEqual(out_lines[-1], 'reformatted f3')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 1 file failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.failed(Path('e2'), 'boom')
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(err_lines[-1], 'error: cannot format e2: boom')
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.path_ignored(Path('wat'), 'no match')
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 2 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.done(Path('f4'), black.Changed.NO)
            self.assertEqual(len(out_lines), 2)
            self.assertEqual(len(err_lines), 2)
            self.assertEqual(unstyle(str(report)), '2 files reformatted, 3 files left unchanged, 2 files failed to reformat.')
            self.assertEqual(report.return_code, 123)
            report.check = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')
            report.check = False
            report.diff = True
            self.assertEqual(unstyle(str(report)), '2 files would be reformatted, 3 files would be left unchanged, 2 files would fail to reformat.')

    def test_lib2to3_parse(self):
        with self.assertRaises(black.InvalidInput):
            black.lib2to3_parse('invalid syntax')
        straddling = 'x + y'
        black.lib2to3_parse(straddling)
        black.lib2to3_parse(straddling, {TargetVersion.PY36})
        py2_only = 'print x'
        with self.assertRaises(black.InvalidInput):
            black.lib2to3_parse(py2_only, {TargetVersion.PY36})
        py3_only = 'exec(x, end=y)'
        black.lib2to3_parse