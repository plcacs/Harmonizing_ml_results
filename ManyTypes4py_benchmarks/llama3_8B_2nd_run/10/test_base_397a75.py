import io
import json
import os
import sys
import types
from pathlib import Path
import click
import pytest
from faust.cli import AppCommand, Command, call_command
from faust.cli.base import DEFAULT_LOGLEVEL, _Group, _prepare_cli, argument, compat_option, find_app, option
from faust.types._env import CONSOLE_PORT
from mode import Worker
from mode.utils.mocks import AsyncMock, Mock, call, patch

class test_argument:
    def test_repr(self) -> None:
        assert repr(argument(default=1))  # type: ignore

class test_option:
    def test_repr(self) -> None:
        assert repr(option('--foo', '--bar', default=1))  # type: ignore

def test_call_command() -> None:
    with patch('faust.cli.base.cli') as cli:
        cli.side_effect = SystemExit(3)
        exitcode, stdout, stderr = call_command('foo', ['x', 'y'])
        cli.assert_called_once_with(args=['foo', 'x', 'y'], side_effects=False, stdout=stdout, stderr=stderr)
        assert exitcode == 3

def test_call_command__no_exit() -> None:
    with patch('faust.cli.base.cli') as cli:
        exitcode, stdout, stderr = call_command('foo', ['x', 'y'])
        cli.assert_called_once_with(args=['foo', 'x', 'y'], side_effects=False, stdout=stdout, stderr=stderr)
        assert exitcode == 0

def test_call_command__custom_ins() -> None:
    o_out = io.StringIO()
    o_err = io.StringIO()
    with patch('faust.cli.base.cli') as cli:
        exitcode, stdout, stderr = call_command('foo', ['x', 'y'], stdout=o_out, stderr=o_err)
        cli.assert_called_once_with(args=['foo', 'x', 'y'], side_effects=False, stdout=stdout, stderr=stderr)
        assert exitcode == 0
        assert stdout is o_out
        assert stderr is o_err

class test_Group:
    @pytest.fixture()
    def group(self) -> _Group:
        return _Group()

    def test_get_help(self, group: _Group) -> None:
        ctx = group.make_context('prog', ['x'])
        group._maybe_import_app = Mock()
        group.get_help(ctx)
        group._maybe_import_app.assert_called_once_with()

    def test_get_usage(self, group: _Group) -> None:
        ctx = group.make_context('prog', ['x'])
        group._maybe_import_app = Mock()
        group.get_usage(ctx)
        group._maybe_import_app.assert_called_once_with()

class test_Command:
    class TestCommand(Command):
        options = [click.option('--quiet/--no-quiet')]

    @pytest.fixture()
    def ctx(self) -> Mock:
        return Mock(name='ctx')

    @pytest.fixture()
    def command(self, ctx: Mock) -> TestCommand:
        return self.TestCommand(ctx=ctx)

    # ... and so on
