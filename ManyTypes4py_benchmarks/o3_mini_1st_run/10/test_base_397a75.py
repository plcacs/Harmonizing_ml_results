#!/usr/bin/env python3
from __future__ import annotations
import io
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import click
import pytest
from faust.cli import AppCommand, Command, call_command
from faust.cli.base import DEFAULT_LOGLEVEL, _Group, _prepare_cli, argument, compat_option, find_app, option
from faust.types._env import CONSOLE_PORT
from mode import Worker
from mode.utils.mocks import AsyncMock, Mock, call, patch

class test_argument:
    def test_repr(self) -> None:
        assert repr(argument(default=1))


class test_option:
    def test_repr(self) -> None:
        assert repr(option('--foo', '--bar', default=1))


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
    o_out: io.StringIO = io.StringIO()
    o_err: io.StringIO = io.StringIO()
    with patch('faust.cli.base.cli') as cli:
        exitcode, stdout, stderr = call_command('foo', ['x', 'y'], stdout=o_out, stderr=o_err)
        cli.assert_called_once_with(args=['foo', 'x', 'y'], side_effects=False, stdout=stdout, stderr=stderr)
        assert exitcode == 0
        assert stdout is o_out
        assert stderr is o_err


def test_compat_option() -> None:
    opt = compat_option('--foo', default=1, state_key='foo')
    ctx: Mock[Any] = Mock(name='ctx')
    param: Mock[Any] = Mock(name='param')
    state: Any = ctx.ensure_object.return_value
    state.foo = 33
    # The following prints for debugging purposes.
    print(dir(opt(ctx)))
    # Test callback with None value (should default to state's foo)
    assert opt(ctx)._callback(ctx, param, None) == 33
    # When value is provided, should override state's foo.
    assert opt(ctx)._callback(ctx, param, 44) == 44
    state.foo = None
    assert opt(ctx)._callback(ctx, param, 44) == 44


def test_find_app() -> None:
    imp: Mock[Any] = Mock(name='imp')
    symbol_by_name: Mock[Any] = Mock(name='symbol_by_name')
    with patch('faust.cli.base.prepare_app') as prepare_app:
        res: Any = find_app('foo', imp=imp, symbol_by_name=symbol_by_name)
        assert res is prepare_app.return_value
        symbol_by_name.assert_called_once_with('foo', imp=imp)
        prepare_app.assert_called_once_with(symbol_by_name.return_value, 'foo')


def test_find_app__attribute_error() -> None:
    imp: Mock[Any] = Mock(name='imp')
    symbol_by_name: Mock[Any] = Mock(name='symbol_by_name')
    symbol_by_name.side_effect = AttributeError()
    with patch('faust.cli.base.prepare_app') as prepare_app:
        res: Any = find_app('foo', imp=imp, symbol_by_name=symbol_by_name)
        assert res is prepare_app.return_value
        symbol_by_name.assert_called_once_with('foo', imp=imp)
        imp.assert_called_once_with('foo')
        prepare_app.assert_called_once_with(imp.return_value, 'foo')


def test_find_app__app_is_module() -> None:
    imp: Mock[Any] = Mock(name='imp')
    symbol_by_name: Mock[Any] = Mock(name='symbol_by_name')
    symbol_by_name.side_effect = AttributeError()
    module_app = types.ModuleType('foo')
    module_app.app = types.ModuleType('foo.app')
    imp.return_value = module_app
    with pytest.raises(AttributeError):
        find_app('foo', imp=imp, symbol_by_name=symbol_by_name)


def test_find_app__app_is_module_but_has_app() -> None:
    imp: Mock[Any] = Mock(name='imp')
    symbol_by_name: Mock[Any] = Mock(name='symbol_by_name')
    symbol_by_name.side_effect = AttributeError()
    module_app = types.ModuleType('foo')
    module_app.app = Mock(name='app')
    imp.return_value = module_app
    with patch('faust.cli.base.prepare_app') as prepare_app:
        find_app('foo', imp=imp, symbol_by_name=symbol_by_name)
        prepare_app.assert_called_once_with(imp.return_value.app, 'foo')


class test_Group:

    @pytest.fixture()
    def group(self) -> _Group:
        return _Group()

    def test_get_help(self, *, group: _Group) -> None:
        ctx: click.Context = group.make_context('prog', ['x'])
        group._maybe_import_app = Mock()
        group.get_help(ctx)
        group._maybe_import_app.assert_called_once_with()

    def test_get_usage(self, *, group: _Group) -> None:
        ctx: click.Context = group.make_context('prog', ['x'])
        group._maybe_import_app = Mock()
        group.get_usage(ctx)
        group._maybe_import_app.assert_called_once_with()

    @pytest.mark.parametrize('argv,expected_chdir,expected_app', [
        (['--foo', '-W', '/foo'], '/foo', None),
        (['--foo', '--workdir', '/foo'], '/foo', None),
        (['--foo', '--workdir=/foo'], '/foo', None),
        (['--foo', '-A', 'foo'], None, 'foo'),
        (['--foo', '--app', 'foo'], None, 'foo'),
        (['--foo', '--app=foo'], None, 'foo'),
        (['--foo', '--app=foo', '--workdir', '/foo'], '/foo', 'foo'),
        ([], None, None)
    ])
    def test_maybe_import_app(self, argv: List[str], expected_chdir: Optional[str], expected_app: Optional[str], *, group: _Group) -> None:
        with patch('os.chdir') as chdir:
            with patch('faust.cli.base.find_app') as find_app:
                group._maybe_import_app(argv)
                if expected_chdir:
                    chdir.assert_called_once_with(Path(expected_chdir).absolute())
                else:
                    chdir.assert_not_called()
                if expected_app:
                    find_app.assert_called_once_with(expected_app)
                else:
                    find_app.assert_not_called()

    def test_maybe_import_app__missing_argument(self, *, group: _Group) -> None:
        with pytest.raises(click.UsageError):
            group._maybe_import_app(['--foo', '-A'])
        with pytest.raises(click.UsageError):
            group._maybe_import_app(['--foo', '--app'])


def test__prepare_cli() -> None:
    ctx: Mock[Any] = Mock(name='context')
    # Ensure state is set on the context object's ensure_object.
    state: Any = ctx.ensure_object.return_value = Mock(name='state')
    root: Any = ctx.find_root.return_value = Mock(name='root')
    root.side_effects = False
    app: Any = Mock(name='app')
    try:
        _prepare_cli(ctx, app=app, quiet=False, debug=True, workdir='/foo', datadir='/data', json=True, no_color=False, loop='foo')
        assert state.app is app
        assert not state.quiet
        assert state.debug
        assert state.workdir == '/foo'
        assert state.datadir == '/data'
        assert state.json
        assert not state.no_color
        assert state.loop == 'foo'
        root.side_effects = True
        with patch('os.chdir') as chdir:
            with patch('faust.cli.base.enable_all_colors') as eac:
                with patch('faust.utils.terminal.isatty') as isatty:
                    with patch('faust.cli.base.disable_all_colors') as dac:
                        isatty.return_value = True
                        _prepare_cli(ctx, app=app, quiet=False, debug=True, workdir='/foo', datadir='/data', json=False, no_color=False, loop='foo')
                        chdir.assert_called_with(Path('/foo').absolute())
                        eac.assert_called_once_with()
                        dac.assert_not_called()
                        _prepare_cli(ctx, app=app, quiet=False, debug=True, workdir='/foo', datadir='/data', json=True, no_color=False, loop='foo')
                        dac.assert_called_once_with()
                        dac.reset_mock()
                        eac.reset_mock()
                        _prepare_cli(ctx, app=app, quiet=False, debug=True, workdir='/foo', datadir='/data', json=False, no_color=True, loop='foo')
                        eac.assert_not_called()
                        dac.assert_called_once_with()
                        _prepare_cli(ctx, app=app, quiet=False, debug=True, workdir=None, datadir=None, json=False, no_color=True, loop='foo')
    finally:
        os.environ.pop('F_DATADIR', None)
        os.environ.pop('F_WORKDIR', None)


class test_Command:
    class TestCommand(Command):
        options = [click.option('--quiet/--no-quiet')]

    @pytest.fixture()
    def ctx(self) -> Mock[Any]:
        return Mock(name='ctx')

    @pytest.fixture()
    def command(self, *, ctx: Mock[Any]) -> test_Command.TestCommand:
        return self.TestCommand(ctx=ctx)

    @pytest.mark.asyncio
    async def test_run(self, *, command: test_Command.TestCommand) -> None:
        await command.run()

    @pytest.mark.asyncio
    async def test_execute(self, *, command: test_Command.TestCommand) -> None:
        command.run = AsyncMock()
        command.on_stop = AsyncMock()
        await command.execute(2, kw=3)
        command.run.assert_called_once_with(2, kw=3)
        command.on_stop.assert_called_once_with()

    def test_parse(self, *, command: test_Command.TestCommand) -> None:
        with pytest.raises(click.UsageError):
            _ = command.parse(['foo', '--quiet'])['quiet']

    def test__parse(self, *, command: test_Command.TestCommand) -> None:
        assert command._parse

    @pytest.mark.asyncio
    async def test_on_stop(self, *, command: test_Command.TestCommand) -> None:
        await command.on_stop()

    def test__call__(self, *, command: test_Command.TestCommand) -> None:
        command.run_using_worker = Mock()
        command(2, kw=2)
        command.run_using_worker.assert_called_once_with(2, kw=2)

    def test_run_using_worker(self, *, command: test_Command.TestCommand) -> None:
        command.args = (2,)
        command.kwargs = {'kw': 1, 'yw': 6}
        command.as_service = Mock()
        command.worker_for_service = Mock()
        worker_mock: Any = command.worker_for_service.return_value
        worker_mock.execute_from_commandline.side_effect = KeyError()
        with patch('asyncio.get_event_loop') as get_event_loop:
            with pytest.raises(KeyError):
                command.run_using_worker(1, kw=2)
                command.as_service.assert_called_once_with(get_event_loop(), 2, 1, kw=2, yw=6)
                command.worker_for_service.assert_called_once_with(command.worker_for_service.return_value, get_event_loop.return_value)
                worker_mock.execute_from_commandline.assert_called_once_with()

    def test_on_worker_created(self, *, command: test_Command.TestCommand) -> None:
        assert command.on_worker_created(Mock(name='worker')) is None

    def test_as_service(self, *, command: test_Command.TestCommand) -> None:
        loop: Any = Mock()
        command.execute = Mock()
        with patch('faust.cli.base.Service') as Service:
            res: Any = command.as_service(loop, 1, kw=2)
            assert res is Service.from_awaitable.return_value
            Service.from_awaitable.assert_called_once_with(command.execute.return_value, name=type(command).__name__, loop=loop)
            command.execute.assert_called_once_with(1, kw=2)

    def test_worker_for_service(self, *, command: test_Command.TestCommand) -> None:
        with patch('faust.cli.base.Worker') as WorkerCls:
            service: Any = Mock(name='service')
            loop: Any = Mock(name='loop')
            res: Any = command.worker_for_service(service, loop)
            assert res is WorkerCls.return_value
            WorkerCls.assert_called_once_with(
                service,
                debug=command.debug,
                quiet=command.quiet,
                stdout=command.stdout,
                stderr=command.stderr,
                loglevel=command.loglevel,
                logfile=command.logfile,
                blocking_timeout=command.blocking_timeout,
                console_port=command.console_port,
                redirect_stdouts=command.redirect_stdouts or False,
                redirect_stdouts_level=command.redirect_stdouts_level,
                loop=loop,
                daemon=command.daemon,
            )

    def test__Worker(self, *, command: test_Command.TestCommand) -> None:
        assert command._Worker is Worker

    def test_tabulate__when_text(self, *, command: test_Command.TestCommand) -> None:
        command.json = False
        data: List[List[str]] = [['A', 'B', 'C'], ['D', 'E', 'F']]
        headers: Optional[List[str]] = ['a', 'b', 'c']
        text_result1 = command.tabulate(data, headers=headers)
        text_result2 = command.tabulate(data, headers=None)
        text_result3 = command.tabulate(data, headers=None, wrap_last_row=False)
        assert text_result1
        assert text_result2
        assert text_result3

    def test_tabulate__when_json(self, *, command: test_Command.TestCommand) -> None:
        command.json = True
        data: List[List[str]] = [['A', 'B', 'C'], ['D', 'E', 'F']]
        headers: Optional[List[str]] = ['a', 'b', 'c']
        json_result: Any = command.tabulate(data, headers=headers)
        assert json.loads(json_result) == [{'a': 'A', 'b': 'B', 'c': 'C'}, {'a': 'D', 'b': 'E', 'c': 'F'}]

    def test_tabulate_json(self, *, command: test_Command.TestCommand) -> None:
        data: List[List[str]] = [['A', 'B', 'C'], ['D', 'E', 'F']]
        json_result: Any = command._tabulate_json(data, headers=None)
        assert json.loads(json_result) == data

    def test_tabulate_json__headers(self, *, command: test_Command.TestCommand) -> None:
        data: List[List[str]] = [['A', 'B', 'C'], ['D', 'E', 'F']]
        headers: List[str] = ['a', 'b', 'c']
        json_result: Any = command._tabulate_json(data, headers=headers)
        assert json.loads(json_result) == [{'a': 'A', 'b': 'B', 'c': 'C'}, {'a': 'D', 'b': 'E', 'c': 'F'}]

    def test_table(self, *, command: test_Command.TestCommand) -> None:
        with patch('faust.utils.terminal.table') as table:
            data: List[List[str]] = [['A', 'B', 'C']]
            t: Any = command.table(data, title='foo')
            assert t is table.return_value
            table.assert_called_once_with(data, title='foo', target=sys.stdout)

    def test_color(self, *, command: test_Command.TestCommand) -> Any:
        return command.color('blue', 'text')

    def test_dark(self, *, command: test_Command.TestCommand) -> Any:
        return command.dark('text')

    def test_bold(self, *, command: test_Command.TestCommand) -> Any:
        return command.bold('text')

    def test_bold_tail(self, *, command: test_Command.TestCommand) -> Any:
        return command.bold_tail('foo.bar.baz')

    def test_table_wrap(self, *, command: test_Command.TestCommand) -> None:
        table_obj: Any = command.table([['A', 'B', 'C']])
        assert command._table_wrap(table_obj, 'fooawqe' * 100)

    def test_say(self, *, command: test_Command.TestCommand) -> None:
        with patch('faust.cli.base.echo') as echo:
            command.quiet = True
            command.say('foo')
            echo.assert_not_called()
            command.quiet = False
            command.say('foo')
            echo.assert_called_once_with('foo', file=command.stdout, err=command.stderr)

    def test_carp(self, *, command: test_Command.TestCommand) -> None:
        command.say = Mock()
        command.debug = False
        command.carp('foo')
        command.say.assert_not_called()
        command.debug = True
        command.carp('foo')
        command.say.assert_called_once()

    def test_dumps(self, *, command: test_Command.TestCommand) -> Any:
        with patch('faust.utils.json.dumps') as dumps:
            obj: Any = Mock(name='obj')
            result: Any = command.dumps(obj)
            dumps.assert_called_once_with(obj)
            return result

    def test_loglevel(self, *, command: test_Command.TestCommand, ctx: Mock[Any]) -> None:
        assert command.loglevel == ctx.ensure_object().loglevel
        command.loglevel = 'FOO'
        assert command.loglevel == 'FOO'
        command.loglevel = None
        assert command.loglevel == DEFAULT_LOGLEVEL

    def test_blocking_timeout(self, *, command: test_Command.TestCommand, ctx: Mock[Any]) -> None:
        assert command.blocking_timeout == ctx.ensure_object().blocking_timeout
        command.blocking_timeout = 32.41
        assert command.blocking_timeout == 32.41
        command.blocking_timeout = None
        assert command.blocking_timeout == 0.0

    def test_console_port(self, *, command: test_Command.TestCommand, ctx: Mock[Any]) -> None:
        assert command.console_port == ctx.ensure_object().console_port
        command.console_port = 3241
        assert command.console_port == 3241
        command.console_port = None
        assert command.console_port == CONSOLE_PORT


class test_AppCommand:
    @pytest.fixture()
    def ctx(self) -> Mock[Any]:
        return Mock(name='ctx')

    @pytest.fixture()
    def command(self, *, app: Any, ctx: Mock[Any]) -> AppCommand:
        return AppCommand(app=app, ctx=ctx)

    def test_finalize_app__str(self, *, command: AppCommand) -> None:
        command._app_from_str = Mock()
        command.state.app = 'foo'
        res: Any = command._finalize_app(None)
        assert res is command._app_from_str.return_value
        command._app_from_str.assert_called_once_with('foo')

    def test_finalize_app__concrete(self, *, command: AppCommand, app: Any) -> None:
        command._finalize_concrete_app = Mock()
        res: Any = command._finalize_app(app)
        assert res is command._finalize_concrete_app.return_value
        command._finalize_concrete_app.assert_called_once_with(app)

    def test_blocking_timeout(self, *, command: AppCommand, ctx: Mock[Any]) -> None:
        assert command.blocking_timeout == ctx.ensure_object().blocking_timeout
        command.blocking_timeout = 32.41
        assert command.blocking_timeout == 32.41
        command.blocking_timeout = None
        assert command.blocking_timeout == command.app.conf.blocking_timeout

    def test_app_from_str(self, *, command: AppCommand) -> None:
        with patch('faust.cli.base.find_app') as find_app:
            res: Any = command._app_from_str('foo')
            assert res is find_app.return_value
            find_app.assert_called_once_with('foo')

    def test_app_from_str__empty(self, *, command: AppCommand) -> None:
        command.require_app = False
        assert command._app_from_str(None) is None
        command.require_app = True
        with pytest.raises(command.UsageError):
            command._app_from_str(None)

    def test_finalize_concrete_app(self, *, command: AppCommand, app: Any) -> None:
        with patch('sys.argv', []):
            command._finalize_concrete_app(app)

    @pytest.mark.asyncio
    async def test_on_stop(self, *, command: AppCommand, ctx: Mock[Any]) -> None:
        app: Any = ctx.find_root().app
        app._producer = None
        app._http_client = None
        app.started = False
        await command.on_stop()
        app._producer = Mock(stop=AsyncMock())
        app._maybe_close_http_client = AsyncMock()
        app._http_client = Mock()
        app.stop = AsyncMock()
        app.started = True
        await command.on_stop()
        app._producer.stop.assert_called_once_with()
        app.stop.assert_called_once_with()
        app._maybe_close_http_client.assert_called_once_with()

    def test_to_key(self, *, command: AppCommand) -> Any:
        command.to_model = Mock()
        res: Any = command.to_key('typ', 'key')
        assert res is command.to_model.return_value
        command.to_model.assert_called_once_with('typ', 'key', command.key_serializer)
        return res

    def test_to_value(self, *, command: AppCommand) -> Any:
        command.to_model = Mock()
        res: Any = command.to_value('typ', 'value')
        assert res is command.to_model.return_value
        command.to_model.assert_called_once_with('typ', 'value', command.value_serializer)
        return res

    def test_to_model(self, *, command: AppCommand) -> Any:
        command.import_relative_to_app = Mock()
        res: Any = command.to_model('typ', 'value', 'json')
        model: Any = command.import_relative_to_app.return_value
        assert res is model.loads.return_value
        model.loads.assert_called_once_with(b'value', serializer='json')
        return res

    def test_to_model__bytes(self, *, command: AppCommand) -> bytes:
        result: bytes = command.to_model(None, 'value', 'json')
        return result

    def test_import_relative_to_app(self, *, command: AppCommand, ctx: Mock[Any]) -> Any:
        with patch('faust.cli.base.symbol_by_name') as symbol_by_name:
            res: Any = command.import_relative_to_app('foo')
            assert res is symbol_by_name.return_value
            symbol_by_name.assert_called_once_with('foo')
            return res

    def test_import_relative_to_app__no_origin(self, *, command: AppCommand, ctx: Mock[Any]) -> None:
        app: Any = ctx.find_root().app
        app.conf.origin = None
        with patch('faust.cli.base.symbol_by_name') as symbol_by_name:
            symbol_by_name.side_effect = ImportError()
            with pytest.raises(ImportError):
                command.import_relative_to_app('foo')

    def test_import_relative_to_app__with_origin(self, *, command: AppCommand, ctx: Mock[Any]) -> None:
        app: Any = ctx.find_root().app
        app.conf.origin = 'root.moo:bar'
        with patch('faust.cli.base.symbol_by_name') as symbol_by_name:
            symbol_by_name.side_effect = ImportError()
            with pytest.raises(ImportError):
                command.import_relative_to_app('foo')
            symbol_by_name.assert_has_calls([call('foo'), call('root.moo.models.foo'), call('root.moo.foo')])

    def test_import_relative_to_app__with_origin_l1(self, *, command: AppCommand, ctx: Mock[Any]) -> None:
        app: Any = ctx.find_root().app
        app.conf.origin = 'root.moo:bar'
        with patch('faust.cli.base.symbol_by_name') as symbol_by_name:
            symbol_by_name.side_effect = [ImportError(), ImportError(), 'x']
            ret: Any = command.import_relative_to_app('foo')
            assert ret == 'x'
            symbol_by_name.assert_has_calls([call('foo'), call('root.moo.models.foo'), call('root.moo.foo')])

    def test_import_relative_to_app__with_origin_l2(self, *, command: AppCommand, ctx: Mock[Any]) -> None:
        app: Any = ctx.find_root().app
        app.conf.origin = 'root.moo:bar'
        with patch('faust.cli.base.symbol_by_name') as symbol_by_name:
            symbol_by_name.side_effect = [ImportError(), 'x']
            ret: Any = command.import_relative_to_app('foo')
            assert ret == 'x'
            symbol_by_name.assert_has_calls([call('foo'), call('root.moo.models.foo')])

    def test_to_topic__missing(self, *, command: AppCommand) -> None:
        with pytest.raises(command.UsageError):
            command.to_topic(None)

    def test_to_topic__agent_prefix(self, *, command: AppCommand) -> Any:
        command.import_relative_to_app = Mock()
        res: Any = command.to_topic('@agent')
        assert res is command.import_relative_to_app.return_value
        command.import_relative_to_app.assert_called_once_with('agent')
        return res

    def test_to_topic__topic_name(self, *, command: AppCommand, ctx: Mock[Any]) -> Any:
        app: Any = ctx.find_root().app
        app.topic = Mock()
        res: Any = command.to_topic('foo')
        assert res is app.topic.return_value
        app.topic.assert_called_once_with('foo')
        return res

    def test_abbreviate_fqdn(self, *, command: AppCommand, ctx: Mock[Any]) -> Any:
        with patch('mode.utils.text.abbr_fqdn') as abbr_fqdn:
            ret: Any = command.abbreviate_fqdn('foo.bar.baz', prefix='prefix')
            assert ret is abbr_fqdn.return_value
            abbr_fqdn.assert_called_once_with(ctx.find_root().app.conf.origin, 'foo.bar.baz', prefix='prefix')
            return ret

    def test_abbreviate_fqdn__no_origin(self, *, command: AppCommand, ctx: Mock[Any]) -> str:
        command.app.conf.origin = None
        return_value: str = command.abbreviate_fqdn('foo')
        assert return_value == ''
        return return_value

    def test_from_handler_no_params(self, *, command: AppCommand) -> None:
        @command.from_handler()
        async def takes_no_args() -> None:
            ...

        @command.from_handler()
        async def takes_self_arg(self: Any) -> None:
            ...