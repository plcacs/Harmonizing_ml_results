from __future__ import annotations
import logging
from collections import namedtuple
from typing import TYPE_CHECKING, Generator
import pytest
from click.testing import CliRunner
from kedro.framework.cli.cli import KedroCLI
from kedro.framework.cli.hooks import cli_hook_impl, get_cli_hook_manager, manager

if TYPE_CHECKING:
    from kedro.framework.startup import ProjectMetadata
    from unittest.mock import MagicMock

logger = logging.getLogger(__name__)
FakeDistribution = namedtuple('FakeDistribution', ['entry_points', 'metadata', 'version'])

@pytest.fixture(autouse=True)
def reset_hook_manager() -> Generator[None, None, None]:
    """Due to singleton nature of the `_cli_hook_manager`, the `_cli_hook_manager`
    must be reset to `None` so that a new `CLIHookManager` gets created at the point
    where `FakeEntryPoint` and `fake_plugin_distribution` exist within the same scope.
    Additionally, this prevents `CLIHookManager` to be set from scope outside of this
    testing module.
    """
    manager._cli_hook_manager = None
    yield
    hook_manager = get_cli_hook_manager()
    plugins = hook_manager.get_plugins()
    for plugin in plugins:
        hook_manager.unregister(plugin)

class FakeEntryPoint:
    name: str = 'fake-plugin'
    group: str = 'kedro.cli_hooks'
    value: str = 'hooks:cli_hooks'

    def load(self) -> type:
        class FakeCLIHooks:
            @cli_hook_impl
            def before_command_run(self, project_metadata: ProjectMetadata, command_args: list[str]) -> None:
                print(f'Before command `{" ".join(command_args)}` run for project {project_metadata}')

            @cli_hook_impl
            def after_command_run(self, project_metadata: ProjectMetadata, command_args: list[str], exit_code: int) -> None:
                print(f'After command `{" ".join(command_args)}` run for project {project_metadata} (exit: {exit_code})')
        return FakeCLIHooks()

@pytest.fixture
def fake_plugin_distribution(mocker: MagicMock) -> FakeDistribution:
    fake_entrypoint = FakeEntryPoint()
    fake_distribution = FakeDistribution(entry_points=(fake_entrypoint,), metadata={'name': fake_entrypoint.name}, version='0.1')
    mocker.patch('importlib.metadata.distributions', return_value=[fake_distribution])
    return fake_distribution

class TestKedroCLIHooks:

    @pytest.mark.parametrize('command, exit_code', [('-V', 0), ('info', 0), ('pipeline list', 2), ('starter', 0)])
    def test_kedro_cli_should_invoke_cli_hooks_from_plugin(self, caplog: pytest.LogCaptureFixture, command: str, exit_code: int, mocker: MagicMock, fake_metadata: ProjectMetadata, fake_plugin_distribution: FakeDistribution, entry_points: list) -> None:
        caplog.set_level(logging.DEBUG, logger='kedro')
        mocker.patch('kedro.framework.cli.cli._is_project', return_value=True)
        mocker.patch('kedro.framework.cli.cli.bootstrap_project', return_value=fake_metadata)
        kedro_cli = KedroCLI(fake_metadata.project_path)
        result = CliRunner().invoke(kedro_cli, [command])
        assert f'Registered CLI hooks from 1 installed plugin(s): {fake_plugin_distribution.metadata["name"]}-{fake_plugin_distribution.version}' in caplog.text
        assert f'Before command `{command}` run for project {fake_metadata}' in result.output
        assert f'After command `{command}` run for project {fake_metadata} (exit: {exit_code})' in result.output
