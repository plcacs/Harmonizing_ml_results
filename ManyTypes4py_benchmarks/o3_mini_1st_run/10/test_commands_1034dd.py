import shutil
import pytest
from typing import Any, Tuple, List, Set
from dbt.artifacts.resources.types import NodeType
from dbt.cli.main import dbtRunner
from dbt.cli.types import Command
from dbt.events.types import NoNodesSelected
from dbt.tests.util import run_dbt
from tests.utils import EventCatcher

# Testing different commands against the happy path fixture

commands_to_skip: Set[str] = {
    'clone', 'generate', 'server', 'init', 'list',
    'run-operation', 'show', 'snapshot', 'freshness'
}
commands: List[str] = [command.value for command in Command if command.value not in commands_to_skip]

class TestRunCommands:

    @pytest.fixture(scope='class', autouse=True)
    def drop_snapshots(self, happy_path_project: Any, project_root: str) -> None:
        """
        The snapshots are erroring out, so let's drop them.

        Note: that the `happy_path_fixture_files` are a _class_ based fixture. Thus although this fixture _modifies_ the
        files available to the happy path project, it doesn't affect that fixture for tests in other test classes.
        """
        shutil.rmtree(f'{project_root}/snapshots')

    @pytest.mark.parametrize('dbt_command', [(command,) for command in commands])
    def test_run_commmand(self, happy_path_project: Any, dbt_command: Tuple[str, ...]) -> None:
        run_dbt([dbt_command])

# Testing command interactions with specific node types

skipped_resource_types: Set[str] = {
    'analysis', 'operation', 'rpc', 'sql_operation', 'doc', 'macro',
    'exposure', 'group', 'unit_test', 'fixture'
}
resource_types: List[str] = [node_type.value for node_type in NodeType if node_type.value not in skipped_resource_types]

class TestSelectResourceType:

    @pytest.fixture(scope='function')
    def catcher(self) -> EventCatcher:
        return EventCatcher(event_to_catch=NoNodesSelected)

    @pytest.fixture(scope='function')
    def runner(self, catcher: EventCatcher) -> dbtRunner:
        return dbtRunner(callbacks=[catcher.catch])

    @pytest.mark.parametrize('resource_type', resource_types)
    def test_select_by_resource_type(
        self,
        resource_type: str,
        happy_path_project: Any,
        runner: dbtRunner,
        catcher: EventCatcher
    ) -> None:
        runner.invoke(['list', '--select', f'resource_type:{resource_type}'])
        assert len(catcher.caught_events) == 0