from typing import Any, Dict, Union
import pytest
from dbt.cli.main import dbtRunner, dbtRunnerResult
from dbt.events.types import DeprecatedModel
from dbt.flags import get_flags
from dbt.tests.util import run_dbt, update_config_file
from dbt_common.events.base_types import EventLevel
from tests.utils import EventCatcher

ModelsDictSpec = Dict[str, Union[str, 'ModelsDictSpec']]

my_model_sql: str = "SELECT 1 AS id, 'cats are cute' AS description"
schema_yml: str = '\nversion: 2\nmodels:\n  - name: my_model\n    deprecation_date: 2020-01-01\n'

@pytest.fixture(scope='class')
def models() -> ModelsDictSpec:
    return {'my_model.sql': my_model_sql, 'schema.yml': schema_yml}

@pytest.fixture(scope='function')
def catcher() -> EventCatcher:
    return EventCatcher(event_to_catch=DeprecatedModel)

@pytest.fixture(scope='function')
def runner(catcher: EventCatcher) -> dbtRunner:
    return dbtRunner(callbacks=[catcher.catch])

def assert_deprecation_warning(result: dbtRunnerResult, catcher: EventCatcher) -> None:
    assert result.success
    assert result.exception is None
    assert len(catcher.caught_events) == 1
    assert catcher.caught_events[0].info.level == EventLevel.WARN.value

def assert_deprecation_error(result: dbtRunnerResult) -> None:
    assert not result.success
    assert result.exception is not None
    assert 'Model my_model has passed its deprecation date of' in str(result.exception)

class TestWarnErrorOptionsFromCLI:

    def test_can_silence(self, project: Any, catcher: EventCatcher, runner: dbtRunner) -> None:
        result: dbtRunnerResult = runner.invoke(['run'])
        assert_deprecation_warning(result, catcher)
        catcher.flush()
        result = runner.invoke(['run', '--warn-error-options', "{'silence': ['DeprecatedModel']}"])
        assert result.success
        assert len(catcher.caught_events) == 0

    def test_can_raise_warning_to_error(self, project: Any, catcher: EventCatcher, runner: dbtRunner) -> None:
        result: dbtRunnerResult = runner.invoke(['run'])
        assert_deprecation_warning(result, catcher)
        catcher.flush()

        result = runner.invoke(['run', '--warn-error-options', "{'include': ['DeprecatedModel']}"])
        assert_deprecation_error(result)
        catcher.flush()

        result = runner.invoke(['run', '--warn-error-options', "{'include': 'all'}"])
        assert_deprecation_error(result)
        catcher.flush()

        result = runner.invoke(['run', '--warn-error-options', "{'error': ['DeprecatedModel']}"])
        assert_deprecation_error(result)
        catcher.flush()

        result = runner.invoke(['run', '--warn-error-options', "{'error': 'all'}"])
        assert_deprecation_error(result)

    def test_can_exclude_specific_event(self, project: Any, catcher: EventCatcher, runner: dbtRunner) -> None:
        result: dbtRunnerResult = runner.invoke(['run', '--warn-error-options', "{'include': 'all'}"])
        assert_deprecation_error(result)
        catcher.flush()

        result = runner.invoke(['run', '--warn-error-options', "{'include': 'all', 'exclude': ['DeprecatedModel']}"])
        assert_deprecation_warning(result, catcher)
        catcher.flush()

        result = runner.invoke(['run', '--warn-error-options', "{'include': 'all', 'warn': ['DeprecatedModel']}"])
        assert_deprecation_warning(result, catcher)

    def test_cant_set_both_include_and_error(self, project: Any, runner: dbtRunner) -> None:
        result: dbtRunnerResult = runner.invoke(['run', '--warn-error-options', "{'include': 'all', 'error': 'all'}"])
        assert not result.success
        assert result.exception is not None
        assert 'Only `error` or `include` can be specified' in str(result.exception)

    def test_cant_set_both_exclude_and_warn(self, project: Any, runner: dbtRunner) -> None:
        result: dbtRunnerResult = runner.invoke(
            ['run', '--warn-error-options', "{'include': 'all', 'exclude': ['DeprecatedModel'], 'warn': ['DeprecatedModel']}"]
        )
        assert not result.success
        assert result.exception is not None
        assert 'Only `warn` or `exclude` can be specified' in str(result.exception)

class TestWarnErrorOptionsFromProject:

    @pytest.fixture(scope='function')
    def clear_project_flags(self, project_root: str) -> None:
        flags: Dict[str, Any] = {'flags': {}}
        update_config_file(flags, project_root, 'dbt_project.yml')

    def test_can_silence(self, project: Any, clear_project_flags: None, project_root: str, catcher: EventCatcher, runner: dbtRunner) -> None:
        result: dbtRunnerResult = runner.invoke(['run'])
        assert_deprecation_warning(result, catcher)
        silence_options: Dict[str, Any] = {'flags': {'warn_error_options': {'silence': ['DeprecatedModel']}}}
        update_config_file(silence_options, project_root, 'dbt_project.yml')
        catcher.flush()
        result = runner.invoke(['run'])
        assert result.success
        assert len(catcher.caught_events) == 0

    def test_can_raise_warning_to_error(self, project: Any, clear_project_flags: None, project_root: str, catcher: EventCatcher, runner: dbtRunner) -> None:
        result: dbtRunnerResult = runner.invoke(['run'])
        assert_deprecation_warning(result, catcher)
        include_options: Dict[str, Any] = {'flags': {'warn_error_options': {'include': ['DeprecatedModel']}}}
        update_config_file(include_options, project_root, 'dbt_project.yml')
        catcher.flush()
        result = runner.invoke(['run'])
        assert_deprecation_error(result)

        include_options = {'flags': {'warn_error_options': {'include': 'all'}}}
        update_config_file(include_options, project_root, 'dbt_project.yml')
        catcher.flush()
        result = runner.invoke(['run'])
        assert_deprecation_error(result)

    def test_can_exclude_specific_event(self, project: Any, clear_project_flags: None, project_root: str, catcher: EventCatcher, runner: dbtRunner) -> None:
        include_options: Dict[str, Any] = {'flags': {'warn_error_options': {'include': 'all'}}}
        update_config_file(include_options, project_root, 'dbt_project.yml')
        result: dbtRunnerResult = runner.invoke(['run'])
        assert_deprecation_error(result)
        exclude_options: Dict[str, Any] = {'flags': {'warn_error_options': {'include': 'all', 'exclude': ['DeprecatedModel']}}}
        update_config_file(exclude_options, project_root, 'dbt_project.yml')
        catcher.flush()
        result = runner.invoke(['run'])
        assert_deprecation_warning(result, catcher)

    def test_cant_set_both_include_and_error(self, project: Any, clear_project_flags: None, project_root: str, runner: dbtRunner) -> None:
        options: Dict[str, Any] = {'flags': {'warn_error_options': {'include': 'all', 'error': 'all'}}}
        update_config_file(options, project_root, 'dbt_project.yml')
        result: dbtRunnerResult = runner.invoke(['run'])
        assert not result.success
        assert result.exception is not None
        assert 'Only `error` or `include` can be specified' in str(result.exception)

    def test_cant_set_both_exclude_and_warn(self, project: Any, clear_project_flags: None, project_root: str, runner: dbtRunner) -> None:
        options: Dict[str, Any] = {'flags': {'warn_error_options': {'include': 'all', 'exclude': ['DeprecatedModel'], 'warn': ['DeprecatedModel']}}}
        update_config_file(options, project_root, 'dbt_project.yml')
        result: dbtRunnerResult = runner.invoke(['run'])
        assert not result.success
        assert result.exception is not None
        assert 'Only `warn` or `exclude` can be specified' in str(result.exception)

class TestEmptyWarnError:

    @pytest.fixture(scope='class')
    def models(self) -> ModelsDictSpec:
        return {'my_model.sql': my_model_sql, 'schema.yml': schema_yml}

    def test_project_flags(self, project: Any) -> None:
        project_flags: Dict[str, Any] = {
            'flags': {
                'send_anonymous_usage_stats': False,
                'warn_error_options': {'warn': None, 'error': None, 'silence': ['TestsConfigDeprecation']}
            }
        }
        update_config_file(project_flags, project.project_root, 'dbt_project.yml')
        run_dbt(['run'])
        flags: Any = get_flags()
        assert flags.warn_error_options.silence == ['TestsConfigDeprecation']