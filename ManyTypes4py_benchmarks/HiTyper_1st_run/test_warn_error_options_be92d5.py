from typing import Dict, Union
import pytest
from dbt.cli.main import dbtRunner, dbtRunnerResult
from dbt.events.types import DeprecatedModel
from dbt.flags import get_flags
from dbt.tests.util import run_dbt, update_config_file
from dbt_common.events.base_types import EventLevel
from tests.utils import EventCatcher
ModelsDictSpec = Dict[str, Union[str, 'ModelsDictSpec']]
my_model_sql = "SELECT 1 AS id, 'cats are cute' AS description"
schema_yml = '\nversion: 2\nmodels:\n  - name: my_model\n    deprecation_date: 2020-01-01\n'

@pytest.fixture(scope='class')
def models() -> dict[typing.Text, typing.Union[str,dict[str, str]]]:
    return {'my_model.sql': my_model_sql, 'schema.yml': schema_yml}

@pytest.fixture(scope='function')
def catcher() -> EventCatcher:
    return EventCatcher(event_to_catch=DeprecatedModel)

@pytest.fixture(scope='function')
def runner(catcher: Union[str, Tracer, dict]) -> dbtRunner:
    return dbtRunner(callbacks=[catcher.catch])

def assert_deprecation_warning(result: Union[typing.Callable, bool, typing.Iterable[typing.Callable]], catcher: Union[list[dict[str, typing.Any]], list[dict], list[list[int]]]) -> None:
    assert result.success
    assert result.exception is None
    assert len(catcher.caught_events) == 1
    assert catcher.caught_events[0].info.level == EventLevel.WARN.value

def assert_deprecation_error(result: Union[dict, apistar.types.ReturnValue]) -> None:
    assert not result.success
    assert result.exception is not None
    assert 'Model my_model has passed its deprecation date of' in str(result.exception)

class TestWarnErrorOptionsFromCLI:

    def test_can_silence(self, project: Union[bool, typing.Callable[typing.Any, None], None, str], catcher: Union[tuple[str], bool, str], runner: Union[typing.MutableSequence, str]) -> None:
        result = runner.invoke(['run'])
        assert_deprecation_warning(result, catcher)
        catcher.flush()
        result = runner.invoke(['run', '--warn-error-options', "{'silence': ['DeprecatedModel']}"])
        assert result.success
        assert len(catcher.caught_events) == 0

    def test_can_raise_warning_to_error(self, project: Union[bool, typing.Callable[typing.Any, None], None, str], catcher: Union[bool, str], runner: tests.e2e.Helper) -> None:
        result = runner.invoke(['run'])
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

    def test_can_exclude_specific_event(self, project: Union[typing.Callable[typing.Any, None], None, bool, str], catcher: Union[str, bool], runner: Union[tests.e2e.Helper, dict[str, str], None]) -> None:
        result = runner.invoke(['run', '--warn-error-options', "{'include': 'all'}"])
        assert_deprecation_error(result)
        catcher.flush()
        result = runner.invoke(['run', '--warn-error-options', "{'include': 'all', 'exclude': ['DeprecatedModel']}"])
        assert_deprecation_warning(result, catcher)
        catcher.flush()
        result = runner.invoke(['run', '--warn-error-options', "{'include': 'all', 'warn': ['DeprecatedModel']}"])
        assert_deprecation_warning(result, catcher)

    def test_cant_set_both_include_and_error(self, project: Union[str, typing.Callable[typing.Any, None], None, bool], runner: Union[str, bool]) -> None:
        result = runner.invoke(['run', '--warn-error-options', "{'include': 'all', 'error': 'all'}"])
        assert not result.success
        assert result.exception is not None
        assert 'Only `error` or `include` can be specified' in str(result.exception)

    def test_cant_set_both_exclude_and_warn(self, project: Union[str, typing.Callable[typing.Any, None], None, bool], runner: Union[bool, str]) -> None:
        result = runner.invoke(['run', '--warn-error-options', "{'include': 'all', 'exclude': ['DeprecatedModel'], 'warn': ['DeprecatedModel']}"])
        assert not result.success
        assert result.exception is not None
        assert 'Only `warn` or `exclude` can be specified' in str(result.exception)

class TestWarnErrorOptionsFromProject:

    @pytest.fixture(scope='function')
    def clear_project_flags(self, project_root: str) -> None:
        flags = {'flags': {}}
        update_config_file(flags, project_root, 'dbt_project.yml')

    def test_can_silence(self, project: Union[bool, typing.Callable[typing.Any, None], None, str], clear_project_flags: Union[bool, typing.Callable[typing.Any, None], None, str], project_root: Union[str, bool, typing.Callable[typing.Any, None], None], catcher: Union[tuple[str], bool, str], runner: Union[typing.MutableSequence, str]) -> None:
        result = runner.invoke(['run'])
        assert_deprecation_warning(result, catcher)
        silence_options = {'flags': {'warn_error_options': {'silence': ['DeprecatedModel']}}}
        update_config_file(silence_options, project_root, 'dbt_project.yml')
        catcher.flush()
        result = runner.invoke(['run'])
        assert result.success
        assert len(catcher.caught_events) == 0

    def test_can_raise_warning_to_error(self, project: Union[bool, typing.Callable[typing.Any, None], None, str], clear_project_flags: Union[bool, typing.Callable[typing.Any, None], None, str], project_root: Union[str, bool], catcher: Union[bool, str], runner: tests.e2e.Helper) -> None:
        result = runner.invoke(['run'])
        assert_deprecation_warning(result, catcher)
        include_options = {'flags': {'warn_error_options': {'include': ['DeprecatedModel']}}}
        update_config_file(include_options, project_root, 'dbt_project.yml')
        catcher.flush()
        result = runner.invoke(['run'])
        assert_deprecation_error(result)
        include_options = {'flags': {'warn_error_options': {'include': 'all'}}}
        update_config_file(include_options, project_root, 'dbt_project.yml')
        catcher.flush()
        result = runner.invoke(['run'])
        assert_deprecation_error(result)

    def test_can_exclude_specific_event(self, project: Union[typing.Callable[typing.Any, None], None, bool, str], clear_project_flags: Union[typing.Callable[typing.Any, None], None, bool, str], project_root: str, catcher: Union[str, bool], runner: Union[tests.e2e.Helper, dict[str, str], None]) -> None:
        include_options = {'flags': {'warn_error_options': {'include': 'all'}}}
        update_config_file(include_options, project_root, 'dbt_project.yml')
        result = runner.invoke(['run'])
        assert_deprecation_error(result)
        exclude_options = {'flags': {'warn_error_options': {'include': 'all', 'exclude': ['DeprecatedModel']}}}
        update_config_file(exclude_options, project_root, 'dbt_project.yml')
        catcher.flush()
        result = runner.invoke(['run'])
        assert_deprecation_warning(result, catcher)

    def test_cant_set_both_include_and_error(self, project: Union[str, typing.Callable[typing.Any, None], None, bool], clear_project_flags: Union[str, typing.Callable[typing.Any, None], None, bool], project_root: Union[str, typing.Callable[typing.Any, None], None, bool], runner: Union[str, bool]) -> None:
        exclude_options = {'flags': {'warn_error_options': {'include': 'all', 'error': 'all'}}}
        update_config_file(exclude_options, project_root, 'dbt_project.yml')
        result = runner.invoke(['run'])
        assert not result.success
        assert result.exception is not None
        assert 'Only `error` or `include` can be specified' in str(result.exception)

    def test_cant_set_both_exclude_and_warn(self, project: Union[str, typing.Callable[typing.Any, None], None, bool], clear_project_flags: Union[str, typing.Callable[typing.Any, None], None, bool], project_root: Union[str, typing.Callable[str, None]], runner: Union[bool, str]) -> None:
        exclude_options = {'flags': {'warn_error_options': {'include': 'all', 'exclude': ['DeprecatedModel'], 'warn': ['DeprecatedModel']}}}
        update_config_file(exclude_options, project_root, 'dbt_project.yml')
        result = runner.invoke(['run'])
        assert not result.success
        assert result.exception is not None
        assert 'Only `warn` or `exclude` can be specified' in str(result.exception)

class TestEmptyWarnError:

    @pytest.fixture(scope='class')
    def models(self) -> dict[typing.Text, typing.Union[str,dict[str, str]]]:
        return {'my_model.sql': my_model_sql, 'schema.yml': schema_yml}

    def test_project_flags(self, project: Any) -> None:
        project_flags = {'flags': {'send_anonymous_usage_stats': False, 'warn_error_options': {'warn': None, 'error': None, 'silence': ['TestsConfigDeprecation']}}}
        update_config_file(project_flags, project.project_root, 'dbt_project.yml')
        run_dbt(['run'])
        flags = get_flags()
        assert flags.warn_error_options.silence == ['TestsConfigDeprecation']