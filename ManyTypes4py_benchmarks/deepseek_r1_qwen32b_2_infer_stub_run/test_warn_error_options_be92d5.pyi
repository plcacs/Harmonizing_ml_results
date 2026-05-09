from typing import Dict, Union, List, Optional
import pytest
from dbt.cli.main import dbtRunner, dbtRunnerResult
from dbt.events.types import DeprecatedModel
from dbt.flags import get_flags
from dbt.tests.util import run_dbt, update_config_file
from dbt_common.events.base_types import EventLevel
from tests.utils import EventCatcher

ModelsDictSpec = Dict[str, Union[str, 'ModelsDictSpec']]

@pytest.fixture(scope='class')
def models() -> ModelsDictSpec:
    ...

@pytest.fixture(scope='function')
def catcher() -> EventCatcher:
    ...

@pytest.fixture(scope='function')
def runner(catcher: EventCatcher) -> dbtRunner:
    ...

def assert_deprecation_warning(result: dbtRunnerResult, catcher: EventCatcher) -> None:
    ...

def assert_deprecation_error(result: dbtRunnerResult) -> None:
    ...

class TestWarnErrorOptionsFromCLI:
    def test_can_silence(self, project: object, catcher: EventCatcher, runner: dbtRunner) -> None:
        ...

    def test_can_raise_warning_to_error(self, project: object, catcher: EventCatcher, runner: dbtRunner) -> None:
        ...

    def test_can_exclude_specific_event(self, project: object, catcher: EventCatcher, runner: dbtRunner) -> None:
        ...

    def test_cant_set_both_include_and_error(self, project: object, runner: dbtRunner) -> None:
        ...

    def test_cant_set_both_exclude_and_warn(self, project: object, runner: dbtRunner) -> None:
        ...

class TestWarnErrorOptionsFromProject:
    @pytest.fixture(scope='function')
    def clear_project_flags(self, project_root: object) -> None:
        ...

    def test_can_silence(self, project: object, clear_project_flags: object, project_root: object, catcher: EventCatcher, runner: dbtRunner) -> None:
        ...

    def test_can_raise_warning_to_error(self, project: object, clear_project_flags: object, project_root: object, catcher: EventCatcher, runner: dbtRunner) -> None:
        ...

    def test_can_exclude_specific_event(self, project: object, clear_project_flags: object, project_root: object, catcher: EventCatcher, runner: dbtRunner) -> None:
        ...

    def test_cant_set_both_include_and_error(self, project: object, clear_project_flags: object, project_root: object, runner: dbtRunner) -> None:
        ...

    def test_cant_set_both_exclude_and_warn(self, project: object, clear_project_flags: object, project_root: object, runner: dbtRunner) -> None:
        ...

class TestEmptyWarnError:
    @pytest.fixture(scope='class')
    def models(self) -> ModelsDictSpec:
        ...

    def test_project_flags(self, project: object) -> None:
        ...