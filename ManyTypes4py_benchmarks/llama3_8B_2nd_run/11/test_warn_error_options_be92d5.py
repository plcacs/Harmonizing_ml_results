from typing import Dict, Union, Fixture, Any
import pytest
from dbt.cli.main import dbtRunner, dbtRunnerResult
from dbt.events.types import DeprecatedModel
from dbt.flags import get_flags
from dbt.tests.util import run_dbt, update_config_file
from dbt_common.events.base_types import EventLevel
from tests.utils import EventCatcher

ModelsDictSpec: Dict[str, Union[str, 'ModelsDictSpec']] = Dict[str, Union[str, 'ModelsDictSpec']]
my_model_sql: str = "SELECT 1 AS id, 'cats are cute' AS description"
schema_yml: str = '\nversion: 2\nmodels:\n  - name: my_model\n    deprecation_date: 2020-01-01\n'

@pytest.fixture(scope='class')
def models() -> Dict[str, str]:
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
    # ...

class TestWarnErrorOptionsFromProject:
    # ...

class TestEmptyWarnError:
    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'my_model.sql': my_model_sql, 'schema.yml': schema_yml}

    # ...
